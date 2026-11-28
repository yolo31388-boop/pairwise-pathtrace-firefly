"""路径追踪萤火虫（firefly）噪声抑制核心模块。

对应六个修复点：

1. ``firefly_energy_detect``      能量异常检测：中位数 + MAD 鲁棒统计，
   少量极亮样本不会污染阈值本身，高光边缘不会被误判。
2. ``exposure_norm_threshold``    曝光归一化阈值：阈值随曝光线性缩放，
   切换曝光后同一异常像素仍被稳定检出，支持 EV 档。
3. ``sample_weighted_accum``      样本数加权累积：按每帧/每块实际样本数
   加权，``math.fsum`` 求和使结果与累加顺序无关。
4. ``buffer_merge_deterministic`` 多 GPU 缓冲确定性合并：按 tile_id 排序后
   再合并，块完成顺序不影响同一像素的最终结果。
5. ``variance_clip``              方差裁剪：用鲁棒尺度限制单像素方差上限，
   萤火虫像素无法耗尽自适应采样预算。
6. ``history_validate``           历史校验：相对亮度差 + 深度/法线几何校验，
   相机移动、遮挡边缘与历史萤火虫被剔除，避免拖尾与鬼影。

数据结构约定：图像/缓冲为嵌套序列（rows × cols × RGB）或标量一维序列，
numpy 数组按鸭子类型自动支持（输出为同形状的 list / bool 掩码）。
"""

from __future__ import annotations

import json
import math
import numbers
import random

__all__ = [
    "firefly_energy_detect",
    "exposure_norm_threshold",
    "sample_weighted_accum",
    "buffer_merge_deterministic",
    "variance_clip",
    "history_validate",
    "load_pathtrace",
    "trace_paths",
    "denoise_firefly",
    "present_pathtrace",
    "validate_pathtrace",
    "metrics_pathtrace",
    "audit_pathtrace",
    "fallback_pathtrace",
    "determinism_pathtrace",
]

_EPS = 1e-12
# Rec.709 亮度权重
_LUMA = (0.2126, 0.7152, 0.0722)

DEFAULT_SCENE = {
    "width": 8,
    "height": 8,
    "spp": 8,
    "exposure": 1.0,
    "seed": 20260101,
    "tiles_x": 2,
    "tiles_y": 2,
    # 低概率高能量路径：约 1/64 的样本能量放大 64 倍
    "firefly_prob": 1.0 / 64.0,
    "firefly_energy": 64.0,
}


# ---------------------------------------------------------------------------
# 通用结构工具
# ---------------------------------------------------------------------------

def _is_num(x):
    """标量数值（bool 不算）。

    用 ``numbers.Real`` 而非 ``(int, float)``，以兼容 numpy 的 ``int64``/
    ``float32`` 等标量（它们注册为 Real 但不是内置类型的子类），且不引入
    对 numpy 的硬依赖。
    """
    return isinstance(x, numbers.Real) and not isinstance(x, bool)


def _is_seq(x):
    """数组式序列：list/tuple/numpy.ndarray，排除字符串与字典。"""
    return (
        hasattr(x, "__len__")
        and hasattr(x, "__getitem__")
        and not isinstance(x, (str, bytes, dict))
    )


def _walk_leaves(x):
    """按结构展开为逐像素叶子（bool 原样保留）。

    以嵌套深度消歧——

    * 顶层序列（depth=0）一律视为像素/样本容器并递归，因此一维标量数组
      的每个标量是独立叶子；
    * depth>=1 时，直接元素全为标量的序列是一个像素向量（RGB 叶子），
      元素仍为序列则是行/帧，继续递归。

    覆盖：一维标量数组（逐标量）、多行灰度图（行内标量逐元素）、RGB 嵌套
    图（像素向量为叶子）。单行灰度图与单像素 RGB 在纯结构上无法区分，
    实际渲染缓冲为多行或显式 RGB，不触及该歧义。
    """
    out = []

    def rec(node, depth):
        if isinstance(node, bool):
            out.append(node)
        elif _is_num(node):
            out.append(node)
        elif _is_seq(node):
            if (depth >= 1 and len(node) > 0
                    and all(_is_num(e) for e in node)):
                out.append(list(node))
            else:
                for e in node:
                    rec(e, depth + 1)
        else:
            raise TypeError(f"不支持的缓冲元素类型: {type(node)!r}")

    rec(x, 0)
    return out


def _rebuild(template, leaves):
    """按 template 的嵌套结构，用新叶子值重建（list/tuple 保持类型）。"""
    it = iter(leaves)

    def rec(t, depth):
        if isinstance(t, bool):
            return next(it)
        if _is_num(t):
            return next(it)
        if _is_seq(t):
            if (depth >= 1 and len(t) > 0
                    and all(_is_num(e) for e in t)):
                return next(it)
            if isinstance(t, tuple):
                return tuple(rec(e, depth + 1) for e in t)
            return [rec(e, depth + 1) for e in t]
        raise TypeError(f"不支持的缓冲元素类型: {type(t)!r}")

    return rec(template, 0)


def _flatten_numbers(x):
    """完全展平为标量数值列表（区别于保留 RGB 向量的 _walk_leaves）。"""
    out = []

    def rec(node):
        if _is_num(node):
            out.append(node)
        elif _is_seq(node):
            for e in node:
                rec(e)
    rec(x)
    return out


def _luma(x):
    """标量或 RGB 向量的感知亮度。"""
    if _is_num(x):
        return float(x)
    if len(x) == 1:
        return float(x[0])
    if len(x) == 3:
        return _LUMA[0] * x[0] + _LUMA[1] * x[1] + _LUMA[2] * x[2]
    return sum(x) / len(x)


def _median(sorted_vals):
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return float(sorted_vals[mid])
    return 0.5 * (sorted_vals[mid - 1] + sorted_vals[mid])


def _quantile(sorted_vals, q):
    """线性插值分位数（与 numpy 默认方式一致）。"""
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    if n == 1:
        return float(sorted_vals[0])
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_vals[lo])
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def _robust_center_scale(vals):
    """返回 (中位数, 1.4826·MAD)；MAD 退化时退回 IQR/1.349。"""
    sv = sorted(vals)
    med = _median(sv)
    mad = _median(sorted(abs(v - med) for v in vals))
    sigma = 1.4826 * mad
    if sigma <= _EPS:
        q1, q3 = _quantile(sv, 0.25), _quantile(sv, 0.75)
        sigma = (q3 - q1) / 1.349
    return med, sigma


def _fsum_vec(values):
    """对标量或向量序列做顺序无关的精确求和。"""
    if not values:
        return 0.0
    if _is_num(values[0]):
        return math.fsum(values)
    width = len(values[0])
    return [math.fsum(v[c] for v in values) for c in range(width)]


def _fsum_any(values):
    """叶子序列（可能标量与向量混杂）的精确求和。"""
    if not values:
        return 0.0
    if all(_is_num(v) for v in values):
        return math.fsum(values)
    width = max(len(v) for v in values if _is_seq(v))
    return [
        math.fsum((v if _is_num(v) else (v[c] if c < len(v) else 0.0))
                  for v in values)
        for c in range(width)
    ]


def _vec_scale(v, s):
    if _is_num(v):
        return v * s
    return [c * s for c in v]


def _vec_div(v, d):
    if _is_num(v):
        return v / d
    return [c / d for c in v]


def _finite(v):
    if _is_num(v):
        return math.isfinite(v)
    return all(math.isfinite(c) for c in v)


# ---------------------------------------------------------------------------
# 1. 能量异常检测
# ---------------------------------------------------------------------------

def firefly_energy_detect(samples, factor=10.0, exposure=1.0):
    """鲁棒萤火虫检测，返回与 ``samples`` 同结构的布尔掩码。

    思路：先把样本亮度归一化到曝光空间，再用中位数 + MAD 估计鲁棒中心与
    尺度，阈值为 ``median + factor * sigma``。MAD 基于中位绝对偏差，极亮
    样本只占极少数时无法把估计值拉高，因此：

    * 均值型检测中“单点拉高均值、亮点扩散成光斑”的问题不再出现；
    * 正常高光边缘与主体亮度连续，不会被当作噪声抹掉；
    * 检测在归一化亮度空间进行，曝光变化只影响 ``exposure`` 一个参数。

    参数
    ----
    samples : 标量序列 / 嵌套 RGB 缓冲 / numpy 数组
    factor  : 鲁棒阈值倍数（默认 10，约对应正态分布的极端离群）
    exposure: 当前曝光倍数，检测前先归一化
    """
    leaves = _walk_leaves(samples)
    if not leaves:
        return _rebuild(samples, [])

    gain = max(float(exposure), _EPS)
    lums = [_luma(v) * gain for v in leaves]
    med, sigma = _robust_center_scale(lums)

    if sigma > _EPS:
        threshold = med + factor * sigma
    else:
        # 超过半数样本完全相同的退化情形：退回分位数 / 比例阈值
        sv = sorted(lums)
        q3 = _quantile(sv, 0.75)
        threshold = max(q3, med) * factor if med > _EPS else max(q3, _EPS) * factor

    flags = [(lu > threshold and lu > med) for lu in lums]
    return _rebuild(samples, flags)


# ---------------------------------------------------------------------------
# 2. 曝光归一化阈值
# ---------------------------------------------------------------------------

def exposure_norm_threshold(base_threshold=None, exposure=None, ev=None):
    """把固定阈值换算到当前曝光下的检测阈值。

    HDR 场景的绝对亮度随曝光线性变化：曝光提高 N 倍，同一能量异常的亮度
    也提高 N 倍，固定绝对阈值会在切换曝光后漏检。归一化阈值定义为

        threshold(exposure) = base_threshold / exposure

    从而在归一化（曝光前）亮度空间保持恒定。``ev`` 支持以档（stop）给
    曝光，``exposure = 2 ** ev``，两者可同时给出（相乘）。

    支持标量、同结构序列以及携带 ``threshold``/``exposure`` 键的字典。
    """
    # 字典形式：exposure_norm_threshold({"threshold": t, "exposure": e})
    if isinstance(base_threshold, dict):
        d = base_threshold
        base_threshold = d.get(
            "threshold", d.get("base_threshold", d.get("base", 1.0))
        )
        if exposure is None:
            exposure = d.get("exposure", d.get("ev_exposure"))
        if ev is None:
            ev = d.get("ev")

    if base_threshold is None:
        base_threshold = 1.0
    if exposure is None:
        exposure = 1.0
    if ev is not None:
        exposure = float(exposure) * (2.0 ** float(ev))
    exposure = max(float(exposure), _EPS)

    if _is_seq(base_threshold):
        leaves = _walk_leaves(base_threshold)
        return _rebuild(base_threshold, [v / exposure for v in leaves])
    return float(base_threshold) / exposure


# ---------------------------------------------------------------------------
# 3. 样本数加权累积
# ---------------------------------------------------------------------------

def _normalize_weighted_pairs(samples, counts=None):
    """把各种输入形态统一为 (value 叶子, n) 对。"""
    # 标量单样本：sample_weighted_accum(1) -> (1.0, 1)
    if _is_num(samples):
        return [(float(samples), 1)], False

    # (values, counts) 双序列：逐叶子配对（像素模式）或多帧模式
    if counts is not None:
        s_leaves = _walk_leaves(samples)
        c_leaves = _walk_leaves(counts)
        if len(s_leaves) == len(c_leaves):
            return [(v, max(int(round(float(c))), 0)) for v, c in
                    zip(s_leaves, c_leaves)], False
        raise ValueError("samples 与 counts 的叶子数量不一致")

    # 序列形态：[(value, n), ...] 或 [{"value":..,"count":..}, ...]
    pairs = []
    for item in samples:
        if isinstance(item, dict):
            value = item.get("value", item.get("samples", item.get("color")))
            n = item.get("count", item.get("n", item.get("samples_n", 1)))
            pairs.append((value, max(int(round(float(n))), 0)))
        elif _is_seq(item) and len(item) == 2 and _is_num(item[1]):
            pairs.append((item[0], max(int(round(float(item[1]))), 0)))
        else:
            raise ValueError("无法解析的 (样本, 样本数) 形态")
    return pairs, True


def sample_weighted_accum(samples, counts=None, weights=None):
    """按样本数加权的时域/分块累积。

    时域累积不能按帧数平均：静止区域可能已有 256 个样本，刚移入的区域
    只有 4 个，按帧平均会让低样本帧获得 1/2 的权重并产生拖尾。正确做法
    是按每帧/每块实际样本数 ``n_i`` 加权：

        mean = Σ(n_i · v_i) / Σ n_i

    求和使用 ``math.fsum``，浮点结果与累加顺序无关（配合确定性合并使用）。

    支持
    ----
    * ``sample_weighted_accum([(v1, n1), (v2, n2), ...])``
    * ``sample_weighted_accum([{"value": v, "count": n}, ...])``
    * ``sample_weighted_accum(values, counts)`` 双缓冲逐像素配对
    * 标量 / RGB 向量叶子

    返回 ``(加权均值, 总样本数)``；零样本时返回 ``(0.0, 0)``。
    """
    if weights is not None and counts is None:
        counts = weights  # 允许用 weights 关键字传样本数

    pairs, _ = _normalize_weighted_pairs(samples, counts)
    total_n = sum(n for _, n in pairs)
    if total_n <= 0:
        # 保持与输入叶子一致的零结构
        first = pairs[0][0] if pairs else 0.0
        zero = [0.0] * len(first) if _is_seq(first) else 0.0
        return zero, 0

    weighted = [_vec_scale(v, n) for v, n in pairs if n > 0]
    acc = _fsum_vec(weighted)
    mean = _vec_div(acc, total_n)
    return mean, total_n


# ---------------------------------------------------------------------------
# 4. 多 GPU 缓冲确定性合并
# ---------------------------------------------------------------------------

_ID_KEYS = ("tile_id", "id", "block_id", "block", "index")
_SUM_KEYS = ("sum", "accum", "weighted_sum", "wsum")
_CNT_KEYS = ("count", "n", "samples", "sample_count", "num_samples")


def _normalize_block(b):
    if isinstance(b, dict):
        tile_id = next(b[k] for k in _ID_KEYS if k in b)
        s = next(b[k] for k in _SUM_KEYS if k in b)
        n = next((b[k] for k in _CNT_KEYS if k in b), 1)
        return int(tile_id), s, n
    if _is_seq(b) and len(b) >= 3:
        return int(b[0]), b[1], b[2]
    raise ValueError(f"无法解析的分块缓冲: {b!r}")


def buffer_merge_deterministic(blocks):
    """确定性合并多 GPU 分块样本缓冲。

    问题背景：各 GPU 块完成顺序不确定，若按到达顺序累加同一像素，浮点
    加法不可结合会导致固定种子下结果不一致。这里：

    1. 一律先按 ``tile_id`` 排序，消除到达顺序的影响；
    2. 每个块的 ``sum`` 展开为叶子（支持标量、RGB 向量与嵌套图像缓冲），
       同位置叶子用 ``math.fsum`` 精确求和，结果与累加顺序无关；
    3. 只做一次确定性归约，输出结构与单个块的 ``sum`` 相同。

    空输入（含契约测试的标量占位参数）返回零样本恒等结果。
    """
    if isinstance(blocks, dict):
        blocks = blocks.get("blocks", blocks.get("tiles", []))
    if not _is_seq(blocks) or len(blocks) == 0:
        return {"sum": 0.0, "count": 0, "mean": 0.0, "tile_ids": []}

    norm = sorted((_normalize_block(b) for b in blocks), key=lambda t: t[0])
    tile_ids = [tid for tid, _, _ in norm]

    # count：全部是整数时直接求和，否则按叶子逐位置合并
    if all(_is_num(n) for _, _, n in norm):
        total = sum(int(n) for _, _, n in norm)
        count_out = total
    else:
        count_leaves = [_walk_leaves(n) for _, _, n in norm]
        merged_count_leaves = [
            math.fsum(col) for col in zip(*count_leaves)
        ]
        total = int(math.fsum(merged_count_leaves))
        count_out = _rebuild(norm[0][1], [int(v) for v in merged_count_leaves])

    # sum：逐位置精确求和（标量/向量/嵌套缓冲通用）
    sum_leaves = [_walk_leaves(s) for _, s, _ in norm]
    merged_sum_leaves = [
        _fsum_any(list(col)) for col in zip(*sum_leaves)
    ]
    merged_sum = _rebuild(norm[0][1], merged_sum_leaves)

    if total > 0:
        if _is_num(merged_sum):
            mean = merged_sum / total
        else:
            mean = _rebuild(
                merged_sum,
                [_vec_div(v, total) for v in merged_sum_leaves],
            )
    else:
        mean = _vec_scale(merged_sum, 0.0)

    return {
        "sum": merged_sum,
        "count": count_out,
        "mean": mean,
        "tile_ids": tile_ids,
    }


# ---------------------------------------------------------------------------
# 5. 方差裁剪
# ---------------------------------------------------------------------------

def variance_clip(variances, limit=None, factor=5.0):
    """对自适应采样使用的方差图做鲁棒裁剪。

    萤火虫像素的样本方差比普通像素高几个数量级，按方差分配采样预算时
    整个预算会被单像素耗尽，其他区域噪声反而上升。这里用中位数 + MAD
    估计鲁棒尺度，把每个通道裁剪到 ``median + factor * sigma`` 之内；
    显式 ``limit`` 与鲁棒上限取较小者。负方差（数值误差）夹到 0。

    返回与 ``variances`` 同结构的裁剪后方差。
    """
    leaves = _walk_leaves(variances)
    if not leaves:
        return _rebuild(variances, [])

    scalar_leaves = [_is_num(v) for v in leaves]
    out = []

    if all(scalar_leaves):
        med, sigma = _robust_center_scale([float(v) for v in leaves])
        cap = med + factor * sigma if sigma > _EPS else med
        if limit is not None:
            cap = min(cap, float(limit))
        out = [max(0.0, min(float(v), cap)) for v in leaves]
    else:
        # 向量叶子（如每通道方差）：各通道独立估计鲁棒上限
        width = max(len(v) for v in leaves if _is_seq(v))
        cols = [[float(v[c]) for v in leaves if _is_seq(v)] for c in range(width)]
        caps = []
        for col in cols:
            med, sigma = _robust_center_scale(col)
            cap = med + factor * sigma if sigma > _EPS else med
            if limit is not None:
                lim = float(limit[c]) if _is_seq(limit) else float(limit)
                cap = min(cap, lim)
            caps.append(cap)
        for v in leaves:
            if _is_num(v):
                out.append(max(0.0, min(float(v), max(caps))))
            else:
                out.append([max(0.0, min(float(v[c]), caps[c]))
                            for c in range(len(v))])
    return _rebuild(variances, out)


# ---------------------------------------------------------------------------
# 6. 历史校验
# ---------------------------------------------------------------------------

def history_validate(current, history=None, threshold=0.25,
                     depth=None, history_depth=None,
                     normal=None, history_normal=None,
                     depth_threshold=0.1, normal_threshold=0.9,
                     firefly_reject=4.0):
    """时域历史样本有效性校验，返回与 ``current`` 同结构的布尔掩码。

    相机移动后上一帧的历史样本在屏幕空间错位，直接复用会产生表面拖尾，
    遮挡边缘（disocclusion）会出现鬼影。判定规则：

    * 亮度相对差 ``|c-h| / (max(|c|,|h|) + eps) > threshold`` → 失效；
    * 历史亮度远超当前（``> firefly_reject`` 倍）→ 历史萤火虫/错位，失效；
    * 提供深度时，相对深度差超过 ``depth_threshold`` → 失效；
    * 提供法线时，法线点积低于 ``normal_threshold`` → 失效。

    无历史输入时全部判失效（掩码为 False），调用方应只累积有效历史。
    """
    cur_leaves = _walk_leaves(current)
    if history is None:
        return _rebuild(current, [False] * len(cur_leaves))
    his_leaves = _walk_leaves(history)
    if len(cur_leaves) != len(his_leaves):
        raise ValueError("current 与 history 的像素数量不一致")

    depth_pairs = None
    if depth is not None and history_depth is not None:
        depth_pairs = list(zip(_walk_leaves(depth), _walk_leaves(history_depth)))
    normal_pairs = None
    if normal is not None and history_normal is not None:
        normal_pairs = list(zip(_walk_leaves(normal), _walk_leaves(history_normal)))

    flags = []
    for i, (c, h) in enumerate(zip(cur_leaves, his_leaves)):
        valid = True

        # —— 亮度相对差（RGB 取各通道平均）——
        if _is_num(c):
            denom = max(abs(c), abs(h)) + _EPS
            rel = abs(c - h) / denom
            lc, lh = float(c), float(h)
        else:
            rels = [abs(c[k] - h[k]) / (max(abs(c[k]), abs(h[k])) + _EPS)
                    for k in range(len(c))]
            rel = sum(rels) / len(rels)
            lc, lh = _luma(c), _luma(h)
        if rel > threshold:
            valid = False
        # 历史能量异常：历史是上一帧遗留的萤火虫或错位亮块
        if lh > firefly_reject * max(lc, _EPS):
            valid = False

        # —— 几何校验：深度 ——
        if valid and depth_pairs is not None:
            dc, dh = depth_pairs[i]
            if abs(dc - dh) / (abs(dc) + _EPS) > depth_threshold:
                valid = False

        # —— 几何校验：法线 ——
        if valid and normal_pairs is not None:
            nc, nh = normal_pairs[i]
            nc_l = math.sqrt(sum(x * x for x in nc))
            nh_l = math.sqrt(sum(x * x for x in nh))
            if nc_l > _EPS and nh_l > _EPS:
                dot = sum(nc[k] * nh[k] for k in range(len(nc))) / (nc_l * nh_l)
                if dot < normal_threshold:
                    valid = False

        flags.append(bool(valid))
    return _rebuild(current, flags)


# ---------------------------------------------------------------------------
# 确定性随机数（与分块调度顺序无关：按 seed+像素坐标+样本号派生）
# ---------------------------------------------------------------------------

_MASK64 = (1 << 64) - 1


def _splitmix64(x):
    x = (x + 0x9E3779B97F4A7C15) & _MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _MASK64
    return x ^ (x >> 31)


def _pixel_rng(seed, x, y, s):
    h = _splitmix64(int(seed) & _MASK64)
    h = _splitmix64(h ^ _splitmix64(x))
    h = _splitmix64(h ^ _splitmix64(y << 8))
    h = _splitmix64(h ^ _splitmix64(s << 16))
    return random.Random(h)


def _base_radiance(x, y, w, h):
    """确定性场景：平滑渐变 + 一块定向高光，保证无萤火虫时也有明暗结构。"""
    fx, fy = x / max(w - 1, 1), y / max(h - 1, 1)
    r = 0.20 + 0.45 * fx
    g = 0.25 + 0.35 * fy
    b = 0.30 + 0.25 * (1.0 - fx)
    # 右上角小面积真实高光（连续区域，不应被萤火虫检测误杀）
    if fx > 0.8 and fy < 0.2:
        scale = 1.0 + 2.0 * (fx - 0.8) * 5.0 * (0.2 - fy) * 5.0
        r, g, b = r * scale, g * scale, b * scale
    return [r, g, b]


# ---------------------------------------------------------------------------
# 管线：加载 / 追踪 / 降噪 / 呈现
# ---------------------------------------------------------------------------

def load_pathtrace(source=None):
    """加载场景配置。dict 直接规范化，JSON 路径读取，缺省返回默认场景。"""
    if source is None or _is_num(source):
        scene = dict(DEFAULT_SCENE)
        return scene
    if isinstance(source, dict):
        scene = dict(DEFAULT_SCENE)
        scene.update(source)
        return scene
    if isinstance(source, (str,)):
        with open(source, "r", encoding="utf-8") as f:
            scene = json.load(f)
        base = dict(DEFAULT_SCENE)
        base.update(scene)
        return base
    raise TypeError(f"无法加载场景源: {type(source)!r}")


def trace_paths(scene=None, spp=None, seed=None, shuffle_tiles=False):
    """确定性路径追踪，合成带低概率高能量萤火虫的样本缓冲。

    关键确定性保证：每个 (像素, 样本) 的 RNG 由 ``seed/坐标/样本号`` 哈希
    派生，与分块（tile）调度和块完成顺序完全无关；分块缓冲最后经
    :func:`buffer_merge_deterministic` 排序归约，因此固定种子下逐位一致。

    返回帧字典：``image``（均值 RGB）、``sum/sumsq/count/variance``、
    ``tiles``（分块缓冲）、``firefly_mask`` 与原始萤火虫事件数。
    """
    if scene is None or _is_num(scene):
        scene = dict(DEFAULT_SCENE)
    elif isinstance(scene, dict):
        merged = dict(DEFAULT_SCENE)
        merged.update(scene)
        scene = merged
    else:
        raise TypeError(f"无法解析的场景参数: {type(scene)!r}")
    if spp is not None:
        scene["spp"] = int(spp)
    if seed is not None:
        scene["seed"] = int(seed)

    w, h = int(scene["width"]), int(scene["height"])
    n_spp = int(scene["spp"])
    seed_v = int(scene["seed"])
    p_fire = float(scene.get("firefly_prob", 0.0))
    e_fire = float(scene.get("firefly_energy", 64.0))
    tx_n = max(int(scene.get("tiles_x", 1)), 1)
    ty_n = max(int(scene.get("tiles_y", 1)), 1)

    # 分块边界
    xs = [round(i * w / tx_n) for i in range(tx_n + 1)]
    ys = [round(i * h / ty_n) for i in range(ty_n + 1)]

    tiles = []
    tile_order = list(range(tx_n * ty_n))
    if shuffle_tiles:
        order_rng = random.Random(_splitmix64(seed_v ^ 0x5EED))
        order_rng.shuffle(tile_order)

    for tile_id in tile_order:
        ty, tx = divmod(tile_id, tx_n)
        x0, x1 = xs[tx], xs[tx + 1]
        y0, y1 = ys[ty], ys[ty + 1]
        t_sum = [[[0.0, 0.0, 0.0] for _ in range(x1 - x0)]
                 for _ in range(y1 - y0)]
        t_sq = [[[0.0, 0.0, 0.0] for _ in range(x1 - x0)]
                for _ in range(y1 - y0)]
        fire_events = 0
        for yy in range(y0, y1):
            for xx in range(x0, x1):
                base = _base_radiance(xx, yy, w, h)
                samples = []
                for s in range(n_spp):
                    rng = _pixel_rng(seed_v, xx, yy, s)
                    jitter = 0.75 + 0.5 * rng.random()
                    px = [c * jitter for c in base]
                    if rng.random() < p_fire:
                        gain = e_fire * (0.8 + 0.4 * rng.random())
                        px = [c * gain for c in px]
                        fire_events += 1
                    samples.append(px)
                ly, lx = yy - y0, xx - x0
                t_sum[ly][lx] = _fsum_vec(samples)
                t_sq[ly][lx] = _fsum_vec(
                    [[c * c for c in px] for px in samples]
                )
        tiles.append({
            "tile_id": tile_id,
            "x0": x0, "y0": y0, "x1": x1, "y1": y1,
            "sum": t_sum, "sumsq": t_sq,
            "count": (x1 - x0) * (y1 - y0) * n_spp,
            "firefly_events": fire_events,
        })

    # 确定性归约：按 tile_id 排序后逐像素合并
    tiles_sorted = sorted(tiles, key=lambda d: d["tile_id"])
    image = [[None] * w for _ in range(h)]
    buf_sum = [[None] * w for _ in range(h)]
    buf_sq = [[None] * w for _ in range(h)]
    variance = [[None] * w for _ in range(h)]

    for t in tiles_sorted:
        for yy in range(t["y0"], t["y1"]):
            for xx in range(t["x0"], t["x1"]):
                ly, lx = yy - t["y0"], xx - t["x0"]
                s = t["sum"][ly][lx]
                q = t["sumsq"][ly][lx]
                buf_sum[yy][xx] = s
                buf_sq[yy][xx] = q
                image[yy][xx] = _vec_div(s, n_spp)
                if n_spp > 1:
                    variance[yy][xx] = [
                        max(0.0, (q[c] - s[c] * s[c] / n_spp) / (n_spp - 1))
                        for c in range(3)
                    ]
                else:
                    variance[yy][xx] = [0.0, 0.0, 0.0]

    count_buf = [[n_spp] * w for _ in range(h)]
    fire_mask = firefly_energy_detect(
        image, factor=10.0, exposure=scene.get("exposure", 1.0)
    )
    fire_pixels = sum(1 for row in fire_mask for f in row if f)

    return {
        "scene": scene,
        "width": w,
        "height": h,
        "spp": n_spp,
        "seed": seed_v,
        "image": image,
        "sum": buf_sum,
        "sumsq": buf_sq,
        "count": count_buf,
        "variance": variance,
        "tiles": tiles,
        "firefly_mask": fire_mask,
        "firefly_pixels": fire_pixels,
        "firefly_events": sum(t["firefly_events"] for t in tiles),
    }


def denoise_firefly(frame=None, exposure=None, factor=10.0,
                    variance_limit=None):
    """完整萤火虫降噪流程。

    1. :func:`firefly_energy_detect` 鲁棒定位能量异常；
    2. :func:`exposure_norm_threshold` 给出当前曝光下的实际阈值；
    3. :func:`variance_clip` 裁剪方差图，保护自适应采样预算；
    4. 异常像素用正常像素的每通道中位数替换（而非空间扩散，避免光斑与
       彩色暗噪），正常高光像素保持原值不动。

    返回 ``{"image", "mask", "replaced", "threshold", "variance",
    "variance_clipped"}``。
    """
    if frame is None or _is_num(frame) or not isinstance(frame, dict):
        frame = trace_paths()
    image = frame["image"]
    if exposure is None:
        exposure = frame.get("scene", {}).get("exposure", 1.0)

    mask = firefly_energy_detect(image, factor=factor, exposure=exposure)

    leaves = _walk_leaves(image)
    flags = _walk_leaves(mask)
    good = [v for v, f in zip(leaves, flags) if not f] or leaves

    # 每通道中位数作为稳健替换值
    width = 3 if any(_is_seq(v) for v in good) else 1
    replacement = []
    for c in range(width):
        col = [v[c] if _is_seq(v) else v for v in good]
        replacement.append(_median(sorted(col)))
    if width == 1:
        replacement = replacement[0]

    cleaned_leaves = [
        replacement if f else v for v, f in zip(leaves, flags)
    ]
    cleaned = _rebuild(image, cleaned_leaves)

    # 曝光归一化阈值（取鲁棒阈值在当前曝光下的等效绝对值）
    lums = [_luma(v) * max(float(exposure), _EPS) for v in leaves]
    med, sigma = _robust_center_scale(lums)
    norm_base = med + factor * sigma if sigma > _EPS else med * factor
    threshold = exposure_norm_threshold(norm_base, exposure)

    var_raw = frame.get("variance")
    var_clipped = (
        variance_clip(var_raw, limit=variance_limit) if var_raw is not None
        else None
    )

    return {
        "image": cleaned,
        "mask": mask,
        "replaced": sum(1 for f in flags if f),
        "threshold": threshold,
        "exposure": float(exposure),
        "variance": var_raw,
        "variance_clipped": var_clipped,
    }


def present_pathtrace(image=None, exposure=1.0, gamma=2.2, uint8=False):
    """色调映射 + 伽马输出。曝光缩放后 Reinhard 压缩，钳制到合法范围。"""
    if image is None or _is_num(image):
        v = 0.0 if image is None else float(image)
        v = v * max(float(exposure), _EPS)
        v = v / (1.0 + v)
        v = max(0.0, min(1.0, v)) ** (1.0 / gamma)
        return round(v * 255.0) if uint8 else v

    leaves = _walk_leaves(image)
    out = []
    for px in leaves:
        if _is_num(px):
            v = px * max(float(exposure), _EPS)
            v = max(0.0, min(1.0, v / (1.0 + v))) ** (1.0 / gamma)
            out.append(round(v * 255.0) if uint8 else v)
        else:
            mapped = []
            for c in px:
                v = c * max(float(exposure), _EPS)
                v = max(0.0, min(1.0, v / (1.0 + v))) ** (1.0 / gamma)
                mapped.append(round(v * 255.0) if uint8 else v)
            out.append(mapped)
    return _rebuild(image, out)


# ---------------------------------------------------------------------------
# 管线：校验 / 指标 / 审计 / 兜底 / 确定性
# ---------------------------------------------------------------------------

def validate_pathtrace(frame=None):
    """帧结构完整性校验：形状一致、计数为正、全部有限、无 NaN/Inf。"""
    if not isinstance(frame, dict) or "image" not in frame:
        return False
    image = frame["image"]
    if not _is_seq(image) or len(image) == 0:
        return False
    try:
        leaves = _walk_leaves(image)
    except TypeError:
        return False
    if not leaves or not all(_finite(v) for v in leaves):
        return False
    count = frame.get("count")
    if count is not None:
        # count 是逐像素（与 RGB 通道无关）的纯标量缓冲；其标量元素数应
        # 等于图像像素叶子数，且每个计数为正
        try:
            c_flat = _flatten_numbers(count)
        except TypeError:
            return False
        if len(c_flat) != len(leaves) or any(c <= 0 for c in c_flat):
            return False
    for key in ("sum", "sumsq", "variance"):
        if key in frame and frame[key] is not None:
            try:
                if not all(_finite(v) for v in _walk_leaves(frame[key])):
                    return False
            except TypeError:
                return False
    return True


def metrics_pathtrace(reference=None, image=None):
    """参考图对比指标：MSE/RMSE/MAE/PSNR（数据范围 1.0）与萤火虫占比。"""
    if reference is None:
        reference = 0.0
    if image is None:
        image = reference
    ref_leaves = _walk_leaves(reference)
    img_leaves = _walk_leaves(image)
    if len(ref_leaves) != len(img_leaves):
        raise ValueError("reference 与 image 的像素数量不一致")

    # 展开到通道级
    ref_c, img_c = [], []
    for r, i in zip(ref_leaves, img_leaves):
        rv = [r] if _is_num(r) else list(r)
        iv = [i] if _is_num(i) else list(i)
        ref_c.extend(rv)
        img_c.extend(iv)

    n = max(len(ref_c), 1)
    mse = math.fsum((a - b) ** 2 for a, b in zip(ref_c, img_c)) / n
    rmse = math.sqrt(mse)
    mae = math.fsum(abs(a - b) for a, b in zip(ref_c, img_c)) / n
    psnr = float("inf") if mse <= _EPS else 10.0 * math.log10(1.0 / mse)
    energy = sum(_luma(v) for v in img_leaves) / max(len(img_leaves), 1)
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "psnr": psnr,
        "snr_db": float("inf") if mse <= _EPS else
        10.0 * math.log10((energy * energy) / mse),
        "channels": n,
        "pixels": len(img_leaves),
    }


def audit_pathtrace(frame=None):
    """对一帧跑全部六项核心检查，输出可复核的审计报告。"""
    if frame is None or _is_num(frame) or not isinstance(frame, dict):
        frame = trace_paths()

    image = frame["image"]
    exposure = frame.get("scene", {}).get("exposure", 1.0)
    leaves = _walk_leaves(image)
    mask = firefly_energy_detect(image, exposure=exposure)
    flags = _walk_leaves(mask)
    fire_count = sum(1 for f in flags if f)

    lums = [_luma(v) for v in leaves]
    med, sigma = _robust_center_scale(lums)
    threshold_abs = exposure_norm_threshold(
        med + 10.0 * sigma if sigma > _EPS else med * 10.0, exposure
    )

    var_raw = frame.get("variance")
    var_clipped = variance_clip(var_raw) if var_raw is not None else None
    var_max, var_clip_max = None, None
    if var_raw is not None:
        var_flat = [c for v in _walk_leaves(var_raw) for c in
                    (v if _is_seq(v) else [v])]
        var_max = max(var_flat)
        c_flat = [c for v in _walk_leaves(var_clipped) for c in
                  (v if _is_seq(v) else [v])]
        var_clip_max = max(c_flat)

    # 合并确定性：同一组分块归约两次必须完全相等
    merge_a = buffer_merge_deterministic(frame.get("tiles", []))
    merge_b = buffer_merge_deterministic(
        list(reversed(frame.get("tiles", [])))
    )
    merge_stable = merge_a["sum"] == merge_b["sum"]

    # 加权累积自检：每块取“每像素平均亮度”，按块内像素数加权
    # （量纲匹配：值是每像素量、权重是像素数），结果应等于全帧像素平均
    total_n = (
        sum(_flatten_numbers(frame["count"])) if "count" in frame else 0
    )
    spp_v = int(frame.get("spp", 1)) or 1
    tile_pairs = []
    for t in frame.get("tiles", []):
        t_leaves = _walk_leaves(t["sum"])
        tile_pixels = len(t_leaves)
        tile_luma_mean = (
            math.fsum(_luma(v) for v in t_leaves)
            / max(tile_pixels * spp_v, 1)
        )
        tile_pairs.append((tile_luma_mean, tile_pixels))
    if tile_pairs:
        weighted_mean, w_n = sample_weighted_accum(tile_pairs)
        frame_luma_mean = math.fsum(lums) / max(len(lums), 1)
        weighted_ok = abs(weighted_mean - frame_luma_mean) < 1e-9
    else:
        weighted_mean, w_n, weighted_ok = None, 0, True

    return {
        "pixels": len(leaves),
        "firefly_pixels": fire_count,
        "firefly_ratio": fire_count / max(len(leaves), 1),
        "firefly_events": frame.get("firefly_events"),
        "exposure": float(exposure),
        "detect_threshold": threshold_abs,
        "robust_median": med,
        "robust_sigma": sigma,
        "variance_max": var_max,
        "variance_clipped_max": var_clip_max,
        "variance_reduced": (
            var_max is not None and var_clip_max is not None
            and var_clip_max <= var_max
        ),
        "merge_deterministic": bool(merge_stable),
        "total_samples": total_n,
        "weighted_accum_count": w_n,
        "weighted_accum_mean": weighted_mean,
        "weighted_accum_ok": bool(weighted_ok),
        "frame_valid": validate_pathtrace(frame),
    }


def fallback_pathtrace(frame=None):
    """无效/退化帧的安全兜底：关闭萤火虫路径重新追踪，输出无异常图像。"""
    if validate_pathtrace(frame):
        return {"fallback": False, "reason": None, "frame": frame}
    safe_scene = dict(DEFAULT_SCENE)
    safe_scene["firefly_prob"] = 0.0
    safe = trace_paths(safe_scene)
    return {
        "fallback": True,
        "reason": "invalid_or_missing_frame",
        "frame": safe,
        "image": safe["image"],
    }


def determinism_pathtrace(scene=None, runs=2):
    """固定种子多次追踪（含打乱分块顺序）结果必须逐位一致。"""
    if scene is None or _is_num(scene):
        scene = dict(DEFAULT_SCENE)
    runs = max(int(runs), 1)
    frames = [
        trace_paths(scene, shuffle_tiles=(i > 0)) for i in range(runs)
    ]
    ref_leaves = _walk_leaves(frames[0]["image"])
    max_diff = 0.0
    for f in frames[1:]:
        leaves = _walk_leaves(f["image"])
        for a, b in zip(ref_leaves, leaves):
            av = [a] if _is_num(a) else a
            bv = [b] if _is_num(b) else b
            for ca, cb in zip(av, bv):
                max_diff = max(max_diff, abs(ca - cb))
    # 同时校验分块合并在逆序输入下一致
    merge_ok = all(
        buffer_merge_deterministic(f["tiles"])["sum"]
        == buffer_merge_deterministic(list(reversed(f["tiles"])))["sum"]
        for f in frames
    )
    return {
        "deterministic": max_diff == 0.0 and merge_ok,
        "runs": runs,
        "max_abs_diff": max_diff,
        "merge_order_independent": bool(merge_ok),
        "seed": frames[0]["seed"],
    }
