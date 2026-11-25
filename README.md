# pairwise-pathtrace-firefly

## 领域

游戏、图形与多媒体

## 主题

路径追踪萤火虫噪声抑制

## 说明

能量异常检测、曝光归一化阈值、样本数加权累积。

初始状态下 `graphics/firefly_filter.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
