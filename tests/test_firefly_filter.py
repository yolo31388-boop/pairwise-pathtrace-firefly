import pytest
import graphics.firefly_filter as mod

CASES = [
    ("firefly_energy_detect", (1,)),
    ("exposure_norm_threshold", (1,)),
    ("sample_weighted_accum", (1,)),
    ("buffer_merge_deterministic", (1,)),
    ("variance_clip", (1,)),
    ("history_validate", (1,)),
    ("load_pathtrace", (1,)),
    ("trace_paths", (1,)),
    ("denoise_firefly", (1,)),
    ("present_pathtrace", (1,)),
    ("validate_pathtrace", (1,)),
    ("metrics_pathtrace", (1,)),
    ("audit_pathtrace", (1,)),
    ("fallback_pathtrace", (1,)),
    ("determinism_pathtrace", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
