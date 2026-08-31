import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from krea2_weighted.attn_patch import APPLY_TO_KEY, _should_apply
from krea2_weighted.tokenize import weight_factors


def test_plan_table():
    assert weight_factors(1.0, 1, "k_bias") == (1.0, 0.0)
    assert weight_factors(1.5, 1, "k_bias") == (1.0, 1.0)
    assert weight_factors(0.5, 1, "k_bias") == (0.5, 0.0)
    assert weight_factors(-1.0, 1, "k_bias") == (-1.0, 0.0)
    assert weight_factors(2.0, 0.5, "k_bias") == (1.0, 1.0)


def test_apply_to_cond():
    assert _should_apply({APPLY_TO_KEY: "cond", "cond_or_uncond": [0]}) is True
    assert _should_apply({APPLY_TO_KEY: "cond", "cond_or_uncond": [1]}) is False
    assert _should_apply({APPLY_TO_KEY: "cond", "cond_or_uncond": [1, 0]}) is True
    assert _should_apply({APPLY_TO_KEY: "uncond", "cond_or_uncond": [1]}) is True
    assert _should_apply({APPLY_TO_KEY: "both", "cond_or_uncond": [1]}) is True
    assert _should_apply({}) is True
