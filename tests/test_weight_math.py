import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from krea2_weighted.attn_patch import APPLY_TO_KEY, _should_apply, compose_attn_mask
from krea2_weighted.tokenize import mix_factors


def test_mix_factors():
    assert mix_factors(0.45, "value_scale") == (0.45, 0.0)
    assert mix_factors(1.0, "value_scale") == (1.0, 0.0)
    assert mix_factors(0.5, "k_bias") == (0.5, 0.0)
    assert mix_factors(1.5, "k_bias") == (1.0, 1.5)
    assert mix_factors(1.5, "both") == (1.375, 1.5)


def test_compose_preserves_ref_boost():
    ref = torch.zeros(1, 1, 4, 4)
    ref[:, :, 2:, 1:3] = 2.0
    kb = torch.tensor([[0.0, 0.5, 0.0, 0.0]])
    out = compose_attn_mask(ref, kb)
    assert float(out[0, 0, 2, 1]) == 2.5
    assert float(out[0, 0, 2, 2]) == 2.0


def test_apply_to_cond():
    assert _should_apply({APPLY_TO_KEY: "cond", "cond_or_uncond": [0]}) is True
    assert _should_apply({APPLY_TO_KEY: "cond", "cond_or_uncond": [1]}) is False
    assert _should_apply({APPLY_TO_KEY: "cond", "cond_or_uncond": [1, 0]}) is True
    assert _should_apply({APPLY_TO_KEY: "uncond", "cond_or_uncond": [1]}) is True
    assert _should_apply({APPLY_TO_KEY: "both", "cond_or_uncond": [1]}) is True
    assert _should_apply({}) is True
