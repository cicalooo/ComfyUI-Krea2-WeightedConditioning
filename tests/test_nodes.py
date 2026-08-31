import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from krea2_weighted.nodes import (
    Krea2ConditioningMix,
    Krea2PromptMix,
    Krea2WeightedConditioning,
    NODE_CLASS_MAPPINGS,
)
from krea2_weighted.tokenize import QWEN_IM_END, QWEN_IM_START, QWEN_NL, QWEN_USER


class FakeAttn:
    pass


class FakeBlock:
    def __init__(self):
        self.attn = FakeAttn()


class FakeDiff:
    def __init__(self, n=4):
        self.blocks = [FakeBlock() for _ in range(n)]


class FakeModel:
    def __init__(self):
        self.model_options = {"transformer_options": {}}
        self.diff = FakeDiff()
        self.patches = []

    def clone(self):
        other = FakeModel()
        other.model_options = {"transformer_options": dict(self.model_options.get("transformer_options", {}))}
        other.diff = self.diff
        return other

    def get_model_object(self, name):
        assert name == "diffusion_model"
        return self.diff

    def add_object_patch(self, path, fn):
        self.patches.append((path, fn))


class FakeClip:
    def tokenize(self, text):
        parts = text.split()
        user = []
        for i, w in enumerate(parts):
            key = (" " + w) if (i > 0 or text.startswith(" ")) else w
            # Isolated "ghost" uses an id that never appears in a longer prompt,
            # simulating a BPE mismatch (the unmatched=error path).
            if w == "ghost" and len(parts) == 1:
                user.append(99999)
            else:
                user.append(200 + (abs(hash(key)) % 50000))
        ids = [QWEN_IM_START, QWEN_USER, QWEN_NL] + user + [QWEN_IM_END]
        return {"qwen3vl_4b": [[(i, 1.0) for i in ids]]}

    def encode_from_tokens_scheduled(self, tok):
        key = next(iter(tok))
        ids = [t[0] for t in tok[key][0]]
        visible = ids[3:]  # drop im_start,user,nl
        t = torch.zeros(1, len(visible), 8)
        return [[t, {}]]


def test_mappings():
    assert "Krea2WeightedConditioning" in NODE_CLASS_MAPPINGS
    assert "Krea2PromptMix" in NODE_CLASS_MAPPINGS
    assert "Krea2ConditioningMix" in NODE_CLASS_MAPPINGS
    types = Krea2WeightedConditioning.INPUT_TYPES()
    assert "clip" in types["required"]
    assert "text" in types["required"]


def test_encode_no_weights_passthrough():
    node = Krea2WeightedConditioning()
    model = FakeModel()
    clip = FakeClip()
    out_model, cond, debug = node.encode(
        clip, model, "a red hat", strength=1.0
    )
    assert out_model is model
    assert cond[0][0].shape[1] > 0
    assert "no (phrase:weight)" in debug


def test_encode_patches_blocks():
    node = Krea2WeightedConditioning()
    model = FakeModel()
    clip = FakeClip()
    out_model, cond, debug = node.encode(
        clip, model, "a (red:1.5) hat", strength=1.0, unmatched="warn"
    )
    assert out_model is not model
    assert out_model.patches
    assert all(p[0].endswith("attn.forward") for p in out_model.patches)
    tw = out_model.model_options["transformer_options"]["krea2_token_weights"]
    assert tw
    assert "red" in debug


def test_prompt_mix_scales_aux_only():
    node = Krea2PromptMix()
    model = FakeModel()
    clip = FakeClip()
    out_model, cond, debug = node.encode(
        clip,
        model,
        text_main="a woman",
        text_aux="cinematic grain",
        aux_strength=0.45,
    )
    assert out_model is not model
    tw = out_model.model_options["transformer_options"]["krea2_token_weights"]
    assert tw
    assert all(abs(p[1] - 0.45) < 1e-6 for p in tw)
    assert "aux id" in debug


def test_conditioning_mix_offsets_positions():
    node = Krea2ConditioningMix()
    model = FakeModel()
    main = [[torch.zeros(1, 4, 8), {}]]
    aux = [[torch.zeros(1, 3, 8), {}]]
    out_model, combined, debug = node.mix(model, main, aux, aux_strength=0.5)
    assert combined[0][0].shape[1] == 7
    tw = out_model.model_options["transformer_options"]["krea2_token_weights"]
    assert [p[0] for p in tw] == [4, 5, 6]


def test_unmatched_error_from_node():
    node = Krea2WeightedConditioning()
    try:
        node.encode(
            FakeClip(), FakeModel(), "a (ghost:2) hat", strength=1.0, unmatched="error"
        )
    except ValueError:
        return
    raise AssertionError("expected ValueError")
