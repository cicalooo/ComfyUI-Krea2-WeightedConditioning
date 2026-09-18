import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from krea2_weighted.attn_patch import compose_attn_mask
from krea2_weighted.nodes import (
    Krea2PromptMix,
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    _grounding_images,
    _k2edit_template,
    _prep_grounding_image,
    K2EDIT_DEFAULT_SYSTEM,
)
from krea2_weighted.tokenize import (
    QWEN_IM_END,
    QWEN_IM_START,
    QWEN_NL,
    QWEN_USER,
)


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


VISION_EXPAND = 4  # tokens each image_pad becomes after Qwen expansion


class FakeClip:
    def __init__(self):
        self.last_images = None
        self.last_template = None

    def tokenize(self, text, images=None, llama_template=None, **kwargs):
        self.last_images = images
        self.last_template = llama_template
        parts = text.split()
        user = []
        for i, w in enumerate(parts):
            key = (" " + w) if (i > 0 or text.startswith(" ")) else w
            user.append(200 + (abs(hash(key)) % 50000))
        vision = []
        if images:
            for img in images:
                vision.extend(
                    [
                        151652,
                        {"type": "image", "data": img, "original_type": "image"},
                        151653,
                    ]
                )
        ids = [QWEN_IM_START, QWEN_USER, QWEN_NL] + vision + user + [QWEN_IM_END]
        return {"qwen3vl_4b": [[(i, 1.0) for i in ids]]}

    def encode_from_tokens_scheduled(self, tok):
        key = next(iter(tok))
        raw = [t[0] for t in tok[key][0]]
        visible = []
        for e in raw[3:]:  # drop im_start,user,nl
            if isinstance(e, dict) and e.get("type") == "image":
                visible.extend([0] * VISION_EXPAND)
            else:
                visible.append(0)
        t = torch.zeros(1, len(visible), 8)
        return [[t, {}]]


def _rgb(h, w, val=0.2):
    return torch.full((1, h, w, 3), val)


def test_mappings():
    assert list(NODE_CLASS_MAPPINGS) == ["Krea2PromptMix"]
    assert NODE_DISPLAY_NAME_MAPPINGS["Krea2PromptMix"] == "Krea2 Prompt Mix"
    types = Krea2PromptMix.INPUT_TYPES()
    assert "clip" in types["required"]
    assert "text_main" in types["required"]
    assert "text_aux" in types["required"]
    assert "image" in types["optional"]
    assert "image_b" in types["optional"]
    assert "grounding_px" in types["optional"]
    assert "system_prompt" in types["optional"]


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


def test_prompt_mix_zero_strength_matches_standard_text_encode():
    node = Krea2PromptMix()
    model = FakeModel()
    clip = FakeClip()

    standard_clip = FakeClip()
    standard_tokens = standard_clip.tokenize("a woman")
    standard_cond = standard_clip.encode_from_tokens_scheduled(standard_tokens)

    out_model, cond, debug = node.encode(
        clip,
        model,
        text_main="a woman",
        text_aux="cinematic grain",
        aux_strength=0.0,
    )

    assert out_model is model
    assert torch.equal(cond[0][0], standard_cond[0][0])
    assert cond[0][0].shape == standard_cond[0][0].shape
    assert debug == "aux_strength=0.0; main only, no patch"


def test_prompt_mix_zero_strength_matches_grounded_empty_aux():
    node = Krea2PromptMix()
    image = _rgb(32, 32)

    zero_model = FakeModel()
    zero_clip = FakeClip()
    zero_out, zero_cond, zero_debug = node.encode(
        zero_clip,
        zero_model,
        text_main="recolor the car",
        text_aux="cinematic grain",
        aux_strength=0.0,
        image=image,
    )

    empty_model = FakeModel()
    empty_clip = FakeClip()
    empty_out, empty_cond, _ = node.encode(
        empty_clip,
        empty_model,
        text_main="recolor the car",
        text_aux="",
        aux_strength=0.45,
        image=image,
    )

    assert zero_out is zero_model
    assert empty_out is empty_model
    assert torch.equal(zero_cond[0][0], empty_cond[0][0])
    assert zero_cond[0][0].shape == empty_cond[0][0].shape
    assert len(zero_clip.last_images) == 1
    assert zero_clip.last_template == empty_clip.last_template
    assert zero_debug == "aux_strength=0.0; main only, no patch; grounded 1 image(s)"


def test_prompt_mix_grounded_one_image():
    node = Krea2PromptMix()
    model = FakeModel()
    clip = FakeClip()
    img = _rgb(32, 32)
    out_model, cond, debug = node.encode(
        clip,
        model,
        text_main="recolor the car",
        text_aux="cinematic grain",
        aux_strength=0.45,
        image=img,
    )
    assert out_model is not model
    assert "grounded 1 image" in debug
    tw = out_model.model_options["transformer_options"]["krea2_token_weights"]
    assert tw
    assert all(abs(p[1] - 0.45) < 1e-6 for p in tw)
    # vision tokens occupy the front of the cond window
    n_vis_markers = 2  # vis_start + vis_end (pad expands)
    vis_block = n_vis_markers + VISION_EXPAND
    aux_pos = [p[0] for p in tw]
    assert min(aux_pos) >= vis_block
    assert clip.last_images is not None and len(clip.last_images) == 1
    assert K2EDIT_DEFAULT_SYSTEM.split()[0] in (clip.last_template or "")


def test_prompt_mix_two_image_ordering():
    node = Krea2PromptMix()
    clip = FakeClip()
    scene = _rgb(32, 32, 0.1)
    subject = _rgb(32, 32, 0.9)
    node.encode(
        clip,
        FakeModel(),
        text_main="put the person in the room",
        text_aux="film still",
        aux_strength=0.4,
        image=scene,
        image_b=subject,
    )
    assert len(clip.last_images) == 2
    # scene (image) first, subject (image_b) second
    assert float(clip.last_images[0].reshape(-1)[0]) < float(clip.last_images[1].reshape(-1)[0])
    vis = "<|vision_start|><|image_pad|><|vision_end|>"
    assert clip.last_template.count(vis) == 2


def test_prompt_mix_aux_after_vision_expansion_only_aux_text():
    node = Krea2PromptMix()
    clip = FakeClip()
    img = _rgb(28, 28)
    out_model, cond, debug = node.encode(
        clip,
        FakeModel(),
        text_main="edit the jacket",
        text_aux="neon",
        aux_strength=0.3,
        image=img,
    )
    tw = out_model.model_options["transformer_options"]["krea2_token_weights"]
    aux_pos = [p[0] for p in tw]
    vis_block = 2 + VISION_EXPAND
    assert min(aux_pos) >= vis_block
    # cond: vis_start, 4 vision, vis_end, main tokens..., aux..., im_end
    # "edit the jacket" = 3 tokens, "neon" = 1, plus im_end
    assert max(aux_pos) < cond[0][0].shape[1]
    # do not weight the vision block
    assert not any(p < vis_block for p in aux_pos)


def test_prompt_mix_empty_aux_still_grounded():
    node = Krea2PromptMix()
    model = FakeModel()
    clip = FakeClip()
    img = _rgb(32, 32)
    out_model, cond, debug = node.encode(
        clip,
        model,
        text_main="recolor the car",
        text_aux="",
        aux_strength=0.45,
        image=img,
    )
    assert out_model is model
    assert "empty aux" in debug
    assert "grounded 1 image" in debug
    assert cond[0][0].shape[1] > 4  # vision expansion present
    assert clip.last_images is not None


def test_prompt_mix_aux_strength_one_grounded_no_patch():
    node = Krea2PromptMix()
    model = FakeModel()
    clip = FakeClip()
    img = _rgb(32, 32)
    out_model, cond, debug = node.encode(
        clip,
        model,
        text_main="recolor the car",
        text_aux="cinematic grain",
        aux_strength=1.0,
        image=img,
    )
    assert out_model is model
    assert "aux_strength=1.0" in debug
    assert "grounded 1 image" in debug
    assert cond[0][0].shape[1] > 4


def test_k2edit_ref_mask_composes_with_key_bias():
    L = 8
    ref = torch.zeros(1, 1, L, L)
    ref[:, :, 5:, 2:4] = 1.5  # K2Edit ref_boost on ref key columns
    kb = torch.zeros(1, L)
    kb[:, 3] = 0.2  # Prompt Mix aux key bias
    out = compose_attn_mask(ref, kb)
    assert abs(float(out[0, 0, 5, 3]) - 1.7) < 1e-5
    assert abs(float(out[0, 0, 5, 2]) - 1.5) < 1e-5
    assert abs(float(out[0, 0, 0, 3]) - 0.2) < 1e-5
    assert compose_attn_mask(None, kb) is kb
    assert compose_attn_mask(ref, None) is ref


def test_prep_28_pixel_grid_and_template():
    big = _rgb(200, 100)
    out = _prep_grounding_image(big, 56)
    h, w = out.shape[1], out.shape[2]
    assert h % 28 == 0 and w % 28 == 0
    assert max(h, w) <= 56
    imgs = _grounding_images(big, _rgb(40, 40, 0.8), 768)
    assert len(imgs) == 2
    tpl = _k2edit_template(2, "")
    assert K2EDIT_DEFAULT_SYSTEM in tpl
    assert tpl.count("<|image_pad|>") == 2


