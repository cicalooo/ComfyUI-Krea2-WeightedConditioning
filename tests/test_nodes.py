import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from krea2_weighted.attn_patch import compose_attn_mask
from krea2_weighted.nodes import (
    GUIDE_DEFAULT_FOCUS,
    IDENTITY_DEFAULT_FOCUS,
    Krea2Encode,
    Krea2EncodeOptions,
    Krea2MultiRef,
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    _compact_refs,
    _grounding_images,
    _k2edit_template,
    _prep_grounding_image,
    K2EDIT_DEFAULT_SYSTEM,
    resolve_layout,
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
    assert list(NODE_CLASS_MAPPINGS) == [
        "Krea2Encode",
        "Krea2EncodeOptions",
        "Krea2MultiRef",
    ]
    assert "Krea2PromptMix" not in NODE_CLASS_MAPPINGS
    assert NODE_DISPLAY_NAME_MAPPINGS["Krea2Encode"] == "Krea2 Split Encode"
    assert NODE_DISPLAY_NAME_MAPPINGS["Krea2EncodeOptions"] == "Split Encode Options"
    assert NODE_DISPLAY_NAME_MAPPINGS["Krea2MultiRef"] == "Split Ref"
    types = Krea2Encode.INPUT_TYPES()
    assert "clip" in types["required"]
    assert "prompt" in types["required"]
    assert "aux" in types["required"]
    assert "identity" in types["optional"]
    assert "scene" in types["optional"]
    assert "image" in types["optional"]
    assert "image_b" in types["optional"]
    assert "refs" in types["optional"]


def test_encode_scales_aux_only():
    node = Krea2Encode()
    model = FakeModel()
    clip = FakeClip()
    out_model, cond, debug = node.encode(
        clip,
        model,
        prompt="a woman",
        aux="cinematic grain",
        aux_strength=0.45,
    )
    assert out_model is not model
    tw = out_model.model_options["transformer_options"]["krea2_token_weights"]
    assert tw
    assert all(abs(p[1] - 0.45) < 1e-6 for p in tw)
    assert "aux id" in debug


def test_encode_zero_strength_matches_standard_text_encode():
    node = Krea2Encode()
    model = FakeModel()
    clip = FakeClip()

    standard_clip = FakeClip()
    standard_tokens = standard_clip.tokenize("a woman")
    standard_cond = standard_clip.encode_from_tokens_scheduled(standard_tokens)

    out_model, cond, debug = node.encode(
        clip,
        model,
        prompt="a woman",
        aux="cinematic grain",
        aux_strength=0.0,
    )

    assert out_model is model
    assert torch.equal(cond[0][0], standard_cond[0][0])
    assert cond[0][0].shape == standard_cond[0][0].shape
    assert debug == "aux_strength=0.0; main only, no patch"


def test_encode_zero_strength_matches_grounded_empty_aux():
    node = Krea2Encode()
    image = _rgb(32, 32)

    zero_model = FakeModel()
    zero_clip = FakeClip()
    zero_out, zero_cond, zero_debug = node.encode(
        zero_clip,
        zero_model,
        prompt="recolor the car",
        aux="cinematic grain",
        aux_strength=0.0,
        image=image,
    )

    empty_model = FakeModel()
    empty_clip = FakeClip()
    empty_out, empty_cond, _ = node.encode(
        empty_clip,
        empty_model,
        prompt="recolor the car",
        aux="",
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


def test_encode_grounded_one_image():
    node = Krea2Encode()
    model = FakeModel()
    clip = FakeClip()
    img = _rgb(32, 32)
    out_model, cond, debug = node.encode(
        clip,
        model,
        prompt="recolor the car",
        aux="cinematic grain",
        aux_strength=0.45,
        image=img,
    )
    assert out_model is not model
    assert "grounded 1 image" in debug
    tw = out_model.model_options["transformer_options"]["krea2_token_weights"]
    assert tw
    assert all(abs(p[1] - 0.45) < 1e-6 for p in tw)
    n_vis_markers = 2
    vis_block = n_vis_markers + VISION_EXPAND
    aux_pos = [p[0] for p in tw]
    assert min(aux_pos) >= vis_block
    assert clip.last_images is not None and len(clip.last_images) == 1
    assert K2EDIT_DEFAULT_SYSTEM.split()[0] in (clip.last_template or "")


def test_encode_two_image_ordering():
    node = Krea2Encode()
    clip = FakeClip()
    scene = _rgb(32, 32, 0.1)
    subject = _rgb(32, 32, 0.9)
    node.encode(
        clip,
        FakeModel(),
        prompt="put the person in the room",
        aux="film still",
        aux_strength=0.4,
        image=scene,
        image_b=subject,
    )
    assert len(clip.last_images) == 2
    assert float(clip.last_images[0].reshape(-1)[0]) < float(clip.last_images[1].reshape(-1)[0])
    vis = "<|vision_start|><|image_pad|><|vision_end|>"
    assert clip.last_template.count(vis) == 2


def test_encode_aux_after_vision_expansion_only_aux_text():
    node = Krea2Encode()
    clip = FakeClip()
    img = _rgb(28, 28)
    out_model, cond, debug = node.encode(
        clip,
        FakeModel(),
        prompt="edit the jacket",
        aux="neon",
        aux_strength=0.3,
        image=img,
    )
    tw = out_model.model_options["transformer_options"]["krea2_token_weights"]
    aux_pos = [p[0] for p in tw]
    vis_block = 2 + VISION_EXPAND
    assert min(aux_pos) >= vis_block
    assert max(aux_pos) < cond[0][0].shape[1]
    assert not any(p < vis_block for p in aux_pos)


def test_encode_empty_aux_still_grounded():
    node = Krea2Encode()
    model = FakeModel()
    clip = FakeClip()
    img = _rgb(32, 32)
    out_model, cond, debug = node.encode(
        clip,
        model,
        prompt="recolor the car",
        aux="",
        aux_strength=0.45,
        image=img,
    )
    assert out_model is model
    assert "empty aux" in debug
    assert "grounded 1 image" in debug
    assert cond[0][0].shape[1] > 4
    assert clip.last_images is not None


def test_encode_aux_strength_one_grounded_no_patch():
    node = Krea2Encode()
    model = FakeModel()
    clip = FakeClip()
    img = _rgb(32, 32)
    out_model, cond, debug = node.encode(
        clip,
        model,
        prompt="recolor the car",
        aux="cinematic grain",
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
    ref[:, :, 5:, 2:4] = 1.5
    kb = torch.zeros(1, L)
    kb[:, 3] = 0.2
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


def test_compact_refs_skips_holes():
    a = _rgb(8, 8, 0.1)
    b = _rgb(8, 8, 0.9)
    out = _compact_refs([
        {"image": a, "strength": 1.0, "focus": "face", "mask": None},
        {"image": None, "strength": 1.0, "focus": "", "mask": None},
        {"image": b, "strength": 0.8, "focus": "pose", "mask": None},
    ])
    assert len(out) == 2
    assert out[0]["focus"] == "face"
    assert out[1]["focus"] == "pose"
    assert abs(out[1]["strength"] - 0.8) < 1e-6


def test_multi_ref_build_compacts():
    node = Krea2MultiRef()
    a = _rgb(8, 8, 0.2)
    c = _rgb(8, 8, 0.8)
    (refs,) = node.build(image_1=a, image_3=c, strength_3=0.4)
    assert len(refs) == 2
    assert abs(refs[1]["strength"] - 0.4) < 1e-6


def test_encode_options_passthrough():
    opts_node = Krea2EncodeOptions()
    (opts,) = opts_node.build(people_count_lock="solo", emphasis_mode="both")
    node = Krea2Encode()
    out_model, cond, debug = node.encode(
        FakeClip(),
        FakeModel(),
        prompt="a woman",
        aux="photo",
        aux_strength=0.45,
        options=opts,
    )
    assert "people_count_lock=solo" in debug
    assert "people_lock injected" in debug


def test_k2edit_template_per_image_focus():
    tpl = _k2edit_template(2, "", image_prompts=["focus on face", "focus on pose"])
    assert "focus on face" in tpl
    assert "focus on pose" in tpl
    assert tpl.index("focus on face") < tpl.index("<|image_pad|>")


def test_layout_scene_identity_guides():
    scene = _rgb(8, 8, 0.1)
    person = _rgb(8, 8, 0.9)
    dress = _rgb(8, 8, 0.3)
    hat = _rgb(8, 8, 0.4)
    dit, qwen, notes = resolve_layout(
        identity=person,
        scene=scene,
        refs=[{"image": dress}, {"image": hat}],
        ref_boost=4.0,
    )
    assert [r["role"] for r in dit] == ["identity"]
    assert [r["role"] for r in qwen] == ["scene", "guide", "guide", "identity"]
    assert abs(dit[0]["boost"] - 4.0) < 1e-6
    assert qwen[1]["focus"] == GUIDE_DEFAULT_FOCUS
    assert qwen[-1]["focus"] == IDENTITY_DEFAULT_FOCUS
    assert "scene is Qwen-only" in "\n".join(notes)


def test_layout_aliases_lose_to_named_sockets():
    scene = _rgb(8, 8, 0.1)
    person = _rgb(8, 8, 0.9)
    wrong_a = _rgb(8, 8, 0.2)
    wrong_b = _rgb(8, 8, 0.3)
    dit, qwen, notes = resolve_layout(
        identity=person,
        scene=scene,
        image=wrong_a,
        image_b=wrong_b,
        ref_boost=4.0,
    )
    assert dit[0]["image"] is scene
    assert dit[1]["image"] is person
    assert any("wins over image_b" in n for n in notes)
    assert any("wins over image" in n for n in notes)


def test_layout_multiref_slot1_becomes_identity_when_missing():
    person = _rgb(8, 8, 0.9)
    dress = _rgb(8, 8, 0.3)
    dit, qwen, notes = resolve_layout(
        refs=[{"image": person, "strength": 1.0}, {"image": dress}],
        ref_boost=4.0,
    )
    assert [r["role"] for r in dit] == ["identity"]
    assert dit[0]["image"] is person
    assert abs(dit[0]["boost"] - 4.0) < 1e-6
    assert [r["role"] for r in qwen] == ["guide", "identity"]
    assert any("slot 1 used as identity" in n for n in notes)


def test_layout_multiref_does_not_replace_connected_identity():
    person = _rgb(8, 8, 0.9)
    dress = _rgb(8, 8, 0.3)
    dit, qwen, _ = resolve_layout(
        identity=person,
        refs=[{"image": dress}],
        ref_boost=4.0,
    )
    assert [r["role"] for r in dit] == ["identity"]
    assert dit[0]["image"] is person
    assert [r["role"] for r in qwen] == ["guide", "identity"]
    assert qwen[0]["image"] is dress


def test_encode_identity_is_last_qwen_image():
    person = _rgb(32, 32, 0.9)
    dress = _rgb(32, 32, 0.1)
    (refs,) = Krea2MultiRef().build(image_1=dress)
    clip = FakeClip()
    Krea2Encode().encode(
        clip,
        FakeModel(),
        prompt="restage this person wearing the dress",
        aux="",
        aux_strength=0.45,
        identity=person,
        refs=refs,
    )
    assert len(clip.last_images) == 2
    assert float(clip.last_images[-1].reshape(-1)[0]) > float(clip.last_images[0].reshape(-1)[0])
    assert GUIDE_DEFAULT_FOCUS.split(";")[0] in (clip.last_template or "")
    assert IDENTITY_DEFAULT_FOCUS.split(";")[0] in (clip.last_template or "")


def test_example_workflow_json():
    path = Path(__file__).resolve().parents[1] / "workflows" / "krea2_encode_identity.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert "nodes" in data and "links" in data
    custom = [
        n["type"]
        for n in data["nodes"]
        if str(n.get("type", "")).startswith("Krea2")
    ]
    allowed = set(NODE_CLASS_MAPPINGS)
    assert custom
    assert set(custom) <= allowed
    assert "Krea2PromptMix" not in custom
    assert "Krea2Encode" in custom
    assert "Krea2MultiRef" in custom
    types = {n["type"] for n in data["nodes"]}
    assert "Krea2EditModelPatch" not in types
    assert "VAEEncode" not in types
