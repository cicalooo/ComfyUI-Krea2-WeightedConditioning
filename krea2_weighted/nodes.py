"""ComfyUI node: Krea2 Prompt Mix (optional Mustyrocks K2Edit grounding)."""

from __future__ import annotations

import torch

from .attn_patch import APPLY_TO_KEY, WEIGHTS_KEY, Krea2WeightPatch
from .tokenize import (
    aux_id_span_in_combined,
    clip_token_ids,
    mix_factors,
    parse_block_range,
    slice_to_cond_pairs,
    slice_to_cond_pairs_after_vision,
    token_ids_from_tok,
    user_span_from_ids,
)


def _patch_model(model, pairs, apply_to, block_range):
    model_clone = model.clone()
    diffusion_model = model_clone.get_model_object("diffusion_model")
    blocks = getattr(diffusion_model, "blocks", None)
    if blocks is None:
        raise RuntimeError(
            "Krea2 Prompt Mix: model has no diffusion_model.blocks "
            "(expected Krea 2 SingleStreamDiT)."
        )
    indices = parse_block_range(block_range, len(blocks))
    transformer_options = model_clone.model_options.get("transformer_options", {}).copy()
    transformer_options[WEIGHTS_KEY] = pairs
    transformer_options[APPLY_TO_KEY] = apply_to
    model_clone.model_options["transformer_options"] = transformer_options
    for idx in indices:
        attn = blocks[idx].attn
        patched = Krea2WeightPatch().__get__(attn, attn.__class__)
        model_clone.add_object_patch(
            "diffusion_model.blocks.{}.attn.forward".format(idx), patched
        )
    return model_clone, indices


# Mustyrocks K2Edit Qwen3-VL grounding (semantic path only — no latent/VAE/fit).
K2EDIT_DEFAULT_SYSTEM = (
    "Describe the image by detailing the color, shape, size, "
    "texture, quantity, text, spatial relationships of the objects and background:"
)


def _k2edit_template(nimg, system_prompt=""):
    sp = (system_prompt or "").strip() or K2EDIT_DEFAULT_SYSTEM
    vis = "<|vision_start|><|image_pad|><|vision_end|>" * nimg
    return (
        "<|im_start|>system\n" + sp + "<|im_end|>\n<|im_start|>user\n"
        + vis + "{}<|im_end|>\n<|im_start|>assistant\n"
    )


def _prep_grounding_image(image, grounding_px):
    """Cap longest side and snap to Qwen3-VL's 28-pixel vision grid (patch 14 × merge 2)."""
    samples = image.movedim(-1, 1)  # B,H,W,C -> B,C,H,W
    h, w = samples.shape[2], samples.shape[3]
    if grounding_px and max(h, w) > grounding_px:
        s = grounding_px / max(h, w)
        nw = max(28, round(w * s) // 28 * 28)
        nh = max(28, round(h * s) // 28 * 28)
        try:
            import comfy.utils

            samples = comfy.utils.common_upscale(samples, nw, nh, "area", "disabled")
        except ImportError:
            samples = torch.nn.functional.interpolate(
                samples.float(), size=(nh, nw), mode="area"
            )
    return samples.movedim(1, -1)[:, :, :, :3]


def _grounding_images(image, image_b, grounding_px):
    """Scene first, subject second — Mustyrocks training order."""
    imgs = []
    if image is not None:
        imgs.append(_prep_grounding_image(image, grounding_px))
    if image_b is not None:
        imgs.append(_prep_grounding_image(image_b, grounding_px))
    return imgs


class Krea2PromptMix:
    """Encode main + aux as one K2 prompt; V-scale only the aux (moodboard) tokens."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "model": ("MODEL",),
                "text_main": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Subject / handwritten prompt. Stays at strength 1.0.",
                    },
                ),
                "text_aux": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Moodboard / style prompt to weaken or boost.",
                    },
                ),
                "aux_strength": (
                    "FLOAT",
                    {
                        "default": 0.45,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": (
                            "Strength of the moodboard tokens only. 1.0 = same as main, "
                            "0.45 = typical weaken, 0 = ignore aux. Not CLIP (prompt:0.5)."
                        ),
                    },
                ),
            },
            "optional": {
                "separator": (["newline", "space", "comma"], {"default": "newline"}),
                "emphasis_mode": (
                    ["value_scale", "k_bias", "both"],
                    {"default": "value_scale"},
                ),
                "block_range": ("STRING", {"default": "all"}),
                "apply_to": (["cond", "uncond", "both"], {"default": "cond"}),
                "image": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "Optional K2Edit source (scene). When connected, encodes "
                            "through Mustyrocks Qwen3-VL grounding (vision + instruction)."
                        ),
                    },
                ),
                "image_b": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "Optional second K2Edit reference (subject). Vision order: "
                            "scene (image), then subject (image_b)."
                        ),
                    },
                ),
                "grounding_px": (
                    "INT",
                    {
                        "default": 768,
                        "min": 0,
                        "max": 4096,
                        "step": 64,
                        "tooltip": "Cap longest side fed to Qwen3-VL; 0 = native. Same as Mustyrocks.",
                    },
                ),
                "system_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Override K2Edit grounding system prompt (empty = training default).",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "STRING")
    RETURN_NAMES = ("model", "conditioning", "debug")
    FUNCTION = "encode"
    CATEGORY = "Krea2/conditioning"
    DESCRIPTION = (
        "Keep a handwritten / edit instruction at 1.0 and scale a second (moodboard) prompt. "
        "Encodes both as one Qwen sequence so you do not ConditioningConcat two chat templates. "
        "Optional K2Edit image / image_b grounding (Mustyrocks Qwen3-VL prep). "
        "Wire both MODEL and CONDITIONING. Identity-edit graphs: CFG as Mustyrocks specifies; "
        "text-only mix still uses CFG 1.0. Do not use with Krea2 Apply Regional."
    )

    def encode(
        self,
        clip,
        model,
        text_main,
        text_aux,
        aux_strength,
        separator="newline",
        emphasis_mode="value_scale",
        block_range="all",
        apply_to="cond",
        image=None,
        image_b=None,
        grounding_px=768,
        system_prompt="",
    ):
        main = (text_main or "").strip()
        aux = (text_aux or "").strip()
        sep = {"newline": "\n", "space": " ", "comma": ", "}[separator]
        imgs = _grounding_images(image, image_b, grounding_px)
        grounded = bool(imgs)
        tok_kw = {}
        if grounded:
            tok_kw = {
                "images": imgs,
                "llama_template": _k2edit_template(len(imgs), system_prompt),
            }

        main_only_reason = None
        if not aux:
            main_only_reason = "empty aux"
        elif float(aux_strength) == 0.0:
            main_only_reason = "aux_strength=0.0"

        if main_only_reason is not None:
            tok = clip.tokenize(main, **tok_kw)
            cond = clip.encode_from_tokens_scheduled(tok)
            msg = "{}; main only, no patch".format(main_only_reason)
            if grounded:
                msg += "; grounded {} image(s)".format(len(imgs))
            return (model, cond, msg)

        combined = main + sep + aux if main else aux
        tok = clip.tokenize(combined, **tok_kw)
        ids = token_ids_from_tok(tok)
        cond = clip.encode_from_tokens_scheduled(tok)
        cond_len = cond[0][0].shape[1]
        visible_start = len(ids) - cond_len

        v_factor, k_bias = mix_factors(aux_strength, emphasis_mode)
        if abs(v_factor - 1.0) < 1e-6 and abs(k_bias) < 1e-6:
            msg = "aux_strength=1.0; no patch"
            if grounded:
                msg += "; grounded {} image(s)".format(len(imgs))
            return (model, cond, msg)

        if not main:
            cs, ce = user_span_from_ids(ids)
            if grounded:
                pairs = slice_to_cond_pairs_after_vision(
                    ids, cs, ce, cond_len, v_factor, k_bias
                )
            else:
                pairs = slice_to_cond_pairs(cs, ce, visible_start, cond_len, v_factor, k_bias)
            debug = "main empty; scaled all user tokens"
        else:
            ids_main = clip_token_ids(clip, main, **tok_kw)
            aux_s, aux_e = aux_id_span_in_combined(ids, ids_main)
            if grounded:
                pairs = slice_to_cond_pairs_after_vision(
                    ids, aux_s, aux_e, cond_len, v_factor, k_bias
                )
            else:
                pairs = slice_to_cond_pairs(
                    aux_s, aux_e, visible_start, cond_len, v_factor, k_bias
                )
            debug = "aux id[{}:{}] → {} cond tokens v={:.4f} k_bias={:.4f}".format(
                aux_s, aux_e, len(pairs), v_factor, k_bias
            )

        if grounded:
            debug += "; grounded {} image(s)".format(len(imgs))

        if not pairs:
            return (model, cond, debug + "\nno aux positions in cond window")

        model_clone, indices = _patch_model(model, pairs, apply_to, block_range)
        return (model_clone, cond, "{}\npatched blocks: {}".format(debug, indices))


NODE_CLASS_MAPPINGS = {
    "Krea2PromptMix": Krea2PromptMix,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Krea2PromptMix": "Krea2 Prompt Mix",
}
