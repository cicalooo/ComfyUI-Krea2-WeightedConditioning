"""ComfyUI nodes: prompt weighting and prompt mix for Krea 2."""

from __future__ import annotations

import logging

import torch

from .attn_patch import APPLY_TO_KEY, WEIGHTS_KEY, Krea2WeightPatch
from .tokenize import (
    aux_id_span_in_combined,
    clip_token_ids,
    mix_factors,
    parse_block_range,
    parse_weighted_terms,
    resolve_weight_pairs,
    slice_to_cond_pairs,
    user_span_from_ids,
)

LOG = logging.getLogger("krea2.weighted")


class Krea2WeightedConditioning:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "model": ("MODEL",),
                "text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": (
                            "Prompt with (phrase:weight). weight<1 scales the token VALUE "
                            "(negative subtracts the concept); weight>1 boosts how much the "
                            "image attends to the token. Plain text = 1.0. Set sampler CFG to 1.0. "
                            "Not compatible with Krea2 Apply Regional (attention mask conflict)."
                        ),
                    },
                ),
                "strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 4.0,
                        "step": 0.05,
                        "tooltip": (
                            "Global multiplier. Effect compounds over patched blocks; "
                            "lower if the image breaks up. Removal (weight<0) is the reliable direction."
                        ),
                    },
                ),
                "emphasis_mode": (
                    ["k_bias", "value_scale", "both"],
                    {"default": "k_bias"},
                ),
                "match_mode": (
                    ["leading_space", "decode_window"],
                    {"default": "leading_space"},
                ),
                "unmatched": (
                    ["warn", "error", "ignore"],
                    {"default": "warn"},
                ),
                "block_range": (
                    "STRING",
                    {
                        "default": "all",
                        "tooltip": "all, 0-27, 4,8,12, or mixed 0-5,10,20-27",
                    },
                ),
                "apply_to": (
                    ["cond", "uncond", "both"],
                    {"default": "cond"},
                ),
            }
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "STRING")
    RETURN_NAMES = ("model", "conditioning", "debug")
    FUNCTION = "encode"
    CATEGORY = "Krea2/conditioning"
    DESCRIPTION = (
        "Per-token prompt weighting for Krea 2 via attention value scaling and k-bias. "
        "Use (word:-1) to remove a concept, (word:1.5) to emphasize one. "
        "Works through the Qwen3-VL encoder where CLIP-style weighting does nothing. "
        "Wire both MODEL and CONDITIONING; set sampler CFG to 1.0. "
        "Incompatible with Krea2 Apply Regional."
    )
    EXPERIMENTAL = True

    def encode(
        self,
        clip,
        model,
        text,
        strength,
        emphasis_mode="k_bias",
        match_mode="leading_space",
        unmatched="warn",
        block_range="all",
        apply_to="cond",
    ):
        terms, clean = parse_weighted_terms(text or "")
        tok = clip.tokenize(clean)
        key = next(iter(tok))
        ids = [t[0] for t in tok[key][0]]
        cond = clip.encode_from_tokens_scheduled(tok)
        cond_len = cond[0][0].shape[1]

        if not terms:
            return (
                model,
                cond,
                "no (phrase:weight) terms; strength is unused. "
                "This node does not scale a whole prompt. "
                "Use Krea2 Prompt Mix to keep a main prompt at 1.0 and weaken a moodboard.",
            )

        pairs, debug = resolve_weight_pairs(
            clip,
            ids,
            cond_len,
            terms,
            strength=strength,
            emphasis_mode=emphasis_mode,
            match_mode=match_mode,
            unmatched=unmatched,
        )
        if not pairs:
            return (model, cond, debug or "no matched phrases")

        LOG.info("Krea2 Weighted Conditioning: %s", debug)
        model_clone, indices = _patch_model(model, pairs, apply_to, block_range)
        debug = "{}\npatched blocks: {}".format(debug, indices)
        return (model_clone, cond, debug)


def _patch_model(model, pairs, apply_to, block_range):
    model_clone = model.clone()
    diffusion_model = model_clone.get_model_object("diffusion_model")
    blocks = getattr(diffusion_model, "blocks", None)
    if blocks is None:
        raise RuntimeError(
            "Krea2 Weighted Conditioning: model has no diffusion_model.blocks "
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


def _ids_from_tok(tok):
    key = next(iter(tok))
    return [t[0] for t in tok[key][0]]


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
            },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "STRING")
    RETURN_NAMES = ("model", "conditioning", "debug")
    FUNCTION = "encode"
    CATEGORY = "Krea2/conditioning"
    DESCRIPTION = (
        "Keep a handwritten prompt at 1.0 and scale a second (moodboard) prompt. "
        "Encodes both as one Qwen sequence so you do not ConditioningConcat two chat templates. "
        "CFG 1.0; wire both MODEL and CONDITIONING. Do not use with Krea2 Apply Regional."
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
    ):
        main = (text_main or "").strip()
        aux = (text_aux or "").strip()
        sep = {"newline": "\n", "space": " ", "comma": ", "}[separator]

        if not aux:
            tok = clip.tokenize(main)
            cond = clip.encode_from_tokens_scheduled(tok)
            return (model, cond, "empty aux; main only, no patch")

        combined = main + sep + aux if main else aux
        tok = clip.tokenize(combined)
        ids = _ids_from_tok(tok)
        cond = clip.encode_from_tokens_scheduled(tok)
        cond_len = cond[0][0].shape[1]
        visible_start = len(ids) - cond_len

        v_factor, k_bias = mix_factors(aux_strength, emphasis_mode)
        if abs(v_factor - 1.0) < 1e-6 and abs(k_bias) < 1e-6:
            return (model, cond, "aux_strength=1.0; no patch")

        if not main:
            cs, ce = user_span_from_ids(ids)
            pairs = slice_to_cond_pairs(cs, ce, visible_start, cond_len, v_factor, k_bias)
            debug = "main empty; scaled all user tokens"
        else:
            ids_main = clip_token_ids(clip, main)
            aux_s, aux_e = aux_id_span_in_combined(ids, ids_main)
            pairs = slice_to_cond_pairs(
                aux_s, aux_e, visible_start, cond_len, v_factor, k_bias
            )
            debug = "aux id[{}:{}] → {} cond tokens v={:.4f} k_bias={:.4f}".format(
                aux_s, aux_e, len(pairs), v_factor, k_bias
            )

        if not pairs:
            return (model, cond, debug + "\nno aux positions in cond window")

        model_clone, indices = _patch_model(model, pairs, apply_to, block_range)
        return (model_clone, cond, "{}\npatched blocks: {}".format(debug, indices))


def _concat_conditioning(cond_to, cond_from):
    """Same layout as ComfyUI ConditioningConcat (seq-dim cat)."""
    out = []
    src = cond_from[0][0]
    for item in cond_to:
        t1, extras = item[0], dict(item[1])
        out.append([torch.cat((t1, src.to(t1.device, t1.dtype)), dim=1), extras])
    return out


class Krea2ConditioningMix:
    """Concat two already-encoded conds and V-scale only the aux token slice."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "conditioning_main": ("CONDITIONING",),
                "conditioning_aux": ("CONDITIONING",),
                "aux_strength": (
                    "FLOAT",
                    {
                        "default": 0.45,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": "Scale aux tokens after concat. Main stays 1.0.",
                    },
                ),
            },
            "optional": {
                "emphasis_mode": (
                    ["value_scale", "k_bias", "both"],
                    {"default": "value_scale"},
                ),
                "block_range": ("STRING", {"default": "all"}),
                "apply_to": (["cond", "uncond", "both"], {"default": "cond"}),
            },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "STRING")
    RETURN_NAMES = ("model", "conditioning", "debug")
    FUNCTION = "mix"
    CATEGORY = "Krea2/conditioning"
    DESCRIPTION = (
        "Drop-in for ConditioningConcat when you want the second prompt weaker. "
        "Main tokens stay at 1.0; aux tokens are V-scaled. "
        "Prefer Krea2 Prompt Mix (one encode) — concat of two K2 CLIP encodes "
        "duplicates the Qwen chat template. CFG 1.0."
    )
    EXPERIMENTAL = True

    def mix(
        self,
        model,
        conditioning_main,
        conditioning_aux,
        aux_strength,
        emphasis_mode="value_scale",
        block_range="all",
        apply_to="cond",
    ):
        v_factor, k_bias = mix_factors(aux_strength, emphasis_mode)
        combined = _concat_conditioning(conditioning_main, conditioning_aux)
        main_len = int(conditioning_main[0][0].shape[1])
        aux_len = int(conditioning_aux[0][0].shape[1])
        pairs = [(main_len + i, v_factor, k_bias) for i in range(aux_len)]
        debug = "concat main_len={} aux_len={} v={:.4f} k_bias={:.4f}".format(
            main_len, aux_len, v_factor, k_bias
        )
        if abs(v_factor - 1.0) < 1e-6 and abs(k_bias) < 1e-6:
            return (model, combined, debug + "\naux_strength=1.0; concat only")
        model_clone, indices = _patch_model(model, pairs, apply_to, block_range)
        return (model_clone, combined, "{}\npatched blocks: {}".format(debug, indices))


NODE_CLASS_MAPPINGS = {
    "Krea2WeightedConditioning": Krea2WeightedConditioning,
    "Krea2PromptMix": Krea2PromptMix,
    "Krea2ConditioningMix": Krea2ConditioningMix,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Krea2WeightedConditioning": "Krea2 Weighted Conditioning",
    "Krea2PromptMix": "Krea2 Prompt Mix",
    "Krea2ConditioningMix": "Krea2 Conditioning Mix",
}
