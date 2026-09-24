"""ComfyUI nodes: Krea2 Encode, Encode Options, Multi Ref."""

from __future__ import annotations

import torch

from .attn_patch import APPLY_TO_KEY, WEIGHTS_KEY, Krea2WeightPatch
from .tokenize import (
    aux_id_span_grounded,
    aux_id_span_in_combined,
    clip_token_ids,
    mix_factors,
    parse_block_range,
    slice_to_cond_pairs,
    slice_to_cond_pairs_after_vision,
    token_ids_from_tok,
    user_span_from_ids,
    vision_cond_pairs_per_image,
)

MAX_REFS = 5
GUIDE_DEFAULT_STRENGTH = 0.45
GUIDE_DEFAULT_GROUNDING_PX = 384
GUIDE_DEFAULT_FOCUS = (
    "clothing or object only; do not copy any person in this image"
)
SCENE_DEFAULT_FOCUS = (
    "scene and layout only; do not copy any face from this image"
)
IDENTITY_DEFAULT_FOCUS = (
    "this is the only person; preserve this face and identity exactly"
)

KREA2_OPTS = "KREA2_ENCODE_OPTS"
KREA2_REFS = "KREA2_REFS"

K2EDIT_DEFAULT_SYSTEM = (
    "Describe the image by detailing the color, shape, size, "
    "texture, quantity, text, spatial relationships of the objects and background:"
)

_PEOPLE_LOCK_PHRASES = {
    "off": "",
    "solo": (
        "exactly one person, solo subject, single subject only, "
        "do not add extra people, no additional figures"
    ),
    "match_refs": (
        "keep the same number of people as in the reference image(s), "
        "do not add extra people or duplicate the subject"
    ),
}

_DEFAULT_OPTS = {
    "separator": "newline",
    "emphasis_mode": "value_scale",
    "block_range": "all",
    "apply_to": "cond",
    "grounding_px": 768,
    "guide_grounding_px": GUIDE_DEFAULT_GROUNDING_PX,
    "system_prompt": "",
    "fit_mode": "fit",
    "people_count_lock": "off",
    "ref_a_strength": 1.0,
    "ref_b_strength": 1.0,
    "ref_a_focus": "",
    "ref_b_focus": "",
    "ref_boost_a": 1.0,
    "ref_boost_mask": None,
    "ref_boost_mask_a": None,
}


def _patch_model(model, pairs, apply_to, block_range):
    model_clone = model.clone()
    diffusion_model = model_clone.get_model_object("diffusion_model")
    blocks = getattr(diffusion_model, "blocks", None)
    if blocks is None:
        raise RuntimeError(
            "Krea2 Encode: model has no diffusion_model.blocks "
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


def _k2edit_template(nimg, system_prompt="", image_prompts=None):
    """Qwen3-VL chat template with optional per-image focus text."""
    sp = (system_prompt or "").strip() or K2EDIT_DEFAULT_SYSTEM
    prompts = list(image_prompts or [])
    chunks = []
    for i in range(nimg):
        label = ""
        if i < len(prompts):
            label = (prompts[i] or "").strip()
        if label:
            chunks.append(label + "\n")
        chunks.append("<|vision_start|><|image_pad|><|vision_end|>")
        if i < nimg - 1:
            chunks.append("\n")
    vis = "".join(chunks)
    return (
        "<|im_start|>system\n" + sp + "<|im_end|>\n<|im_start|>user\n"
        + vis + "\n{}<|im_end|>\n<|im_start|>assistant\n"
    )


def _prep_grounding_image(image, grounding_px):
    """Cap longest side and snap to Qwen3-VL's 28-pixel vision grid (patch 14 × merge 2)."""
    samples = image.movedim(-1, 1)
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


def _merge_opts(options):
    out = dict(_DEFAULT_OPTS)
    if options:
        out.update(options)
    return out


def _compact_refs(slots):
    """Drop empty image slots; preserve remaining order. Cap at MAX_REFS."""
    out = []
    for slot in slots:
        img = slot.get("image")
        if img is None:
            continue
        strength = slot.get("strength")
        if strength is None:
            strength = GUIDE_DEFAULT_STRENGTH
        out.append({
            "image": img,
            "strength": float(strength),
            "focus": slot.get("focus") or "",
            "mask": slot.get("mask"),
        })
        if len(out) >= MAX_REFS:
            break
    return out


def _qwen_entry(image, role, strength, focus, grounding_px):
    return {
        "image": image,
        "role": role,
        "strength": float(strength),
        "focus": focus or "",
        "grounding_px": grounding_px,
    }


def _dit_entry(image, role, boost, mask):
    return {
        "image": image,
        "role": role,
        "boost": float(boost),
        "mask": mask,
    }


def resolve_layout(
    identity=None,
    scene=None,
    image=None,
    image_b=None,
    refs=None,
    opts=None,
    ref_boost=4.0,
):
    """Split appearance (DiT, max 2) from placement (Qwen, all images).

    ``identity`` / ``scene`` win over aliases ``image_b`` / ``image``.
    Multi Ref is Qwen guides only, unless no person is on Encode — then
    Multi Ref slot 1 is identity and the rest are guides.
    """
    opts = _merge_opts(opts)
    ident = identity if identity is not None else image_b
    scn = scene if scene is not None else image
    guides = _compact_refs(refs) if refs is not None else []
    notes = []
    ident_from_guide = None

    if ident is None and guides:
        ident_from_guide = guides[0]
        ident = ident_from_guide["image"]
        guides = guides[1:]
        notes.append(
            "Multi Ref slot 1 used as identity (no identity / image_b on Encode)"
        )

    if ident is not None and identity is None and image_b is not None:
        notes.append("identity from image_b alias")
    if scn is not None and scene is None and image is not None:
        notes.append("scene from image alias")
    if identity is not None and image_b is not None:
        notes.append("identity socket wins over image_b")
    if scene is not None and image is not None:
        notes.append("scene socket wins over image")

    grounding_px = opts["grounding_px"]
    guide_px = opts.get("guide_grounding_px", GUIDE_DEFAULT_GROUNDING_PX)

    # Guides already place clothes through Qwen. A scene VAE frame is fitted onto
    # the output grid and overwrites the face. With guides connected, DiT is the
    # person only (single-ref identity, the path this LoRA holds).
    scene_on_dit = scn is not None and not guides
    dit = []
    qwen = []
    if scn is not None:
        scene_focus = (opts.get("ref_a_focus") or "").strip() or SCENE_DEFAULT_FOCUS
        if scene_on_dit:
            dit.append(_dit_entry(scn, "scene", opts["ref_boost_a"], opts.get("ref_boost_mask_a")))
        qwen.append(_qwen_entry(
            scn, "scene", opts["ref_a_strength"], scene_focus, grounding_px
        ))
    for g in guides:
        focus = (g.get("focus") or "").strip() or GUIDE_DEFAULT_FOCUS
        qwen.append(_qwen_entry(
            g["image"], "guide", g["strength"], focus, guide_px
        ))
    if ident is not None:
        ident_focus = (opts.get("ref_b_focus") or "").strip() or IDENTITY_DEFAULT_FOCUS
        ident_strength = opts["ref_b_strength"]
        ident_mask = opts.get("ref_boost_mask")
        if ident_from_guide is not None:
            if (ident_from_guide.get("focus") or "").strip():
                ident_focus = ident_from_guide["focus"]
            ident_strength = ident_from_guide.get("strength", ident_strength)
            if ident_from_guide.get("mask") is not None:
                ident_mask = ident_from_guide["mask"]
        dit.append(_dit_entry(ident, "identity", ref_boost, ident_mask))
        qwen.append(_qwen_entry(ident, "identity", ident_strength, ident_focus, grounding_px))

    if dit:
        notes.append(
            "DiT refs={} ({})  boosts={}".format(
                len(dit),
                ", ".join(r["role"] for r in dit),
                [r["boost"] for r in dit],
            )
        )
    notes.append(
        "Qwen refs={} ({})".format(
            len(qwen),
            ", ".join(r["role"] for r in qwen) if qwen else "none",
        )
    )
    extra_guides = sum(1 for r in qwen if r["role"] == "guide")
    if extra_guides:
        notes.append(
            "guides are Qwen-only (not VAE frames); they place clothing/objects, "
            "identity stays on the person DiT ref"
        )
        if scn is not None and not scene_on_dit:
            notes.append(
                "scene is Qwen-only while guides are connected "
                "(a scene VAE frame fitted to the output overwrites the face)"
            )
        hot = [r["strength"] for r in qwen if r["role"] == "guide" and r["strength"] >= 0.85]
        if hot:
            notes.append(
                "WARNING: guide vision strength is {:.2f}. Set Multi Ref strength "
                "to ~0.45 or clothing photos compete with the face.".format(hot[0])
            )
    return dit, qwen, notes


def _apply_identity_edit(model, dit_refs, vae, target_latent, fit_mode):
    from .edit_patch import Krea2EditModelPatch

    images = [r["image"] for r in dit_refs]
    boosts = [r["boost"] for r in dit_refs]
    masks = [r["mask"] for r in dit_refs]
    samples = vae.encode(images[0][:, :, :, :3])
    source_latent = {"samples": samples}
    patched = Krea2EditModelPatch().patch(
        model,
        source_latent,
        vae=vae,
        source_images=images,
        fit_mode=fit_mode,
        target_latent=target_latent,
        ref_boosts=boosts,
        ref_boost_masks=masks,
    )[0]
    return patched


def _short(s, n=72):
    s = s or ""
    return s[:n] + ("…" if len(s) > n else "")


def encode_krea2(
    clip,
    model,
    prompt,
    aux,
    aux_strength,
    image=None,
    image_b=None,
    identity=None,
    scene=None,
    vae=None,
    ref_boost=4.0,
    latent=None,
    options=None,
    refs=None,
):
    opts = _merge_opts(options)
    separator = opts["separator"]
    emphasis_mode = opts["emphasis_mode"]
    block_range = opts["block_range"]
    apply_to = opts["apply_to"]
    system_prompt = opts["system_prompt"]
    fit_mode = opts["fit_mode"]
    people_count_lock = opts["people_count_lock"]

    dit_refs, qwen_refs, layout_notes = resolve_layout(
        identity=identity,
        scene=scene,
        image=image,
        image_b=image_b,
        refs=refs,
        opts=opts,
        ref_boost=ref_boost,
    )

    edit_notes = list(layout_notes)
    identity_on = vae is not None and bool(dit_refs)
    if identity_on:
        model = _apply_identity_edit(model, dit_refs, vae, latent, fit_mode)
        edit_notes.append("identity_edit=ON  fit={}".format(fit_mode))
        if latent is None:
            edit_notes.append(
                "NOTE: connect latent (KSampler empty latent) to pre-encode at target size"
            )
        edit_notes.append("NOTE: Identity Edit LoRA must already be on MODEL (strength 1.0)")
    elif qwen_refs:
        edit_notes.append(
            "identity_edit=OFF — Qwen grounding only; connect vae + identity for "
            "in-context source tokens"
        )
    else:
        edit_notes.append("text-only mix")

    text_main = (prompt or "").strip()
    text_aux = (aux or "").strip()
    if identity_on and text_aux and len(text_aux) > 80 and float(aux_strength) > 0:
        edit_notes.append(
            "WARNING: aux is {} chars at strength {:.2f}. Standard Krea Edit has no "
            "moodboard; set aux_strength=0 or clear aux if the face is drifting.".format(
                len(text_aux), float(aux_strength)
            )
        )
    lock_phrase = _PEOPLE_LOCK_PHRASES.get(people_count_lock or "off", "")
    main = text_main
    if lock_phrase:
        main = (main + "\n" + lock_phrase).strip() if main else lock_phrase

    sep = {"newline": "\n", "space": " ", "comma": ", "}[separator]
    imgs = [
        _prep_grounding_image(r["image"], r["grounding_px"])
        for r in qwen_refs
    ]
    grounded = bool(imgs)
    tok_kw = {}
    if grounded:
        tok_kw = {
            "images": imgs,
            "llama_template": _k2edit_template(
                len(imgs),
                system_prompt,
                image_prompts=[r["focus"] for r in qwen_refs],
            ),
        }

    img_strengths = [r["strength"] for r in qwen_refs]

    main_only_reason = None
    if not text_aux:
        main_only_reason = "empty aux"
    elif float(aux_strength) == 0.0:
        main_only_reason = "aux_strength=0.0"

    need_vision_weights = grounded and any(abs(s - 1.0) > 1e-6 for s in img_strengths)
    v_factor_early, k_bias_early = mix_factors(aux_strength, emphasis_mode)
    aux_is_noop = abs(v_factor_early - 1.0) < 1e-6 and abs(k_bias_early) < 1e-6

    def _debug_report(lines, patched_blocks=None, pair_count=0):
        header = [
            "=== Krea2 Encode ===",
            "prompt: {!r}".format(_short(text_main)),
            "aux:    {!r}".format(_short(text_aux)),
            "aux_strength={:.2f}  mode={}  people_count_lock={}".format(
                float(aux_strength), emphasis_mode, people_count_lock
            ),
        ]
        if lock_phrase:
            header.append("people_lock injected into main @1.0")
        header.extend(edit_notes)
        if grounded:
            header.append(
                "vision_strengths={}  roles={}".format(
                    img_strengths, [r["role"] for r in qwen_refs]
                )
            )
        else:
            header.append("refs=0 (text-only)")
        header.extend(lines)
        if patched_blocks is not None:
            header.append("patched blocks: {}".format(patched_blocks))
            header.append("total weighted cond tokens: {}".format(pair_count))
        return "\n".join(header)

    simple = (
        options is None
        and refs is None
        and identity is None
        and scene is None
        and people_count_lock == "off"
        and not identity_on
        and not need_vision_weights
    )

    if main_only_reason is not None and not need_vision_weights:
        tok = clip.tokenize(main, **tok_kw)
        cond = clip.encode_from_tokens_scheduled(tok)
        extra = []
        if grounded:
            extra.append("grounded {} image(s)".format(len(imgs)))
        msg = "{}; main only, no patch".format(main_only_reason)
        if extra:
            msg += "; " + "; ".join(extra)
        if simple:
            return (model, cond, msg)
        return (model, cond, _debug_report([msg]))

    if aux_is_noop and not need_vision_weights and main_only_reason is None:
        combined = main + sep + text_aux if main else text_aux
        tok = clip.tokenize(combined, **tok_kw)
        cond = clip.encode_from_tokens_scheduled(tok)
        msg = "aux_strength=1.0; no patch"
        if grounded:
            msg += "; grounded {} image(s)".format(len(imgs))
        if simple:
            return (model, cond, msg)
        return (model, cond, _debug_report([msg]))

    if main_only_reason is not None:
        combined = main
    else:
        combined = main + sep + text_aux if main else text_aux
    tok = clip.tokenize(combined, **tok_kw)
    ids = token_ids_from_tok(tok)
    cond = clip.encode_from_tokens_scheduled(tok)
    cond_len = cond[0][0].shape[1]
    visible_start = len(ids) - cond_len

    pairs: list = []
    lines: list[str] = []

    if need_vision_weights:
        vis_pairs = vision_cond_pairs_per_image(
            ids, cond_len, img_strengths, emphasis_mode
        )
        pairs.extend(vis_pairs)
        lines.append(
            "vision: strengths={} → {} cond tokens scaled".format(
                img_strengths, len(vis_pairs)
            )
        )
    elif grounded:
        lines.append("vision: strengths={} (all ≈1.0, not scaled)".format(img_strengths))

    v_factor, k_bias = mix_factors(aux_strength, emphasis_mode)
    aux_is_noop = abs(v_factor - 1.0) < 1e-6 and abs(k_bias) < 1e-6

    if main_only_reason is not None:
        lines.append("{} — no aux text weighting".format(main_only_reason))
        aux_pairs = []
    elif aux_is_noop:
        lines.append("aux: strength≈1.0 — no aux text patch")
        aux_pairs = []
    else:
        if not main:
            cs, ce = user_span_from_ids(ids)
            if grounded:
                aux_pairs = slice_to_cond_pairs_after_vision(
                    ids, cs, ce, cond_len, v_factor, k_bias
                )
            else:
                aux_pairs = slice_to_cond_pairs(
                    cs, ce, visible_start, cond_len, v_factor, k_bias
                )
            lines.append(
                "aux: main empty — scaled all user tokens "
                "({} tokens, v={:.3f}, k_bias={:.3f})".format(
                    len(aux_pairs), v_factor, k_bias
                )
            )
        else:
            ids_main = clip_token_ids(clip, main, **tok_kw)
            if grounded:
                aux_s, aux_e = aux_id_span_grounded(ids, ids_main)
                aux_pairs = slice_to_cond_pairs_after_vision(
                    ids, aux_s, aux_e, cond_len, v_factor, k_bias
                )
            else:
                aux_s, aux_e = aux_id_span_in_combined(ids, ids_main)
                aux_pairs = slice_to_cond_pairs(
                    aux_s, aux_e, visible_start, cond_len, v_factor, k_bias
                )
            lines.append(
                "aux id[{}:{}] → {} cond tokens v={:.4f} k_bias={:.4f}".format(
                    aux_s, aux_e, len(aux_pairs), v_factor, k_bias
                )
            )
            if grounded:
                lines.append("grounded {} image(s)".format(len(imgs)))
            if len(aux_pairs) == 0:
                lines.append(
                    "WARNING: 0 aux tokens — join/split failed; "
                    "try different separator or shorter main"
                )
        pairs.extend(aux_pairs)

    if not pairs:
        if simple and main_only_reason is None:
            debug = "aux id[{}:{}] → {} cond tokens v={:.4f} k_bias={:.4f}".format(
                0, 0, 0, v_factor, k_bias
            )
            if grounded:
                debug += "; grounded {} image(s)".format(len(imgs))
            return (model, cond, debug + "\nno aux positions in cond window")
        return (model, cond, _debug_report(lines + ["no weighted positions in cond window"]))

    model_clone, indices = _patch_model(model, pairs, apply_to, block_range)
    if simple:
        debug = lines[0] if lines else "patched"
        aux_line = next((ln for ln in lines if ln.startswith("aux id")), None)
        if aux_line:
            debug = aux_line
            if grounded:
                debug += "; grounded {} image(s)".format(len(imgs))
        return (model_clone, cond, "{}\npatched blocks: {}".format(debug, indices))
    return (
        model_clone,
        cond,
        _debug_report(lines, patched_blocks=indices, pair_count=len(pairs)),
    )


class Krea2Encode:
    """T2I mix, Krea 2 Edit, and prompt mix in one encode."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "model": ("MODEL",),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Subject / edit instruction. Stays at strength 1.0.",
                    },
                ),
                "aux": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Moodboard / enforce prompt. Scaled by aux_strength.",
                    },
                ),
                "aux_strength": (
                    "FLOAT",
                    {
                        "default": 0.45,
                        "min": 0.0,
                        "max": 4.0,
                        "step": 0.05,
                        "tooltip": (
                            "Aux tokens only. 0 = omit aux, 0.35–0.55 typical weaken, "
                            "1.0 = equal, >1 enforce. Not CLIP (prompt:0.5)."
                        ),
                    },
                ),
            },
            "optional": {
                "identity": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "Person / identity photo. This is the DiT appearance ref "
                            "(ref_boost applies here). Prefer this over image_b."
                        ),
                    },
                ),
                "scene": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "Optional scene. With identity: DiT frame 1 = scene, "
                            "frame 2 = person. Prefer this over image."
                        ),
                    },
                ),
                "image": (
                    "IMAGE",
                    {
                        "tooltip": "Alias for scene. Ignored if scene is connected.",
                    },
                ),
                "image_b": (
                    "IMAGE",
                    {
                        "tooltip": "Alias for identity. Ignored if identity is connected.",
                    },
                ),
                "vae": (
                    "VAE",
                    {
                        "tooltip": (
                            "Qwen Image VAE. With identity (or scene), enables the "
                            "Identity Edit pixel path. No manual VAEEncode."
                        ),
                    },
                ),
                "ref_boost": (
                    "FLOAT",
                    {
                        "default": 4.0,
                        "min": 0.0,
                        "max": 1000.0,
                        "step": 0.01,
                        "tooltip": (
                            "Likeness for the person DiT ref only. Ignored without VAE. "
                            "1 = off, ~4 recommended. Multi Ref does not use this."
                        ),
                    },
                ),
                "latent": (
                    "LATENT",
                    {
                        "tooltip": (
                            "Same empty latent as KSampler.latent_image. Optional; tells the "
                            "node the output size so VAE encode happens before sampling."
                        ),
                    },
                ),
                "options": (
                    KREA2_OPTS,
                    {
                        "tooltip": "Krea2 Encode Options. Disconnected = safe defaults.",
                    },
                ),
                "refs": (
                    KREA2_REFS,
                    {
                        "tooltip": (
                            "Krea2 Multi Ref — Qwen placement guides (clothing/objects). "
                            "Not VAE frames. Does not replace identity."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "STRING")
    RETURN_NAMES = ("model", "conditioning", "debug")
    FUNCTION = "encode"
    CATEGORY = "Krea2/conditioning"
    DESCRIPTION = (
        "Text mix + optional K2 Edit. Connect identity (person) and vae for likeness. "
        "Multi Ref is clothing/object placement in Qwen only — it does not steal the "
        "person's VAE frame. Wire MODEL and CONDITIONING. Do not use with Krea2 Apply Regional."
    )

    def encode(
        self,
        clip,
        model,
        prompt,
        aux,
        aux_strength,
        image=None,
        image_b=None,
        identity=None,
        scene=None,
        vae=None,
        ref_boost=4.0,
        latent=None,
        options=None,
        refs=None,
    ):
        return encode_krea2(
            clip,
            model,
            prompt,
            aux,
            aux_strength,
            image=image,
            image_b=image_b,
            identity=identity,
            scene=scene,
            vae=vae,
            ref_boost=ref_boost,
            latent=latent,
            options=options,
            refs=refs,
        )


class Krea2EncodeOptions:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "people_count_lock": (
                    ["off", "solo", "match_refs"],
                    {
                        "default": "off",
                        "tooltip": (
                            "Injects a fixed phrase into prompt at 1.0. "
                            "solo = one person; match_refs = same count as refs."
                        ),
                    },
                ),
                "emphasis_mode": (
                    ["value_scale", "k_bias", "both"],
                    {
                        "default": "value_scale",
                        "tooltip": "value_scale is safest with Identity Edit ref_boost masks.",
                    },
                ),
                "separator": (["newline", "space", "comma"], {"default": "newline"}),
                "block_range": ("STRING", {"default": "all"}),
                "apply_to": (["cond", "uncond", "both"], {"default": "cond"}),
                "grounding_px": (
                    "INT",
                    {
                        "default": 768,
                        "min": 0,
                        "max": 4096,
                        "step": 64,
                        "tooltip": "Qwen cap for scene and identity.",
                    },
                ),
                "guide_grounding_px": (
                    "INT",
                    {
                        "default": GUIDE_DEFAULT_GROUNDING_PX,
                        "min": 0,
                        "max": 4096,
                        "step": 64,
                        "tooltip": "Qwen cap for Multi Ref guides (lower so clothes do not overwrite the face).",
                    },
                ),
                "system_prompt": (
                    "STRING",
                    {"multiline": True, "default": ""},
                ),
                "fit_mode": (
                    ["fit", "crop (legacy)"],
                    {"default": "fit"},
                ),
                "ref_a_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05},
                ),
                "ref_b_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05},
                ),
                "ref_a_focus": ("STRING", {"multiline": True, "default": ""}),
                "ref_b_focus": ("STRING", {"multiline": True, "default": ""}),
                "ref_boost_a": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1000.0,
                        "step": 0.01,
                        "tooltip": "Likeness dial for the scene DiT ref.",
                    },
                ),
                "ref_boost_mask": ("MASK", {"tooltip": "Optional region on the person (e.g. face)."}),
                "ref_boost_mask_a": ("MASK", {"tooltip": "Optional region on the scene."}),
            },
        }

    RETURN_TYPES = (KREA2_OPTS,)
    RETURN_NAMES = ("options",)
    FUNCTION = "build"
    CATEGORY = "Krea2/conditioning"
    DESCRIPTION = "Advanced options for Krea2 Encode. One wire; disconnected Encode uses defaults."

    def build(
        self,
        people_count_lock="off",
        emphasis_mode="value_scale",
        separator="newline",
        block_range="all",
        apply_to="cond",
        grounding_px=768,
        guide_grounding_px=GUIDE_DEFAULT_GROUNDING_PX,
        system_prompt="",
        fit_mode="fit",
        ref_a_strength=1.0,
        ref_b_strength=1.0,
        ref_a_focus="",
        ref_b_focus="",
        ref_boost_a=1.0,
        ref_boost_mask=None,
        ref_boost_mask_a=None,
    ):
        return ({
            "people_count_lock": people_count_lock,
            "emphasis_mode": emphasis_mode,
            "separator": separator,
            "block_range": block_range,
            "apply_to": apply_to,
            "grounding_px": grounding_px,
            "guide_grounding_px": guide_grounding_px,
            "system_prompt": system_prompt,
            "fit_mode": fit_mode,
            "ref_a_strength": ref_a_strength,
            "ref_b_strength": ref_b_strength,
            "ref_a_focus": ref_a_focus,
            "ref_b_focus": ref_b_focus,
            "ref_boost_a": ref_boost_a,
            "ref_boost_mask": ref_boost_mask,
            "ref_boost_mask_a": ref_boost_mask_a,
        },)


def _ref_slot_inputs(n):
    fields = {}
    for i in range(1, n + 1):
        fields["image_{}".format(i)] = (
            "IMAGE",
            {
                "tooltip": (
                    "Guide {}. Clothing / object for Qwen placement. "
                    "Not a VAE identity frame. Empty slots are skipped."
                ).format(i),
            },
        )
        fields["strength_{}".format(i)] = (
            "FLOAT",
            {
                "default": GUIDE_DEFAULT_STRENGTH,
                "min": 0.0,
                "max": 3.0,
                "step": 0.05,
                "tooltip": "Qwen vision-token strength for guide {}.".format(i),
            },
        )
        fields["focus_{}".format(i)] = (
            "STRING",
            {
                "multiline": True,
                "default": "",
                "tooltip": (
                    "What to read from this image. Empty = clothing/object only, "
                    "do not copy any person."
                ),
            },
        )
    return fields


class Krea2MultiRef:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": _ref_slot_inputs(MAX_REFS),
        }

    RETURN_TYPES = (KREA2_REFS,)
    RETURN_NAMES = ("refs",)
    FUNCTION = "build"
    CATEGORY = "Krea2/conditioning"
    DESCRIPTION = (
        "Up to 5 Qwen placement guides (dress, hat, necklace, …). "
        "They do not become Identity Edit VAE frames. Wire the person to Encode.identity."
    )

    def build(self, **kwargs):
        slots = []
        for i in range(1, MAX_REFS + 1):
            slots.append({
                "image": kwargs.get("image_{}".format(i)),
                "strength": kwargs.get("strength_{}".format(i), GUIDE_DEFAULT_STRENGTH),
                "focus": kwargs.get("focus_{}".format(i), ""),
            })
        return (_compact_refs(slots),)


NODE_CLASS_MAPPINGS = {
    "Krea2Encode": Krea2Encode,
    "Krea2EncodeOptions": Krea2EncodeOptions,
    "Krea2MultiRef": Krea2MultiRef,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Krea2Encode": "Krea2 Split Encode",
    "Krea2EncodeOptions": "Split Encode Options",
    "Krea2MultiRef": "Split Ref",
}
