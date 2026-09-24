"""Krea2 identity-edit source patch (in-context VAE tokens).

Vendored/adapted from mustyrocks_k2edit 1.3.0 (Apache-2.0).
Original: https://github.com/lbouaraba/comfyui-krea2edit / mustyrocks pack.
Used internally by Krea2 Encode when VAE + image(s) are connected.
"""
import math

import torch
import torch.nn.functional as F
from einops import rearrange

import comfy.patcher_extension
import comfy.utils
import comfy.ldm.common_dit
from comfy.ldm.flux.layers import timestep_embedding

# fwd_cache holds step-invariant forward work (fit/proj/freqs/bias) for the lifetime
# of the patched model. One full run uses ~4 entries; 8 ≈ two runs' worth, so a
# session cycling resolutions/prompts stays bounded instead of accumulating stale
# GPU tensors (the attention bias alone is ~L^2 x 2 bytes — ~450MB at 1776x1184).
_FWD_CACHE_MAX = 8


def _imgids(bs, frame, h_, w_, device):
    ids = torch.zeros(h_, w_, 3, device=device, dtype=torch.float32)
    ids[..., 0] = frame
    ids[..., 1] = torch.arange(h_, device=device, dtype=torch.float32)[:, None]
    ids[..., 2] = torch.arange(w_, device=device, dtype=torch.float32)[None, :]
    return ids.reshape(1, h_ * w_, 3).repeat(bs, 1, 1)


def _imgids_offset(bs, frame, gh, gw, th, tw, device):
    """Stride-1 positions at the exact centered offset. For `fit` refs the pixels
    are already resampled to target grid density, so the position grid is stride-1
    BY CONSTRUCTION — scaling it again only manufactures skip/collision artifacts.
    Requires gh<=th, gw<=tw (guaranteed by the floor+cap in fit)."""
    # fractional center (2026-07-28): integer floor placed odd-gap refs 8px off
    # their true center — RoPE is continuous, half-token positions are exact.
    off_h, off_w = max(0.0, (th - gh) / 2), max(0.0, (tw - gw) / 2)
    ids = torch.zeros(gh, gw, 3, device=device, dtype=torch.float32)
    ids[..., 0] = frame
    ids[..., 1] = (torch.arange(gh, device=device, dtype=torch.float32) + off_h)[:, None]
    ids[..., 2] = (torch.arange(gw, device=device, dtype=torch.float32) + off_w)[None, :]
    return ids.reshape(1, gh * gw, 3).repeat(bs, 1, 1)


def _to_4d(v):
    """(B,C,T,H,W) -> (B*T,C,H,W); pass 4D through. Images use T=1."""
    if v.ndim == 5:
        b, c, t, h, w = v.shape
        return v.reshape(b * t, c, h, w)
    return v


def _fit_src(src, H, W):
    """Fit a source latent to the target grid the way TRAINING did: center-crop to
    the target aspect ratio, then resize. A plain interpolate (the pre-fix behavior)
    STRETCHES mixed-AR sources — users saw stretched people whenever their input AR
    differed from the output resolution."""
    sh, sw = src.shape[-2:]
    if (sh, sw) == (H, W):
        return src
    s = max(H / sh, W / sw)
    ch, cw = min(sh, int(round(H / s))), min(sw, int(round(W / s)))
    y0, x0 = (sh - ch) // 2, (sw - cw) // 2
    src = src[..., y0:y0 + ch, x0:x0 + cw]
    return F.interpolate(src.float(), size=(H, W), mode="bilinear")


def _encode_vae_fallback(vae, img):
    """VAE-encode with a tiled retry on CUDA OOM. A full-resolution encode is the
    biggest transient VRAM spike in this pack; tiling bounds it (the VAE is
    local-conv, so tiles are value-preserving up to the overlap seam)."""
    try:
        return vae.encode(img)
    except RuntimeError as e:
        if "out of memory" not in str(e).lower():
            raise
        print(f"[krea2edit] VAE encode OOM at {tuple(img.shape)} — retrying tiled", flush=True)
        torch.cuda.empty_cache()
        if hasattr(vae, "encode_tiled"):          # current ComfyUI (VAE in comfy/sd.py)
            return vae.encode_tiled(img)
        from comfy.ldm.vae import tiled_encode    # legacy ComfyUI module function
        return tiled_encode(vae, img)


def _fit_encode_image(image, vae, H, W, cache, key, fit_mode="crop"):
    """Pixel-space source prep (blur-proof path): center-crop the IMAGE to the
    target AR, resize to the exact target pixel grid, VAE-encode. Latent-space
    resizing (the old fallback) softens VAE latents — this path never resizes
    latents at all. Cached per target resolution (encode once, not per step)."""
    key = key + (fit_mode,)
    if key in cache:
        return cache[key]
    print(f"[krea2edit] _fit_encode_image: mode={fit_mode} in={tuple(image.shape)} target_latent={H}x{W}", flush=True)
    px_h, px_w = H * 8, W * 8
    img = image.movedim(-1, 1)  # B,H,W,C -> B,C,H,W
    ih, iw = img.shape[-2:]
    if fit_mode == "fit":
        # "bilinear" answer to scale mismatch: resample CONTENT (pixel space, bicubic)
        # to the target's grid density instead of moving positions. AR-preserving
        # fit-inside, no crop, no grey canvas — the forward places it at an integer
        # centered offset (scaled-pos with s=1 -> stride 1, no rounding artifacts).
        sc = min(px_h / ih, px_w / iw)
        if sc < 0.5:
            print(f"[krea2edit] WARNING: source is smaller than the target grid "
                  f"(upscaled x{1 / sc:.1f}) — outside the trained range; expect degraded "
                  f"fidelity.", flush=True)
        # NEAR-MATCHED AR: fill the target grid EXACTLY via a minimal center-crop.
        # Fit-inside margins of 1-2 tokens are not harmless: target edge columns
        # with no ref correspondence get filled by repeating adjacent ref content
        # (2026-07-14 edge-duplication bug: ref (74,54) vs target (74,56)).
        # This also restores the design promise fit == crop at matched AR.
        CROP_TOL = 0.08
        if ih * sc >= px_h * (1 - CROP_TOL) and iw * sc >= px_w * (1 - CROP_TOL):
            s = max(px_h / ih, px_w / iw)
            ch, cw = min(ih, int(round(px_h / s))), min(iw, int(round(px_w / s)))
            y0, x0 = (ih - ch) // 2, (iw - cw) // 2
            img = img[..., y0:y0 + ch, x0:x0 + cw]
            nh, nw = px_h, px_w
        else:
            # genuine AR mismatch: MUST match the trainer's _fit_prep EXACTLY
            # (krea2_edit.py) — /16 floor snap capped at the target's /16 floor.
            # The model is trained on this geometry; a /8-round node grid would
            # produce a different ref latent size -> different centered offset ->
            # a visible margin-boundary seam even from a well-trained model
            # (train/infer geometry must be byte-identical). 2026-07-15 alignment.
            nh = min(max(16, int(ih * sc) // 16 * 16), max(16, px_h // 16 * 16))
            nw = min(max(16, int(iw * sc) // 16 * 16), max(16, px_w // 16 * 16))
            # CROP-TO-GRID (2026-07-28 seam-doubling RCA): resizing ih*sc -> floor16
            # SQUASHES content by up to 15px; the misregistration peaks exactly at
            # the ref band edges — the outpaint seam — and renders as a doubled
            # band (proven causal: 754px vs 753px input A/B, one pixel flips
            # clean<->worst). Center-crop the source so the fitted axis lands on
            # the /16 grid at scale sc EXACTLY: zero squash, stride-1 stays true.
            ch2, cw2 = min(ih, max(1, int(round(nh / sc)))), min(iw, max(1, int(round(nw / sc))))
            y0, x0 = (ih - ch2) // 2, (iw - cw2) // 2
            img = img[..., y0:y0 + ch2, x0:x0 + cw2]
        img = F.interpolate(img.float(), size=(nh, nw), mode="bicubic", antialias=True)
        lat = _encode_vae_fallback(vae, img.movedim(1, -1)[..., :3].clamp(0, 1))
        cache[key] = lat
        return lat
    # crop (default / "v1 legacy"): center-crop to the target AR, then resize.
    s = max(px_h / ih, px_w / iw)
    if s > 2.0:
        print(f"[krea2edit] WARNING: source is smaller than the target grid "
              f"(upscaled x{s:.1f}) — outside the trained range; expect degraded "
              f"fidelity.", flush=True)
    ch, cw = min(ih, int(round(px_h / s))), min(iw, int(round(px_w / s)))
    y0, x0 = (ih - ch) // 2, (iw - cw) // 2
    img = img[..., y0:y0 + ch, x0:x0 + cw]
    img = F.interpolate(img.float(), size=(px_h, px_w), mode="bicubic", antialias=True)
    lat = _encode_vae_fallback(vae, img.movedim(1, -1)[..., :3].clamp(0, 1))
    cache[key] = lat
    return lat


def _ref_attn_bias(boosts, boost_masks, txtlen, slens, tgtlen, mask_hw, device, dtype):
    """Additive attention-logit bias on the [text | refs... | target] sequence.

    boosts: per-ref factor on target->ref attention, aligned with the source blocks
    (last entry = last ref = the subject by workflow convention). Equivalent to
    multiplying those keys' post-softmax attention weight before renormalization.
    boost_masks: per-ref ComfyUI MASKs (ref-image pixel space) restricting each ref's
    boost to a region (e.g. the face); None entries = whole reference. A legacy
    single mask may be passed instead of a list — it applies to the LAST ref.
    """
    nsrc = len(slens)
    if not isinstance(boost_masks, (list, tuple)):
        boost_masks = [None] * (nsrc - 1) + [boost_masks]
    offs = [txtlen]
    for sl in slens:
        offs.append(offs[-1] + sl)
    rows0 = offs[-1]
    L = rows0 + tgtlen
    bias = torch.zeros(1, 1, L, L, device=device, dtype=dtype)
    for i, b in enumerate(boosts):
        if b == 1.0:
            continue
        off, sl = offs[i], slens[i]
        mask = boost_masks[i] if i < len(boost_masks) else None
        if mask is not None and mask_hw is not None:
            m0 = mask[:1]
            if m0.ndim == 2:
                m0 = m0[None]
            m0 = F.interpolate(m0[None].float(), size=mask_hw[i], mode="area")[0, 0]
            cols = off + torch.nonzero(m0.reshape(-1) > 0.5, as_tuple=True)[0].to(device)
        else:
            cols = torch.arange(off, off + sl, device=device)
        if cols.numel() == 0:
            continue
        bias[:, :, rows0:, cols] = math.log(max(b, 1e-4))
    return bias


def krea2_edit_forward(m, x, timesteps, context, src_latent, transformer_options,
                       ref_boost=1.0, ref_boost_a=1.0, ref_boost_mask=None,
                       ref_boost_mask_a=None, ref_native=False, pos_mode="anchor",
                       fwd_cache=None, ref_boosts=None, ref_boost_masks=None):
    """Krea2 SingleStreamDiT._forward, but with source block(s) prepended.

    m           : the SingleStreamDiT (LoRA-patched at sample time)
    x           : (B,C,H,W) or (B,C,T,H,W) noisy TARGET latent
    src_latent  : clean SOURCE latent (VAE-encoded), 4D/5D — or a LIST of them
                  (multi-ref: [scene, subject], frames 1..N, training-matched)
    context     : (B, seq, txtlayers*txtdim) — the 12-layer Qwen3-VL stack
    fwd_cache   : optional dict for cross-step caching. The source never changes
                  mid-sample, so fit+pad, ref projection, RoPE freqs and the
                  attention bias are all step-invariant; pass {} from the patch
                  node to compute each once instead of every denoise step.
                  None = pure per-call behavior (tests / one-shot use).
    """
    patch = m.patch
    warned = fwd_cache.setdefault("_warned", {}) if fwd_cache is not None else {}

    # Mirror ComfyUI _forward: latents may arrive 5D (B,C,T,H,W) for this model.
    temporal = x.ndim == 5
    if temporal:
        b5, c5, t5, h5, w5 = x.shape
    x = _to_4d(x)
    bs, c, H_orig, W_orig = x.shape

    # Native _forward copies transformer_options before annotating it (block_index,
    # img_slice, ...) — mirror that so the sampler-owned dict is never mutated.
    transformer_options = (transformer_options or {}).copy()

    x = comfy.ldm.common_dit.pad_to_patch_size(x, (patch, patch), padding_mode="replicate")
    H, W = x.shape[-2], x.shape[-1]
    h_, w_ = H // patch, W // patch

    # source(s) -> (bs, C, H, W): flatten temporal, match batch, fit to the target grid
    # (center-crop to target AR then resize — training-matched; never stretch).
    src_list = src_latent if isinstance(src_latent, (list, tuple)) else [src_latent]
    srcs_raw = [_to_4d(sl) for sl in src_list]

    if bs > 1 and any(s.shape[0] == 1 for s in srcs_raw) and "batch_broadcast" not in warned:
        print("[krea2edit] WARNING: batch size > 1 but only one source image given — "
              "every target in the batch will use the SAME reference.", flush=True)
        warned["batch_broadcast"] = True

    # The cache outlives a single sampling run (the wrapper holds it for the node's
    # lifetime). A new run — different resolution, source or prompt length — mints
    # NEW keys and the old tensors would accumulate forever: the attention bias alone
    # is ~L^2 x 2 bytes (~450MB at 1776x1184 with a long grounded prompt). Two bounds:
    #   * run signature change (geometry/source/dtype) -> drop everything now;
    #   * entry cap -> evict oldest, so same-geometry runs with variable-length
    #     prompts (TextGenerate) can't grow it without limit either.
    if fwd_cache is not None:
        run_sig = ("run", tuple(s.shape for s in srcs_raw), H, W, bool(ref_native), str(x.dtype))
        if fwd_cache.get("_run_sig") != run_sig:
            warned_keep = fwd_cache.get("_warned")
            fwd_cache.clear()
            if warned_keep is not None:
                fwd_cache["_warned"] = warned_keep
            fwd_cache["_run_sig"] = run_sig
        reserved = ("_warned", "_run_sig")
        keys = [k for k in fwd_cache if k not in reserved]
        excess = len(keys) - _FWD_CACHE_MAX
        for k in keys[:max(0, excess)]:
            del fwd_cache[k]

    # Fit + pad is step-invariant (the source never changes mid-sample): once per run.
    fit_key = ("fit", tuple(s.shape for s in srcs_raw), H, W, bool(ref_native), str(x.dtype))
    if fwd_cache is not None and fit_key in fwd_cache:
        srcs = fwd_cache[fit_key]
    else:
        srcs = []
        for sl in srcs_raw:
            src = sl.to(x.device, x.dtype)
            if src.shape[0] != bs:
                src = src[:1].expand(bs, *src.shape[1:])
            if not ref_native and src.shape[-2:] != (H, W):
                sh_, sw_ = src.shape[-2:]
                scale = max(H / sh_, W / sw_)
                msg = f"[krea2edit] LATENT-PATH fit_src (crop): src={tuple(src.shape[-2:])} -> {H}x{W}"
                if scale > 2.0:
                    msg += " — source upscaled >2x (outside the trained range; consider the pixel path: connect vae + source_image)"
                print(msg, flush=True)
                src = _fit_src(src, H, W).to(x.dtype)
            srcs.append(comfy.ldm.common_dit.pad_to_patch_size(src, (patch, patch), padding_mode="replicate"))
        if fwd_cache is not None:
            fwd_cache[fit_key] = srcs
    src_grids = [(s_.shape[-2] // patch, s_.shape[-1] // patch) for s_ in srcs]

    context = m._unpack_context(context)                       # (B, seq, 12, 2560)

    tgt_img = m.first(rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch))

    # Ref projection is step-invariant (m.first weights are static within a sample).
    proj_key = ("refproj", tuple(s.shape for s in srcs), str(x.dtype))
    if fwd_cache is not None and proj_key in fwd_cache:
        src_imgs = fwd_cache[proj_key]
    else:
        src_imgs = [m.first(rearrange(s_, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch))
                    for s_ in srcs]
        if fwd_cache is not None:
            fwd_cache[proj_key] = src_imgs

    t = m.tmlp(timestep_embedding(timesteps, m.tdim).unsqueeze(1).to(tgt_img.dtype))
    tvec = m.tproj(t)

    context = m.txtfusion(context, mask=None, transformer_options=transformer_options)
    context = m.txtmlp(context)

    txtlen, tgtlen = context.shape[1], tgt_img.shape[1]
    srclen = sum(si.shape[1] for si in src_imgs)
    combined = torch.cat([context] + src_imgs + [tgt_img], dim=1)  # [text | refs... | target]

    device = combined.device

    # RoPE positions depend only on the sequence geometry: compute once.
    freq_key = ("freqs", bs, txtlen, tuple(src_grids), tgtlen, pos_mode, bool(ref_native))
    if fwd_cache is not None and freq_key in fwd_cache:
        freqs = fwd_cache[freq_key]
    else:
        if pos_mode == "stride1" and ref_native:
            print(f"[krea2edit] STRIDE1-POS fit: ref grids {src_grids} centered in ({h_},{w_})", flush=True)
            warned["stride1"] = True
            ref_ids = [_imgids_offset(bs, i + 1, gh, gw, h_, w_, device)
                       for i, (gh, gw) in enumerate(src_grids)]
        else:
            ref_ids = [_imgids(bs, i + 1, gh, gw, device) for i, (gh, gw) in enumerate(src_grids)]
        pos = torch.cat([
            torch.zeros(bs, txtlen, 3, device=device, dtype=torch.float32)]   # text @ 0
            + ref_ids
            + [_imgids(bs, 0, h_, w_, device)],                                # target frame=0
            dim=1)
        freqs = m.pe_embedder(pos)
        if fwd_cache is not None:
            fwd_cache[freq_key] = freqs

    attn_bias = None
    n = len(src_imgs)
    if ref_boosts is not None:
        boosts = list(ref_boosts) + [1.0] * max(0, n - len(ref_boosts))
        boosts = boosts[:n]
        if ref_boost_masks is not None:
            masks = list(ref_boost_masks) + [None] * max(0, n - len(ref_boost_masks))
            masks = masks[:n]
        else:
            masks = [None] * n
    else:
        # last ref = subject (single-ref: the only ref); earlier refs (scene) get ref_boost_a
        boosts = [ref_boost_a] * (n - 1) + [ref_boost]
        masks = ([ref_boost_mask_a] + [None] * (n - 2) + [ref_boost_mask]) if n > 1 else [ref_boost_mask]
    if any(b != 1.0 for b in boosts):
        for i, mk in enumerate(masks):
            tag = f"ref{i + 1}"
            if mk is not None and boosts[i] != 1.0 and (mk > 0.5).sum() == 0 \
                    and f"mask_empty_{tag}" not in warned:
                print(f"[krea2edit] WARNING: ref_boost_mask for {tag} has no active pixels "
                      f"(>0.5) — the boost is a no-op.", flush=True)
                warned[f"mask_empty_{tag}"] = True
        bias_key = ("bias", tuple(boosts), txtlen, tuple(si.shape[1] for si in src_imgs),
                    tgtlen, tuple(id(mk) for mk in masks))
        if fwd_cache is not None and bias_key in fwd_cache:
            attn_bias = fwd_cache[bias_key]
        else:
            attn_bias = _ref_attn_bias(boosts, masks, txtlen,
                                       [si.shape[1] for si in src_imgs], tgtlen,
                                       src_grids, device, combined.dtype)
            if fwd_cache is not None:
                fwd_cache[bias_key] = attn_bias

    # Same per-block bookkeeping as native _forward (other patches may read these).
    transformer_options["total_blocks"] = len(m.blocks)
    transformer_options["block_type"] = "single"
    transformer_options["img_slice"] = [txtlen, combined.shape[1]]
    for i, block in enumerate(m.blocks):
        transformer_options["block_index"] = i
        combined = block(combined, tvec, freqs, attn_bias, transformer_options=transformer_options)

    final = m.last(combined, t)
    out = final[:, txtlen + srclen: txtlen + srclen + tgtlen, :]         # target tokens only
    out = rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)",
                    h=h_, w=w_, ph=patch, pw=patch, c=m.channels)
    out = out[:, :, :H_orig, :W_orig]
    if temporal:
        out = out.reshape(b5, t5, m.channels, H_orig, W_orig).movedim(1, 2)
    return out


class Krea2EditModelPatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "source_latent": ("LATENT",),
        }, "optional": {
            "source_latent_b": ("LATENT", {"tooltip": "2nd reference (subject photo) for multi-ref LoRAs -> RoPE frame=2, training-matched order: scene first, subject second"}),
            "ref_boost": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.01, "round": 0.001,
                                     "tooltip": "reference-fidelity dial: multiplies target->reference attention. Applies to the LAST ref (= the subject in two-ref workflows, the only ref in single-ref). 1.0 = off, >1 pulls harder toward the reference's appearance, <1 loosens. Optimal value is model-specific"}),
            "ref_boost_a": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.01, "round": 0.001,
                                       "tooltip": "same dial for the FIRST ref (= the scene in two-ref workflows). No effect in single-ref workflows. 1.0 = off"}),
            "fit_mode": (["fit", "crop (legacy)"], {"default": "fit",
                          "tooltip": "how an image source fits a mismatched output aspect ratio (needs vae + source_image connected): fit = resample the source to the target grid at a centered offset — matches how this model was trained (default, use this); crop (legacy) = center-crop to the target AR then resize (v1/v1.1 geometry, only for older weights)"}),
            "ref_boost_mask": ("MASK", {"tooltip": "optional region on the (last) reference to boost, e.g. the face; empty = whole reference"}),
            "ref_boost_mask_a": ("MASK", {"tooltip": "optional region on the FIRST reference to boost (the scene in two-ref workflows); empty = whole reference"}),
            "vae": ("VAE", {"tooltip": "RECOMMENDED with source_image: enables the blur-proof pixel-space path (crop+resize in pixels, encode internally) — immune to input/output resolution mismatches"}),
            "source_image": ("IMAGE", {"tooltip": "source as IMAGE (with vae connected): overrides source_latent with exact pixel-space fitting — fixes blurry results from mismatched resolutions"}),
            "source_image_b": ("IMAGE", {"tooltip": "2nd reference as IMAGE (with vae)"}),
            "target_latent": ("LATENT", {"tooltip": "the same latent that feeds KSampler.latent_image (e.g. EmptySD3LatentImage): pre-encodes the source at the exact target size BEFORE sampling, so the VAE is not pulled onto the GPU mid-sampling where it can evict part of the diffusion model and slow every remaining step"}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "mustyrocks_k2edit"
    DESCRIPTION = "Adds the krea2_edit in-context source-preservation path (source latent as frame=1 tokens) to a Krea2 model."

    def patch(self, model, source_latent, source_latent_b=None, ref_boost=1.0, ref_boost_a=1.0,
              ref_boost_mask=None, ref_boost_mask_a=None, vae=None, source_image=None,
              source_image_b=None, fit_mode="fit", target_latent=None,
              source_images=None, ref_boosts=None, ref_boost_masks=None, **_future):
        if _future:
            print(f"[krea2edit] WARNING: workflow provides inputs this node version does not "
                  f"know ({', '.join(sorted(_future))}). The workflow is newer than the "
                  f"installed node pack. Update comfyui-krea2edit (Manager -> Update, or git "
                  f"pull) and restart ComfyUI. Continuing without them.", flush=True)
        m = model.clone()

        # Fail fast with a clear message instead of an AttributeError on step 1.
        # ComfyUI core support (commit 2a610155) wraps the raw SingleStreamDiT in a
        # model-base class (comfy.model_base.Krea2); BaseModel stores the raw DiT at
        # .diffusion_model, and that wrapper is what ModelPatcher.model holds at patch
        # time. The sampling-time forward receives the RAW DiT (BaseModel._apply_model
        # calls self.diffusion_model(...), whose own WrapperExecutor passes itself as
        # class_obj), so validate that inner object; older ComfyUI hands us the raw
        # DiT directly.
        _required = ("first", "last", "tmlp", "tproj", "txtfusion", "txtmlp",
                     "pe_embedder", "blocks", "channels", "patch", "tdim", "_unpack_context")
        dm0 = m.model
        inner = getattr(dm0, "diffusion_model", None)
        if inner is None:
            inner = getattr(dm0, "model", None)
        if inner is not None and all(hasattr(inner, a) for a in _required):
            dm0 = inner
        missing = [a for a in _required if not hasattr(dm0, a)]
        if missing:
            outer = type(m.model)
            where = f"{outer.__module__}.{outer.__name__}"
            if dm0 is m.model and inner is not None:
                # wrapper present but its raw model isn't a Krea2 DiT — name both
                where += f" (raw diffusion model: {type(inner).__module__}.{type(inner).__name__})"
            raise RuntimeError(
                f"[krea2edit] the connected model is not a Krea 2 SingleStreamDiT — its inner "
                f"model is {where}, missing attributes: {', '.join(missing)}. Connect a "
                f"model ComfyUI loads as SingleStreamDiT (Krea 2 Raw/Turbo, Kroma, or a "
                f"krea2 fine-tune), with the krea2_edit LoRA applied, to this node.")

        # A second Krea2EditModelPatch downstream would otherwise silently overwrite
        # this one (same wrapper key) — refuse instead.
        existing = m.model_options.get("transformer_options", {}) \
            .get("wrappers", {}).get(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, {}) \
            .get("krea2_edit")
        if existing:
            raise RuntimeError(
                "[krea2edit] the model already carries a krea2_edit patch — only one "
                "Krea2EditModelPatch per graph is supported (a second one would silently "
                "replace the first). Remove it or reconnect your graph.")

        # The target latent reaches the diffusion model already scaled (process_latent_in);
        # scale the source(s) the same way so all share one latent space.
        src_samples = model.model.process_latent_in(source_latent["samples"])
        if source_latent_b is not None:
            src_samples = [src_samples, model.model.process_latent_in(source_latent_b["samples"])]

        px_images = list(source_images) if source_images else []
        if not px_images:
            if source_image is not None:
                px_images.append(source_image)
            if source_image_b is not None:
                px_images.append(source_image_b)

        px_cache = {}   # pixel-path encoded sources, keyed per target resolution
        fwd_cache = {}  # step-invariant forward work (fit/proj/freqs/bias) — see krea2_edit_forward
        warned = {}     # once-only wrapper-level warnings
        mm = model.model  # for process_latent_in on the pixel path

        if fit_mode == "fit" and (vae is None or not px_images):
            print(f"[krea2edit] WARNING: fit_mode='fit' has NO EFFECT — it needs both "
                  f"'vae' and 'source_image' connected (the pixel path). Falling back to the "
                  f"latent crop path.", flush=True)

        # Pre-encode OUTSIDE the sampling window. vae.encode -> load_models_gpu ->
        # free_memory(keep_loaded=[]), which partially unloads whatever is resident —
        # including the diffusion model, if the first call lands inside the sampler.
        # Nothing re-expands it (sampler_helpers loads once, before the loop), so the
        # rest of the run streams weights from CPU every step. Running the encode here,
        # at node-execution time, restores the ordinary VAEEncode -> KSampler order
        # where the sampler evicts the VAE instead of the reverse.
        if vae is not None and px_images:
            if target_latent is not None:
                Hh, Ww = target_latent["samples"].shape[-2], target_latent["samples"].shape[-1]
                print(f"[krea2edit] pre-encoding sources at target {Hh * 8}x{Ww * 8}px "
                      f"(before sampling, fit_mode={fit_mode})", flush=True)
                for i, img in enumerate(px_images):
                    _fit_encode_image(img, vae, Hh, Ww, px_cache, (i, Hh, Ww), fit_mode)
            else:
                print("[krea2edit] NOTE: connect 'target_latent' (the same latent that feeds "
                      "KSampler.latent_image) to pre-encode the source here instead of on the "
                      "first sampling step. Without it the VAE is loaded mid-sampling and can "
                      "evict part of the diffusion model, slowing every remaining step.",
                      flush=True)

        def wrapper(executor, x, timesteps, context, *wargs, **kwargs):
            # ComfyUI signature drift (2026-07-19, commit c9602625 adds ref_latents):
            #   old: execute(x, t, ctx, attention_mask, transformer_options)
            #   new: execute(x, t, ctx, attention_mask, ref_latents, transformer_options)
            # Accept both: transformer_options is the trailing dict; any native
            # ref_latents are ignored (this patch supplies its own source path).
            transformer_options = kwargs.pop("transformer_options", None)
            if transformer_options is None:
                transformer_options = {}
                for a in reversed(wargs):
                    if isinstance(a, dict):
                        transformer_options = a
                        break
            # attention_mask is the first non-dict positional after context (both the old
            # and new signatures). Native _forward ignores it (txtfusion gets mask=None),
            # so we do too — but say so ONCE if upstream ever starts passing a real one.
            attn_mask = kwargs.get("attention_mask")
            if attn_mask is None:
                for a in wargs:
                    if not isinstance(a, dict):
                        attn_mask = a
                        break
            if attn_mask is not None and "attn_mask" not in warned:
                print("[krea2edit] NOTE: upstream passed a non-None attention_mask; this node "
                      "ignores it (native Krea2 _forward does the same — txtfusion uses "
                      "mask=None). If you see artifacts, report it.", flush=True)
                warned["attn_mask"] = True
            dm = executor.class_obj  # the SingleStreamDiT instance
            src = src_samples
            if vae is not None and px_images:
                if not px_cache:
                    print(f"[krea2edit] pixel path ACTIVE (fit_mode={fit_mode})", flush=True)
                xx = _to_4d(x)
                Hh, Ww = xx.shape[-2], xx.shape[-1]
                lats = [
                    mm.process_latent_in(
                        _fit_encode_image(img, vae, Hh, Ww, px_cache, (i, Hh, Ww), fit_mode)
                    )
                    for i, img in enumerate(px_images)
                ]
                src = lats[0] if len(lats) == 1 else lats
            v = krea2_edit_forward(dm, x, timesteps, context, src, transformer_options,
                                   ref_boost=ref_boost, ref_boost_a=ref_boost_a,
                                   ref_boost_mask=ref_boost_mask,
                                   ref_boost_mask_a=ref_boost_mask_a,
                                   ref_native=(fit_mode == "fit" and vae is not None
                                               and bool(px_images)),
                                   pos_mode=("stride1" if fit_mode == "fit" else "anchor"),
                                   fwd_cache=fwd_cache,
                                   ref_boosts=ref_boosts,
                                   ref_boost_masks=ref_boost_masks)
            return v

        to = m.model_options.setdefault("transformer_options", {})
        comfy.patcher_extension.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, "krea2_edit", wrapper, to
        )
        return (m,)


