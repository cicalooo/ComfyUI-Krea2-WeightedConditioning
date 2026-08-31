"""Krea 2 self-attention patch: per-token V scale and k-bias."""

from __future__ import annotations

import types

import torch

WEIGHTS_KEY = "krea2_token_weights"
APPLY_TO_KEY = "krea2_weight_apply_to"


def compose_attn_mask(existing, key_bias):
    """Add Prompt Mix key bias onto an existing additive / bool attention mask.

    K2Edit ``ref_boost`` passes a (1, 1, L, L) additive mask into attn.forward.
    Key bias is a per-key vector ``(1, L)``. They add (broadcast) instead of
    replacing, so identity-edit fidelity and aux emphasis can coexist.
    """
    if key_bias is None:
        return existing
    if existing is None:
        return key_bias

    e = existing
    if e.dtype == torch.bool:
        fill = torch.finfo(key_bias.dtype).min
        e = torch.zeros(e.shape, dtype=key_bias.dtype, device=e.device)
        e = e.masked_fill(~existing.bool(), fill)

    kb = key_bias.to(device=e.device, dtype=e.dtype)
    # (1, L) -> (1, 1, L) -> (1, 1, 1, L) so it adds on the key axis of (1, 1, L, L).
    while kb.ndim < e.ndim:
        if kb.ndim == 1:
            kb = kb.unsqueeze(0)
        else:
            kb = kb.unsqueeze(-2)
    return e + kb


def _should_apply(transformer_options: dict) -> bool:
    apply_to = transformer_options.get(APPLY_TO_KEY, "cond")
    if apply_to == "both":
        return True
    cou = transformer_options.get("cond_or_uncond")
    if cou is None:
        return True
    if apply_to == "uncond":
        return 1 in cou
    # cond
    return 0 in cou


def krea2_attn_forward_weight(self, x, freqs=None, mask=None, transformer_options=None):
    from einops import rearrange
    from comfy.ldm.flux.math import apply_rope
    from comfy.ldm.modules.attention import attention_pytorch, optimized_attention

    if transformer_options is None:
        transformer_options = {}

    q, k, v, gate = self.wq(x), self.wk(x), self.wv(x), self.gate(x)
    q = rearrange(q, "B L (H D) -> B H L D", H=self.heads)
    k = rearrange(k, "B L (H D) -> B H L D", H=self.kvheads)
    v = rearrange(v, "B L (H D) -> B H L D", H=self.kvheads)

    weights = transformer_options.get(WEIGHTS_KEY)
    apply = bool(weights) and _should_apply(transformer_options)
    if apply:
        v = v.clone()
        seq_v = v.shape[2]
        for pos, v_factor, _ in weights:
            if v_factor != 1.0 and 0 <= pos < seq_v:
                v[:, :, pos] = v[:, :, pos] * v_factor

    q, k = self.qknorm(q, k)
    if freqs is not None:
        q, k = apply_rope(q, k, freqs)
    if self.kvheads != self.heads:
        rep = self.heads // self.kvheads
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)

    bias = None
    if apply and any(kb != 0.0 for _, _, kb in weights):
        bias = q.new_zeros(1, k.shape[2])
        seq_k = bias.shape[1]
        for pos, _, kb in weights:
            if kb != 0.0 and 0 <= pos < seq_k:
                bias[:, pos] = kb

    attn_mask = compose_attn_mask(mask, bias) if bias is not None else mask
    if bias is not None:
        out = attention_pytorch(q, k, v, self.heads, mask=attn_mask, skip_reshape=True)
    else:
        # value_scale-only: keep the caller's mask (K2Edit ref_boost).
        out = optimized_attention(
            q, k, v, self.heads, mask=mask, skip_reshape=True,
            transformer_options=transformer_options,
        )
    return self.wo(out * torch.sigmoid(gate))


class Krea2WeightPatch:
    def __get__(self, obj, objtype=None):
        return types.MethodType(krea2_attn_forward_weight, obj)
