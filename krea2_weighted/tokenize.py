"""Map main/aux prompt spans onto Krea 2 / Qwen3-VL token ids (including vision pads)."""

from __future__ import annotations

from typing import Optional, Sequence

import torch

# Qwen chat specials used by Krea 2's template.
QWEN_IM_START = 151644
QWEN_USER = 872
QWEN_NL = 198
QWEN_IM_END = 151645
QWEN_IMAGE_PAD = 151655  # <|image_pad|>; vision tower expands this placeholder.


def user_content_span(ids: Sequence[int]) -> tuple[Optional[int], Optional[int]]:
    """Token span of the user prompt between ``<|im_start|>user\\n`` and ``<|im_end|>``."""
    n = len(ids)
    for i in range(n - 2):
        if ids[i] == QWEN_IM_START and ids[i + 1] == QWEN_USER and ids[i + 2] == QWEN_NL:
            start = i + 3
            end = start
            while end < n and ids[end] != QWEN_IM_END:
                end += 1
            return start, end
    return None, None


def is_image_marker(elem) -> bool:
    """True for Qwen image placeholders (id, dict, or tensor-backed embed)."""
    if isinstance(elem, dict) and elem.get("type") == "image":
        return True
    if isinstance(elem, dict) and elem.get("original_type") == "image":
        return True
    if torch.is_tensor(elem):
        return True
    if isinstance(elem, (int, float)) and not isinstance(elem, bool):
        return int(elem) == QWEN_IMAGE_PAD
    return False


def normalize_token_id(elem):
    """Stable id for prefix matching. Image embeds collapse to ``QWEN_IMAGE_PAD``."""
    if is_image_marker(elem):
        return QWEN_IMAGE_PAD
    if isinstance(elem, (int, float)) and not isinstance(elem, bool):
        return int(elem)
    return ("embed", type(elem).__name__)


def token_ids_from_tok(tok) -> list:
    key = next(iter(tok))
    return [normalize_token_id(t[0]) for t in tok[key][0]]


def clip_token_ids(clip, text: str, **tokenize_kwargs) -> list:
    tok = clip.tokenize(text, **tokenize_kwargs)
    return token_ids_from_tok(tok)


def user_span_from_ids(ids: Sequence[int]) -> tuple[int, int]:
    start, end = user_content_span(ids)
    if start is None:
        return 0, len(ids)
    return start, end


def parse_block_range(spec: str, n_blocks: int) -> list[int]:
    """Parse ``all``, ``0-27``, ``4,8,12``, or mixed ``0-5,10,20-27``."""
    spec = (spec or "all").strip().lower()
    if spec in ("", "all", "*"):
        return list(range(n_blocks))
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            lo, hi = int(a), int(b)
            if lo > hi:
                lo, hi = hi, lo
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(part))
    seen = set()
    ordered = []
    for i in out:
        if i < 0 or i >= n_blocks or i in seen:
            continue
        seen.add(i)
        ordered.append(i)
    if not ordered:
        raise ValueError("block_range {!r} selected no blocks in 0..{}".format(spec, n_blocks - 1))
    return ordered


def longest_prefix_len(a: Sequence[int], b: Sequence[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def aux_id_span_in_combined(ids_combined: Sequence[int], ids_main: Sequence[int]) -> tuple[int, int]:
    """Return ``(aux_start, aux_end)`` in combined token ids (user content suffix).

    Encodes main and main+aux separately, then treats the unmatched suffix of the
    combined user span as the aux/moodboard tokens. Survives BPE glue at the join.
    """
    cs, ce = user_span_from_ids(ids_combined)
    ms, me = user_span_from_ids(ids_main)
    main_user = list(ids_main[ms:me])
    comb_user = list(ids_combined[cs:ce])
    prefix = longest_prefix_len(comb_user, main_user)
    aux_start = cs + prefix
    return aux_start, ce


def slice_to_cond_pairs(
    id_start: int,
    id_end: int,
    visible_start: int,
    cond_len: int,
    v_factor: float,
    k_bias: float,
) -> list[tuple[int, float, float]]:
    pairs = []
    for i in range(id_start, id_end):
        cp = i - visible_start
        if 0 <= cp < cond_len:
            pairs.append((cp, v_factor, k_bias))
    return pairs


def vision_expansion_extra(ids: Sequence, cond_len: int) -> int:
    """Extra cond tokens Qwen inserts when expanding ``<|image_pad|>`` placeholders."""
    start, _ = user_span_from_ids(ids)
    unexpanded_visible = len(ids) - start
    return max(0, int(cond_len) - unexpanded_visible)


def slice_to_cond_pairs_after_vision(
    ids: Sequence,
    id_start: int,
    id_end: int,
    cond_len: int,
    v_factor: float,
    k_bias: float,
) -> list[tuple[int, float, float]]:
    """Map an unexpanded id span onto cond positions after vision-token expansion.

    Image placeholders (and system / chat-template tokens outside the span) are
    never weighted. Expansion extras are attributed to ``QWEN_IMAGE_PAD`` slots
    so aux text sits after the vision block.
    """
    start, _ = user_span_from_ids(ids)
    extra = vision_expansion_extra(ids, cond_len)
    image_ix = [j for j, t in enumerate(ids) if t == QWEN_IMAGE_PAD]
    extras_at: dict[int, int] = {}
    n = len(image_ix)
    if n and extra:
        base, rem = divmod(extra, n)
        for k, j in enumerate(image_ix):
            extras_at[j] = base + (1 if k >= n - rem else 0)

    pairs: list[tuple[int, float, float]] = []
    acc_extra = 0
    for i, tok in enumerate(ids):
        if id_start <= i < id_end and tok != QWEN_IMAGE_PAD:
            cp = i + acc_extra - start
            if 0 <= cp < cond_len:
                pairs.append((cp, v_factor, k_bias))
        acc_extra += extras_at.get(i, 0)
    return pairs


def mix_factors(aux_strength: float, emphasis_mode: str = "value_scale") -> tuple[float, float]:
    """Map a 0..N aux strength to (v_factor, k_bias). 1.0 is a no-op."""
    s = float(aux_strength)
    if emphasis_mode == "k_bias":
        if s > 1.0:
            return 1.0, (s - 1.0) * 2.0
        return s, 0.0
    if emphasis_mode == "both":
        if s > 1.0:
            return 1.0 + (s - 1.0) * 0.5, (s - 1.0) * 2.0
        return s, 0.0
    return s, 0.0
