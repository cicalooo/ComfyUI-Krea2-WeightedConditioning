"""Parse (phrase:weight) and map phrases onto Krea 2 / Qwen3-VL token ids."""

from __future__ import annotations

import logging
import re
from typing import Callable, Iterable, Optional, Sequence

LOG = logging.getLogger("krea2.weighted")

# Qwen chat specials used by Krea 2's template.
QWEN_IM_START = 151644
QWEN_USER = 872
QWEN_NL = 198
QWEN_IM_END = 151645

WEIGHT_RE = re.compile(r"\(([^():]+):(-?\d*\.?\d+)\)")

DecodeFn = Callable[[Sequence[int]], str]


def parse_weighted_terms(text: str) -> tuple[list[tuple[str, float]], str]:
    """Return ([(phrase, weight), ...], cleaned_text) with parentheses stripped."""
    terms = [(m.group(1).strip(), float(m.group(2))) for m in WEIGHT_RE.finditer(text)]
    clean = WEIGHT_RE.sub(lambda m: m.group(1), text)
    return terms, clean


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


def find_subsequence(seq: Sequence[int], sub: Sequence[int], lo: int, hi: int) -> list[int]:
    out: list[int] = []
    n = len(sub)
    if n == 0:
        return out
    last = hi - n + 1
    for i in range(lo, last):
        if list(seq[i : i + n]) == list(sub):
            out.append(i)
    return out


def clip_token_ids(clip, text: str) -> list[int]:
    tok = clip.tokenize(text)
    key = next(iter(tok))
    return [t[0] for t in tok[key][0]]


def user_span_from_ids(ids: Sequence[int]) -> tuple[int, int]:
    start, end = user_content_span(ids)
    if start is None:
        return 0, len(ids)
    return start, end


def phrase_sub_ids(clip, phrase: str) -> Optional[list[int]]:
    """Token ids of ``phrase`` inside a chat-templated encode, user span only."""
    ids = clip_token_ids(clip, phrase)
    start, end = user_content_span(ids)
    if start is None:
        return ids if ids else None
    sub = list(ids[start:end])
    return sub or None


def match_leading_space(
    clip, ids: Sequence[int], start: int, end: int, phrase: str
) -> tuple[list[int], list[int]]:
    """Return (hit_starts, sub_ids) of ``phrase`` in ``ids[start:end]``."""
    for variant in (" " + phrase, phrase):
        sub = phrase_sub_ids(clip, variant)
        if not sub:
            continue
        hits = find_subsequence(ids, sub, start, end)
        if hits:
            return hits, sub
    return [], []


def match_decode_window(
    ids: Sequence[int],
    start: int,
    end: int,
    phrase: str,
    decode_fn: DecodeFn,
) -> tuple[list[int], list[int]]:
    """Find token windows whose decode equals the phrase (handles BPE splits)."""
    target = phrase.strip()
    if not target or decode_fn is None:
        return [], []
    n = end - start
    # Exact-window search, shortest first so we don't swallow extra tokens.
    for length in range(1, n + 1):
        for i in range(start, end - length + 1):
            window = list(ids[i : i + length])
            try:
                decoded = decode_fn(window).strip()
            except Exception:
                continue
            if decoded == target:
                return [i], window
    # Containment fallback: first shortest window that contains the phrase.
    for length in range(1, n + 1):
        for i in range(start, end - length + 1):
            window = list(ids[i : i + length])
            try:
                decoded = decode_fn(window)
            except Exception:
                continue
            if target in decoded:
                return [i], window
    return [], []


def try_clip_decode_fn(clip) -> Optional[DecodeFn]:
    """Best-effort id→text decoder from a Comfy CLIP object."""
    candidates = []
    tokenizer = getattr(clip, "tokenizer", None)
    if tokenizer is not None:
        candidates.append(tokenizer)
        inner = getattr(tokenizer, "tokenizer", None)
        if inner is not None:
            candidates.append(inner)
    csm = getattr(clip, "cond_stage_model", None)
    if csm is not None:
        for attr in ("tokenizer", "llama", "qwen"):
            obj = getattr(csm, attr, None)
            if obj is None:
                continue
            candidates.append(obj)
            tok = getattr(obj, "tokenizer", None)
            if tok is not None:
                candidates.append(tok)

    for obj in candidates:
        decode = getattr(obj, "decode", None)
        if callable(decode):
            def _fn(ids, _decode=decode):
                return _decode(list(ids))
            return _fn
    return None


def match_phrase(
    clip,
    ids: Sequence[int],
    start: int,
    end: int,
    phrase: str,
    match_mode: str = "leading_space",
    decode_fn: Optional[DecodeFn] = None,
) -> tuple[list[int], list[int]]:
    """Return (hit_starts, sub_ids) for ``phrase`` in the user span."""
    if match_mode == "decode_window":
        fn = decode_fn or try_clip_decode_fn(clip)
        if fn is not None:
            hits, sub = match_decode_window(ids, start, end, phrase, fn)
            if hits:
                return hits, sub
        # fall through
    return match_leading_space(clip, ids, start, end, phrase)


def weight_factors(w: float, strength: float, emphasis_mode: str) -> tuple[float, float]:
    """(v_factor, k_bias) from a user weight and global strength."""
    strength = float(strength)
    w = float(w)
    if emphasis_mode == "value_scale":
        return 1.0 + strength * (w - 1.0), 0.0
    if w > 1.0:
        k_bias = strength * (w - 1.0) * 2.0
        if emphasis_mode == "both":
            v_factor = 1.0 + strength * (w - 1.0) * 0.5
        else:
            v_factor = 1.0
        return v_factor, k_bias
    return 1.0 + strength * (w - 1.0), 0.0


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


def resolve_weight_pairs(
    clip,
    ids: Sequence[int],
    cond_len: int,
    terms: Iterable[tuple[str, float]],
    strength: float,
    emphasis_mode: str = "k_bias",
    match_mode: str = "leading_space",
    unmatched: str = "warn",
    decode_fn: Optional[DecodeFn] = None,
) -> tuple[list[tuple[int, float, float]], str]:
    """Map weighted phrases onto conditioning positions.

    Returns (weight_pairs, debug_text). Each pair is
    ``(cond_pos, v_factor, k_bias)``.
    """
    visible_start = len(ids) - cond_len
    start, end = user_content_span(ids)
    if start is None:
        start, end = visible_start, len(ids)

    pairs: list[tuple[int, float, float]] = []
    debug_lines: list[str] = []
    misses: list[str] = []

    for phrase, w in terms:
        v_factor, k_bias = weight_factors(w, strength, emphasis_mode)
        hits, sub = match_phrase(
            clip, ids, start, end, phrase, match_mode=match_mode, decode_fn=decode_fn
        )
        positions: list[int] = []
        for mi in hits:
            for off in range(len(sub)):
                cp = mi + off - visible_start
                if 0 <= cp < cond_len:
                    positions.append(cp)
                    pairs.append((cp, v_factor, k_bias))
        if not positions:
            misses.append(phrase)
            debug_lines.append("{!r} w={} → not found".format(phrase, w))
            continue
        debug_lines.append(
            "{!r} w={} → pos={} v={:.4f} k_bias={:.4f}".format(
                phrase, w, positions, v_factor, k_bias
            )
        )

    if misses:
        msg = "Krea2 Weighted Conditioning: phrase(s) not found: {}".format(
            ", ".join(repr(p) for p in misses)
        )
        if unmatched == "error":
            raise ValueError(msg)
        if unmatched != "ignore":
            LOG.warning(msg)

    return pairs, "\n".join(debug_lines)


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
