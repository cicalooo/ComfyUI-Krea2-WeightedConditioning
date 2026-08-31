import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from krea2_weighted.tokenize import (
    QWEN_IM_END,
    QWEN_IM_START,
    QWEN_NL,
    QWEN_USER,
    find_subsequence,
    parse_block_range,
    parse_weighted_terms,
    resolve_weight_pairs,
    user_content_span,
    weight_factors,
)


class FakeClip:
    """Maps a string to a chat-wrapped id sequence: template + user ids + im_end."""

    def __init__(self, vocab=None):
        # word -> single id (or list)
        self.vocab = vocab or {}
        self._next = 10

    def _ids_for(self, text):
        if text in self.vocab:
            v = self.vocab[text]
            return list(v) if isinstance(v, (list, tuple)) else [v]
        # split on spaces; assign stable ids
        parts = text.split(" ") if text else []
        out = []
        for i, p in enumerate(parts):
            key = (" " + p) if i > 0 or (text.startswith(" ") and i == 0) else p
            if key not in self.vocab:
                # leading-space form of the same word shares a distinct id
                if key not in self.vocab:
                    self.vocab[key] = self._next
                    self._next += 1
            v = self.vocab[key]
            out.extend(v if isinstance(v, list) else [v])
        if text == "" or (not parts and text):
            if text not in self.vocab:
                self.vocab[text] = [self._next]
                self._next += 1
            v = self.vocab[text]
            return list(v) if isinstance(v, list) else [v]
        return out

    def tokenize(self, text):
        user = self._ids_for(text)
        ids = [QWEN_IM_START, QWEN_USER, QWEN_NL] + user + [QWEN_IM_END]
        pairs = [(i, 1.0) for i in ids]
        return {"qwen3vl_4b": [pairs]}


def test_parse_weighted_terms():
    terms, clean = parse_weighted_terms("a (red:1.5) (hat:-1) dress")
    assert terms == [("red", 1.5), ("hat", -1.0)]
    assert clean == "a red hat dress"


def test_user_content_span():
    ids = [1, QWEN_IM_START, QWEN_USER, QWEN_NL, 10, 11, QWEN_IM_END, 99]
    start, end = user_content_span(ids)
    assert (start, end) == (4, 6)


def test_find_subsequence():
    seq = [1, 2, 3, 2, 3, 4]
    assert find_subsequence(seq, [2, 3], 0, 6) == [1, 3]


def test_weight_factors_table():
    rows = [
        (1.0, 1.0, "k_bias", 1.0, 0.0),
        (1.5, 1.0, "k_bias", 1.0, 1.0),
        (0.5, 1.0, "k_bias", 0.5, 0.0),
        (-1.0, 1.0, "k_bias", -1.0, 0.0),
        (2.0, 0.5, "k_bias", 1.0, 1.0),
        (1.5, 1.0, "value_scale", 1.5, 0.0),
        (2.0, 1.0, "both", 1.5, 2.0),
    ]
    for w, s, mode, vf, kb in rows:
        got = weight_factors(w, s, mode)
        assert got == (vf, kb), (w, s, mode, got)


def test_parse_block_range():
    assert parse_block_range("all", 4) == [0, 1, 2, 3]
    assert parse_block_range("0-2", 8) == [0, 1, 2]
    assert parse_block_range("4,8,12", 16) == [4, 8, 12]
    assert parse_block_range("0-1,3", 5) == [0, 1, 3]


def test_resolve_weight_pairs_match_and_error():
    clip = FakeClip({"red": 20, " hat": 21, "hat": 21, "a": 19, " dress": 22, "dress": 22})
    clip.vocab[" red"] = 30
    clip.vocab["red"] = 20
    clip.vocab[" hat"] = 21
    clip.vocab["hat"] = 21

    clip_ids = clip_token_ids_via(clip, "a red hat dress")
    # cond_len = everything after template start (ids minus first 3)
    cond_len = len(clip_ids) - 3
    terms = [("red", 1.5), ("hat", -1.0)]
    pairs, debug = resolve_weight_pairs(
        clip, clip_ids, cond_len, terms, strength=1.0, unmatched="warn"
    )
    assert pairs, debug
    positions = [p[0] for p in pairs]
    assert len(positions) >= 2
    # hat is removal
    hat_pairs = [p for p in pairs if p[1] < 1]
    red_pairs = [p for p in pairs if p[2] != 0]
    assert hat_pairs
    assert red_pairs


def clip_token_ids_via(clip, text):
    tok = clip.tokenize(text)
    key = next(iter(tok))
    return [t[0] for t in tok[key][0]]


def test_unmatched_error():
    clip = FakeClip({"hello": 5})
    ids = clip_token_ids_via(clip, "hello")
    cond_len = len(ids) - 3
    try:
        resolve_weight_pairs(
            clip, ids, cond_len, [("missing", 2.0)], strength=1.0, unmatched="error"
        )
    except ValueError as e:
        assert "missing" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_aux_span_suffix():
    from krea2_weighted.tokenize import aux_id_span_in_combined, mix_factors

    main_user = [10, 11]
    aux_user = [20, 21, 22]
    prefix = [QWEN_IM_START, QWEN_USER, QWEN_NL]
    suffix = [QWEN_IM_END]
    ids_main = prefix + main_user + suffix
    ids_comb = prefix + main_user + aux_user + suffix
    s, e = aux_id_span_in_combined(ids_comb, ids_main)
    assert list(ids_comb[s:e]) == aux_user
    assert mix_factors(0.45, "value_scale") == (0.45, 0.0)
    assert mix_factors(1.0, "value_scale") == (1.0, 0.0)


def test_decode_window():
    clip = FakeClip()
    clip.vocab["red"] = 20
    clip.vocab[" hat"] = 21
    ids = [QWEN_IM_START, QWEN_USER, QWEN_NL, 20, 21, QWEN_IM_END]
    table = {20: "red", 21: " hat"}

    def decode(window):
        return "".join(table[i] for i in window)

    from krea2_weighted.tokenize import match_decode_window

    hits, sub = match_decode_window(ids, 3, 5, "red hat", decode)
    assert hits == [3]
    assert sub == [20, 21]
