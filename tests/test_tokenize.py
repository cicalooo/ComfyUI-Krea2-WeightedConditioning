import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from krea2_weighted.tokenize import (
    QWEN_IM_END,
    QWEN_IM_START,
    QWEN_NL,
    QWEN_USER,
    parse_block_range,
    user_content_span,
)


def test_user_content_span():
    ids = [1, QWEN_IM_START, QWEN_USER, QWEN_NL, 10, 11, QWEN_IM_END, 99]
    start, end = user_content_span(ids)
    assert (start, end) == (4, 6)


def test_parse_block_range():
    assert parse_block_range("all", 4) == [0, 1, 2, 3]
    assert parse_block_range("0-2", 8) == [0, 1, 2]
    assert parse_block_range("4,8,12", 16) == [4, 8, 12]
    assert parse_block_range("0-1,3", 5) == [0, 1, 3]


def test_normalize_image_markers_prefix():
    import torch
    from krea2_weighted.tokenize import (
        QWEN_IMAGE_PAD,
        aux_id_span_in_combined,
        normalize_token_id,
        slice_to_cond_pairs_after_vision,
    )

    img_a = {"type": "image", "data": torch.zeros(1, 2, 2, 3)}
    img_b = {"type": "image", "data": torch.ones(1, 2, 2, 3)}
    assert normalize_token_id(img_a) == QWEN_IMAGE_PAD
    assert normalize_token_id(img_b) == QWEN_IMAGE_PAD
    assert normalize_token_id(torch.zeros(4)) == QWEN_IMAGE_PAD

    prefix = [QWEN_IM_START, QWEN_USER, QWEN_NL, 151652, QWEN_IMAGE_PAD, 151653]
    main_user = [10, 11]
    aux_user = [20, 21]
    suffix = [QWEN_IM_END]
    ids_main = prefix + main_user + suffix
    ids_comb = prefix + main_user + aux_user + suffix
    s, e = aux_id_span_in_combined(ids_comb, ids_main)
    assert list(ids_comb[s:e]) == aux_user

    user_start = 3
    cond_len = (len(ids_comb) - user_start) + 3
    pairs = slice_to_cond_pairs_after_vision(ids_comb, s, e, cond_len, 0.45, 0.0)
    pos = [p[0] for p in pairs]
    assert pos == [i - user_start + 3 for i in range(s, e)]
    assert min(pos) >= 2 + 4


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
