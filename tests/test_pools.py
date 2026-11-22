"""Tests for the pools built from screen rows.

The module is pure. Rows come from `screening.ScreenRow`, which is the schema
the screen writes to `rows.jsonl`, and no model or file is involved.
"""

from __future__ import annotations

import pytest

from rlvr.pools import DEFAULT_PROMOTION_BAR, Pools, PoolsError, from_rows
from rlvr.screening import ScreenRow


def row(prompt_id: str, passed: int, *, samples: int = 16) -> ScreenRow:
    return ScreenRow(
        prompt_id=prompt_id,
        samples=samples,
        passed=passed,
        pass_rate=passed / samples,
    )


def test_zero_passes_go_to_parked() -> None:
    pools = from_rows([row("q0", 0)], samples=16)
    assert pools.parked == ("q0",)
    assert pools.train == ()
    assert pools.retired == ()


def test_every_pass_goes_to_retired() -> None:
    pools = from_rows([row("q16", 16)], samples=16)
    assert pools.retired == ("q16",)
    assert pools.train == ()
    assert pools.parked == ()


def test_a_partial_count_goes_to_train() -> None:
    for passed in (1, 8, 15):
        pools = from_rows([row("q", passed)], samples=16)
        assert pools.train == ("q",)
        assert pools.retired == ()
        assert pools.parked == ()


def test_the_three_boundaries_split_into_three_pools() -> None:
    pools = from_rows(
        [
            row("none", 0),
            row("one", 1),
            row("almost", 15),
            row("all", 16),
        ],
        samples=16,
    )
    assert pools == Pools(
        train=("almost", "one"),
        retired=("all",),
        parked=("none",),
        samples=16,
    )


def test_a_row_with_another_sample_count_names_the_prompt() -> None:
    rows = [row("good", 8), row("short", 4, samples=8)]
    with pytest.raises(PoolsError) as caught:
        from_rows(rows, samples=16)
    assert str(caught.value) == "prompt 'short' used 8 samples, not 16"


def test_a_row_with_more_samples_than_configured_names_the_prompt() -> None:
    rows = [row("good", 8), row("long", 8, samples=32)]
    with pytest.raises(PoolsError) as caught:
        from_rows(rows, samples=16)
    assert str(caught.value) == "prompt 'long' used 32 samples, not 16"


def test_empty_rows_give_three_empty_tuples() -> None:
    pools = from_rows([], samples=16)
    assert pools == Pools(train=(), retired=(), parked=(), samples=16)


def test_pools_are_sorted_by_prompt_id() -> None:
    pools = from_rows(
        [
            row("train-c", 8),
            row("parked-b", 0),
            row("train-a", 4),
            row("retired-d", 16),
            row("parked-a", 0),
        ],
        samples=16,
    )
    assert pools.train == ("train-a", "train-c")
    assert pools.parked == ("parked-a", "parked-b")
    assert pools.retired == ("retired-d",)


def test_input_order_does_not_change_the_pools() -> None:
    rows = [
        row("b", 0),
        row("a", 16),
        row("d", 8),
        row("c", 3),
    ]
    assert from_rows(rows, samples=16) == from_rows(list(reversed(rows)), samples=16)


def test_json_rows_with_the_screen_schema_are_accepted() -> None:
    rows = [
        {"prompt_id": "json-a", "samples": 16, "passed": 0, "pass_rate": 0.0},
        {"prompt_id": "json-b", "samples": 16, "passed": 16, "pass_rate": 1.0},
        {"prompt_id": "json-c", "samples": 16, "passed": 5, "pass_rate": 0.3125},
    ]
    pools = from_rows(rows, samples=16)
    assert pools.parked == ("json-a",)
    assert pools.retired == ("json-b",)
    assert pools.train == ("json-c",)


def test_a_duplicate_prompt_id_is_rejected() -> None:
    with pytest.raises(PoolsError) as caught:
        from_rows([row("same", 0), row("same", 16)], samples=16)
    assert "same" in str(caught.value)


def test_a_pass_count_above_the_sample_count_is_rejected() -> None:
    with pytest.raises(PoolsError) as caught:
        from_rows([row("odd", 17)], samples=16)
    assert "odd" in str(caught.value)


def test_the_sample_count_is_kept_on_the_pools() -> None:
    assert from_rows([], samples=16).samples == 16


def test_the_default_promotion_bar_is_two() -> None:
    assert DEFAULT_PROMOTION_BAR == 2


def test_a_promotion_bar_above_the_samples_is_allowed() -> None:
    pools = from_rows([], samples=1, promotion_bar=2)
    assert pools == Pools(train=(), retired=(), parked=(), samples=1)


def test_empty_rows_with_a_non_default_bar_give_three_empty_tuples() -> None:
    pools = from_rows([], samples=16, promotion_bar=5)
    assert pools == Pools(train=(), retired=(), parked=(), samples=16)


def test_a_promotion_bar_below_one_is_rejected() -> None:
    with pytest.raises(PoolsError):
        from_rows([], samples=16, promotion_bar=0)
