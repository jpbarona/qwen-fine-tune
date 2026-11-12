"""Tests for the fixed evaluation subset and pass@1.

The checkpoint subset is chosen once per run from the test pool with a seed
that is separate from the data split seed. It must not be the first questions
in file order.
"""

from __future__ import annotations

import pytest

from rlvr.evaluation import pass_at_1, select_eval_subset, subset_selector
from rlvr.experiment import Prompt


def make_prompts(count: int) -> list[Prompt]:
    return [
        Prompt(prompt_id=f"mbpp-{index:04d}", problem=f"problem {index}")
        for index in range(count)
    ]


def test_the_subset_is_fixed_for_a_seed() -> None:
    pool = make_prompts(250)
    first = select_eval_subset(pool, count=100, seed=2026)
    second = select_eval_subset(pool, count=100, seed=2026)
    assert first == second
    assert len(first) == 100


def test_the_subset_is_not_the_first_questions_in_file_order() -> None:
    pool = make_prompts(250)
    subset = select_eval_subset(pool, count=100, seed=2026)
    assert {prompt.prompt_id for prompt in subset} != {
        prompt.prompt_id for prompt in pool[:100]
    }


def test_the_subset_keeps_the_pool_order() -> None:
    pool = make_prompts(250)
    subset = select_eval_subset(pool, count=100, seed=2026)
    positions = [pool.index(prompt) for prompt in subset]
    assert positions == sorted(positions)


def test_a_different_seed_picks_a_different_subset() -> None:
    pool = make_prompts(250)
    first = select_eval_subset(pool, count=100, seed=2026)
    second = select_eval_subset(pool, count=100, seed=2027)
    assert first != second


def test_a_pool_smaller_than_the_request_returns_every_question() -> None:
    pool = make_prompts(3)
    assert select_eval_subset(pool, count=100, seed=1) == tuple(pool)


def test_selection_rejects_an_empty_pool() -> None:
    with pytest.raises(ValueError, match="pool"):
        select_eval_subset([], count=100, seed=1)


@pytest.mark.parametrize("count", [0, -1])
def test_selection_rejects_a_bad_count(count: int) -> None:
    with pytest.raises(ValueError, match="count"):
        select_eval_subset(make_prompts(3), count=count, seed=1)


def test_pass_at_1_is_the_fraction_that_passed() -> None:
    assert pass_at_1([True, False, True, True]) == pytest.approx(0.75)


def test_pass_at_1_of_none_passing_is_zero() -> None:
    assert pass_at_1([False, False]) == 0.0


def test_pass_at_1_rejects_an_empty_result() -> None:
    with pytest.raises(ValueError, match="at least one"):
        pass_at_1([])


def test_the_selector_records_the_method_seed_and_sizes() -> None:
    pool = make_prompts(250)
    subset = select_eval_subset(pool, count=100, seed=9)
    selector = subset_selector(pool, count=100, seed=9, subset=subset)
    assert selector == {
        "method": "seeded_sample",
        "seed": 9,
        "requested": 100,
        "selected": 100,
        "pool_size": 250,
    }
