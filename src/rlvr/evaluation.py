"""The fixed checkpoint evaluation subset and pass@1.

The test pool is fixed before any arm runs. Checkpoint evaluation uses a subset
of that pool, chosen once per run by a seed that is separate from the data split
seed. The subset is not the first questions in file order. The seed and the
sizes are recorded in `eval.json`, so a reader can rebuild the same subset from
the pool.

Selection happens without replacement. The chosen questions keep their pool
order, so two runs with the same seed list the subset in the same order. The
selection is deterministic across machines because `random.Random` is used with
an integer seed and no other input.

pass@1 is the only metric. It is the fraction of evaluated questions whose
greedy generation passed every task test.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:  # pragma: no cover - import cycle guard for type checkers
    from .experiment import Prompt

__all__ = [
    "SUBSET_METHOD",
    "pass_at_1",
    "select_eval_subset",
    "subset_selector",
]

#: Name recorded in `eval.json` for the subset method.
SUBSET_METHOD = "seeded_sample"


def select_eval_subset(
    prompts: Sequence[Prompt], *, count: int, seed: int
) -> tuple[Prompt, ...]:
    """Pick `count` questions from the pool with a seeded sample.

    A pool no larger than `count` is returned whole, in pool order. Questions
    keep their pool order in the result. Raises ValueError for an empty pool or
    a count below one.
    """
    if count < 1:
        raise ValueError(f"count must be at least 1, got {count}")
    pool = tuple(prompts)
    if not pool:
        raise ValueError("cannot select an evaluation subset from an empty prompt pool")
    if count >= len(pool):
        return pool

    indices = random.Random(seed).sample(range(len(pool)), count)
    return tuple(pool[index] for index in sorted(indices))


def subset_selector(
    prompts: Sequence[Prompt], *, count: int, seed: int, subset: Sequence[Prompt]
) -> dict[str, Any]:
    """The selector record written to `eval.json`.

    It names the method, the seed, the requested and actual sizes, and the pool
    size, so the subset can be rebuilt from the same test pool.
    """
    return {
        "method": SUBSET_METHOD,
        "seed": seed,
        "requested": count,
        "selected": len(subset),
        "pool_size": len(prompts),
    }


def pass_at_1(passed: Sequence[bool]) -> float:
    """The fraction of evaluated questions that passed every test."""
    results = tuple(passed)
    if not results:
        raise ValueError("pass@1 needs at least one evaluated question")
    return sum(1 for result in results if result) / len(results)
