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

`band_eval_prompts` reads a band from the training split, which carries the
difficulty labels. It keeps the prompts labelled with that band and applies the
same seeded sample, so a medium pass@1 can be measured without the test split
and without a run folder.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from .stages import BANDS

if TYPE_CHECKING:  # pragma: no cover - import cycle guard for type checkers
    from .experiment import Prompt

__all__ = [
    "SUBSET_METHOD",
    "band_eval_prompts",
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


def band_eval_prompts(
    prompts: Sequence[Prompt],
    labels: Mapping[str, str],
    band: str,
    count: int,
    seed: int,
) -> tuple[Prompt, ...]:
    """Pick `count` prompts from one labelled difficulty band.

    Only prompts whose id carries the requested band are kept. The kept prompts
    keep their pool order, and `select_eval_subset` samples them with the given
    seed. Raises ValueError for an unknown band, and when the band holds fewer
    prompts than `count`, so a short band is not silently scored on a smaller
    set.
    """
    if band not in BANDS:
        raise ValueError(
            f"unknown band {band!r}, expected one of {', '.join(BANDS)}"
        )
    in_band = tuple(
        prompt for prompt in prompts if labels.get(prompt.prompt_id) == band
    )
    if len(in_band) < count:
        raise ValueError(
            f"band {band!r} holds {len(in_band)} prompts, "
            f"fewer than the requested {count}"
        )
    return select_eval_subset(in_band, count=count, seed=seed)


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
