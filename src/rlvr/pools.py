"""Pools of prompt ids, built only from measured pass counts.

The screen samples each training prompt a fixed number of times and records how
many samples passed. This module turns those rows into the three pools the
trainer uses:

    parked      no sample passed, so the prompt leaves the training pool
    rehearsal   every sample passed, so the prompt is kept for maintenance
    train       some but not all samples passed, so the prompt has spread

Membership comes only from the measured counts. Bands, labels and the model name
are not inputs, so no other part of the codebase decides which pool holds a
prompt. Each pool is sorted by prompt id, so two runs over the same rows give
the same result whatever order the rows arrive in.

A row is an object or a mapping with `prompt_id`, `samples` and `passed`. That is
the schema `screening.ScreenRow` writes to `rows.jsonl`. `screening` is not
imported here, because the training path will import this module and `screening`
already imports the training path's prompt type.

The promotion bar is the count a later re-screen uses to promote a parked
prompt. It does not change the first build of the pools, because a prompt starts
in `train` on any partial count. It is checked for a positive integer here. It is
not compared to the sample count, because a screen count and a later rescreen
count can differ.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

__all__ = [
    "DEFAULT_PROMOTION_BAR",
    "Pools",
    "PoolsError",
    "from_rows",
]

#: Passes needed to promote a parked prompt, out of sixteen samples or more.
DEFAULT_PROMOTION_BAR = 2


class PoolsError(ValueError):
    """A screen row or a pool setting is invalid."""


@dataclass(frozen=True)
class Pools:
    """The three prompt pools and the sample count they were measured at."""

    train: tuple[str, ...]
    rehearsal: tuple[str, ...]
    parked: tuple[str, ...]
    samples: int


def _field(row: object, name: str) -> object:
    """One field from a row object or a row mapping."""
    if isinstance(row, Mapping):
        return row.get(name)
    return getattr(row, name, None)


def from_rows(
    rows: Iterable[object],
    *,
    samples: int,
    promotion_bar: int = DEFAULT_PROMOTION_BAR,
) -> Pools:
    """Build the three pools from screen rows measured at `samples` samples.

    A row with zero passes is parked, a row with every pass is rehearsal, and a
    row with any other count is train. A row measured at a different sample
    count is an error that names the prompt, because mixing two screens in one
    pool would make the counts incomparable. No rows gives three empty pools.
    """
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise PoolsError(
            f"samples must be an integer of at least 1, got {samples!r}"
        )
    if (
        isinstance(promotion_bar, bool)
        or not isinstance(promotion_bar, int)
        or promotion_bar < 1
    ):
        raise PoolsError(
            f"promotion_bar must be an integer of at least 1, got {promotion_bar!r}"
        )

    train: list[str] = []
    rehearsal: list[str] = []
    parked: list[str] = []
    seen: set[str] = set()

    for position, row in enumerate(rows, start=1):
        prompt_id = _field(row, "prompt_id")
        if not isinstance(prompt_id, str) or not prompt_id.strip():
            raise PoolsError(
                f"row {position}: 'prompt_id' must be a non-empty string"
            )
        if prompt_id in seen:
            raise PoolsError(f"row {position}: duplicate prompt id {prompt_id!r}")
        seen.add(prompt_id)

        row_samples = _field(row, "samples")
        if (
            isinstance(row_samples, bool)
            or not isinstance(row_samples, int)
            or row_samples < 1
        ):
            raise PoolsError(
                f"prompt {prompt_id!r}: 'samples' must be an integer of at least 1"
            )
        if row_samples != samples:
            raise PoolsError(
                f"prompt {prompt_id!r} used {row_samples} samples, not {samples}"
            )

        passed = _field(row, "passed")
        if (
            isinstance(passed, bool)
            or not isinstance(passed, int)
            or not 0 <= passed <= row_samples
        ):
            raise PoolsError(
                f"prompt {prompt_id!r}: 'passed' must be an integer "
                f"between 0 and {row_samples}"
            )

        if passed == 0:
            parked.append(prompt_id)
        elif passed == row_samples:
            rehearsal.append(prompt_id)
        else:
            train.append(prompt_id)

    return Pools(
        train=tuple(sorted(train)),
        rehearsal=tuple(sorted(rehearsal)),
        parked=tuple(sorted(parked)),
        samples=samples,
    )
