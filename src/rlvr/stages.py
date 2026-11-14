"""Ordered curriculum stages built from labelled questions.

The bands are fixed and ordered: very easy, medium, hard, very hard. There is
no reordering by config. A run starts at a chosen band and moves forward only.

The builder is pure. It takes labelled rows, the ordered band names, a carry
over ratio and a seed, and returns one `Stage` per band from the start band on.
Each stage holds its own training pool and the held out slice for evaluation.
The split inside a band is 80 percent train and 20 percent held out, chosen by
the seed, so a rerun picks the same questions whatever the input order is.

Carry over takes a share of the previous band's own training pool and puts it
at the front of the current stage's training pool. It never takes from the next
band and never takes from a held out slice. With a start band other than the
first, the first stage of the run carries nothing, because the run never saw
the band before it.

The difficulty labels are owned by another task. This module only defines the
seam: `load_band_labels` maps a prompt id to a band name. A missing file returns
an empty mapping, so a run without labels still works and falls back to the
start band everywhere.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, Iterable, Mapping, Sequence, TypeVar

__all__ = [
    "BANDS",
    "DEFAULT_CARRY_OVER_RATIO",
    "DEFAULT_LABELS_PATH",
    "HELD_OUT_FRACTION",
    "UNSTAGED_STAGE",
    "LabelledRow",
    "Stage",
    "StageError",
    "build_stages",
    "load_band_labels",
]

#: The band order. This is fixed by the study design, not by config.
BANDS: tuple[str, ...] = ("very easy", "medium", "hard", "very hard")

#: Share of the previous band's training pool carried into the next stage.
DEFAULT_CARRY_OVER_RATIO = 0.2

#: Share of each band held out for evaluation.
HELD_OUT_FRACTION = 0.2

#: Repo-relative path of the difficulty label file. Task 15 owns its contents.
DEFAULT_LABELS_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "difficulty_labels.jsonl"
)

#: Used when a run has no stages at all, so records still carry a string.
UNSTAGED_STAGE = "unstaged"

ItemT = TypeVar("ItemT")


class StageError(ValueError):
    """The stage request or the label file is invalid."""


@dataclass(frozen=True)
class LabelledRow(Generic[ItemT]):
    """One question with its band. `item` is whatever the caller trains on."""

    prompt_id: str
    band: str
    item: ItemT


@dataclass(frozen=True)
class Stage(Generic[ItemT]):
    """One ordered stage: the training pool and the held out slice."""

    name: str
    train: tuple[ItemT, ...]
    held_out: tuple[ItemT, ...] = ()


def _checked_ratio(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StageError(f"carry over ratio must be a number, got {value!r}")
    ratio = float(value)
    if not math.isfinite(ratio):
        raise StageError(f"carry over ratio must be finite, got {value!r}")
    if not 0.0 <= ratio < 1.0:
        raise StageError(f"carry over ratio must be at least 0 and below 1, got {value!r}")
    return ratio


def _held_out_count(size: int) -> int:
    """The 20 percent slice size, leaving at least one training question."""
    if size <= 1:
        return 0
    held = int(math.floor(size * HELD_OUT_FRACTION + 0.5))
    return min(max(held, 0), size - 1)


def _carry_count(size: int, ratio: float) -> int:
    """The rounded share of a pool to carry, clamped to the pool."""
    if size <= 0:
        return 0
    return min(max(int(math.floor(size * ratio + 0.5)), 0), size)


def build_stages(
    rows: Iterable[LabelledRow[ItemT]],
    bands: Sequence[str] = BANDS,
    carry_over_ratio: float = DEFAULT_CARRY_OVER_RATIO,
    *,
    seed: int = 0,
    start_stage: str | None = None,
) -> tuple[Stage[ItemT], ...]:
    """Build the ordered stages from labelled rows.

    Raises `StageError` on an unknown band, a duplicate prompt id, an empty or
    repeated band list, a bad ratio, or an unknown start band. Input order does
    not matter: rows are sorted by prompt id before the seeded shuffle.
    """
    definitions = tuple(bands)
    if not definitions:
        raise StageError("bands must not be empty")
    if any(not isinstance(band, str) or not band.strip() for band in definitions):
        raise StageError("every band must be a non-empty string")
    if len(set(definitions)) != len(definitions):
        raise StageError("bands must not repeat")

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise StageError(f"seed must be an integer, got {seed!r}")
    ratio = _checked_ratio(carry_over_ratio)

    if start_stage is None:
        start_stage = definitions[0]
    if start_stage not in definitions:
        raise StageError(
            f"unknown start stage {start_stage!r}, expected one of "
            f"{', '.join(definitions)}"
        )
    start_index = definitions.index(start_stage)

    grouped: dict[str, list[tuple[str, ItemT]]] = {band: [] for band in definitions}
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, LabelledRow):
            raise StageError("every row must be a LabelledRow")
        prompt_id = row.prompt_id
        if not isinstance(prompt_id, str) or not prompt_id.strip():
            raise StageError("every row must have a non-empty prompt id")
        if prompt_id in seen:
            raise StageError(f"duplicate prompt id {prompt_id!r}")
        seen.add(prompt_id)
        if row.band not in grouped:
            raise StageError(
                f"unknown band {row.band!r} for {prompt_id!r}, expected one of "
                f"{', '.join(definitions)}"
            )
        grouped[row.band].append((prompt_id, row.item))

    splits: dict[str, tuple[list[tuple[str, ItemT]], list[tuple[str, ItemT]]]] = {}
    for band in definitions:
        entries = sorted(grouped[band], key=lambda pair: pair[0])
        random.Random(f"{seed}:{band}").shuffle(entries)
        held = _held_out_count(len(entries))
        cutoff = len(entries) - held
        splits[band] = (entries[:cutoff], entries[cutoff:])

    stages: list[Stage[ItemT]] = []
    previous_train: list[tuple[str, ItemT]] | None = None
    for index, band in enumerate(definitions):
        own_train, held_out = splits[band]
        train = list(own_train)
        if previous_train is not None and index > start_index:
            carried = previous_train[: _carry_count(len(previous_train), ratio)]
            train = list(carried) + train
        if index >= start_index:
            stages.append(
                Stage(
                    name=band,
                    train=tuple(item for _, item in train),
                    held_out=tuple(item for _, item in held_out),
                )
            )
        previous_train = own_train

    return tuple(stages)


def load_band_labels(path: str | Path = DEFAULT_LABELS_PATH) -> Mapping[str, str]:
    """Read a prompt id to band mapping from a JSONL label file.

    A missing file returns an empty mapping. That is the clean degradation
    this ticket promises while the label data is owned by another task. A file
    that exists but is malformed or names an unknown band raises `StageError`,
    so bad data cannot silently change the curriculum.
    """
    file_path = Path(path)
    if not file_path.is_file():
        return {}

    labels: dict[str, str] = {}
    try:
        lines = file_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise StageError(f"cannot read difficulty labels {file_path}: {error}") from error

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise StageError(
                f"difficulty labels line {line_number}: not valid JSON: {error}"
            ) from error
        if not isinstance(row, dict):
            raise StageError(f"difficulty labels line {line_number}: row must be an object")

        prompt_id = row.get("prompt_id")
        if not isinstance(prompt_id, str) or not prompt_id.strip():
            raise StageError(
                f"difficulty labels line {line_number}: 'prompt_id' must be a non-empty string"
            )
        band = row.get("band")
        if not isinstance(band, str) or band not in BANDS:
            raise StageError(
                f"difficulty labels line {line_number}: unknown band {band!r}, "
                f"expected one of {', '.join(BANDS)}"
            )
        if prompt_id in labels:
            raise StageError(
                f"difficulty labels line {line_number}: duplicate prompt id {prompt_id!r}"
            )
        labels[prompt_id] = band

    return labels
