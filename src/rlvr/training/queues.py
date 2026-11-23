"""One saved queue per prompt pool.

A queue is the rotation for a pool. A step takes a block from the front and
puts the block at the back in the same order. A full pass therefore shows every
id once before any id repeats. After that the queue cycles in the same order,
oldest first.

`build` seeds the order once, so a run and its resume draw the same order when
the pool membership is the same. A queue is saved in the run's label file after
each step. `restore_queue` rebuilds a saved order against the current pool. Ids
that left the pool are dropped. Pool ids that the saved queue does not hold are
appended in sorted order, so a hand-edited or older row cannot lose an id.
"""

from __future__ import annotations

import random
from typing import Any, Iterable, Mapping, Sequence

__all__ = ["PromptQueue", "queues_from_row", "restore_queue"]


def _queue_id(value: object) -> str:
    """One prompt id from a queue input. A bad id is an error."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"a queue id must be a non-empty string, got {value!r}"
        )
    return value


class PromptQueue:
    """An ordered list of prompt ids with one pass before a repeat.

    A queue starts empty. `build` sets the order from a seeded shuffle. A
    restored queue is built from `restore_queue` instead, which keeps the saved
    order and reconciles it with the pool.
    """

    def __init__(self, ids: Sequence[str] = ()) -> None:
        self._ids: list[str] = []
        for prompt_id in ids:
            self.append(prompt_id)

    def build(self, ids: Iterable[str], seed: int) -> None:
        """Set the order to a seeded shuffle of `ids`.

        The ids are sorted before the shuffle, so the same pool gives the same
        order whatever order the caller collected it in. Duplicate ids are
        shown once.
        """
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"seed must be an integer, got {seed!r}")
        order = sorted({_queue_id(prompt_id) for prompt_id in ids})
        random.Random(seed).shuffle(order)
        self._ids = order

    def take(self, n: int) -> list[str]:
        """The first `n` ids, which move to the back in the same order.

        When `n` is at least the queue length, the whole queue comes back once
        and the order does not change. That is a full pass with no repeat.
        """
        if isinstance(n, bool) or not isinstance(n, int) or n < 0:
            raise ValueError(f"take needs an integer of at least 0, got {n!r}")
        if not self._ids or n == 0:
            return []
        count = min(n, len(self._ids))
        taken = self._ids[:count]
        self._ids = self._ids[count:] + taken
        return taken

    def keep(self, ids: Iterable[str]) -> None:
        """Drop every id that is not in `ids`, keeping the queue order."""
        keep = {_queue_id(prompt_id) for prompt_id in ids}
        self._ids = [prompt_id for prompt_id in self._ids if prompt_id in keep]

    def append(self, prompt_id: str) -> bool:
        """Put one id at the back, unless the queue already holds it."""
        prompt_id = _queue_id(prompt_id)
        if prompt_id in self._ids:
            return False
        self._ids.append(prompt_id)
        return True

    def remove(self, prompt_id: str) -> bool:
        """Take one id out of the queue. A missing id is not an error."""
        prompt_id = _queue_id(prompt_id)
        try:
            self._ids.remove(prompt_id)
        except ValueError:
            return False
        return True

    def as_list(self) -> list[str]:
        """The queue order as a new list."""
        return list(self._ids)

    def __len__(self) -> int:
        return len(self._ids)


def queues_from_row(
    row: Mapping[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
    """The parked and train queue ids in a saved row, or None.

    A row must hold both lists and every id must be a non-empty string. A
    missing or malformed list makes the whole row unusable, so a future row
    name cannot restore half a queue.
    """
    parked = row.get("parked_queue")
    train = row.get("train_queue")
    if not isinstance(parked, (list, tuple)):
        return None
    if not isinstance(train, (list, tuple)):
        return None
    for prompt_id in (*parked, *train):
        if not isinstance(prompt_id, str) or not prompt_id.strip():
            return None
    return tuple(parked), tuple(train)


def restore_queue(saved_ids: Iterable[str], pool_ids: Iterable[str]) -> PromptQueue:
    """Rebuild a saved queue against the current pool.

    Saved ids that left the pool are dropped. Pool ids that the saved queue
    does not hold are appended in sorted order. The saved order is kept for the
    ids that are still in the pool.
    """
    queue = PromptQueue()
    for prompt_id in saved_ids:
        queue.append(prompt_id)
    pool = sorted({_queue_id(prompt_id) for prompt_id in pool_ids})
    queue.keep(pool)
    present = set(queue.as_list())
    for prompt_id in pool:
        if prompt_id not in present:
            queue.append(prompt_id)
    return queue
