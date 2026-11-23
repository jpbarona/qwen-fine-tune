"""Tests for the saved prompt queues.

A queue is the rotation for one pool. A block comes off the front and goes to
the back, so a full pass shows every id once before any id repeats. The run
saves the queue after each step and restores it against the current pool on a
resume.
"""

from __future__ import annotations

from rlvr.core.config import load_config
from rlvr.training.experiment import Experiment, Prompt
from rlvr.training.pools import Pools
from rlvr.training.queues import PromptQueue, queues_from_row, restore_queue

CONFIG_TEXT = """\
[model]
name = "Qwen/Qwen3-0.6B"
revision = "r"

[judge]
model = "fake-judge-model"

[data]
split_seed = 1

[fake]
steps = 1

[training]
generations_per_prompt = 16
"""


def test_take_returns_the_first_ids_and_moves_them_to_the_back() -> None:
    queue = PromptQueue(["a", "b", "c"])

    assert queue.take(2) == ["a", "b"]
    assert queue.as_list() == ["c", "a", "b"]

    assert queue.take(2) == ["c", "a"]
    assert queue.as_list() == ["b", "c", "a"]


def test_a_pool_is_covered_before_any_id_repeats() -> None:
    ids = [f"p{index}" for index in range(7)]
    queue = PromptQueue()
    queue.build(ids, seed=11)

    seen: list[str] = []
    while len(seen) < len(ids):
        seen.extend(queue.take(3))

    assert sorted(seen[: len(ids)]) == ids
    assert len(set(seen[: len(ids)])) == len(ids)


def test_the_oldest_block_comes_back_after_a_full_pass() -> None:
    queue = PromptQueue(["a", "b", "c", "d"])

    first = queue.take(2)
    second = queue.take(2)
    assert queue.as_list() == ["a", "b", "c", "d"]

    assert queue.take(2) == first
    assert queue.take(2) == second


def test_take_of_the_whole_pool_returns_every_id_once() -> None:
    queue = PromptQueue(["a", "b", "c"])

    assert queue.take(3) == ["a", "b", "c"]
    assert queue.as_list() == ["a", "b", "c"]

    assert queue.take(10) == ["a", "b", "c"]
    assert queue.as_list() == ["a", "b", "c"]


def test_keep_drops_ids_that_left_the_pool() -> None:
    queue = PromptQueue(["a", "b", "c", "d"])

    queue.keep(["a", "c"])
    assert queue.as_list() == ["a", "c"]

    queue.keep(["c", "a", "new"])
    assert queue.as_list() == ["a", "c"]


def test_append_ignores_an_id_already_in_the_queue() -> None:
    queue = PromptQueue(["a", "b"])

    assert queue.append("c") is True
    assert queue.append("b") is False
    assert queue.as_list() == ["a", "b", "c"]


def test_remove_takes_one_id_out() -> None:
    queue = PromptQueue(["a", "b", "c"])

    assert queue.remove("b") is True
    assert queue.remove("b") is False
    assert queue.as_list() == ["a", "c"]


def test_build_uses_the_seed_and_ignores_the_input_order() -> None:
    first = PromptQueue()
    first.build(["b", "a", "c"], seed=7)
    second = PromptQueue()
    second.build(["c", "b", "a"], seed=7)

    assert first.as_list() == second.as_list()
    assert sorted(first.as_list()) == ["a", "b", "c"]


def test_restore_drops_removed_ids_and_appends_new_ids_in_sorted_order() -> None:
    restored = restore_queue(
        saved_ids=["b", "a", "gone"],
        pool_ids=["c", "a", "b"],
    )

    assert restored.as_list() == ["b", "a", "c"]


def test_a_queues_row_parses_both_lists() -> None:
    row = {"step": 1, "parked_queue": ["a"], "train_queue": ["b", "c"]}

    assert queues_from_row(row) == (("a",), ("b", "c"))
    assert queues_from_row({"step": 1, "parked_queue": []}) is None
    bad_train = {"step": 1, "parked_queue": ["a"], "train_queue": [3]}
    assert queues_from_row(bad_train) is None


class _LabelStore:
    """A label store that keeps the written rows in memory."""

    def __init__(self, rows: list[dict] | None = None) -> None:
        self.rows = list(rows or [])

    def append_label(self, row: dict) -> None:
        self.rows.append(row)

    def read_labels(self) -> list[dict]:
        return list(self.rows)


class _QueueSchedule:
    """A schedule stub with saved queues and no rotation of its own."""

    prompts_per_step = 2
    name = "fake-queues"

    def __init__(self) -> None:
        self.restored: tuple[list[str], list[str]] | None = None
        self.parked_queue = PromptQueue()
        self.train_queue = PromptQueue()

    def plan(
        self, step: int, train_pool, parked_pool
    ):  # pragma: no cover - the tests never plan a step
        raise AssertionError("the fake schedule is not asked to plan")

    def observe(self, step: int, outcomes):  # pragma: no cover - unused
        return ()

    def state_row(self, step: int) -> dict:
        return {
            "step": step,
            "parked_queue": self.parked_queue.as_list(),
            "train_queue": self.train_queue.as_list(),
        }

    def restore_state(self, parked_queue, train_queue) -> None:
        self.parked_queue = parked_queue
        self.train_queue = train_queue
        self.restored = (parked_queue.as_list(), train_queue.as_list())


def _prompts() -> list[Prompt]:
    return [
        Prompt(prompt_id=prompt_id, problem=f"problem {prompt_id}")
        for prompt_id in ("a", "b", "c", "d", "e")
    ]


def _pools() -> Pools:
    return Pools(train=("a", "b", "c"), retired=(), parked=("d", "e"), samples=16)


def _experiment(tmp_path, store: _LabelStore, schedule: _QueueSchedule) -> Experiment:
    path = tmp_path / "config.toml"
    path.write_text(CONFIG_TEXT, encoding="utf-8")
    return Experiment(
        arm="code-score",
        config=load_config(path),
        generator=None,
        judge=None,
        test_runner=None,
        store=store,
        prompts=_prompts(),
        pools=_pools(),
        schedule=schedule,
    )


def test_a_step_writes_one_queues_row(tmp_path) -> None:
    schedule = _QueueSchedule()
    schedule.parked_queue.build(["d", "e"], seed=0)
    schedule.train_queue.build(["a", "b", "c"], seed=0)
    store = _LabelStore()
    experiment = _experiment(tmp_path, store, schedule)
    experiment._schedule = schedule

    experiment._record_schedule_state(4)

    assert store.rows == [
        {
            "step": 4,
            "parked_queue": schedule.parked_queue.as_list(),
            "train_queue": schedule.train_queue.as_list(),
        }
    ]


def test_a_saved_queue_row_restores_with_new_and_removed_ids(tmp_path) -> None:
    schedule = _QueueSchedule()
    store = _LabelStore(
        [
            {
                "step": 2,
                "parked_queue": ["d", "gone"],
                "train_queue": ["b", "a", "gone"],
            }
        ]
    )
    experiment = _experiment(tmp_path, store, schedule)
    experiment.prompts, experiment._parked_prompts = experiment._pool_prompts()
    experiment._schedule = schedule

    experiment._load_labels()

    assert schedule.restored == (["d", "e"], ["b", "a", "c"])
