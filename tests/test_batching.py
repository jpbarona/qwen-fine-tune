"""Tests for batched generation: the size policy, grouping, metrics, and resume.

No torch and no GPU are involved. The policy is pure, and the Experiment runs
against small stubs. The fake path is covered end to end with the deterministic
fake backends.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from rlvr.batching import (
    DEFAULT_MEMORY_FLOOR_MIB,
    MIN_PROMPTS,
    START_PROMPTS,
    BatchSizePolicy,
    free_memory_mib,
    is_cuda_out_of_memory,
)
from rlvr.checkpoint import find_latest_checkpoint, read_checkpoint_manifest
from rlvr.config import default_config_path, load_config
from rlvr.experiment import Experiment, Generation, Outcome, Prompt
from rlvr.fake import FakeGenerator, FakeJudge, FakeTestRunner, fake_prompts
from rlvr.memory import MemoryReport
from rlvr.qwen import group_advantages
from rlvr.runs import RunStore

TIMESTAMP = "20261112T000000Z"


def config_text(
    steps: int = 2,
    *,
    checkpoint_every_steps: int = 100,
    memory_floor_mib: int | None = None,
    entropy_every_steps: int | None = None,
) -> str:
    text = f"""\
[model]
name = "Qwen/Qwen3-0.6B"
revision = "r"

[judge]
model = "fake-judge-model"

[data]
split_seed = 1

[fake]
steps = {steps}

[run]
checkpoint_every_steps = {checkpoint_every_steps}
"""
    training = []
    if memory_floor_mib is not None:
        training.append(f"memory_floor_mib = {memory_floor_mib}")
    if entropy_every_steps is not None:
        training.append(f"entropy_every_steps = {entropy_every_steps}")
    if training:
        text += "\n[training]\n" + "\n".join(training) + "\n"
    return text


def report(total_mib: float = 10000.0, free_mib: float = 8000.0) -> MemoryReport:
    """A report whose reserved total leaves exactly `free_mib` free."""
    reserved = total_mib - free_mib
    return MemoryReport(
        live_mib=reserved, cached_mib=0.0, fragmented_mib=0.0, total_mib=total_mib
    )


class RecordingStore:
    def __init__(self, checkpoints_dir: Path) -> None:
        self.rollouts = []
        self.metrics = []
        self.eval = None
        self.meta = None
        self.timestamp = TIMESTAMP
        self.run_dir = checkpoints_dir.parent
        self.checkpoints_dir = checkpoints_dir

    def write_meta(self, meta) -> None:
        self.meta = meta

    def append_rollout(self, record) -> None:
        self.rollouts.append(record)

    def append_metric(self, row) -> None:
        self.metrics.append(row)

    def write_eval(self, payload) -> None:
        self.eval = payload


class GroupGenerator:
    """Samples a fixed group and records each update with its own rewards."""

    def __init__(self) -> None:
        self.groups: list[tuple[str, tuple[float, ...]]] = []
        self.saves: list[str] = []
        self.updates = 0

    def generate(self, prompt: Prompt, *, n: int, temperature: float):
        return [
            Generation(
                reasoning=f"trace {prompt.prompt_id}",
                code=f"# code {prompt.prompt_id}",
                prompt_tokens=10,
                completion_tokens=20,
            )
            for _ in range(n)
        ]

    def generate_batch(self, prompts, *, n: int, temperature: float):
        return [
            self.generate(prompt, n=n, temperature=temperature) for prompt in prompts
        ]

    def accumulate(self, prompt, generations, rewards, *, measure_entropy: bool = False):
        self.groups.append((prompt.prompt_id, tuple(rewards)))
        return 0.5 if measure_entropy else None

    def apply_update(self) -> None:
        self.updates += 1

    def save_checkpoint(self, directory) -> None:
        self.saves.append(str(directory))

    def load_checkpoint(self, directory) -> None:
        pass


class OomOnceGenerator(GroupGenerator):
    """Fails the first training call with a CUDA out-of-memory error."""

    def __init__(self) -> None:
        super().__init__()
        self.failed = False

    def generate(self, prompt: Prompt, *, n: int, temperature: float):
        if temperature != 0.0 and not self.failed:
            self.failed = True
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return super().generate(prompt, n=n, temperature=temperature)


class AlwaysOomGenerator(GroupGenerator):
    """Fails every training call, but still serves greedy evaluation."""

    def generate(self, prompt: Prompt, *, n: int, temperature: float):
        if temperature != 0.0:
            raise RuntimeError("CUDA out of memory. Tried to allocate 1.00 GiB")
        return super().generate(prompt, n=n, temperature=temperature)


class PromptRunner:
    """Full pass for the named prompts, one of four tests for the rest."""

    __test__ = False

    def __init__(self, full: set[str] | None = None) -> None:
        self.full = full or set()

    def run(self, prompt: Prompt, code: str) -> Outcome:
        if prompt.prompt_id in self.full:
            return Outcome(tests_passed=4, tests_total=4, error_class="pass")
        return Outcome(tests_passed=1, tests_total=4, error_class="assertion")


class ExplodingJudge:
    def score_output(self, prompt, code):
        raise AssertionError("the code arm must never judge")

    def score_process(self, prompt, reasoning):
        raise AssertionError("the code arm must never judge")

    def score_outputs(self, items):
        raise AssertionError("the code arm must never judge")

    def score_processes(self, items):
        raise AssertionError("the code arm must never judge")


def make_prompts(count: int) -> list[Prompt]:
    return [
        Prompt(prompt_id=f"mbpp-{index:04d}", problem=f"problem {index}")
        for index in range(count)
    ]


def run_experiment(
    tmp_path: Path,
    *,
    prompts: list[Prompt],
    steps: int,
    memory=None,
    batch_policy: BatchSizePolicy | None = None,
    generator=None,
    runner=None,
    config_steps: int | None = None,
    checkpoint_every_steps: int = 100,
    memory_floor_mib: int | None = None,
    entropy_every_steps: int | None = None,
    store=None,
    resume_step: int = 0,
    resume_prompt_cursor: int = 0,
    prior_evaluations=(),
):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "config.toml"
    path.write_text(
        config_text(
            steps=steps if config_steps is None else config_steps,
            checkpoint_every_steps=checkpoint_every_steps,
            memory_floor_mib=memory_floor_mib,
            entropy_every_steps=entropy_every_steps,
        ),
        encoding="utf-8",
    )
    config = load_config(path)
    store = store or RecordingStore(tmp_path / "checkpoints")
    experiment = Experiment(
        arm="code-score",
        config=config,
        generator=generator or GroupGenerator(),
        judge=ExplodingJudge(),
        test_runner=runner or PromptRunner(),
        store=store,
        prompts=prompts,
        eval_prompts=[Prompt("eval-0000", "eval problem")],
        memory=memory,
        batch_policy=batch_policy,
        resume_step=resume_step,
        resume_prompt_cursor=resume_prompt_cursor,
        prior_evaluations=prior_evaluations,
    )
    return experiment, store


# --- the size policy ------------------------------------------------------


def test_the_batch_starts_at_two_prompts_and_never_below_one() -> None:
    policy = BatchSizePolicy()
    assert START_PROMPTS == 2
    assert MIN_PROMPTS == 1
    assert policy.count(10) == 2


def test_the_batch_grows_by_one_while_free_memory_is_above_the_floor() -> None:
    policy = BatchSizePolicy(floor_mib=512)
    policy.observe(report(free_mib=8000.0), prompts=2)
    assert policy.current == 3
    assert policy.largest_fit == 2
    policy.observe(report(free_mib=8000.0), prompts=3)
    assert policy.current == 4
    assert policy.largest_fit == 3


def test_the_batch_settles_at_the_largest_size_that_fit() -> None:
    policy = BatchSizePolicy(floor_mib=512)
    policy.observe(report(free_mib=8000.0), prompts=2)
    policy.observe(report(free_mib=8000.0), prompts=3)
    policy.observe(report(free_mib=100.0), prompts=4)

    assert policy.settled is True
    assert policy.current == 3

    # Another good measurement does not start probing upward again.
    policy.observe(report(free_mib=8000.0), prompts=3)
    assert policy.current == 3


def test_a_step_that_does_not_fit_at_two_drops_to_one() -> None:
    policy = BatchSizePolicy(floor_mib=512)
    policy.observe(report(free_mib=0.0), prompts=2)
    assert policy.settled is True
    assert policy.current == MIN_PROMPTS


def test_the_floor_is_inclusive() -> None:
    policy = BatchSizePolicy(floor_mib=512)
    policy.observe(report(free_mib=512.0), prompts=2)
    assert policy.current == 3


def test_the_floor_rises_to_one_measured_group_footprint() -> None:
    policy = BatchSizePolicy(floor_mib=512)

    # First step: no earlier measurement, so the configured floor applies.
    policy.observe(report(total_mib=10000.0, free_mib=8000.0), prompts=2)
    assert policy.group_footprint_mib == 0.0
    assert policy.effective_floor_mib == 512

    # The third prompt raised the reserved total by 1000 MiB, so one group is
    # measured at 1000 MiB and the floor rises to match it.
    policy.observe(report(total_mib=10000.0, free_mib=7000.0), prompts=3)
    assert policy.group_footprint_mib == pytest.approx(1000.0)
    assert policy.effective_floor_mib == pytest.approx(1000.0)

    # 800 MiB free clears the old 512 floor but not one group's footprint.
    policy.observe(report(total_mib=10000.0, free_mib=800.0), prompts=3)
    assert policy.settled is True
    assert policy.current == 3


def test_the_configured_floor_wins_when_it_is_larger() -> None:
    policy = BatchSizePolicy(floor_mib=2048)
    policy.observe(report(total_mib=10000.0, free_mib=8000.0), prompts=2)
    policy.observe(report(total_mib=10000.0, free_mib=7000.0), prompts=3)

    assert policy.group_footprint_mib == pytest.approx(1000.0)
    assert policy.effective_floor_mib == pytest.approx(2048.0)


def test_settle_keeps_the_largest_size_that_fit() -> None:
    policy = BatchSizePolicy(floor_mib=512)
    policy.observe(report(free_mib=8000.0), prompts=2)

    policy.settle()
    assert policy.settled is True
    assert policy.current == 2

    # A good measurement afterwards does not resume growth.
    policy.observe(report(free_mib=8000.0), prompts=2)
    assert policy.current == 2


def test_cuda_out_of_memory_is_detected_without_torch() -> None:
    class OutOfMemoryError(RuntimeError):
        pass

    assert is_cuda_out_of_memory(
        OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")
    )
    assert not is_cuda_out_of_memory(ValueError("bad input"))


def test_no_memory_measurement_holds_the_current_size() -> None:
    policy = BatchSizePolicy()
    policy.observe(MemoryReport.zeros(), prompts=2)
    assert policy.current == START_PROMPTS
    assert policy.settled is False


def test_a_short_pool_does_not_lower_the_target_size() -> None:
    policy = BatchSizePolicy(current=6)
    policy.observe(report(free_mib=8000.0), prompts=1)
    assert policy.current == 6
    assert policy.count(10) == 6


def test_the_prompt_count_is_clamped_to_the_pool_and_the_pin() -> None:
    policy = BatchSizePolicy(current=8)
    assert policy.count(3) == 3
    assert policy.count(1) == 1

    pinned = BatchSizePolicy(current=8, max_prompts=2)
    assert pinned.count(10) == 2


def test_free_memory_is_the_part_the_allocator_does_not_hold() -> None:
    assert free_memory_mib(report(total_mib=1000.0, free_mib=250.0)) == pytest.approx(250.0)
    over = MemoryReport(
        live_mib=1200.0, cached_mib=0.0, fragmented_mib=0.0, total_mib=1000.0
    )
    assert free_memory_mib(over) == 0.0


def test_bad_policy_arguments_are_rejected() -> None:
    with pytest.raises(ValueError, match="memory floor"):
        BatchSizePolicy(floor_mib=-1)
    with pytest.raises(ValueError, match="prompt count"):
        BatchSizePolicy(current=0)
    with pytest.raises(ValueError, match="available prompts"):
        BatchSizePolicy().count(0)
    with pytest.raises(ValueError, match="observed prompts"):
        BatchSizePolicy().observe(MemoryReport.zeros(), prompts=0)


# --- grouping and metrics -------------------------------------------------


def test_one_step_takes_two_prompt_groups_and_records_both_counts(tmp_path) -> None:
    generator = GroupGenerator()
    experiment, store = run_experiment(
        tmp_path, prompts=make_prompts(6), steps=1, generator=generator
    )
    experiment.run()

    row = store.metrics[0]
    assert row["prompts"] == 2
    assert row["sequences"] == 16
    assert row["rollouts"] == 16
    assert len(store.rollouts) == 16

    # One update per prompt, each with only that prompt's eight rewards.
    assert [prompt_id for prompt_id, _ in generator.groups] == ["mbpp-0000", "mbpp-0001"]
    assert all(len(rewards) == 8 for _, rewards in generator.groups)


def test_each_group_is_updated_with_its_own_rewards(tmp_path) -> None:
    generator = GroupGenerator()
    experiment, _ = run_experiment(
        tmp_path,
        prompts=make_prompts(4),
        steps=1,
        generator=generator,
        runner=PromptRunner(full={"mbpp-0000"}),
    )
    experiment.run()

    first, second = generator.groups
    assert first[0] == "mbpp-0000"
    assert first[1] == pytest.approx((1.0,) * 8)
    assert second[0] == "mbpp-0001"
    assert second[1] == pytest.approx((0.25,) * 8)


def test_advantage_is_normalised_inside_a_group_not_across_the_batch(
    tmp_path,
) -> None:
    generator = GroupGenerator()
    experiment, _ = run_experiment(
        tmp_path,
        prompts=make_prompts(4),
        steps=1,
        generator=generator,
        runner=PromptRunner(full={"mbpp-0000"}),
    )
    experiment.run()

    first, second = (rewards for _, rewards in generator.groups)
    # Each group is flat, so each has zero advantage on its own. Mixing the
    # groups would invent a difference that the prompt did not have.
    assert group_advantages(first) == [0.0] * 8
    assert group_advantages(second) == [0.0] * 8
    mixed = group_advantages(list(first) + list(second))
    assert any(advantage != 0.0 for advantage in mixed)


def test_the_entropy_cadence_counts_steps_not_prompt_updates(tmp_path) -> None:
    experiment, store = run_experiment(
        tmp_path,
        prompts=make_prompts(6),
        steps=3,
        generator=GroupGenerator(),
        batch_policy=BatchSizePolicy(current=2, max_prompts=2),
        entropy_every_steps=2,
    )
    experiment.run()

    # Four prompt updates ran in the first two steps, but the cadence is per
    # step, so only step 2 carries an entropy value.
    assert [row["prompts"] for row in store.metrics] == [2, 2, 2]
    assert ["mean_entropy" in row for row in store.metrics] == [False, True, False]
    assert store.metrics[1]["mean_entropy"] == pytest.approx(0.5)


# --- memory-driven growth -------------------------------------------------


def test_the_batch_grows_across_steps_while_memory_allows(tmp_path) -> None:
    experiment, store = run_experiment(
        tmp_path,
        prompts=make_prompts(8),
        steps=4,
        memory=lambda: report(free_mib=8000.0),
    )
    experiment.run()

    assert [row["prompts"] for row in store.metrics] == [2, 3, 4, 5]
    assert [row["sequences"] for row in store.metrics] == [16, 24, 32, 40]


def test_the_batch_holds_the_largest_size_that_fit(tmp_path) -> None:
    reports = iter(
        [
            report(free_mib=8000.0),
            report(free_mib=8000.0),
            report(free_mib=100.0),
            report(free_mib=8000.0),
            report(free_mib=8000.0),
        ]
    )
    experiment, store = run_experiment(
        tmp_path, prompts=make_prompts(8), steps=5, memory=lambda: next(reports)
    )
    experiment.run()

    assert [row["prompts"] for row in store.metrics] == [2, 3, 4, 3, 3]


def test_a_short_pool_caps_the_batch_at_one_prompt(tmp_path) -> None:
    experiment, store = run_experiment(
        tmp_path, prompts=make_prompts(1), steps=2, memory=lambda: report(free_mib=8000.0)
    )
    experiment.run()

    assert [row["prompts"] for row in store.metrics] == [1, 1]
    assert [row["sequences"] for row in store.metrics] == [8, 8]


def test_the_configured_memory_floor_decides_growth(tmp_path) -> None:
    experiment, store = run_experiment(
        tmp_path,
        prompts=make_prompts(4),
        steps=2,
        memory=lambda: report(free_mib=8000.0),
        memory_floor_mib=8500,
    )
    experiment.run()

    assert [row["prompts"] for row in store.metrics] == [2, 1]


def test_without_a_memory_reader_the_batch_holds_its_start(tmp_path) -> None:
    experiment, store = run_experiment(tmp_path, prompts=make_prompts(6), steps=3)
    experiment.run()

    assert [row["prompts"] for row in store.metrics] == [2, 2, 2]


def test_the_default_floor_is_an_absolute_number() -> None:
    config = load_config(default_config_path())
    assert config.memory_floor_mib == DEFAULT_MEMORY_FLOOR_MIB == 512


def test_a_cuda_oom_settles_the_batch_and_continues(tmp_path) -> None:
    experiment, store = run_experiment(
        tmp_path,
        prompts=make_prompts(4),
        steps=2,
        generator=OomOnceGenerator(),
        memory=lambda: report(free_mib=8000.0),
    )
    result = experiment.run()

    # The first step asked for two prompts, hit the guard, and retried at one.
    assert [row["prompts"] for row in store.metrics] == [1, 1]
    assert result.stop_reason == "completed"


def test_a_cuda_oom_at_one_prompt_stops_cleanly(tmp_path) -> None:
    experiment, store = run_experiment(
        tmp_path,
        prompts=make_prompts(4),
        steps=3,
        generator=AlwaysOomGenerator(),
        batch_policy=BatchSizePolicy(current=1, max_prompts=1),
    )
    result = experiment.run()

    assert result.stop_reason == "out_of_memory"
    assert result.steps == 1
    assert store.eval["stop_reason"] == "out_of_memory"
    assert store.metrics == []


# --- the fake path --------------------------------------------------------


def test_the_fake_path_takes_two_prompts_per_step(tmp_path) -> None:
    generator = FakeGenerator()
    experiment, store = run_experiment(
        tmp_path,
        prompts=fake_prompts(4),
        steps=2,
        generator=generator,
        runner=FakeTestRunner(),
    )
    experiment.judge = FakeJudge()
    experiment.run()

    assert [row["prompts"] for row in store.metrics] == [2, 2]
    assert [row["sequences"] for row in store.metrics] == [16, 16]
    assert len(generator.update_calls) == 4
    assert all(len(rewards) == 8 for _, rewards in generator.update_calls)
    assert [prompt_id for prompt_id, _ in generator.update_calls] == [
        "mbpp-0000",
        "mbpp-0001",
        "mbpp-0002",
        "mbpp-0003",
    ]
    # Two groups per step, one weight update per step.
    assert generator.update_groups == [2, 2]
    assert generator.step == 2


def test_the_fake_batched_run_is_deterministic(tmp_path) -> None:
    first, first_store = run_experiment(
        tmp_path / "a",
        prompts=fake_prompts(4),
        steps=2,
        generator=FakeGenerator(),
        runner=FakeTestRunner(),
    )
    first.judge = FakeJudge()
    first.run()

    second, second_store = run_experiment(
        tmp_path / "b",
        prompts=fake_prompts(4),
        steps=2,
        generator=FakeGenerator(),
        runner=FakeTestRunner(),
    )
    second.judge = FakeJudge()
    second.run()

    assert [record.to_dict() for record in first_store.rollouts] == [
        record.to_dict() for record in second_store.rollouts
    ]


# --- resume on a batch boundary ------------------------------------------


def _run_pairs(store) -> list[tuple[int, str]]:
    rows = [
        json.loads(line)
        for line in store.rollouts_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    return [(row["step"], row["prompt_id"]) for row in rows]


def _metrics(store) -> list[dict]:
    return [
        json.loads(line)
        for line in store.metrics_path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def test_resume_on_a_batch_boundary_keeps_the_same_schedule(tmp_path) -> None:
    memory = lambda: report(free_mib=8000.0)

    # The uninterrupted reference run.
    full_dir = tmp_path / "full"
    full_dir.mkdir()
    full_store = RunStore(full_dir, "code-score", timestamp=TIMESTAMP)
    full, _ = run_experiment(
        full_dir,
        prompts=make_prompts(6),
        steps=4,
        memory=memory,
        store=full_store,
        checkpoint_every_steps=1,
    )
    full.run()
    full_pairs = _run_pairs(full_store)
    assert [row["prompts"] for row in _metrics(full_store)] == [2, 3, 4, 5]

    # Stop after step 2, then resume from its checkpoint.
    part_dir = tmp_path / "part"
    part_dir.mkdir()
    part_store = RunStore(part_dir, "code-score", timestamp=TIMESTAMP)
    part, _ = run_experiment(
        part_dir,
        prompts=make_prompts(6),
        steps=2,
        memory=memory,
        store=part_store,
        checkpoint_every_steps=1,
    )
    part.run()

    checkpoint = find_latest_checkpoint(part_store.checkpoints_dir)
    assert checkpoint is not None and checkpoint.step == 2
    manifest = read_checkpoint_manifest(checkpoint.directory)
    assert manifest["prompt_cursor"] == 5
    assert manifest["batch"] == {"current": 4, "largest_fit": 3, "settled": False}

    config_path = part_dir / "config.toml"
    resumed_config = replace(load_config(config_path), steps=4)
    resumed = Experiment(
        arm="code-score",
        config=resumed_config,
        generator=GroupGenerator(),
        judge=ExplodingJudge(),
        test_runner=PromptRunner(),
        store=part_store,
        prompts=make_prompts(6),
        eval_prompts=[Prompt("eval-0000", "eval problem")],
        memory=memory,
        resume_step=2,
        resume_prompt_cursor=manifest["prompt_cursor"],
        batch_policy=BatchSizePolicy(
            floor_mib=resumed_config.memory_floor_mib,
            current=manifest["batch"]["current"],
            largest_fit=manifest["batch"]["largest_fit"],
            settled=manifest["batch"]["settled"],
        ),
    )
    resumed.run()

    resumed_pairs = [(step, prompt_id) for step, prompt_id in _run_pairs(part_store) if step > 2]
    assert resumed_pairs == [(step, prompt_id) for step, prompt_id in full_pairs if step > 2]
