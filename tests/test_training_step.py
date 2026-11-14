"""Tests for one optimizer update per step and several prompts per sample call.

The real generator needs torch and a GPU, so these tests use small fakes. They
cover the batch call, the split back into prompt groups, the per group rewards
that keep advantage normalisation inside a group, the single optimizer step,
and the step counter that must match the number of weight updates.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rlvr.batching import BatchSizePolicy
from rlvr.checkpoint import find_latest_checkpoint, read_checkpoint_manifest
from rlvr.config import load_config
from rlvr.experiment import Experiment, Generation, Outcome, Prompt
from rlvr.fake import FakeGenerator, FakeJudge, FakeTestRunner, fake_prompts
from rlvr.runs import RunStore
from rlvr.schema import RolloutRecord

CONFIG_TEXT = """\
[model]
name = "Qwen/Qwen3-0.6B"
revision = "r"

[judge]
model = "fake-judge-model"

[data]
split_seed = 1

[fake]
steps = {steps}
"""


class RecordingGenerator:
    """Records the batch calls, the per group rewards, and the update count."""

    def __init__(self) -> None:
        self.batches: list[tuple[str, ...]] = []
        self.groups: list[tuple[str, tuple[float, ...]]] = []
        self.measure_flags: list[bool] = []
        self.updates = 0

    def generate(
        self, prompt: Prompt, *, n: int, temperature: float
    ) -> list[Generation]:
        return [
            Generation(
                reasoning=f"trace for {prompt.prompt_id}",
                code=f"# code for {prompt.prompt_id}",
                prompt_tokens=10,
                completion_tokens=20,
            )
            for _ in range(n)
        ]

    def generate_batch(self, prompts, *, n: int, temperature: float):
        self.batches.append(tuple(prompt.prompt_id for prompt in prompts))
        return [
            self.generate(prompt, n=n, temperature=temperature) for prompt in prompts
        ]

    def accumulate(
        self,
        prompt: Prompt,
        generations,
        rewards,
        *,
        measure_entropy: bool = False,
    ) -> float | None:
        self.groups.append((prompt.prompt_id, tuple(rewards)))
        self.measure_flags.append(measure_entropy)
        return 0.5 if measure_entropy else None

    def apply_update(self) -> None:
        self.updates += 1

    def save_checkpoint(self, directory) -> None:
        pass

    def load_checkpoint(self, directory) -> None:
        pass


class ShortBatchGenerator(RecordingGenerator):
    """Drops one group and one sample, to exercise the shape guards."""

    def __init__(self, *, drop_group: bool = True) -> None:
        super().__init__()
        self.drop_group = drop_group

    def generate_batch(self, prompts, *, n: int, temperature: float):
        groups = super().generate_batch(prompts, n=n, temperature=temperature)
        if self.drop_group:
            return groups[:-1]
        groups[0] = groups[0][:-1]
        return groups


class PassCountRunner:
    """A fixed full pass count per prompt, out of four tests."""

    __test__ = False

    def __init__(self, passes: dict[str, int] | None = None) -> None:
        self.passes = passes or {}
        self.calls: list[str] = []

    def run(self, prompt: Prompt, code: str) -> Outcome:
        self.calls.append(prompt.prompt_id)
        passed = self.passes.get(prompt.prompt_id, 4)
        return Outcome(
            tests_passed=passed,
            tests_total=4,
            error_class="pass" if passed == 4 else "wrong_answer",
        )


class RecordingStore:
    def __init__(self, run_dir: Path | None = None) -> None:
        self.rollouts: list[RolloutRecord] = []
        self.metrics: list[dict] = []
        self.eval: dict | None = None
        self.meta: dict | None = None
        self.timestamp = "20261112T000000Z"
        self.run_dir = run_dir or Path("runs/test/20261112T000000Z")
        self.checkpoints_dir = self.run_dir / "checkpoints"

    def write_meta(self, meta: dict) -> None:
        self.meta = meta

    def append_rollout(self, record: RolloutRecord) -> None:
        self.rollouts.append(record)

    def append_metric(self, row: dict) -> None:
        self.metrics.append(row)

    def write_eval(self, payload: dict) -> None:
        self.eval = payload


def make_prompts(count: int) -> list[Prompt]:
    return [
        Prompt(prompt_id=f"mbpp-{index:04d}", problem=f"problem {index}")
        for index in range(count)
    ]


def build(
    tmp_path,
    *,
    generator,
    prompts,
    steps: int = 2,
    per_step: int | None = None,
    judge=None,
    runner=None,
    store=None,
    config_text: str = CONFIG_TEXT,
):
    config = load_config(_write_config(tmp_path, config_text, steps=steps))
    policy = (
        None
        if per_step is None
        else BatchSizePolicy(current=per_step, max_prompts=per_step)
    )
    return Experiment(
        arm="code-score",
        config=config,
        generator=generator,
        judge=judge or FakeJudge(),
        test_runner=runner or PassCountRunner(),
        store=store or RecordingStore(),
        prompts=prompts,
        eval_prompts=prompts,
        batch_policy=policy,
    )


def _write_config(tmp_path, text: str, *, steps: int) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(text.format(steps=steps), encoding="utf-8")
    return path


def test_a_step_samples_every_prompt_in_one_batch_call(tmp_path) -> None:
    generator = RecordingGenerator()
    prompts = make_prompts(3)
    store = RecordingStore()
    result = build(
        tmp_path,
        generator=generator,
        prompts=prompts,
        steps=1,
        per_step=3,
        store=store,
    ).run()

    assert generator.batches == [("mbpp-0000", "mbpp-0001", "mbpp-0002")]
    assert result.steps == 1
    assert store.metrics[0]["prompts"] == 3
    assert store.metrics[0]["sequences"] == 24


def test_a_step_applies_exactly_one_optimizer_update(tmp_path) -> None:
    generator = RecordingGenerator()
    store = RecordingStore()
    result = build(
        tmp_path,
        generator=generator,
        prompts=make_prompts(3),
        steps=2,
        per_step=3,
        store=store,
    ).run()

    # Two steps, three groups each, one weight update per step.
    assert generator.updates == 2
    assert len(generator.groups) == 6
    assert result.steps == 2
    assert len(store.metrics) == 2


def test_each_group_keeps_its_own_rewards(tmp_path) -> None:
    generator = RecordingGenerator()
    runner = PassCountRunner({"mbpp-0000": 4, "mbpp-0001": 1})
    build(
        tmp_path,
        generator=generator,
        prompts=make_prompts(2),
        steps=1,
        per_step=2,
        runner=runner,
    ).run()

    # The first group passes every test, the second passes one of four. The
    # rewards never mix, so the advantage is normalised inside each group.
    assert generator.groups == [
        ("mbpp-0000", (1.0,) * 8),
        ("mbpp-0001", (0.25,) * 8),
    ]


def test_pooling_the_groups_would_invent_an_advantage() -> None:
    from rlvr.qwen import group_advantages

    # Both groups are uniform, so neither holds a signal for GRPO.
    first, second = [1.0] * 8, [0.25] * 8
    assert group_advantages(first) == [0.0] * 8
    assert group_advantages(second) == [0.0] * 8
    # Normalising across the step would make every sample look informative.
    assert any(value != 0.0 for value in group_advantages(first + second))


def test_entropy_is_the_mean_over_the_groups_in_the_step(tmp_path) -> None:
    generator = RecordingGenerator()
    store = RecordingStore()
    config_text = CONFIG_TEXT + "\n[training]\nentropy_every_steps = 1\n"
    build(
        tmp_path,
        generator=generator,
        prompts=make_prompts(2),
        steps=1,
        per_step=2,
        store=store,
        config_text=config_text,
    ).run()

    assert generator.measure_flags == [True, True]
    assert store.metrics[0]["mean_entropy"] == pytest.approx(0.5)


def test_a_short_batch_is_rejected_before_any_scoring(tmp_path) -> None:
    generator = ShortBatchGenerator(drop_group=True)
    runner = PassCountRunner()
    with pytest.raises(ValueError, match="groups"):
        build(
            tmp_path,
            generator=generator,
            prompts=make_prompts(2),
            steps=1,
            per_step=2,
            runner=runner,
        ).run()
    assert runner.calls == []
    assert generator.updates == 0


def test_a_short_group_is_rejected_before_any_scoring(tmp_path) -> None:
    generator = ShortBatchGenerator(drop_group=False)
    runner = PassCountRunner()
    with pytest.raises(ValueError, match="generations"):
        build(
            tmp_path,
            generator=generator,
            prompts=make_prompts(2),
            steps=1,
            per_step=2,
            runner=runner,
        ).run()
    assert runner.calls == []
    assert generator.updates == 0


def test_the_fake_step_counter_matches_the_weight_updates(tmp_path) -> None:
    generator = FakeGenerator()
    store = RecordingStore()
    result = build(
        tmp_path,
        generator=generator,
        prompts=fake_prompts(4),
        steps=2,
        per_step=2,
        runner=FakeTestRunner(),
        store=store,
    ).run()

    assert generator.update_groups == [2, 2]
    assert generator.step == 2 == result.steps
    # One accumulate call per prompt group, not per step.
    assert [prompt_id for prompt_id, _ in generator.update_calls] == [
        "mbpp-0000",
        "mbpp-0001",
        "mbpp-0002",
        "mbpp-0003",
    ]


def test_the_fake_records_one_step_number_per_weight_update(tmp_path) -> None:
    store = RecordingStore()
    build(
        tmp_path,
        generator=FakeGenerator(),
        prompts=fake_prompts(4),
        steps=3,
        per_step=2,
        runner=FakeTestRunner(),
        store=store,
    ).run()

    assert sorted({record.step for record in store.rollouts}) == [1, 2, 3]


def test_the_checkpoint_step_matches_the_weight_update_count(tmp_path) -> None:
    generator = FakeGenerator()
    store = RunStore(tmp_path / "runs", "code-score", timestamp="20261112T000000Z")
    config_text = CONFIG_TEXT + "\n[run]\ncheckpoint_every_steps = 2\n"
    build(
        tmp_path,
        generator=generator,
        prompts=fake_prompts(4),
        steps=2,
        per_step=2,
        runner=FakeTestRunner(),
        store=store,
        config_text=config_text,
    ).run()

    checkpoint = find_latest_checkpoint(store.checkpoints_dir)
    assert checkpoint is not None and checkpoint.step == 2
    manifest = read_checkpoint_manifest(checkpoint.directory)
    assert manifest["step"] == 2
    assert generator.step == 2


def test_the_meta_records_one_update_per_step(tmp_path) -> None:
    store = RecordingStore()
    build(tmp_path, generator=RecordingGenerator(), prompts=make_prompts(2), store=store).run()
    assert store.meta["trainer"]["optimizer_updates_per_step"] == 1
