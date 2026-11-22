"""Tests for editing a run through the control file, between steps.

The run holds its training pool and test lists in memory, so these tests check
the only supported way to change them mid-run: a control file polled at the top
of every step. No GPU, Docker, or model is involved.
"""

from __future__ import annotations

import json
from pathlib import Path

from rlvr.batching import BatchSizePolicy
from rlvr.config import load_config
from rlvr.control import ControlWatcher
from rlvr.experiment import Experiment, Outcome, Prompt

CONFIG_TEXT = """\
[model]
name = "Qwen/Qwen3-0.6B"
revision = "r"

[judge]
model = "fake-judge-model"

[data]
split_seed = 1

[fake]
steps = 3
"""


class WritingGenerator:
    """A generator that can write the control file during a step.

    `on_accumulate` runs once per prompt group, which is inside a step, so a
    file written there is first seen by the next step.
    """

    def __init__(self) -> None:
        self.steps_seen: list[str] = []
        self.on_accumulate = None

    def generate(self, prompt, *, n, temperature):
        from rlvr.experiment import Generation

        return [
            Generation(
                reasoning="",
                code=f"def f():\n    return 1  # {prompt.prompt_id}",
                prompt_tokens=5,
                completion_tokens=5,
            )
            for _ in range(n)
        ]

    def generate_batch(self, prompts, *, n, temperature, thinking_budget=None):
        return [self.generate(prompt, n=n, temperature=temperature) for prompt in prompts]

    def accumulate(self, prompt, generations, rewards, *, measure_entropy=False):
        self.steps_seen.append(prompt.prompt_id)
        if self.on_accumulate is not None:
            callback, self.on_accumulate = self.on_accumulate, None
            callback()
        return None

    def apply_update(self) -> None:
        pass

    def save_checkpoint(self, directory) -> None:
        pass

    def load_checkpoint(self, directory) -> None:
        pass


class StubRunner:
    __test__ = False

    def __init__(self, passing: set[str] | None = None) -> None:
        self.passing = passing or set()
        self.tests: dict[str, tuple[str, ...]] = {}
        self.calls: list[str] = []

    def run(self, prompt: Prompt, code: str) -> Outcome:
        self.calls.append(prompt.prompt_id)
        if prompt.prompt_id in self.passing:
            return Outcome(tests_passed=2, tests_total=2, error_class="pass")
        return Outcome(tests_passed=1, tests_total=2, error_class="wrong_answer")

    def replace_tests(self, prompt_id: str, tests) -> None:
        self.tests[prompt_id] = tuple(tests)


class StubJudge:
    def score_output(self, prompt, code):
        raise AssertionError("the code arm must not judge")

    score_process = score_output
    score_outputs = score_output
    score_processes = score_output


class RecordingStore:
    def __init__(self) -> None:
        self.rollouts = []
        self.metrics = []
        self.control_events = []
        self.meta = None
        self.eval = None
        self.timestamp = "20261112T000000Z"
        self.run_dir = Path("runs/test/20261112T000000Z")
        self.checkpoints_dir = self.run_dir / "checkpoints"

    def write_meta(self, meta) -> None:
        self.meta = meta

    def append_rollout(self, record) -> None:
        self.rollouts.append(record)

    def append_metric(self, row) -> None:
        self.metrics.append(row)

    def append_control_event(self, event) -> None:
        self.control_events.append(event)

    def write_eval(self, payload) -> None:
        self.eval = payload


def prompt(name: str) -> Prompt:
    return Prompt(prompt_id=name, problem=f"problem {name}")


def write_control(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def build(
    tmp_path: Path,
    prompts,
    *,
    passing: set[str] | None = None,
    steps: int = 3,
    resume_step: int = 0,
    control: bool = True,
):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        CONFIG_TEXT.replace("steps = 3", f"steps = {steps}"), encoding="utf-8"
    )
    control_path = tmp_path / "control.json"
    store = RecordingStore()
    runner = StubRunner(passing)
    generator = WritingGenerator()
    experiment = Experiment(
        arm="code-score",
        config=load_config(config_path),
        generator=generator,
        judge=StubJudge(),
        test_runner=runner,
        store=store,
        prompts=list(prompts),
        eval_prompts=[prompt("eval-0000")],
        batch_policy=BatchSizePolicy(current=1, max_prompts=1),
        control=ControlWatcher(control_path) if control else None,
        resume_step=resume_step,
    )
    return experiment, store, runner, generator, control_path


def prompts_by_step(store: RecordingStore) -> list[str]:
    result: list[str] = []
    seen: set[int] = set()
    for record in store.rollouts:
        if record.step not in seen:
            seen.add(record.step)
            result.append(record.prompt_id)
    return result


def test_a_dropped_prompt_is_not_sampled_again(tmp_path: Path) -> None:
    experiment, store, _runner, _gen, control = build(
        tmp_path, [prompt("q0"), prompt("q1"), prompt("q2")], steps=3
    )
    write_control(control, {"drop": ["q1"]})

    experiment.run()

    assert prompts_by_step(store) == ["q0", "q2", "q0"]


def test_dropping_the_whole_pool_stops_the_run(tmp_path: Path) -> None:
    experiment, store, _runner, _gen, control = build(
        tmp_path, [prompt("q0"), prompt("q1")], steps=2
    )
    write_control(control, {"drop": ["q0", "q1"]})

    result = experiment.run()

    assert result.stop_reason == "completed"
    assert store.rollouts == []


def test_unknown_prompt_ids_are_ignored(tmp_path: Path) -> None:
    experiment, store, _runner, _gen, control = build(
        tmp_path, [prompt("q0")], steps=2
    )
    write_control(control, {"drop": ["not-a-prompt"]})

    experiment.run()

    assert prompts_by_step(store) == ["q0", "q0"]


def test_a_test_replacement_reaches_the_runner(tmp_path: Path) -> None:
    experiment, _store, runner, _gen, control = build(
        tmp_path, [prompt("q0")], steps=2
    )
    write_control(control, {"tests": {"q0": ["assert f() == 1"]}})

    experiment.run()

    assert runner.tests == {"q0": ("assert f() == 1",)}


def test_one_event_is_written_per_change_not_per_step(tmp_path: Path) -> None:
    experiment, store, _runner, _gen, control = build(
        tmp_path, [prompt("q0"), prompt("q1")], steps=4
    )
    write_control(control, {"drop": ["q1"]})

    experiment.run()

    assert len(store.control_events) == 1
    event = store.control_events[0]
    assert event["action"] == "apply"
    assert event["drop"] == ["q1"]
    assert event["step"] == 1


def test_a_broken_file_is_reported_and_the_run_continues(tmp_path: Path) -> None:
    experiment, store, _runner, _gen, control = build(
        tmp_path, [prompt("q0"), prompt("q1")], steps=2
    )
    control.write_text("{not json", encoding="utf-8")

    result = experiment.run()

    assert result.stop_reason == "completed"
    assert any(event["action"] == "error" for event in store.control_events)
    assert store.rollouts


def test_a_file_written_mid_run_is_seen_on_the_next_step(tmp_path: Path) -> None:
    experiment, store, _runner, generator, control = build(
        tmp_path, [prompt("q0"), prompt("q1"), prompt("q2")], steps=2
    )
    generator.on_accumulate = lambda: write_control(control, {"drop": ["q1", "q2"]})

    experiment.run()

    # Step one samples q0 and writes the file, step two sees the drop.
    assert prompts_by_step(store) == ["q0", "q0"]


def test_a_resume_applies_the_drop_before_the_first_step(tmp_path: Path) -> None:
    experiment, store, _runner, _gen, control = build(
        tmp_path,
        [prompt("q0"), prompt("q1"), prompt("q2")],
        steps=8,
        resume_step=5,
    )
    write_control(control, {"drop": ["q0", "q1"]})

    experiment.run()

    assert {row["stage"] for row in store.metrics} == {"unstaged"}
    assert set(prompts_by_step(store)) == {"q2"}


def test_no_control_file_means_no_events(tmp_path: Path) -> None:
    experiment, store, _runner, _gen, _control = build(
        tmp_path, [prompt("q0")], steps=2, control=False
    )

    experiment.run()

    assert store.control_events == []


def test_a_test_edit_is_applied_once_and_not_every_step(tmp_path: Path) -> None:
    experiment, store, runner, _gen, control = build(
        tmp_path, [prompt("q0")], steps=3
    )
    write_control(control, {"tests": {"q0": ["assert f() == 1"]}})

    experiment.run()

    assert runner.tests == {"q0": ("assert f() == 1",)}
    applied = [e for e in store.control_events if e["action"] == "apply"]
    assert len(applied) == 1
