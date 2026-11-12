"""Tests for resume: opening the latest folder, truncating past the checkpoint,
and not duplicating records. These use the fake generator and real files."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rlvr.checkpoint import checkpoint_dir_name, find_latest_checkpoint, write_checkpoint_manifest
from rlvr.config import load_config
from rlvr.experiment import Experiment, Prompt
from rlvr.fake import FakeGenerator, FakeJudge, FakeTestRunner
from rlvr.runs import RunStore
from rlvr.schema import RolloutRecord

TIMESTAMP = "20261112T000000Z"

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
prompts_per_step = 1

[run]
checkpoint_every_steps = 2
"""


def _record(step: int) -> RolloutRecord:
    return RolloutRecord(
        arm="code-score",
        step=step,
        prompt_id="mbpp-0000",
        reasoning="r",
        code="c",
        prompt_tokens=1,
        completion_tokens=1,
        mean_entropy=0.5,
        tests_passed=1,
        tests_total=1,
        error_class="pass",
        code_score=1.0,
        judge_score=None,
        process_score=None,
        reward=1.0,
        cache_hit=False,
    )


def _experiment(tmp_path: Path, store: RunStore, *, resume_step: int = 0, prior=()):
    config = load_config(_config_path(tmp_path))
    return Experiment(
        arm="code-score",
        config=config,
        generator=FakeGenerator(),
        judge=FakeJudge(),
        test_runner=FakeTestRunner(),
        store=store,
        prompts=[Prompt(prompt_id="mbpp-0000", problem="p")],
        eval_prompts=[Prompt(prompt_id="eval-0000", problem="p")],
        resume_step=resume_step,
        prior_evaluations=prior,
    )


def _config_path(tmp_path: Path) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(CONFIG_TEXT, encoding="utf-8")
    return path


def _rollout_steps(run_dir: Path) -> list[int]:
    lines = (run_dir / "rollouts.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line)["step"] for line in lines if line]


def test_resume_opens_the_latest_run_folder(tmp_path) -> None:
    first = RunStore(tmp_path / "runs", "code-score", timestamp="20260101T000000Z")
    first.write_meta({"arm": "code-score"})
    second = RunStore(tmp_path / "runs", "code-score", timestamp="20260201T000000Z")
    second.write_meta({"arm": "code-score"})
    store = RunStore.resume(tmp_path / "runs", "code-score")
    assert store.timestamp == "20260201T000000Z"


def test_resume_raises_without_a_run_folder(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="no run folder"):
        RunStore.resume(tmp_path / "runs", "code-score")


def test_truncate_drops_rollouts_and_metrics_past_a_step(tmp_path) -> None:
    store = RunStore(tmp_path / "runs", "code-score", timestamp=TIMESTAMP)
    for step in (1, 2, 3):
        for _ in range(8):
            store.append_rollout(_record(step))
        store.append_metric({"step": step})

    store.truncate_to_step(2)

    assert _rollout_steps(store.run_dir) == [1] * 8 + [2] * 8
    metric_steps = [
        json.loads(line)["step"]
        for line in (store.run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert metric_steps == [1, 2]


def test_truncate_drops_evaluations_past_a_step(tmp_path) -> None:
    store = RunStore(tmp_path / "runs", "code-score", timestamp=TIMESTAMP)
    store.write_eval(
        {
            "arm": "code-score",
            "timestamp": TIMESTAMP,
            "step": 3,
            "pass_at_1": 0.5,
            "count": 1,
            "subset": {},
            "evaluations": [
                {"step": 2, "kind": "checkpoint", "pass_at_1": 0.4, "count": 1,
                 "prompt_ids": ["eval-0000"], "subset": {}},
                {"step": 3, "kind": "final", "pass_at_1": 0.5, "count": 1,
                 "prompt_ids": ["eval-0000"], "subset": None},
            ],
        }
    )

    store.truncate_to_step(2)

    evaluations = store.read_evaluations()
    assert [point["step"] for point in evaluations] == [2]


def test_read_evaluations_is_empty_without_eval_json(tmp_path) -> None:
    store = RunStore(tmp_path / "runs", "code-score", timestamp=TIMESTAMP)
    assert store.read_evaluations() == []


def test_resume_continues_without_duplicating_records(tmp_path) -> None:
    root = tmp_path / "runs"
    store = RunStore(root, "code-score", timestamp=TIMESTAMP)
    store.write_meta({"arm": "code-score"})
    # A run that wrote steps 1 and 2, saved a checkpoint at step 2, then
    # crashed partway through step 3, leaving one phantom step-3 record.
    for step in (1, 2):
        for _ in range(8):
            store.append_rollout(_record(step))
        store.append_metric({"step": step})
    store.append_rollout(_record(3))
    write_checkpoint_manifest(
        store.checkpoints_dir / checkpoint_dir_name(2),
        step=2,
        arm="code-score",
        timestamp=TIMESTAMP,
    )

    resumed = RunStore.resume(root, "code-score")
    checkpoint = find_latest_checkpoint(resumed.checkpoints_dir)
    assert checkpoint is not None and checkpoint.step == 2
    resumed.truncate_to_step(checkpoint.step)
    prior = resumed.read_evaluations()

    _experiment(tmp_path, resumed, resume_step=checkpoint.step, prior=prior).run()

    assert _rollout_steps(resumed.run_dir) == [1] * 8 + [2] * 8 + [3] * 8
