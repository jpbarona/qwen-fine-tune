"""Tests for resume: opening the latest folder, truncating past the checkpoint,
and not duplicating records. These use the fake generator and real files."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest

from rlvr.core.checkpoint import (
    checkpoint_dir_name,
    find_latest_checkpoint,
    read_checkpoint_manifest,
    write_checkpoint_manifest,
)
from rlvr.core.config import load_config
from rlvr.experiment import Experiment, Prompt
from rlvr.fake import FakeGenerator, FakeJudge, FakeTestRunner
from rlvr.resume import ResumeError, validate_resume
from rlvr.runs import RunStore
from rlvr.core.schema import RolloutRecord

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

[run]
checkpoint_every_steps = 2
"""


def _record(step: int) -> RolloutRecord:
    return RolloutRecord(
        arm="code-score",
        step=step,
        stage="very easy",
        prompt_id="mbpp-0000",
        reasoning="r",
        code="c",
        prompt_tokens=1,
        completion_tokens=1,
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


def test_a_resumed_store_routes_writes_to_an_attached_mirror(tmp_path) -> None:
    root = tmp_path / "runs"
    RunStore(root, "code-score", timestamp=TIMESTAMP).write_meta({"arm": "code-score"})

    class Recorder:
        def __init__(self) -> None:
            self.steps: list[int] = []
            self.started = 0

        def start(self, meta) -> None:
            self.started += 1

        def step(self, row) -> None:
            self.steps.append(row["step"])

        def eval(self, payload) -> None:
            pass

        def labels(self) -> None:
            pass

        def gpu_sample(self, row) -> None:
            pass

        def rebuild(self) -> None:
            pass

        def close(self, status="FINISHED") -> None:
            pass

    store = RunStore.resume(root, "code-score")
    recorder = Recorder()
    store.attach_mirror(recorder)
    store.append_metric({"step": 2})

    assert recorder.steps == [2]
    # A resumed store never writes meta again, so it never starts a new mirror run.
    assert recorder.started == 0


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


def test_an_old_checkpoint_with_stage_fields_still_resumes(tmp_path) -> None:
    from rlvr.resume import manifest_count, resume_batch_policy

    root = tmp_path / "runs"
    store = RunStore(root, "code-score", timestamp=TIMESTAMP)
    store.write_meta({"arm": "code-score"})
    for step in (1, 2):
        for _ in range(8):
            store.append_rollout(_record(step))
        store.append_metric({"step": step})

    # A checkpoint written before the pool change carries a stage position.
    write_checkpoint_manifest(
        store.checkpoints_dir / checkpoint_dir_name(2),
        step=2,
        arm="code-score",
        timestamp=TIMESTAMP,
        prompt_cursor=2,
        batch={"current": 1, "largest_fit": 1, "settled": True},
        stage={"index": 1, "step": 3, "pool": ["mbpp-0000"]},
    )

    resumed = RunStore.resume(root, "code-score")
    checkpoint = find_latest_checkpoint(resumed.checkpoints_dir)
    assert checkpoint is not None and checkpoint.step == 2

    manifest = read_checkpoint_manifest(checkpoint.directory)
    assert manifest["stage"]["index"] == 1
    assert manifest_count(manifest, "prompt_cursor", 0) == 2
    config = load_config(_config_path(tmp_path))
    policy = resume_batch_policy(config, manifest)
    assert policy is not None and policy.current == 1

    resumed.truncate_to_step(checkpoint.step)
    result = _experiment(
        tmp_path,
        resumed,
        resume_step=checkpoint.step,
        prior=resumed.read_evaluations(),
    ).run()

    assert result.steps == 3
    assert _rollout_steps(resumed.run_dir) == [1] * 8 + [2] * 8 + [3] * 8


def _pooled_resume(tmp_path: Path):
    text = (
        CONFIG_TEXT
        + "\n[training]\n"
        + 'pool_policy = "spread"\n'
        + 'screen_rows = "screens/a"\n'
        + "generations_per_prompt = 16\n"
    )
    config_path = tmp_path / "pooled.toml"
    config_path.write_text(text, encoding="utf-8")
    config = load_config(config_path)
    meta_path = tmp_path / "meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "arm": "code-score",
                "mode": "fake",
                "split_seed": 1,
                "config": tomllib.loads(text),
                "pool_policy": "spread",
                "screen_rows": "screens/a",
            }
        ),
        encoding="utf-8",
    )
    return config, text, meta_path


@pytest.mark.parametrize(
    ("old", "new", "field"),
    (
        ('pool_policy = "spread"', 'pool_policy = "sweep"', "pool_policy"),
        ('screen_rows = "screens/a"', 'screen_rows = "screens/b"', "screen_rows"),
    ),
)
def test_resume_rejects_a_changed_pool_policy_or_screen(
    tmp_path: Path, old: str, new: str, field: str
) -> None:
    config, text, meta_path = _pooled_resume(tmp_path)
    validate_resume(meta_path, config, arm="code-score", fake=True)

    changed = text.replace(old, new)
    assert changed != text
    changed_path = tmp_path / "changed.toml"
    changed_path.write_text(changed, encoding="utf-8")

    with pytest.raises(ResumeError, match=field):
        validate_resume(
            meta_path, load_config(changed_path), arm="code-score", fake=True
        )


def _step_size_resume(tmp_path: Path):
    text = (
        CONFIG_TEXT
        + "\n[training]\n"
        + 'pool_policy = "spread"\n'
        + 'screen_rows = "screens/a"\n'
        + "generations_per_prompt = 8\n"
        + "rollouts_per_step = 608\n"
    )
    config_path = tmp_path / "step-size.toml"
    config_path.write_text(text, encoding="utf-8")
    meta = {
        "arm": "code-score",
        "mode": "fake",
        "split_seed": 1,
        "config": tomllib.loads(text),
        "pool_policy": "spread",
        "screen_rows": "screens/a",
    }
    meta_path = tmp_path / "step-size-meta.json"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    return load_config(config_path), text, meta_path, meta


@pytest.mark.parametrize(
    ("old", "new", "field"),
    (
        ("rollouts_per_step = 608", "rollouts_per_step = 304", "rollouts_per_step"),
        (
            "generations_per_prompt = 8",
            "generations_per_prompt = 16",
            "generations_per_prompt",
        ),
    ),
)
def test_resume_rejects_a_changed_step_size(
    tmp_path: Path, old: str, new: str, field: str
) -> None:
    config, text, meta_path, _ = _step_size_resume(tmp_path)
    validate_resume(meta_path, config, arm="code-score", fake=True)

    changed = text.replace(old, new)
    assert changed != text
    changed_path = tmp_path / "changed.toml"
    changed_path.write_text(changed, encoding="utf-8")

    with pytest.raises(ResumeError, match=field):
        validate_resume(
            meta_path, load_config(changed_path), arm="code-score", fake=True
        )


def test_older_metadata_without_step_size_still_resumes(tmp_path: Path) -> None:
    config, _, meta_path, meta = _step_size_resume(tmp_path)
    training = meta["config"]["training"]
    del training["rollouts_per_step"]
    del training["generations_per_prompt"]
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    validate_resume(meta_path, config, arm="code-score", fake=True)
