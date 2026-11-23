"""Tests for the command line interface."""

from __future__ import annotations

import json
import importlib
import os
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from rlvr.cli import (
    build_judge,
    build_prompts,
    dashboard_argv,
    dashboard_command,
    main,
    run_command,
    sync_mlflow_command,
    sync_mlflow_settings,
)
from rlvr.config import load_config
from rlvr.data import load_split, prompt_id_for
from rlvr.fake import FakeJudge
from rlvr.memory import ALLOCATOR_ENV, EXPANDABLE_SEGMENTS
from rlvr.mlflow_log import NullMirror
from rlvr.synthetic import load_synthetic_questions

CONFIG = """\
[model]
name = "Qwen/Qwen3-0.6B"
revision = "r"

[judge]
model = "fake-judge-model"

[data]
split_seed = 1234

[fake]
steps = 2
"""


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(CONFIG, encoding="utf-8")
    return path


@pytest.fixture
def judge_config_path(tmp_path: Path) -> Path:
    """A fake run long enough to reach a prompt the fake runner fully passes.

    With no curriculum the pool starts at the split order, and the first fully
    passing prompt is the seventh one. Two-prompt steps need four steps to reach
    it, which is what the judge arms need for a judge call.
    """
    path = tmp_path / "judge_config.toml"
    path.write_text(CONFIG.replace("steps = 2", "steps = 4"), encoding="utf-8")
    return path


def test_run_fake_exits_zero_for_the_code_score_arm(config_path, tmp_path, capsys) -> None:
    runs_root = tmp_path / "runs"
    assert main(["run", "--arm", "code-score", "--fake", "--config", str(config_path), "--runs-root", str(runs_root)]) == 0
    assert (runs_root / "code-score").is_dir()
    assert "runs/code-score" in capsys.readouterr().out


def test_a_run_sets_expandable_segments_before_torch(
    config_path, tmp_path, monkeypatch
) -> None:
    monkeypatch.delenv(ALLOCATOR_ENV, raising=False)
    runs_root = tmp_path / "runs"

    assert main(
        ["run", "--arm", "code-score", "--fake", "--config", str(config_path), "--runs-root", str(runs_root)]
    ) == 0
    assert os.environ[ALLOCATOR_ENV] == EXPANDABLE_SEGMENTS


def test_a_user_allocator_setting_survives_the_run(
    config_path, tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv(ALLOCATOR_ENV, "max_split_size_mb:64")
    runs_root = tmp_path / "runs"

    assert main(
        ["run", "--arm", "code-score", "--fake", "--config", str(config_path), "--runs-root", str(runs_root)]
    ) == 0
    assert os.environ[ALLOCATOR_ENV] == "max_split_size_mb:64"


def test_a_fake_run_writes_zero_memory_on_every_metric_row(config_path, tmp_path) -> None:
    runs_root = tmp_path / "runs"
    assert main(
        ["run", "--arm", "code-score", "--fake", "--config", str(config_path), "--runs-root", str(runs_root)]
    ) == 0
    run_dir = next((runs_root / "code-score").iterdir())
    rows = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert rows
    for row in rows:
        assert row["gpu_live_mib"] == 0.0
        assert row["gpu_cached_mib"] == 0.0
        assert row["gpu_fragmented_mib"] == 0.0
        assert row["gpu_total_mib"] == 0.0
        assert row["gpu_unaccounted_mib"] == 0.0


@pytest.mark.parametrize("arm", ["hybrid", "gated-judge"])
def test_judge_arms_run_fake_and_populate_judge_scores(arm, judge_config_path, tmp_path) -> None:
    runs_root = tmp_path / "runs"
    assert main(
        ["run", "--arm", arm, "--fake", "--config", str(judge_config_path), "--runs-root", str(runs_root)]
    ) == 0
    run_dir = next((runs_root / arm).iterdir())
    records = [
        json.loads(line)
        for line in (run_dir / "rollouts.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert all(record["process_score"] is None for record in records)
    for record in records:
        if record["judge_score"] is None:
            assert record["reward"] == 0.0
        else:
            assert 0.0 <= record["judge_score"] <= 1.0
            assert 0.0 <= record["reward"] <= 1.0
    assert any(record["judge_score"] is not None for record in records)


def test_gated_process_runs_fake_and_populates_both_scores(
    judge_config_path, tmp_path
) -> None:
    runs_root = tmp_path / "runs"
    assert main(
        [
            "run",
            "--arm",
            "gated-process",
            "--fake",
            "--config",
            str(judge_config_path),
            "--runs-root",
            str(runs_root),
        ]
    ) == 0
    run_dir = next((runs_root / "gated-process").iterdir())
    records = [
        json.loads(line)
        for line in (run_dir / "rollouts.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    passing = [record for record in records if record["process_score"] is not None]
    assert passing
    for record in passing:
        assert 0.0 <= record["process_score"] <= 1.0
        assert 0.0 <= record["judge_score"] <= 1.0
        assert record["reward"] == pytest.approx(
            0.5 * record["process_score"] + 0.5 * record["judge_score"]
        )
    for record in records:
        if record["process_score"] is None:
            assert record["judge_score"] is None
            assert record["reward"] == 0.0


@pytest.mark.parametrize("arm", ["hybrid", "gated-judge"])
def test_a_judge_arm_fails_before_the_run_folder_when_the_key_is_missing(
    arm, config_path, tmp_path, monkeypatch, capsys
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    runs_root = tmp_path / "runs"
    code = main(
        ["run", "--arm", arm, "--config", str(config_path), "--runs-root", str(runs_root)]
    )
    captured = capsys.readouterr()
    assert code != 0
    assert "OPENROUTER_API_KEY" in captured.err
    assert not (runs_root / arm).exists()


def test_build_judge_only_builds_a_client_for_arms_that_judge(
    write_config, monkeypatch
) -> None:
    from rlvr import cli

    built = []

    class StubClient:
        @classmethod
        def from_config(cls, config, **kwargs):
            built.append(config)
            return FakeJudge()

    monkeypatch.setattr(cli, "JudgeClient", StubClient)
    config = load_config(write_config())

    build_judge("code-score", config, fake=False)
    assert built == []

    build_judge("hybrid", config, fake=False)
    build_judge("gated-judge", config, fake=False)
    build_judge("gated-process", config, fake=False)
    assert built == [config, config, config]

    build_judge("hybrid", config, fake=True)
    assert built == [config, config, config]


def test_run_writes_the_four_files(config_path, tmp_path) -> None:
    runs_root = tmp_path / "runs"
    main(["run", "--arm", "code-score", "--fake", "--config", str(config_path), "--runs-root", str(runs_root)])
    run_dir = next((runs_root / "code-score").iterdir())
    assert sorted(p.name for p in run_dir.iterdir()) == [
        "checkpoints",
        "control.json",
        "eval.json",
        "gpu.jsonl",
        "meta.json",
        "metrics.jsonl",
        "mlflow.json",
        "rollouts.jsonl",
    ]
    assert (run_dir / "meta.json").read_text(encoding="utf-8").startswith("{")


def test_a_sampler_that_cannot_start_does_not_stop_the_run(
    config_path, tmp_path, monkeypatch, capsys
) -> None:
    from rlvr import cli

    def fail_start(self):
        raise RuntimeError("cannot create a thread")

    monkeypatch.setattr(cli.GpuSampler, "start", fail_start)
    runs_root = tmp_path / "runs"
    assert main(
        ["run", "--arm", "code-score", "--fake", "--config", str(config_path), "--runs-root", str(runs_root)]
    ) == 0
    captured = capsys.readouterr()
    assert "gpu sampler did not start" in captured.err
    run_dir = next((runs_root / "code-score").iterdir())
    assert (run_dir / "metrics.jsonl").is_file()
    assert not (run_dir / "gpu.jsonl").exists()


def test_unknown_arm_is_reported_on_stderr(config_path, tmp_path, capsys) -> None:
    code = main(
        [
            "run",
            "--arm",
            "reward-shaping",
            "--fake",
            "--config",
            str(config_path),
            "--runs-root",
            str(tmp_path / "runs"),
        ]
    )
    captured = capsys.readouterr()
    assert code != 0
    assert captured.out == ""
    assert "reward-shaping" in captured.err
    assert "code-score" in captured.err and "gated-process" in captured.err


def test_missing_config_is_reported_on_stderr(tmp_path, capsys) -> None:
    code = main(
        ["run", "--arm", "code-score", "--fake", "--config", str(tmp_path / "absent.toml")]
    )
    captured = capsys.readouterr()
    assert code != 0
    assert "absent.toml" in captured.err
    assert "config" in captured.err


def test_invalid_config_is_reported_on_stderr(tmp_path, capsys) -> None:
    bad = tmp_path / "bad.toml"
    bad.write_text("[judge]\nmodel = 5\n", encoding="utf-8")
    code = main(["run", "--arm", "code-score", "--fake", "--config", str(bad)])
    captured = capsys.readouterr()
    assert code != 0
    assert "config" in captured.err


def test_unparseable_config_is_reported_on_stderr(tmp_path, capsys) -> None:
    bad = tmp_path / "bad.toml"
    bad.write_text("this is not toml at all ===\n", encoding="utf-8")
    code = main(["run", "--arm", "code-score", "--fake", "--config", str(bad)])
    captured = capsys.readouterr()
    assert code != 0
    assert "config" in captured.err


def test_without_fake_the_command_asks_for_the_train_extra(
    config_path, tmp_path, capsys, monkeypatch
) -> None:
    import rlvr.qwen
    import rlvr.vllm_backend as backend

    def missing(*args, **kwargs):
        raise ImportError("simulated missing torch")

    class StubEngine:
        def __init__(self, **kwargs) -> None:
            pass

        def close(self) -> None:
            pass

    monkeypatch.setattr(rlvr.qwen, "QwenGenerator", missing)
    monkeypatch.setattr(backend, "VLLMEngine", StubEngine)
    runs_root = tmp_path / "runs"
    code = main(
        ["run", "--arm", "code-score", "--config", str(config_path), "--runs-root", str(runs_root)]
    )
    captured = capsys.readouterr()
    assert code != 0
    assert "pip install -e '.[vllm]'" in captured.err
    assert "--fake" in captured.err
    # The generator fails before the run folder is made, so nothing is left behind.
    assert not (runs_root / "code-score").exists()


def test_arm_is_required(tmp_path, capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["run", "--fake", "--runs-root", str(tmp_path / "runs")])
    assert excinfo.value.code != 0
    assert "--arm" in capsys.readouterr().err


def test_unknown_subcommand_fails(tmp_path) -> None:
    with pytest.raises(SystemExit):
        main(["train", "--arm", "code-score", "--fake"])


def test_the_parser_accepts_eval_band() -> None:
    from rlvr.cli import build_parser

    args = build_parser().parse_args(["eval-band", "--band", "medium"])

    assert args.command == "eval-band"
    assert args.band == "medium"
    assert args.count == 100
    assert args.seed == 2026
    assert args.checkpoint is None
    assert args.out is None


def test_eval_band_requires_a_band(capsys) -> None:
    from rlvr.cli import build_parser

    with pytest.raises(SystemExit) as excinfo:
        build_parser().parse_args(["eval-band"])

    assert excinfo.value.code != 0
    assert "--band" in capsys.readouterr().err


def test_runs_root_defaults_to_runs_in_the_current_directory(config_path, tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["run", "--arm", "code-score", "--fake", "--config", str(config_path)]) == 0
    assert (tmp_path / "runs" / "code-score").is_dir()


def test_default_config_is_used_when_no_flag_is_given(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["run", "--arm", "code-score", "--fake", "--runs-root", str(tmp_path / "runs")]) == 0
    run_dir = next((tmp_path / "runs" / "code-score").iterdir())
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["model_revision"]


def test_console_script_runs_end_to_end(config_path, tmp_path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "rlvr",
            "run",
            "--arm",
            "code-score",
            "--fake",
            "--config",
            str(config_path),
            "--runs-root",
            str(tmp_path / "runs"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "runs" / "code-score").is_dir()


def test_console_script_reports_unknown_arm_without_a_traceback(tmp_path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "rlvr", "run", "--arm", "bogus", "--fake"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "bogus" in result.stderr
    assert "Traceback" not in result.stderr


def test_summary_prints_the_metrics_and_writes_the_report(
    config_path, tmp_path, capsys
) -> None:
    runs_root = tmp_path / "runs"
    main(
        ["run", "--arm", "code-score", "--fake", "--config", str(config_path), "--runs-root", str(runs_root)]
    )
    capsys.readouterr()

    report = tmp_path / "report.html"
    code = main(["summary", str(runs_root), "--html", str(report)])

    printed = capsys.readouterr().out
    assert code == 0
    for title in ("reward per step", "pass@1 per step", "entropy per step", "length per step"):
        assert title in printed
    assert "code-score" in printed
    assert f"report: {report}" in printed
    assert report.is_file()
    assert "<!DOCTYPE html>" in report.read_text(encoding="utf-8")


def test_summary_accepts_more_than_one_run_folder(config_path, tmp_path, capsys) -> None:
    runs_root = tmp_path / "runs"
    run_dirs = []
    for arm in ("code-score", "hybrid"):
        main(
            ["run", "--arm", arm, "--fake", "--config", str(config_path), "--runs-root", str(runs_root)]
        )
        run_dirs.append(str(next((runs_root / arm).iterdir())))
    capsys.readouterr()

    assert main(["summary", *run_dirs, "--no-html"]) == 0
    printed = capsys.readouterr().out
    assert "code-score" in printed
    assert "hybrid" in printed


def test_summary_without_html_does_not_write_a_file(config_path, tmp_path, capsys) -> None:
    runs_root = tmp_path / "runs"
    main(
        ["run", "--arm", "code-score", "--fake", "--config", str(config_path), "--runs-root", str(runs_root)]
    )
    capsys.readouterr()
    assert main(["summary", str(runs_root), "--no-html"]) == 0
    assert not (tmp_path / "report.html").exists()


def test_summary_rejects_a_folder_that_is_not_a_run(tmp_path, capsys) -> None:
    code = main(["summary", str(tmp_path / "nothing"), "--no-html"])
    captured = capsys.readouterr()
    assert code != 0
    assert "not a run folder" in captured.err


def test_summary_needs_at_least_one_path(capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["summary", "--no-html"])
    assert excinfo.value.code != 0


def test_publish_pushes_the_report_to_a_local_bare_repo(tmp_path, capsys) -> None:
    bare = tmp_path / "pages.git"
    subprocess.run(["git", "init", "--bare", str(bare)], check=True, capture_output=True)
    report = tmp_path / "report.html"
    report.write_text("<!DOCTYPE html><p>study</p>\n", encoding="utf-8")

    code = main(["publish", "--report", str(report), "--remote", str(bare)])

    assert code == 0
    assert "published" in capsys.readouterr().out
    shown = subprocess.run(
        ["git", "--git-dir", str(bare), "show", "gh-pages:index.html"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert shown.stdout.strip() == "<!DOCTYPE html><p>study</p>"


def test_publish_reports_a_missing_report(tmp_path, capsys) -> None:
    code = main(
        ["publish", "--report", str(tmp_path / "absent.html"), "--remote", str(tmp_path / "bare.git")]
    )
    captured = capsys.readouterr()
    assert code != 0
    assert "report not found" in captured.err


SPLIT_CONFIG = """\
[model]
name = "Qwen/Qwen3-0.6B"
revision = "r"

[judge]
model = "fake-judge-model"

[data]
split_seed = 7

[fake]
steps = 1

[run]
eval_every_steps = 1
eval_questions = 100
eval_subset_seed = 2026
"""


def write_split_config(tmp_path) -> Path:
    path = tmp_path / "split.toml"
    path.write_text(SPLIT_CONFIG, encoding="utf-8")
    return path


def test_build_prompts_separates_the_train_and_test_questions(tmp_path) -> None:
    config = load_config(write_split_config(tmp_path))
    train, evaluation = build_prompts(config)

    assert len(train) == 711 + 50
    assert len(evaluation) == 250
    assert {prompt.prompt_id for prompt in train}.isdisjoint(
        {prompt.prompt_id for prompt in evaluation}
    )
    split = load_split(seed=7)
    synthetic_ids = {
        question.prompt_id for question in load_synthetic_questions()
    }
    assert {prompt.prompt_id for prompt in train} == {
        prompt_id_for(question) for question in split.train
    } | synthetic_ids
    assert {prompt.prompt_id for prompt in evaluation} == {
        prompt_id_for(question) for question in split.test
    }
    assert train[0].problem.startswith("Write a Python function")


def test_build_prompts_puts_the_synthetic_questions_in_training(tmp_path) -> None:
    train, evaluation = build_prompts(load_config(write_split_config(tmp_path)))
    train_ids = {prompt.prompt_id for prompt in train}
    synthetic_ids = {
        question.prompt_id for question in load_synthetic_questions()
    }

    assert len(synthetic_ids) == 50
    assert synthetic_ids <= train_ids
    assert synthetic_ids.isdisjoint(
        {prompt.prompt_id for prompt in evaluation}
    )


def test_a_fake_run_evaluates_the_held_out_test_split(tmp_path, capsys) -> None:
    runs_root = tmp_path / "runs"
    assert (
        main(
            [
                "run",
                "--arm",
                "code-score",
                "--fake",
                "--config",
                str(write_split_config(tmp_path)),
                "--runs-root",
                str(runs_root),
            ]
        )
        == 0
    )
    capsys.readouterr()

    run_dir = next((runs_root / "code-score").iterdir())
    payload = json.loads((run_dir / "eval.json").read_text(encoding="utf-8"))
    split = load_split(seed=7)
    train_ids = {prompt_id_for(question) for question in split.train}
    test_ids = {prompt_id_for(question) for question in split.test}

    final = payload["evaluations"][-1]
    assert payload["step"] == 1
    assert payload["count"] == 250
    assert final["kind"] == "final"
    assert final["count"] == 250
    assert set(final["prompt_ids"]) == test_ids

    checkpoints = [
        point for point in payload["evaluations"] if point["kind"] == "checkpoint"
    ]
    assert len(checkpoints) == 1
    assert checkpoints[0]["count"] == 100
    assert len(set(checkpoints[0]["prompt_ids"])) == 100
    assert set(checkpoints[0]["prompt_ids"]) <= test_ids

    records = [
        json.loads(line)
        for line in (run_dir / "rollouts.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    # One step trains one prompt group (8 identical rollouts), so the rollout
    # prompt ids are a subset of the train ids or the synthetic easy ids, never
    # a test id.
    synthetic_ids = {
        question.prompt_id for question in load_synthetic_questions()
    }
    record_ids = {record["prompt_id"] for record in records}
    assert record_ids <= train_ids | synthetic_ids
    assert record_ids.isdisjoint(test_ids)


def test_a_bad_split_stops_before_the_run_folder(
    config_path, tmp_path, monkeypatch, capsys
) -> None:
    from rlvr import cli
    from rlvr.data import DatasetError

    def fail(*args, **kwargs):
        raise DatasetError("dataset file not found")

    monkeypatch.setattr(cli, "load_split", fail)
    runs_root = tmp_path / "runs"
    code = main(
        ["run", "--arm", "code-score", "--fake", "--config", str(config_path), "--runs-root", str(runs_root)]
    )
    captured = capsys.readouterr()
    assert code != 0
    assert "MBPP split" in captured.err
    assert not (runs_root / "code-score").exists()


RESUME_CONFIG = """\
[model]
name = "Qwen/Qwen3-0.6B"
revision = "r"

[judge]
model = "fake-judge-model"

[data]
split_seed = 7

[fake]
steps = 3

[run]
checkpoint_every_steps = 2
"""


def _rollout_steps(run_dir: Path) -> list[int]:
    return [
        json.loads(line)["step"]
        for line in (run_dir / "rollouts.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]


def test_resume_continues_from_the_last_checkpoint_without_duplicates(
    tmp_path, capsys
) -> None:
    runs_root = tmp_path / "runs"
    config_path = tmp_path / "resume.toml"
    config_path.write_text(RESUME_CONFIG, encoding="utf-8")

    assert main(
        ["run", "--arm", "code-score", "--fake", "--config", str(config_path), "--runs-root", str(runs_root)]
    ) == 0
    capsys.readouterr()

    run_dir = next((runs_root / "code-score").iterdir())
    before = _rollout_steps(run_dir)
    before_lines = (run_dir / "rollouts.jsonl").read_text(encoding="utf-8").splitlines()
    # Every step takes the two prompt batch, so each step has sixteen records.
    assert before.count(1) == 16
    assert before.count(2) == 16
    assert before.count(3) == 16

    # The checkpoint carries the batch boundary and the prompt cursor, so a
    # resume can pick them up. There is no stage position.
    manifest = json.loads(
        (run_dir / "checkpoints" / "step-00000002" / "checkpoint.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["batch"]["current"] == 2
    assert "prompt_cursor" in manifest
    assert "stage" not in manifest

    assert main(
        ["run", "--arm", "code-score", "--fake", "--resume", "--config", str(config_path), "--runs-root", str(runs_root)]
    ) == 0
    capsys.readouterr()

    # Still one run folder. Step 3 was truncated and redone from the saved
    # batch and stage state, so the run is identical to the uninterrupted one.
    assert len(list((runs_root / "code-score").iterdir())) == 1
    after_lines = (run_dir / "rollouts.jsonl").read_text(encoding="utf-8").splitlines()
    assert after_lines == before_lines
    assert _rollout_steps(run_dir) == before


def test_resume_without_a_checkpoint_fails_cleanly(tmp_path, capsys) -> None:
    runs_root = tmp_path / "runs"
    (runs_root / "code-score").mkdir(parents=True)
    (runs_root / "code-score" / "20261112T000000Z").mkdir()
    (runs_root / "code-score" / "20261112T000000Z" / "meta.json").write_text(
        '{"arm": "code-score"}\n', encoding="utf-8"
    )

    code = main(
        ["run", "--arm", "code-score", "--fake", "--resume", "--runs-root", str(runs_root)]
    )
    captured = capsys.readouterr()
    assert code != 0
    assert "no checkpoint" in captured.err


EVAL_LIMIT_CONFIG = """\
[model]
name = "Qwen/Qwen3-0.6B"
revision = "r"

[judge]
model = "fake-judge-model"

[data]
split_seed = 7

[fake]
steps = 1

[run]
eval_every_steps = 1
eval_questions = 5
eval_final_questions = 10
"""


def test_eval_final_questions_caps_the_final_eval_pool(tmp_path, capsys) -> None:
    runs_root = tmp_path / "runs"
    config_path = tmp_path / "limit.toml"
    config_path.write_text(EVAL_LIMIT_CONFIG, encoding="utf-8")

    assert main(
        ["run", "--arm", "code-score", "--fake", "--config", str(config_path), "--runs-root", str(runs_root)]
    ) == 0
    capsys.readouterr()

    run_dir = next((runs_root / "code-score").iterdir())
    payload = json.loads((run_dir / "eval.json").read_text(encoding="utf-8"))
    assert payload["count"] == 10
    assert payload["evaluations"][-1]["count"] == 10


def test_the_fake_run_carries_the_unstaged_value(
    config_path, tmp_path, capsys
) -> None:
    runs_root = tmp_path / "runs"
    assert main(
        ["run", "--arm", "code-score", "--fake", "--config", str(config_path), "--runs-root", str(runs_root)]
    ) == 0
    capsys.readouterr()

    run_dir = next((runs_root / "code-score").iterdir())
    metrics = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    records = [
        json.loads(line)
        for line in (run_dir / "rollouts.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    payload = json.loads((run_dir / "eval.json").read_text(encoding="utf-8"))
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))

    assert {row["stage"] for row in metrics} == {"unstaged"}
    assert {record["stage"] for record in records} == {"unstaged"}
    assert {point["stage"] for point in payload["evaluations"]} == {"unstaged"}
    assert meta["pools"]["train"] > 0
    assert meta["pools"]["parked"] == 0
    assert meta["pools"]["samples"] is None
    assert meta["pools"]["measured"] is False
    assert "curriculum" not in meta


def test_build_test_runner_uses_fakes_or_docker(write_config) -> None:
    from rlvr.cli import build_test_runner
    from rlvr.docker_runner import DockerTestRunner
    from rlvr.fake import FakeTestRunner

    config = load_config(write_config())
    assert isinstance(build_test_runner(config, fake=True), FakeTestRunner)
    # The real runner is built from the merged train and test questions and the
    # synthetic easy set. No Docker binary is invoked at construction, so this
    # is safe on CPU.
    runner = build_test_runner(config, fake=False)
    assert isinstance(runner, DockerTestRunner)
    synthetic_ids = {
        question.prompt_id for question in load_synthetic_questions()
    }
    assert synthetic_ids <= set(runner._questions)


def test_the_run_creates_an_empty_control_file(config_path, tmp_path) -> None:
    import json as _json

    runs_root = tmp_path / "runs"
    main(["run", "--arm", "code-score", "--fake", "--config", str(config_path), "--runs-root", str(runs_root)])
    run_dir = next((runs_root / "code-score").iterdir())
    payload = _json.loads((run_dir / "control.json").read_text(encoding="utf-8"))
    assert payload == {"drop": [], "tests": {}}


def test_a_control_path_can_be_given(config_path, tmp_path) -> None:
    import json as _json

    runs_root = tmp_path / "runs"
    control = tmp_path / "elsewhere" / "edit.json"
    main(
        [
            "run",
            "--arm",
            "code-score",
            "--fake",
            "--config",
            str(config_path),
            "--runs-root",
            str(runs_root),
            "--control",
            str(control),
        ]
    )
    assert _json.loads(control.read_text(encoding="utf-8")) == {"drop": [], "tests": {}}


@pytest.mark.parametrize(
    "old, new, expected",
    [('Qwen/Qwen3-0.6B', 'LiquidAI/LFM2.5-2.6B', 'model.name'),
     ('revision = "r"', 'revision = "different"', 'model.revision'),
     ('split_seed = 7', 'split_seed = 8', 'data.split_seed')],
)
def test_resume_rejects_changed_identity_before_changing_records(tmp_path, capsys, old, new, expected):
    config_path = tmp_path / "resume.toml"
    config_path.write_text(RESUME_CONFIG, encoding="utf-8")
    runs_root = tmp_path / "runs"
    args = ["run", "--arm", "code-score", "--fake", "--config", str(config_path),
            "--runs-root", str(runs_root)]
    assert main(args) == 0
    capsys.readouterr()
    run_dir = next((runs_root / "code-score").iterdir())
    before = {p.name: p.read_bytes() for p in run_dir.iterdir() if p.is_file()}
    changed = RESUME_CONFIG.replace(old, new)
    assert changed != RESUME_CONFIG
    config_path.write_text(changed, encoding="utf-8")
    assert main(args + ["--resume"]) == 2
    assert expected in capsys.readouterr().err
    assert before == {p.name: p.read_bytes() for p in run_dir.iterdir() if p.is_file()}


@pytest.mark.parametrize("error_type", [ValueError, RuntimeError, EOFError])
def test_failed_checkpoint_load_preserves_existing_records(tmp_path, monkeypatch, capsys, error_type):
    from rlvr import cli

    config_path = tmp_path / "resume.toml"
    config_path.write_text(RESUME_CONFIG, encoding="utf-8")
    runs_root = tmp_path / "runs"
    args = ["run", "--arm", "code-score", "--fake", "--config", str(config_path),
            "--runs-root", str(runs_root)]
    assert main(args) == 0
    run_dir = next((runs_root / "code-score").iterdir())
    before = {p.name: p.read_bytes() for p in run_dir.iterdir() if p.is_file()}

    def fail_load(self, directory):
        raise error_type("incomplete checkpoint")

    monkeypatch.setattr(cli.FakeGenerator, "load_checkpoint", fail_load)
    assert main(args + ["--resume"]) == 2
    assert "incomplete checkpoint" in capsys.readouterr().err
    assert before == {p.name: p.read_bytes() for p in run_dir.iterdir() if p.is_file()}


def write_screen_config(tmp_path, screen_rows) -> Path:
    path = tmp_path / "screen-config.toml"
    path.write_text(
        SPLIT_CONFIG
        + "\n[training]\n"
        + "generations_per_prompt = 16\n"
        + f'screen_rows = "{screen_rows.as_posix()}"\n',
        encoding="utf-8",
    )
    return path


def test_build_pools_returns_none_without_screen_rows(tmp_path) -> None:
    from rlvr.cli import build_pools

    config = load_config(write_split_config(tmp_path))
    train, _ = build_prompts(config)

    assert build_pools(config, train) is None


def test_build_pools_reads_the_screen_rows(tmp_path) -> None:
    from rlvr.cli import build_pools
    from rlvr.screening import ScreenRow, ScreenStore

    config = load_config(write_split_config(tmp_path))
    train, _ = build_prompts(config)
    directory = tmp_path / "screen"
    store = ScreenStore(directory)
    expected = {"train": 0, "retired": 0, "parked": 0}
    for index, prompt in enumerate(train):
        passed = (0, 8, 16)[index % 3]
        store.append_row(
            ScreenRow(
                prompt_id=prompt.prompt_id,
                samples=16,
                passed=passed,
                pass_rate=passed / 16,
            )
        )
        expected[("parked", "train", "retired")[index % 3]] += 1

    pools = build_pools(
        load_config(write_screen_config(tmp_path, directory)), train
    )

    assert pools is not None
    assert pools.samples == 16
    assert len(pools.train) == expected["train"]
    assert len(pools.retired) == expected["retired"]
    assert len(pools.parked) == expected["parked"]


def test_build_pools_rejects_a_screen_at_another_group_size(tmp_path) -> None:
    from rlvr.cli import UsageError, build_pools
    from rlvr.screening import ScreenRow, ScreenStore

    directory = tmp_path / "screen"
    store = ScreenStore(directory)
    config = load_config(write_screen_config(tmp_path, directory))
    train, _ = build_prompts(config)
    for prompt in train:
        store.append_row(
            ScreenRow(
                prompt_id=prompt.prompt_id,
                samples=8,
                passed=4,
                pass_rate=0.5,
            )
        )

    with pytest.raises(UsageError) as error:
        build_pools(config, train)

    message = str(error.value)
    assert "8 samples per prompt" in message
    assert "training.generations_per_prompt" in message
    assert "16" in message


def test_build_pools_accepts_the_rows_file(tmp_path) -> None:
    from rlvr.cli import build_pools
    from rlvr.screening import ScreenRow, ScreenStore

    config = load_config(write_split_config(tmp_path))
    train, _ = build_prompts(config)
    directory = tmp_path / "screen"
    store = ScreenStore(directory)
    for prompt in train:
        store.append_row(
            ScreenRow(
                prompt_id=prompt.prompt_id,
                samples=16,
                passed=8,
                pass_rate=0.5,
            )
        )

    pools = build_pools(
        load_config(write_screen_config(tmp_path, store.rows_path)), train
    )

    assert pools is not None
    assert len(pools.train) == len(train)


def test_an_empty_screen_fails_loudly(tmp_path) -> None:
    from rlvr.cli import UsageError, build_pools

    config = load_config(write_split_config(tmp_path))
    train, _ = build_prompts(config)
    empty = tmp_path / "screen"
    empty.mkdir()

    with pytest.raises(UsageError, match="screen rows are empty"):
        build_pools(load_config(write_screen_config(tmp_path, empty)), train)


def test_a_screen_row_outside_the_training_split_fails_loudly(tmp_path) -> None:
    from rlvr.cli import UsageError, build_pools
    from rlvr.screening import ScreenRow, ScreenStore

    config = load_config(write_split_config(tmp_path))
    train, _ = build_prompts(config)
    directory = tmp_path / "screen"
    store = ScreenStore(directory)
    store.append_row(
        ScreenRow(prompt_id="not-in-split", samples=16, passed=8, pass_rate=0.5)
    )

    with pytest.raises(UsageError, match="not in the training split"):
        build_pools(load_config(write_screen_config(tmp_path, directory)), train)


def test_two_sample_counts_in_one_screen_fail_loudly(tmp_path) -> None:
    from rlvr.cli import UsageError, build_pools
    from rlvr.screening import ScreenRow, ScreenStore

    config = load_config(write_split_config(tmp_path))
    train, _ = build_prompts(config)
    directory = tmp_path / "screen"
    store = ScreenStore(directory)
    store.append_row(
        ScreenRow(
            prompt_id=train[0].prompt_id, samples=16, passed=8, pass_rate=0.5
        )
    )
    store.append_row(
        ScreenRow(
            prompt_id=train[1].prompt_id, samples=8, passed=4, pass_rate=0.5
        )
    )

    with pytest.raises(UsageError, match="cannot build the pools"):
        build_pools(load_config(write_screen_config(tmp_path, directory)), train)


def test_a_screen_path_must_be_a_folder_or_rows_file(tmp_path) -> None:
    from rlvr.cli import UsageError, build_pools

    config = load_config(write_split_config(tmp_path))
    train, _ = build_prompts(config)
    other = tmp_path / "scores.json"
    other.write_text("{}\n", encoding="utf-8")

    with pytest.raises(UsageError, match="screen folder or a rows.jsonl file"):
        build_pools(load_config(write_screen_config(tmp_path, other)), train)


def test_a_fake_run_uses_the_measured_pool_and_the_spread_schedule(
    tmp_path, capsys
) -> None:
    from rlvr.screening import ScreenRow, ScreenStore

    directory = tmp_path / "screen"
    path = tmp_path / "measured.toml"
    path.write_text(
        SPLIT_CONFIG
        + f'\n[training]\nscreen_rows = "{directory.as_posix()}"\n'
        + 'pool_policy = "spread"\ngenerations_per_prompt = 16\n'
        + "rollouts_per_step = 64\n",
        encoding="utf-8",
    )
    config = load_config(path)
    train, _ = build_prompts(config)
    store = ScreenStore(directory)
    for index, prompt in enumerate(train):
        passed = 0 if index < 10 else 8
        store.append_row(
            ScreenRow(
                prompt_id=prompt.prompt_id,
                samples=16,
                passed=passed,
                pass_rate=passed / 16,
            )
        )

    runs_root = tmp_path / "runs"

    assert (
        main(
            [
                "run",
                "--arm",
                "code-score",
                "--fake",
                "--config",
                str(path),
                "--runs-root",
                str(runs_root),
            ]
        )
        == 0
    )
    capsys.readouterr()

    run_dir = next((runs_root / "code-score").iterdir())
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    metric = json.loads((run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()[0])

    assert meta["pools"]["measured"] is True
    assert meta["pools"]["samples"] == 16
    assert meta["pools"]["parked"] == 10
    assert meta["pool_policy"] == "spread"
    assert meta["screen_rows"] == directory.as_posix()
    assert metric["pool_policy"] == "spread"
    assert metric["train_prompts"] == 1
    assert metric["parked_prompts"] == 3


def test_a_small_parked_pool_warns_about_the_short_block(tmp_path) -> None:
    from dataclasses import replace

    from rlvr.cli import spread_share_warning
    from rlvr.pools import Pools

    config = replace(
        load_config(write_split_config(tmp_path)),
        pool_policy="spread",
        screen_rows="screens/a",
        generations_per_prompt=16,
        rollouts_per_step=608,
    )
    small = Pools(
        train=("t",),
        retired=(),
        parked=tuple(f"p{index}" for index in range(10)),
        samples=16,
    )
    message = spread_share_warning(config, small)

    assert message is not None
    assert "holds 10 prompts" in message
    assert "block of 19" in message
    assert "19/38" in message

    enough = Pools(
        train=("t",),
        retired=(),
        parked=tuple(f"p{index}" for index in range(19)),
        samples=16,
    )
    assert spread_share_warning(config, enough) is None


def test_a_partial_screen_fails_loudly(tmp_path) -> None:
    from rlvr.cli import UsageError, build_pools
    from rlvr.screening import ScreenRow, ScreenStore

    config = load_config(write_split_config(tmp_path))
    train, _ = build_prompts(config)
    directory = tmp_path / "screen"
    store = ScreenStore(directory)
    for prompt in train[:-3]:
        store.append_row(
            ScreenRow(
                prompt_id=prompt.prompt_id,
                samples=16,
                passed=8,
                pass_rate=0.5,
            )
        )

    with pytest.raises(UsageError, match="3 missing"):
        build_pools(load_config(write_screen_config(tmp_path, directory)), train)


def test_training_steps_uses_the_schedule_prompt_count(tmp_path) -> None:
    from rlvr.cli import training_steps
    from rlvr.pools import Pools

    path = tmp_path / "steps.toml"
    path.write_text(
        SPLIT_CONFIG
        + '\n[training]\npool_policy = "spread"\n'
        'screen_rows = "rows.jsonl"\ngenerations_per_prompt = 16\n'
        "rollouts_per_step = 64\n",
        encoding="utf-8",
    )
    config = load_config(path)
    prompts = [object() for _ in range(200)]
    pools = Pools(
        train=tuple(f"p{index}" for index in range(100)),
        retired=(),
        parked=("p100",),
        samples=16,
    )

    # 64 rollouts at group size 16 is four prompts per step. Five epochs over
    # 100 train prompts is 25 steps per epoch.
    assert config.epochs == 5
    assert training_steps(config, prompts, pools) == 125


def test_training_steps_keeps_the_starting_batch_without_a_policy(tmp_path) -> None:
    from rlvr.cli import training_steps

    config = load_config(write_split_config(tmp_path))
    prompts = [object() for _ in range(200)]

    # The old path starts at two prompts per step.
    assert training_steps(config, prompts, None) == config.epochs * 100


def test_dashboard_prints_the_remote_store(capsys, monkeypatch) -> None:
    def fail_exec(path: str, argv: list[str]) -> None:
        raise AssertionError("the remote path must not start a server")

    monkeypatch.setattr("rlvr.cli.os.execv", fail_exec)
    args = Namespace(local=False, port=None, runs_root=Path("runs"))
    assert dashboard_command(args) == 0
    assert "http://127.0.0.1:8081" in capsys.readouterr().out


def test_dashboard_prints_the_uri_for_a_namespace_with_a_port(
    tmp_path, capsys, monkeypatch
) -> None:
    """The function does not reject a port. Only `main` sees an explicit one."""

    def fail_exec(path: str, argv: list[str]) -> None:
        raise AssertionError("the remote path must not start a server")

    monkeypatch.setattr("rlvr.cli.os.execv", fail_exec)
    path = tmp_path / "config.toml"
    path.write_text(CONFIG, encoding="utf-8")
    args = Namespace(local=False, port=8081, runs_root=Path("runs"), config=path)
    assert dashboard_command(args) == 0
    assert "http://127.0.0.1:8081" in capsys.readouterr().out


def test_dashboard_falls_back_when_the_packaged_config_cannot_be_read(
    capsys, monkeypatch
) -> None:
    def fail_load(path: str | Path):
        raise OSError("no packaged config")

    monkeypatch.setattr("rlvr.cli.load_config", fail_load)
    args = Namespace(local=False, port=None, runs_root=Path("runs"))
    assert dashboard_command(args) == 0
    assert "http://127.0.0.1:8081" in capsys.readouterr().out


def test_dashboard_reads_the_packaged_config_tracking_uri(
    tmp_path, capsys, monkeypatch
) -> None:
    path = tmp_path / "default_config.toml"
    path.write_text(
        CONFIG + '\n[mlflow]\ntracking_uri = "http://127.0.0.1:9100"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr("rlvr.cli.default_config_path", lambda: path)
    args = Namespace(local=False, port=None, runs_root=Path("runs"))
    assert dashboard_command(args) == 0
    assert "http://127.0.0.1:9100" in capsys.readouterr().out


def test_dashboard_prints_the_configured_store(tmp_path, capsys, monkeypatch) -> None:
    def fail_exec(path: str, argv: list[str]) -> None:
        raise AssertionError("the remote path must not start a server")

    monkeypatch.setattr("rlvr.cli.os.execv", fail_exec)
    path = tmp_path / "config.toml"
    path.write_text(
        CONFIG + '\n[mlflow]\ntracking_uri = "http://127.0.0.1:9000"\n',
        encoding="utf-8",
    )
    args = Namespace(local=False, port=None, runs_root=Path("runs"), config=path)
    assert dashboard_command(args) == 0
    assert "http://127.0.0.1:9000" in capsys.readouterr().out


def test_dashboard_rejects_a_bad_config(tmp_path, capsys) -> None:
    path = tmp_path / "config.toml"
    path.write_text("[model]\nname = 3\n", encoding="utf-8")
    assert main(["dashboard", "--config", str(path)]) == 2
    assert "config" in capsys.readouterr().err


def test_dashboard_port_needs_local(capsys) -> None:
    assert main(["dashboard", "--port", "8099"]) == 2
    assert "--port needs --local" in capsys.readouterr().err


def test_dashboard_local_still_starts_a_server(tmp_path, monkeypatch) -> None:
    seen: list[tuple[str, list[str]]] = []

    def fake_exec(path: str, argv: list[str]) -> None:
        seen.append((path, argv))

    monkeypatch.setattr("rlvr.cli.os.execv", fake_exec)
    runs_root = tmp_path / "runs"

    assert main(["dashboard", "--local", "--runs-root", str(runs_root)]) == 0

    assert len(seen) == 1
    path, argv = seen[0]
    assert path == sys.executable
    assert argv[argv.index("--backend-store-uri") + 1] == (
        f"sqlite:///{runs_root / 'mlflow.db'}"
    )
    assert argv[argv.index("--host") + 1] == "127.0.0.1"
    assert argv[argv.index("--workers") + 1] == "1"
    # The local server defaults to a port that cannot collide with the remote.
    assert argv[argv.index("--port") + 1] == "8099"


def test_dashboard_command_runs_mlflow_as_a_module(tmp_path, monkeypatch) -> None:
    seen: list[tuple[str, list[str]]] = []

    def fake_exec(path: str, argv: list[str]) -> None:
        seen.append((path, argv))

    monkeypatch.setattr("rlvr.cli.os.execv", fake_exec)
    monkeypatch.setenv("PATH", "")

    runs_root = tmp_path / "runs"
    assert (
        main(
            [
                "dashboard",
                "--local",
                "--runs-root",
                str(runs_root),
                "--port",
                "8081",
            ]
        )
        == 0
    )

    assert len(seen) == 1
    path, argv = seen[0]
    assert path == sys.executable
    assert argv[0] == sys.executable
    assert argv[1:4] == ["-m", "mlflow", "server"]
    assert argv[argv.index("--host") + 1] == "127.0.0.1"
    assert argv[argv.index("--port") + 1] == "8081"
    assert argv[argv.index("--workers") + 1] == "1"
    assert argv[argv.index("--allowed-hosts") + 1] == "*"
    # The module form does not need the mlflow console script on PATH.
    assert "mlflow" not in os.environ.get("PATH", "")
    assert dashboard_argv(runs_root, 8081) == argv


def test_dashboard_allowed_hosts_can_be_set(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RLVR_ALLOWED_HOSTS", "example.test:8083")
    argv = dashboard_argv(tmp_path, 8081)
    assert argv[argv.index("--allowed-hosts") + 1] == "example.test:8083"


def test_dashboard_help_mentions_mlflow() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "rlvr", "dashboard", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "mlflow" in result.stdout.lower()


def test_the_dashboard_module_is_gone() -> None:
    sys.modules.pop("rlvr.dashboard", None)
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("rlvr.dashboard")


IMPORT_TIMESTAMP = "20260101T000000Z"


def make_run(runs_root: Path) -> Path:
    """A run folder with meta.json and two metrics rows, ready to import."""
    run_dir = runs_root / "code-score" / IMPORT_TIMESTAMP
    run_dir.mkdir(parents=True)
    meta = {"arm": "code-score", "timestamp": IMPORT_TIMESTAMP, "mode": "fake"}
    (run_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    rows = [
        {"step": 1, "pass_rate": 0.25},
        {"step": 2, "pass_rate": 0.5},
    ]
    (run_dir / "metrics.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    return run_dir


def count_runs() -> int:
    """The run count in the store the test mirror pointed MLflow at."""
    return len(store_runs())


def store_runs() -> list:
    """Every run in the store the test mirror pointed MLflow at."""
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    experiment = client.get_experiment_by_name("rlvr")
    if experiment is None:
        return []
    return client.search_runs([experiment.experiment_id])


def parents_and_children() -> tuple[list, list]:
    """The parent runs and the GPU child runs in the test store."""
    runs = store_runs()
    parents = [run for run in runs if "rlvr/run_dir" in run.data.tags]
    children = [run for run in runs if "mlflow.parentRunId" in run.data.tags]
    return parents, children


def test_sync_mlflow_twice_leaves_one_run(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv(
        "MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'runs' / 'mlflow.db'}"
    )
    folder = make_run(tmp_path / "runs")
    args = Namespace(run_dir=folder, config=None)
    assert sync_mlflow_command(args) == 0
    first = count_runs()
    assert first > 0
    assert sync_mlflow_command(args) == 0

    assert count_runs() == first
    parents, children = parents_and_children()
    assert len(parents) == 1
    assert len(children) == 1
    assert children[0].data.tags["mlflow.parentRunId"] == parents[0].info.run_id


def test_sync_mlflow_uses_the_tracking_uri_in_meta(tmp_path, monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    remote = tmp_path / "remote"
    remote.mkdir()
    folder = make_run(tmp_path / "runs")
    payload = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    payload["config"] = {
        "mlflow": {"tracking_uri": f"sqlite:///{remote / 'mlflow.db'}"}
    }
    (folder / "meta.json").write_text(json.dumps(payload), encoding="utf-8")

    args = Namespace(run_dir=folder, config=None)
    assert sync_mlflow_command(args) == 0

    assert (remote / "mlflow.db").is_file()
    assert not (tmp_path / "runs" / "mlflow.db").exists()
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=f"sqlite:///{remote / 'mlflow.db'}")
    experiment = client.get_experiment_by_name("rlvr")
    assert experiment is not None
    assert len(client.search_runs([experiment.experiment_id])) == 2


def test_sync_mlflow_keeps_the_checkpoints_tag(tmp_path, monkeypatch):
    """A repaired run must keep the bucket pointer its files live under."""
    monkeypatch.setenv(
        "MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'runs' / 'mlflow.db'}"
    )
    folder = make_run(tmp_path / "runs")
    payload = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    payload["config"] = {"mlflow": {"bucket": "s3://bucket/runs/v1"}}
    (folder / "meta.json").write_text(json.dumps(payload), encoding="utf-8")

    args = Namespace(run_dir=folder, config=None)
    assert sync_mlflow_command(args) == 0
    assert sync_mlflow_command(args) == 0

    parents, children = parents_and_children()
    assert len(parents) == 1
    assert len(children) == 1
    assert parents[0].data.tags["rlvr/checkpoints_uri"] == (
        f"s3://bucket/runs/v1/code-score/{IMPORT_TIMESTAMP}"
    )


def test_sync_mlflow_settings_fall_back_to_the_http_store(monkeypatch) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    settings = sync_mlflow_settings(None, {})
    assert settings.tracking_uri == "http://127.0.0.1:8081"


def test_sync_mlflow_settings_explicit_config_wins(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:8888")
    path = tmp_path / "config.toml"
    path.write_text(
        CONFIG + '\n[mlflow]\ntracking_uri = "http://127.0.0.1:9000"\n',
        encoding="utf-8",
    )
    settings = sync_mlflow_settings(path, {"config": {"mlflow": {}}})
    assert settings.tracking_uri == "http://127.0.0.1:9000"


def test_sync_mlflow_prints_the_store_it_used(
    tmp_path, capsys, monkeypatch
) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    folder = make_run(tmp_path / "runs")
    monkeypatch.setattr("rlvr.cli.import_run", lambda *args, **kwargs: "run-1")
    args = Namespace(run_dir=folder, config=None)

    assert sync_mlflow_command(args) == 0

    assert "store: http://127.0.0.1:8081" in capsys.readouterr().out


class StubUploader:
    """Records the calls a run makes on the uploader."""

    calls: list[tuple] = []

    def __init__(self, run_dir, arm, started, destination, **kwargs) -> None:
        StubUploader.calls.append(("init", run_dir, arm, started, destination, kwargs))

    def start(self) -> None:
        StubUploader.calls.append(("start",))

    def close(self, timeout: float = 60.0) -> None:
        StubUploader.calls.append(("close",))


class CloseFailsUploader(StubUploader):
    """An uploader whose close raises, like a bucket that went away."""

    def close(self, timeout: float = 60.0) -> None:
        StubUploader.calls.append(("close",))
        raise RuntimeError("the bucket is gone")


class QuietExperiment:
    """A stand-in experiment that writes meta and returns at once."""

    def __init__(self, **kwargs) -> None:
        self.store = kwargs["store"]
        self.arm = kwargs["arm"]

    def run(self):
        from rlvr.experiment import RunSummary

        self.store.write_meta({"arm": self.arm, "timestamp": self.store.timestamp})
        return RunSummary(
            arm=self.arm,
            run_dir=self.store.run_dir,
            steps=0,
            rollouts=0,
            mean_reward=0.0,
            pass_at_1=0.0,
        )


class TagReadingExperiment(QuietExperiment):
    """Reads the checkpoints tag right after the mirror opens the run."""

    seen: list[str] = []

    def run(self):
        summary = super().run()
        from mlflow.tracking import MlflowClient

        run = MlflowClient().get_run(self.store.mirror.run_id)
        TagReadingExperiment.seen.append(run.data.tags.get("rlvr/checkpoints_uri", ""))
        return summary


class FailingExperiment:
    """A stand-in experiment whose training raises."""

    def __init__(self, **kwargs) -> None:
        pass

    def run(self):
        raise RuntimeError("training died")


def write_bucket_config(tmp_path: Path) -> Path:
    path = tmp_path / "bucket.toml"
    path.write_text(CONFIG + '\n[mlflow]\nbucket = "s3://bucket/runs/v1"\n', encoding="utf-8")
    return path


def run_args(tmp_path: Path, config_path: Path) -> Namespace:
    return Namespace(
        arm="code-score",
        config=config_path,
        fake=True,
        runs_root=tmp_path / "runs",
        resume=False,
        control=None,
    )


def test_run_uses_the_environment_tracking_uri(tmp_path, config_path, monkeypatch) -> None:
    """The container gets MLFLOW_TRACKING_URI, and the mirror must honor it."""
    seen: list[str | None] = []

    def fake_open_mirror(runs_root, run_dir, enabled, *, tracking_uri=None, **kwargs):
        seen.append(tracking_uri)
        return NullMirror()

    monkeypatch.setattr("rlvr.cli.open_mirror", fake_open_mirror)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://172.17.0.1:8081")

    assert run_command(run_args(tmp_path, config_path)) == 0

    assert seen == ["http://172.17.0.1:8081"]


def test_a_config_tracking_uri_wins_over_the_environment(
    tmp_path, config_path, monkeypatch
) -> None:
    seen: list[str | None] = []

    def fake_open_mirror(runs_root, run_dir, enabled, *, tracking_uri=None, **kwargs):
        seen.append(tracking_uri)
        return NullMirror()

    monkeypatch.setattr("rlvr.cli.open_mirror", fake_open_mirror)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://172.17.0.1:8081")
    config_path.write_text(
        CONFIG + '\n[mlflow]\ntracking_uri = "http://127.0.0.1:9999"\n',
        encoding="utf-8",
    )

    assert run_command(run_args(tmp_path, config_path)) == 0

    assert seen == ["http://127.0.0.1:9999"]


def test_run_starts_the_uploader_when_a_bucket_is_set(tmp_path, monkeypatch) -> None:
    StubUploader.calls = []
    monkeypatch.setattr("rlvr.cli.RunUploader", StubUploader)
    config_path = write_bucket_config(tmp_path)
    runs_root = tmp_path / "runs"

    assert main(
        ["run", "--arm", "code-score", "--fake", "--config", str(config_path), "--runs-root", str(runs_root)]
    ) == 0

    kinds = [call[0] for call in StubUploader.calls]
    assert kinds == ["init", "start", "close"]
    init = StubUploader.calls[0]
    assert init[2] == "code-score"
    assert init[4] == "s3://bucket/runs/v1"
    assert init[5]["concurrency"] == 2


def test_a_run_tags_the_checkpoints_uri_before_training_ends(
    tmp_path, monkeypatch
) -> None:
    """The tag must exist while the run is live, not after it closes."""
    TagReadingExperiment.seen = []
    StubUploader.calls = []
    monkeypatch.setattr("rlvr.cli.RunUploader", StubUploader)
    monkeypatch.setattr("rlvr.cli.Experiment", TagReadingExperiment)
    config_path = write_bucket_config(tmp_path)
    args = run_args(tmp_path, config_path)

    assert run_command(args) == 0

    started = next((args.runs_root / "code-score").iterdir()).name
    assert TagReadingExperiment.seen == [
        f"s3://bucket/runs/v1/code-score/{started}"
    ]


def test_a_broken_uploader_does_not_stop_the_run(tmp_path, monkeypatch) -> None:
    class BrokenUploader(StubUploader):
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("no boto3 here")

    monkeypatch.setattr("rlvr.cli.RunUploader", BrokenUploader)
    config_path = write_bucket_config(tmp_path)
    runs_root = tmp_path / "runs"

    args = Namespace(
        arm="code-score",
        config=config_path,
        fake=True,
        runs_root=runs_root,
        resume=False,
        control=None,
    )
    assert run_command(args) == 0
    assert (runs_root / "code-score").is_dir()
    # The prefix comes from the config, so it is tagged even with no uploader.
    parents, _ = parents_and_children()
    assert parents[0].data.tags["rlvr/checkpoints_uri"].startswith(
        "s3://bucket/runs/v1/code-score/"
    )


def test_a_close_that_raises_does_not_stop_the_run(tmp_path, monkeypatch) -> None:
    StubUploader.calls = []
    monkeypatch.setattr("rlvr.cli.RunUploader", CloseFailsUploader)
    monkeypatch.setattr("rlvr.cli.Experiment", QuietExperiment)
    args = run_args(tmp_path, write_bucket_config(tmp_path))

    assert run_command(args) == 0
    assert [call[0] for call in StubUploader.calls] == ["init", "start", "close"]


def test_the_uploader_closes_when_the_run_raises(tmp_path, monkeypatch) -> None:
    StubUploader.calls = []
    monkeypatch.setattr("rlvr.cli.RunUploader", StubUploader)
    monkeypatch.setattr("rlvr.cli.Experiment", FailingExperiment)
    args = run_args(tmp_path, write_bucket_config(tmp_path))

    with pytest.raises(RuntimeError, match="training died"):
        run_command(args)

    assert [call[0] for call in StubUploader.calls] == ["init", "start", "close"]
