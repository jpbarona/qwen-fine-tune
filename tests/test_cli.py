"""Tests for the command line interface."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from rlvr.cli import build_judge, main
from rlvr.config import load_config
from rlvr.fake import FakeJudge

CONFIG = """\
[model]
name = "Qwen/Qwen3-0.6B"
revision = "r"

[judge]
model = "fake-judge-model"

[data]
split_seed = 7

[fake]
steps = 2
prompts_per_step = 2
"""


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(CONFIG, encoding="utf-8")
    return path


def test_run_fake_exits_zero_for_the_code_score_arm(config_path, tmp_path, capsys) -> None:
    runs_root = tmp_path / "runs"
    assert main(["run", "--arm", "code-score", "--fake", "--config", str(config_path), "--runs-root", str(runs_root)]) == 0
    assert (runs_root / "code-score").is_dir()
    assert "runs/code-score" in capsys.readouterr().out


@pytest.mark.parametrize("arm", ["hybrid", "gated-judge"])
def test_judge_arms_run_fake_and_populate_judge_scores(arm, config_path, tmp_path) -> None:
    runs_root = tmp_path / "runs"
    assert main(
        ["run", "--arm", arm, "--fake", "--config", str(config_path), "--runs-root", str(runs_root)]
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


def test_pending_arm_exits_with_a_clear_message(config_path, tmp_path, capsys) -> None:
    runs_root = tmp_path / "runs"
    code = main(
        [
            "run",
            "--arm",
            "gated-process",
            "--fake",
            "--config",
            str(config_path),
            "--runs-root",
            str(runs_root),
        ]
    )
    captured = capsys.readouterr()
    assert code != 0
    assert captured.out == ""
    assert "gated-process" in captured.err
    assert "not implemented" in captured.err
    assert not (runs_root / "gated-process").exists()


@pytest.mark.parametrize("arm", ["hybrid", "gated-judge"])
def test_a_judge_arm_fails_before_the_run_folder_when_the_key_is_missing(
    arm, config_path, tmp_path, monkeypatch, capsys
) -> None:
    monkeypatch.delenv("OPENCODE_GO_API_KEY", raising=False)
    runs_root = tmp_path / "runs"
    code = main(
        ["run", "--arm", arm, "--config", str(config_path), "--runs-root", str(runs_root)]
    )
    captured = capsys.readouterr()
    assert code != 0
    assert "OPENCODE_GO_API_KEY" in captured.err
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
    assert built == [config, config]

    build_judge("hybrid", config, fake=True)
    assert built == [config, config]


def test_run_writes_the_four_files(config_path, tmp_path) -> None:
    runs_root = tmp_path / "runs"
    main(["run", "--arm", "code-score", "--fake", "--config", str(config_path), "--runs-root", str(runs_root)])
    run_dir = next((runs_root / "code-score").iterdir())
    assert sorted(p.name for p in run_dir.iterdir()) == [
        "checkpoints",
        "eval.json",
        "meta.json",
        "metrics.jsonl",
        "rollouts.jsonl",
    ]
    assert (run_dir / "meta.json").read_text(encoding="utf-8").startswith("{")


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


def test_without_fake_the_command_explains_that_real_backends_are_missing(config_path, tmp_path, capsys) -> None:
    code = main(["run", "--arm", "code-score", "--config", str(config_path)])
    captured = capsys.readouterr()
    assert code != 0
    assert "--fake" in captured.err


def test_arm_is_required(tmp_path, capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["run", "--fake", "--runs-root", str(tmp_path / "runs")])
    assert excinfo.value.code != 0
    assert "--arm" in capsys.readouterr().err


def test_unknown_subcommand_fails(tmp_path) -> None:
    with pytest.raises(SystemExit):
        main(["train", "--arm", "code-score", "--fake"])


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
