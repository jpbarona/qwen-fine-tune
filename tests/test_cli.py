"""Tests for the command line interface."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from rlvr.cli import main
from rlvr.reward import ARMS

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
prompts_per_step = 1
"""


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(CONFIG, encoding="utf-8")
    return path


@pytest.mark.parametrize("arm", ARMS)
def test_run_fake_exits_zero_for_every_arm(arm, config_path, tmp_path, capsys) -> None:
    runs_root = tmp_path / "runs"
    assert main(["run", "--arm", arm, "--fake", "--config", str(config_path), "--runs-root", str(runs_root)]) == 0
    assert (runs_root / arm).is_dir()
    assert f"runs/{arm}" in capsys.readouterr().out


def test_run_writes_the_four_files(config_path, tmp_path) -> None:
    runs_root = tmp_path / "runs"
    main(["run", "--arm", "hybrid", "--fake", "--config", str(config_path), "--runs-root", str(runs_root)])
    run_dir = next((runs_root / "hybrid").iterdir())
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
        ["run", "--arm", "hybrid", "--fake", "--config", str(tmp_path / "absent.toml")]
    )
    captured = capsys.readouterr()
    assert code != 0
    assert "absent.toml" in captured.err
    assert "config" in captured.err


def test_invalid_config_is_reported_on_stderr(tmp_path, capsys) -> None:
    bad = tmp_path / "bad.toml"
    bad.write_text("[judge]\nmodel = 5\n", encoding="utf-8")
    code = main(["run", "--arm", "hybrid", "--fake", "--config", str(bad)])
    captured = capsys.readouterr()
    assert code != 0
    assert "config" in captured.err


def test_unparseable_config_is_reported_on_stderr(tmp_path, capsys) -> None:
    bad = tmp_path / "bad.toml"
    bad.write_text("this is not toml at all ===\n", encoding="utf-8")
    code = main(["run", "--arm", "hybrid", "--fake", "--config", str(bad)])
    captured = capsys.readouterr()
    assert code != 0
    assert "config" in captured.err


def test_without_fake_the_command_explains_that_real_backends_are_missing(config_path, tmp_path, capsys) -> None:
    code = main(["run", "--arm", "hybrid", "--config", str(config_path)])
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
        main(["train", "--arm", "hybrid", "--fake"])


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
            "gated-judge",
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
    assert (tmp_path / "runs" / "gated-judge").is_dir()


def test_console_script_reports_unknown_arm_without_a_traceback(tmp_path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "rlvr", "run", "--arm", "bogus", "--fake"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "bogus" in result.stderr
    assert "Traceback" not in result.stderr
