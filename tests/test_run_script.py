"""Tests for run.sh argument handling. The launcher execs `rlvr run`, so the
arm validation paths are checked directly and the fake pipeline is run once
end to end to prove extra arguments pass through."""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "run.sh", *args],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )


def test_run_sh_rejects_a_missing_arm() -> None:
    result = _run()
    assert result.returncode == 2
    assert "usage" in result.stderr


def test_run_sh_rejects_an_unknown_arm() -> None:
    result = _run("bogus")
    assert result.returncode == 2
    assert "unknown arm" in result.stderr


def test_run_sh_runs_the_fake_pipeline_with_extra_args(tmp_path) -> None:
    runs_root = tmp_path / "runs"
    result = _run("code-score", "--fake", "--runs-root", str(runs_root))
    assert result.returncode == 0, result.stderr
    assert (runs_root / "code-score").is_dir()
    run_dir = next((runs_root / "code-score").iterdir())
    assert (run_dir / "rollouts.jsonl").is_file()


def test_run_sh_accepts_all_four_arms() -> None:
    # Only the arm name is validated before the run, so a missing key for a
    # judge arm must fail after the launcher passes the arm through. The
    # code-score arm needs no key, so it is the one that reaches the run.
    for arm in ("code-score", "hybrid", "gated-judge", "gated-process"):
        assert "unknown arm" not in _run(arm).stderr


def test_run_sh_honors_a_caller_supplied_config(tmp_path) -> None:
    import json

    config = tmp_path / "custom.toml"
    config.write_text(
        '[model]\nname = "Qwen/Qwen3-0.6B"\nrevision = "r"\n'
        '[judge]\nmodel = "fake-judge-model"\n'
        "[data]\nsplit_seed = 7\n"
        "[fake]\nsteps = 2\nprompts_per_step = 1\n",
        encoding="utf-8",
    )
    runs_root = tmp_path / "runs"
    result = _run(
        "code-score", "--config", str(config), "--fake", "--runs-root", str(runs_root)
    )
    assert result.returncode == 0, result.stderr
    run_dir = next((runs_root / "code-score").iterdir())
    metric_steps = [
        json.loads(line)["step"]
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    # Two steps proves the caller config (steps = 2) was used, not the default
    # real.toml (no [fake] table, so one step).
    assert metric_steps == [1, 2]
