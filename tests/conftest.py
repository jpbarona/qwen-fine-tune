"""Shared fixtures for the rlvr test suite."""

from __future__ import annotations

from pathlib import Path

import pytest

from rlvr.config import load_config
from rlvr.experiment import Experiment
from rlvr.fake import FakeGenerator, FakeJudge, FakeTestRunner, fake_prompts
from rlvr.runs import RunStore

REPO_ROOT = Path(__file__).resolve().parents[1]

MINIMAL_CONFIG = """\
[model]
name = "Qwen/Qwen3-0.6B"
revision = "fake-revision"

[judge]
model = "fake-judge-model"

[data]
split_seed = 1234

[fake]
steps = 3
prompts_per_step = 2
"""

FIXED_TIMESTAMP = "20261112T000000Z"


@pytest.fixture
def write_config(tmp_path: Path):
    """Write a TOML config file in tmp_path and return its path."""

    def _write(text: str = MINIMAL_CONFIG, name: str = "config.toml") -> Path:
        path = tmp_path / name
        path.write_text(text, encoding="utf-8")
        return path

    return _write


@pytest.fixture
def run_fake(tmp_path: Path):
    """Run the fake pipeline and return (run_dir, RunSummary)."""

    def _run(
        arm: str = "code-score",
        *,
        config_text: str = MINIMAL_CONFIG,
        timestamp: str = FIXED_TIMESTAMP,
        runs_root: str | Path = "runs",
    ) -> tuple[Path, object]:
        root = Path(runs_root)
        if not root.is_absolute():
            root = tmp_path / root
        config = load_config(write_config_one(tmp_path, config_text))
        store = RunStore(root, arm, timestamp=timestamp)
        experiment = Experiment(
            arm=arm,
            config=config,
            generator=FakeGenerator(),
            judge=FakeJudge(),
            test_runner=FakeTestRunner(),
            store=store,
            prompts=fake_prompts(config.prompts_per_step),
        )
        return store.run_dir, experiment.run()

    return _run


def write_config_one(tmp_path: Path, text: str, name: str = "config.toml") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def single_run_dir(runs_root: Path, arm: str) -> Path:
    """Return the only run folder for an arm, failing if there is not exactly one."""
    candidates = sorted(p for p in (runs_root / arm).iterdir() if p.is_dir())
    assert len(candidates) == 1, f"expected one run dir for {arm}, got {candidates}"
    return candidates[0]
