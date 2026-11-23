"""Tests for copying old run folders into the local MLflow store."""

from __future__ import annotations

import json
import os
from pathlib import Path

# Silence a one time hint MLflow prints on import, so the test output stays
# readable.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from mlflow.tracking import MlflowClient  # noqa: E402
import pytest  # noqa: E402

from rlvr.cli import main  # noqa: E402
from rlvr.mlflow_import import import_run  # noqa: E402

# MLflow's SQLAlchemy store configures its mappers with a loader that SQLAlchemy
# deprecates. The warning is not this code's problem.
pytestmark = pytest.mark.filterwarnings(
    "ignore:The ``noload`` loader strategy is deprecated:sqlalchemy.exc.SADeprecationWarning"
)

TIMESTAMP = "20260101T000000Z"


def make_run(runs_root: Path, arm: str, timestamp: str = TIMESTAMP) -> Path:
    """A run folder with meta.json and two metrics rows, and no gpu.jsonl."""
    run_dir = runs_root / arm / timestamp
    run_dir.mkdir(parents=True)
    meta = {"arm": arm, "timestamp": timestamp, "mode": "fake"}
    (run_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    rows = [
        {"step": 1, "pass_rate": 0.25, "mean_reward": 0.1},
        {"step": 2, "pass_rate": 0.5, "mean_reward": 0.2},
    ]
    lines = [json.dumps(row) for row in rows]
    (run_dir / "metrics.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return run_dir


def test_import_run_copies_an_old_run(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_dir = make_run(runs_root, "code-score")

    run_id = import_run(runs_root, run_dir)

    client = MlflowClient()
    run = client.get_run(run_id)
    assert run.info.run_id == run_id
    assert run.data.tags["rlvr/arm"] == "code-score"
    history = client.get_metric_history(run_id, "pass_rate")
    assert [point.step for point in history] == [1, 2]
    assert [point.value for point in history] == [0.25, 0.5]
    # A folder with no gpu.jsonl imports without error.
    assert not (run_dir / "mlflow-errors.log").exists()
    payload = json.loads((run_dir / "mlflow.json").read_text(encoding="utf-8"))
    assert payload["run_id"] == run_id


def test_import_mlflow_walks_every_arm(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    make_run(runs_root, "code-score")
    make_run(runs_root, "hybrid", "20260102T000000Z")

    assert main(["import-mlflow", "--runs-root", str(runs_root)]) == 0

    client = MlflowClient()
    experiment = client.get_experiment_by_name("rlvr")
    assert experiment is not None
    runs = client.search_runs([experiment.experiment_id])
    assert {run.data.tags["rlvr/arm"] for run in runs} == {"code-score", "hybrid"}


def test_importing_the_same_folder_twice_skips_it(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_dir = make_run(runs_root, "code-score")

    first = import_run(runs_root, run_dir)
    second = import_run(runs_root, run_dir)

    assert first is not None
    assert second is None
    client = MlflowClient()
    experiment = client.get_experiment_by_name("rlvr")
    assert experiment is not None
    runs = client.search_runs([experiment.experiment_id])
    parents = [run for run in runs if "rlvr/run_dir" in run.data.tags]
    assert len(parents) == 1


def test_import_mlflow_prints_a_skip_line(tmp_path: Path, capsys) -> None:
    runs_root = tmp_path / "runs"
    make_run(runs_root, "code-score")

    assert main(["import-mlflow", "--runs-root", str(runs_root)]) == 0
    capsys.readouterr()
    assert main(["import-mlflow", "--runs-root", str(runs_root)]) == 0

    out = capsys.readouterr().out
    assert "skipped" in out


def test_import_mlflow_honors_the_arm_filter(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    make_run(runs_root, "code-score")
    make_run(runs_root, "hybrid", "20260102T000000Z")

    assert main(
        ["import-mlflow", "--runs-root", str(runs_root), "--arm", "hybrid"]
    ) == 0

    client = MlflowClient()
    experiment = client.get_experiment_by_name("rlvr")
    assert experiment is not None
    runs = client.search_runs([experiment.experiment_id])
    assert {run.data.tags["rlvr/arm"] for run in runs} == {"hybrid"}


def test_import_run_rebuild_replaces_the_run(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_dir = make_run(runs_root, "code-score")

    first = import_run(runs_root, run_dir)
    again = import_run(runs_root, run_dir, rebuild=True)

    assert first is not None
    assert again is not None
    assert again != first
    client = MlflowClient()
    experiment = client.get_experiment_by_name("rlvr")
    assert experiment is not None
    runs = client.search_runs([experiment.experiment_id])
    parents = [run for run in runs if "rlvr/run_dir" in run.data.tags]
    assert [run.info.run_id for run in parents] == [again]
    history = client.get_metric_history(again, "pass_rate")
    assert [point.step for point in history] == [1, 2]
