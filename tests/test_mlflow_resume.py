"""Tests for rebuilding a resumed run and closing dead runs."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Silence a one time hint MLflow prints on import, so the test output stays
# readable.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from mlflow.tracking import MlflowClient  # noqa: E402
import pytest  # noqa: E402

from rlvr.mlflow_log import MlflowMirror  # noqa: E402

# MLflow's SQLAlchemy store configures its mappers with a loader option that
# SQLAlchemy deprecates. The warning comes from the library, not this code.
pytestmark = pytest.mark.filterwarnings(
    "ignore:The ``noload`` loader strategy is deprecated:sqlalchemy.exc.SADeprecationWarning"
)

RUN_NAME = "20260101T000000Z"


def make_run(tmp_path: Path) -> Path:
    """A run folder with the metadata the rebuild reads."""
    run_dir = tmp_path / "runs" / "code-score" / RUN_NAME
    run_dir.mkdir(parents=True)
    meta = {"arm": "code-score", "timestamp": RUN_NAME}
    (run_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return run_dir


def append_metric(run_dir: Path, row: dict) -> None:
    """Append one row to the run's metrics file."""
    with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def write_metrics(run_dir: Path, rows: list[dict]) -> None:
    """Replace the metrics file with the rows the store kept."""
    text = "".join(json.dumps(row) + "\n" for row in rows)
    (run_dir / "metrics.jsonl").write_text(text, encoding="utf-8")


def old_stamp() -> str:
    """A last write time two hours in the past, in the tag's format."""
    return (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()


def test_rebuild_replays_the_local_files_without_duplicate_steps(
    tmp_path: Path,
) -> None:
    run_dir = make_run(tmp_path)
    mirror = MlflowMirror(tmp_path / "runs", run_dir)
    mirror.start({"arm": "code-score", "timestamp": RUN_NAME})
    old_run_id = mirror.run_id
    old_gpu_run_id = mirror.gpu_run_id

    append_metric(run_dir, {"step": 1, "pass_rate": 0.25})
    mirror.step(1)
    append_metric(run_dir, {"step": 2, "pass_rate": 0.5})
    mirror.step(2)
    append_metric(run_dir, {"step": 3, "pass_rate": 0.75})
    mirror.step(3)

    # The store truncated the local files to the last checkpoint. The stale
    # step 3 must not survive the rebuild.
    write_metrics(
        run_dir,
        [
            {"step": 1, "pass_rate": 0.25},
            {"step": 2, "pass_rate": 0.5},
        ],
    )
    with (run_dir / "gpu.jsonl").open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"t": 1.0, "utilization": 50.0}) + "\n")

    mirror.rebuild()

    client = MlflowClient()
    assert mirror.run_id != old_run_id
    assert mirror.gpu_run_id != old_gpu_run_id
    history = client.get_metric_history(mirror.run_id, "pass_rate")
    assert [(point.step, point.value) for point in history] == [(1, 0.25), (2, 0.5)]
    gpu = client.get_metric_history(mirror.gpu_run_id, "utilization")
    assert [point.value for point in gpu] == [50.0]
    mirror.close()


def test_close_dead_marks_a_stale_run_failed(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path)
    mirror = MlflowMirror(tmp_path / "runs", run_dir)
    client = MlflowClient()

    stale = client.create_run(
        mirror.experiment_id,
        run_name="code-score-stale",
        tags={"rlvr/arm": "code-score"},
    )
    client.set_tag(stale.info.run_id, "rlvr/last_write_utc", old_stamp())

    # A run with no last write tag is left alone.
    unknown = client.create_run(
        mirror.experiment_id,
        run_name="code-score-unknown",
        tags={"rlvr/arm": "code-score"},
    )

    mirror.close_dead("code-score")

    assert client.get_run(stale.info.run_id).info.status == "FAILED"
    assert client.get_run(unknown.info.run_id).info.status == "RUNNING"


def test_start_closes_a_stale_run_for_the_same_arm(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path)
    mirror = MlflowMirror(tmp_path / "runs", run_dir)
    client = MlflowClient()

    stale = client.create_run(
        mirror.experiment_id,
        run_name="code-score-stale",
        tags={"rlvr/arm": "code-score"},
    )
    client.set_tag(stale.info.run_id, "rlvr/last_write_utc", old_stamp())

    mirror.start({"arm": "code-score", "timestamp": RUN_NAME})

    assert client.get_run(stale.info.run_id).info.status == "FAILED"
    mirror.close()
