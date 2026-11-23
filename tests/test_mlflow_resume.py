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

import rlvr.mlflow_log as mlflow_log  # noqa: E402
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
    mirror.step({"step": 1, "pass_rate": 0.25})
    append_metric(run_dir, {"step": 2, "pass_rate": 0.5})
    mirror.step({"step": 2, "pass_rate": 0.5})
    append_metric(run_dir, {"step": 3, "pass_rate": 0.75})
    mirror.step({"step": 3, "pass_rate": 0.75})

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


def test_step_writes_the_heartbeat_tags_on_both_runs(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path)
    mirror = MlflowMirror(tmp_path / "runs", run_dir)
    mirror.start({"arm": "code-score", "timestamp": RUN_NAME})

    append_metric(run_dir, {"step": 1, "pass_rate": 0.25})
    mirror.step({"step": 1, "pass_rate": 0.25})

    client = MlflowClient()
    parent_tags = client.get_run(mirror.run_id).data.tags
    child_tags = client.get_run(mirror.gpu_run_id).data.tags
    for tags in (parent_tags, child_tags):
        stamp = tags.get("rlvr/last_write_utc")
        assert stamp is not None
        written = datetime.fromisoformat(stamp)
        assert written.tzinfo is not None
        heartbeat = tags.get("rlvr/heartbeat_seconds")
        assert heartbeat is not None
        assert float(heartbeat) >= 0.0

    # The spec maps the heartbeat to a metric and the last write to a tag.
    heartbeat = parent_tags["rlvr/heartbeat_seconds"]
    metric = client.get_metric_history(mirror.run_id, "rlvr/heartbeat_seconds")
    assert len(metric) == 1
    assert metric[0].step == 1
    assert metric[0].value == pytest.approx(float(heartbeat), abs=0.001)
    mirror.close()


def test_close_dead_sweeps_a_stale_child_run(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path)
    mirror = MlflowMirror(tmp_path / "runs", run_dir)
    client = MlflowClient()

    parent = client.create_run(
        mirror.experiment_id,
        run_name="code-score-parent",
        tags={"rlvr/arm": "code-score"},
    )
    child = client.create_run(
        mirror.experiment_id,
        run_name="code-score-parent-gpu",
        tags={
            "rlvr/arm": "code-score",
            "rlvr/role": "gpu",
            "mlflow.parentRunId": parent.info.run_id,
        },
    )
    client.set_tag(child.info.run_id, "rlvr/last_write_utc", old_stamp())

    mirror.close_dead("code-score")

    assert client.get_run(child.info.run_id).info.status == "FAILED"
    # The parent has no last write tag, so it is left alone.
    assert client.get_run(parent.info.run_id).info.status == "RUNNING"


def test_start_writes_the_heartbeat_tags_on_both_runs(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path)
    mirror = MlflowMirror(tmp_path / "runs", run_dir)
    mirror.start({"arm": "code-score", "timestamp": RUN_NAME})

    client = MlflowClient()
    for run_id in (mirror.run_id, mirror.gpu_run_id):
        tags = client.get_run(run_id).data.tags
        assert tags.get("rlvr/last_write_utc")
        assert tags.get("rlvr/heartbeat_seconds")
    mirror.close()


def test_rebuild_deletes_a_run_with_no_ids_file(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path)
    mirror = MlflowMirror(tmp_path / "runs", run_dir)
    mirror.start({"arm": "code-score", "timestamp": RUN_NAME})
    old_run_id = mirror.run_id
    old_gpu_run_id = mirror.gpu_run_id

    append_metric(run_dir, {"step": 1, "pass_rate": 0.25})
    mirror.step({"step": 1, "pass_rate": 0.25})
    (run_dir / "mlflow.json").unlink()

    mirror.rebuild()

    client = MlflowClient()
    ids = [
        run.info.run_id
        for run in client.search_runs(experiment_ids=[mirror.experiment_id])
    ]
    assert old_run_id not in ids
    assert old_gpu_run_id not in ids
    assert mirror.run_id in ids
    mirror.close()


def test_rebuild_deletes_the_old_runs_and_replays_the_tables(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path)
    payload = {
        "arm": "code-score",
        "timestamp": RUN_NAME,
        "step": 1,
        "stop_reason": "max_steps",
        "evaluations": [
            {"step": 1, "kind": "final", "pass_at_1": 1.0, "count": 1}
        ],
    }
    (run_dir / "eval.json").write_text(json.dumps(payload), encoding="utf-8")
    (run_dir / "labels.jsonl").write_text(
        json.dumps({"step": 1, "prompt_id": "p1", "label": "train"}) + "\n",
        encoding="utf-8",
    )

    mirror = MlflowMirror(tmp_path / "runs", run_dir)
    mirror.start({"arm": "code-score", "timestamp": RUN_NAME})
    old_run_id = mirror.run_id
    old_gpu_run_id = mirror.gpu_run_id

    mirror.rebuild()

    client = MlflowClient()
    ids = [
        run.info.run_id
        for run in client.search_runs(experiment_ids=[mirror.experiment_id])
    ]
    assert old_run_id not in ids
    assert old_gpu_run_id not in ids
    assert mirror.run_id in ids
    names = {item.path for item in client.list_artifacts(mirror.run_id)}
    assert "labels.json" in names
    mirror.close()


def test_rebuild_logs_one_band_point_for_two_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = make_run(tmp_path)
    write_metrics(
        run_dir,
        [
            {"step": 1, "pass_rate": 0.25},
            {"step": 2, "pass_rate": 0.5},
        ],
    )
    dashboard = {
        "runs": [
            {
                "series": {"steps": [1, 2], "pass_rate": [0.25, 0.5]},
                "bands": [{"band": "easy", "pass_rate": 0.75}],
                "stage_progress": {
                    "points": [
                        {"step": 1, "band": "easy", "solved": 1, "total": 4},
                        {"step": 2, "band": "easy", "solved": 2, "total": 4},
                    ]
                },
                "questions": {"rows": []},
                "distributions": {},
                "dropped": {},
            }
        ]
    }
    monkeypatch.setattr(mlflow_log, "collect_dashboard", lambda paths: dashboard)

    mirror = MlflowMirror(tmp_path / "runs", run_dir)
    mirror.start({"arm": "code-score", "timestamp": RUN_NAME})
    mirror.rebuild()

    client = MlflowClient()
    band = client.get_metric_history(mirror.run_id, "band/easy/pass_rate")
    assert [(point.step, point.value) for point in band] == [(2, 0.75)]
    solved = client.get_metric_history(mirror.run_id, "stage/easy/solved")
    assert [(point.step, point.value) for point in solved] == [(1, 1.0), (2, 2.0)]
    mirror.close()
