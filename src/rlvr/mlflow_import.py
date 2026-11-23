"""Copy old run folders into the local MLflow store.

The run folder stays the record. This module reads the files a finished run
wrote and replays them through the same mirror the live trainer uses. The
mirror writes `mlflow.json` in the run folder and appends to
`mlflow-errors.log` when the store fails.

A folder with no `gpu.jsonl` imports without error. A folder with no `eval.json`
imports without pushing tables. The walker only takes folders that hold a
`meta.json`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from .dashboard_io import _read_json
from .mlflow_log import MlflowMirror

__all__ = ["import_run", "run_dirs"]

META_NAME = "meta.json"


def run_dirs(runs_root: Path, arm: str | None = None) -> Iterator[Path]:
    """Yield every run folder under the runs root, in arm and timestamp order.

    The layout is `runs/<arm>/<timestamp>/meta.json`. A folder without a
    `meta.json` is not a run and is skipped. With `arm` set, only that arm is
    walked.
    """
    root = Path(runs_root)
    if arm is not None:
        arms = [root / arm]
    else:
        arms = sorted(child for child in root.iterdir() if child.is_dir())
    for arm_dir in arms:
        if not arm_dir.is_dir():
            continue
        for run_dir in sorted(child for child in arm_dir.iterdir() if child.is_dir()):
            if (run_dir / META_NAME).is_file():
                yield run_dir


def import_run(runs_root: Path, run_dir: Path) -> str:
    """Replay one old run folder into the store and return its MLflow run id.

    The mirror opens the parent run and the GPU child run. The replay copies
    every `metrics.jsonl` row at its step, pushes the tables once and closes the
    run. The replay helper is the same one a resumed run uses, so an old run and
    a rebuilt run follow one code path.
    """
    root = Path(runs_root)
    folder = Path(run_dir)
    meta = _read_json(folder / META_NAME)
    mirror = MlflowMirror(root, folder)
    mirror.start(meta)
    if not mirror.run_id:
        raise RuntimeError(
            f"cannot open an MLflow run for {folder}. "
            f"See {folder / 'mlflow-errors.log'}."
        )
    try:
        mirror._replay()
    except Exception:
        mirror.close(status="FAILED")
        raise
    mirror.close()
    return mirror.run_id
