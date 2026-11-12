"""The run folder writer.

Layout, from SPEC.md:

    runs/<arm>/<timestamp>/
        meta.json
        metrics.jsonl
        rollouts.jsonl
        eval.json
        checkpoints/

Everything is append-as-you-go. A rollout is on disk before the next one is
generated, so a crash costs at most one record. Each line is flushed on write.
Later tasks that need speed can hold the handle open, as long as records stay
visible.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .schema import RolloutRecord

__all__ = ["RunStore", "utc_timestamp"]

META_NAME = "meta.json"
METRICS_NAME = "metrics.jsonl"
ROLLOUTS_NAME = "rollouts.jsonl"
EVAL_NAME = "eval.json"
CHECKPOINTS_NAME = "checkpoints"


def utc_timestamp() -> str:
    """A UTC timestamp that is safe as one path segment."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _dumps(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)


class RunStore:
    """Writes one run folder. The concrete implementation of the Store seam."""

    def __init__(self, runs_root: str | Path, arm: str, timestamp: str | None = None) -> None:
        self.arm = arm
        self.timestamp = utc_timestamp() if timestamp is None else timestamp
        if "/" in self.timestamp or self.timestamp in ("", ".", ".."):
            raise ValueError(f"timestamp must be one path segment, got {self.timestamp!r}")
        self.runs_root = Path(runs_root)
        self.run_dir = self.runs_root / arm / self.timestamp
        if self.run_dir.exists():
            raise FileExistsError(f"run folder already exists: {self.run_dir}")
        self.run_dir.mkdir(parents=True)
        (self.run_dir / CHECKPOINTS_NAME).mkdir(parents=True, exist_ok=True)

    @classmethod
    def resume(cls, runs_root: str | Path, arm: str) -> "RunStore":
        """Open the latest existing run folder for `arm` without creating one.

        The folder is chosen by the largest timestamp name, which is the UTC
        `YYYYMMDDTHHMMSSZ` sort order. Raises FileNotFoundError when the arm has
        no run folder.
        """
        root = Path(runs_root)
        arm_dir = root / arm
        if not arm_dir.is_dir():
            raise FileNotFoundError(f"no run folder for arm {arm!r} under {root}")
        candidates = [
            child
            for child in arm_dir.iterdir()
            if child.is_dir() and (child / META_NAME).is_file()
        ]
        if not candidates:
            raise FileNotFoundError(f"no run folder for arm {arm!r} under {root}")
        run_dir = max(candidates, key=lambda child: child.name)
        store = cls.__new__(cls)
        store.arm = arm
        store.timestamp = run_dir.name
        store.runs_root = root
        store.run_dir = run_dir
        return store

    def truncate_to_step(self, step: int) -> None:
        """Drop every record and evaluation past `step`, for a clean resume.

        A crash can leave records for steps beyond the last checkpoint. Those
        steps are regenerated after resume, so their records, metrics, and
        evaluations are removed here to avoid duplicates.
        """
        self._truncate_rollouts(step)
        self._truncate_metrics(step)
        self._truncate_eval(step)

    def read_evaluations(self) -> list[dict[str, Any]]:
        """The evaluations list from eval.json, or an empty list."""
        if not self.eval_path.exists():
            return []
        try:
            payload = json.loads(self.eval_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        evaluations = payload.get("evaluations")
        if not isinstance(evaluations, list):
            return []
        return [dict(point) for point in evaluations if isinstance(point, dict)]

    def _truncate_rollouts(self, step: int) -> None:
        if not self.rollouts_path.exists():
            return
        kept: list[str] = []
        for line in self.rollouts_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record_step = json.loads(line).get("step")
            except json.JSONDecodeError:
                continue
            if isinstance(record_step, int) and not isinstance(record_step, bool) and record_step <= step:
                kept.append(line)
        self._write_atomic(
            self.rollouts_path, "\n".join(kept) + ("\n" if kept else "")
        )

    def _truncate_metrics(self, step: int) -> None:
        if not self.metrics_path.exists():
            return
        kept: list[str] = []
        for line in self.metrics_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row_step = json.loads(line).get("step")
            except json.JSONDecodeError:
                continue
            if isinstance(row_step, int) and not isinstance(row_step, bool) and row_step <= step:
                kept.append(line)
        self._write_atomic(
            self.metrics_path, "\n".join(kept) + ("\n" if kept else "")
        )

    def _truncate_eval(self, step: int) -> None:
        if not self.eval_path.exists():
            return
        try:
            payload = json.loads(self.eval_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(payload, dict):
            return
        evaluations = payload.get("evaluations")
        if not isinstance(evaluations, list):
            evaluations = []
        kept = [
            dict(point)
            for point in evaluations
            if isinstance(point, dict)
            and isinstance(point.get("step"), int)
            and not isinstance(point.get("step"), bool)
            and point["step"] <= step
        ]
        next_payload = dict(payload)
        next_payload["evaluations"] = kept
        if kept:
            last = kept[-1]
            next_payload["step"] = last.get("step")
            next_payload["pass_at_1"] = last.get("pass_at_1")
            next_payload["count"] = last.get("count")
        else:
            next_payload.pop("step", None)
            next_payload.pop("pass_at_1", None)
            next_payload.pop("count", None)
        self.write_eval(next_payload)

    @property
    def meta_path(self) -> Path:
        return self.run_dir / META_NAME

    @property
    def metrics_path(self) -> Path:
        return self.run_dir / METRICS_NAME

    @property
    def rollouts_path(self) -> Path:
        return self.run_dir / ROLLOUTS_NAME

    @property
    def eval_path(self) -> Path:
        return self.run_dir / EVAL_NAME

    @property
    def checkpoints_dir(self) -> Path:
        return self.run_dir / CHECKPOINTS_NAME

    def write_meta(self, meta: Mapping[str, Any]) -> None:
        """Write meta.json. Called once, before the first rollout."""
        self._write_atomic(self.meta_path, _dumps(dict(meta)) + "\n")

    def append_rollout(self, record: RolloutRecord) -> None:
        """Append one rollout record."""
        self._append_line(self.rollouts_path, record.to_json())

    def append_metric(self, row: Mapping[str, Any]) -> None:
        """Append one metrics row."""
        self._append_line(
            self.metrics_path,
            json.dumps(dict(row), ensure_ascii=False, allow_nan=False),
        )

    def write_eval(self, payload: Mapping[str, Any]) -> None:
        """Replace eval.json atomically.

        The file is rewritten after every evaluation, checkpoint and final. A
        reader sees the previous complete file or the new complete file, never
        a half-written one.
        """
        self._write_atomic(self.eval_path, _dumps(dict(payload)) + "\n")

    @staticmethod
    def _write_atomic(path: Path, text: str) -> None:
        """Write `text` to `path` through a temporary file in the same folder."""
        descriptor, temporary = tempfile.mkstemp(
            dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        except BaseException:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise

    @staticmethod
    def _append_line(path: Path, line: str) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
