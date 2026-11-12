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
        self.meta_path.write_text(_dumps(dict(meta)) + "\n", encoding="utf-8")

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
        """Write eval.json."""
        self.eval_path.write_text(_dumps(dict(payload)) + "\n", encoding="utf-8")

    @staticmethod
    def _append_line(path: Path, line: str) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
