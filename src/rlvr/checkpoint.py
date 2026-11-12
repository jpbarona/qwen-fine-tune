"""Checkpoint directory and manifest handling.

A checkpoint is one directory under `run_dir/checkpoints/` named `step-XXXXXXXX`
(an 8-digit, zero-padded step number). The trainer writes its LoRA adapter and
optimizer state into that directory. A small `checkpoint.json` manifest records
the step, arm, and timestamp so a resume can find the latest checkpoint without
importing torch.

This module is pure Python. The real state dicts live in `qwen.py` and are the
only part that needs torch.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

__all__ = [
    "MANIFEST_NAME",
    "Checkpoint",
    "CheckpointError",
    "checkpoint_dir_name",
    "find_latest_checkpoint",
    "parse_checkpoint_step",
    "read_checkpoint_manifest",
    "write_checkpoint_manifest",
]

MANIFEST_NAME = "checkpoint.json"

#: Number of digits in a checkpoint directory name.
_STEP_WIDTH = 8


class CheckpointError(ValueError):
    """A checkpoint directory or manifest is missing or unreadable."""


@dataclass(frozen=True)
class Checkpoint:
    """One checkpoint: its step number and directory."""

    step: int
    directory: Path


def checkpoint_dir_name(step: int) -> str:
    """The directory name for one step, zero-padded to 8 digits."""
    if isinstance(step, bool) or not isinstance(step, int) or step < 1:
        raise ValueError(f"step must be a positive integer, got {step!r}")
    return f"step-{step:0{_STEP_WIDTH}d}"


def parse_checkpoint_step(name: str) -> int | None:
    """The step encoded in a directory name, or None when it is not one."""
    if not name.startswith("step-"):
        return None
    digits = name[len("step-") :]
    if len(digits) != _STEP_WIDTH or not digits.isdigit():
        return None
    step = int(digits)
    return step if step >= 1 else None


def write_checkpoint_manifest(
    directory: str | Path,
    *,
    step: int,
    arm: str,
    timestamp: str,
) -> Path:
    """Write `checkpoint.json` into `directory`, creating it. Returns the path."""
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "arm": arm,
        "timestamp": timestamp,
    }
    path = folder / MANIFEST_NAME
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return path


def read_checkpoint_manifest(directory: str | Path) -> dict[str, Any]:
    """Read and validate a checkpoint manifest. Raises CheckpointError."""
    folder = Path(directory)
    path = folder / MANIFEST_NAME
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise CheckpointError(f"checkpoint manifest not found: {path}") from None
    except (OSError, json.JSONDecodeError) as error:
        raise CheckpointError(f"cannot read checkpoint manifest {path}: {error}") from error

    if not isinstance(payload, dict):
        raise CheckpointError(f"checkpoint manifest {path} must be an object")
    step = payload.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 1:
        raise CheckpointError(f"checkpoint manifest {path} has a bad step {step!r}")
    arm = payload.get("arm")
    if not isinstance(arm, str) or not arm:
        raise CheckpointError(f"checkpoint manifest {path} has a bad arm {arm!r}")
    return payload


def find_latest_checkpoint(checkpoints_dir: str | Path) -> Checkpoint | None:
    """The checkpoint with the highest step under `checkpoints_dir`, or None.

    Directories that do not name a valid step or that carry a bad manifest are
    skipped, so one corrupt checkpoint cannot hide an earlier good one.
    """
    root = Path(checkpoints_dir)
    if not root.is_dir():
        return None

    best: Checkpoint | None = None
    for child in root.iterdir():
        if not child.is_dir():
            continue
        step = parse_checkpoint_step(child.name)
        if step is None:
            continue
        try:
            read_checkpoint_manifest(child)
        except CheckpointError:
            continue
        if best is None or step > best.step:
            best = Checkpoint(step=step, directory=child)
    return best
