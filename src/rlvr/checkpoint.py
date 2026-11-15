"""Checkpoint directory and manifest handling.

A checkpoint is one directory under `run_dir/checkpoints/` named `step-XXXXXXXX`
(an 8-digit, zero-padded step number). The trainer writes its LoRA adapter and
optimizer state into that directory. A small `checkpoint.json` manifest records
the step, arm, and timestamp so a resume can find the latest checkpoint without
importing torch.

The run folder keeps ten checkpoints: the newest five consecutive and older ones
at doubling gaps, so an early collapse still has a checkpoint to resume from.
`select_checkpoints_to_keep` picks them and `prune_checkpoints` deletes the
rest. Retention only touches the run folder, never persistent storage.

This module is pure Python. The real state dicts live in `qwen.py` and are the
only part that needs torch.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

__all__ = [
    "DEFAULT_KEEP_RECENT",
    "DEFAULT_KEEP_TOTAL",
    "MANIFEST_NAME",
    "Checkpoint",
    "CheckpointError",
    "checkpoint_dir_name",
    "find_latest_checkpoint",
    "parse_checkpoint_step",
    "prune_checkpoints",
    "read_checkpoint_manifest",
    "select_checkpoints_to_keep",
    "write_checkpoint_manifest",
]

MANIFEST_NAME = "checkpoint.json"

#: Number of digits in a checkpoint directory name.
_STEP_WIDTH = 8

#: Retention: ten checkpoints, the newest five consecutive and older ones back
#: at doubling gaps. One checkpoint is about 100 MB, so ten keep an early
#: collapse recoverable without filling a 30 GB disk.
DEFAULT_KEEP_RECENT = 5
DEFAULT_KEEP_TOTAL = 10

#: Cumulative positions back from the newest for the older retained
#: checkpoints, doubling at 2, 4, 8, 16 and 32 beyond the newest block. The
#: first skip of 6 counts the newest five plus one.
_RETENTION_OFFSETS = (6, 10, 18, 34, 66)


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


def select_checkpoints_to_keep(
    steps: Sequence[int],
    *,
    recent: int = DEFAULT_KEEP_RECENT,
    total: int = DEFAULT_KEEP_TOTAL,
) -> list[int]:
    """The step numbers to retain, oldest first.

    The newest `recent` checkpoints stay. Older ones step back at doubling gaps
    of 2, 4, 8, 16 and 32 positions from the newest, which is cumulative 6, 10,
    18, 34 and 66. When the gaps run past the oldest checkpoint before `total`
    is reached, the remaining slots take the oldest unkept checkpoints, working
    forward, so a short run keeps a checkpoint near its start. A run with no
    more than `total` checkpoints keeps all of them.
    """
    if isinstance(recent, bool) or recent < 1:
        raise ValueError(f"recent must be at least 1, got {recent!r}")
    if isinstance(total, bool) or total < 1:
        raise ValueError(f"total must be at least 1, got {total!r}")
    if recent > total:
        raise ValueError(f"recent ({recent}) must not exceed total ({total})")

    ordered = sorted({int(step) for step in steps})
    if not ordered:
        return []

    keep = set(ordered[-recent:])
    newest_index = len(ordered) - 1
    for offset in _RETENTION_OFFSETS:
        if len(keep) >= total:
            break
        index = newest_index - offset
        if index < 0:
            break
        keep.add(ordered[index])

    for step in ordered:
        if len(keep) >= total:
            break
        keep.add(step)

    return sorted(keep)


def prune_checkpoints(directory: str | Path, keep: Sequence[int]) -> list[Path]:
    """Delete `step-XXXXXXXX` directories whose step is not in `keep`.

    Only directories named for a step are considered. Files, other directories
    and symlinks are left alone, so README notes or a partial copy survive.
    Returns the deleted paths, sorted by name. A missing directory is a no-op.
    """
    root = Path(directory)
    if not root.is_dir():
        return []

    kept = {int(step) for step in keep}
    deleted: list[Path] = []
    for child in sorted(root.iterdir(), key=lambda path: path.name):
        if child.is_symlink() or not child.is_dir():
            continue
        step = parse_checkpoint_step(child.name)
        if step is None or step in kept:
            continue
        shutil.rmtree(child)
        deleted.append(child)
    return deleted


def write_checkpoint_manifest(
    directory: str | Path,
    *,
    step: int,
    arm: str,
    timestamp: str,
    prompt_cursor: int | None = None,
    batch: Mapping[str, Any] | None = None,
    stage: Mapping[str, Any] | None = None,
) -> Path:
    """Write `checkpoint.json` into `directory`, creating it. Returns the path.

    `prompt_cursor` is the position in the prompt list the next batch starts
    from. `batch` holds the batch policy state, so a resume keeps the chosen
    size. `stage` holds the curriculum position and remaining pool. All three
    are optional, so older callers and older manifests still work.
    """
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "step": step,
        "arm": arm,
        "timestamp": timestamp,
    }
    if prompt_cursor is not None:
        payload["prompt_cursor"] = prompt_cursor
    if batch is not None:
        payload["batch"] = dict(batch)
    if stage is not None:
        payload["stage"] = dict(stage)
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
