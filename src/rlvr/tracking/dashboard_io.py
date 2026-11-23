"""Tolerant file readers for dashboard snapshots of actively written runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator


def _read_json(path: Path) -> dict[str, Any]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file, skipping a line that is still being written."""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    rows: list[dict[str, Any]] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _read_tail_lines(path: Path, max_bytes: int) -> list[str]:
    """The last bytes of a file, as lines. The first partial line is dropped."""
    try:
        size = path.stat().st_size
    except OSError:
        return []
    if size == 0:
        return []
    try:
        with path.open("rb") as handle:
            if size > max_bytes:
                handle.seek(size - max_bytes)
            data = handle.read()
    except OSError:
        return []
    lines = data.decode("utf-8", errors="replace").splitlines()
    if size > max_bytes and lines:
        lines = lines[1:]
    return lines


def _all_rollout_records(path: Path) -> Iterator[dict[str, Any]]:
    """Yield every rollout record, one line at a time.

    Used by the panels that report totals over the whole run. The file is read
    as a stream, so a long run costs one line of memory at a time, not the
    whole file. A line still being written is skipped as a JSON error.
    """
    try:
        handle = path.open("r", encoding="utf-8", errors="replace")
    except OSError:
        return
    with handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                yield record
