"""A one second GPU sampler that runs inside the training process.

The run folder stays the record. Every tick reads `nvidia-smi` for utilization,
memory and temperature, reads the torch allocator for the five memory splits,
appends one row to `gpu.jsonl`, and then hands the row to the MLflow mirror. A
mirror that fails loses a copy and never a row.

The `nvidia-smi` reader is copied from `runread.py`. The dashboard copy stays
until the web layer is removed. The duplication is deliberate: the read layer
and the sampler are moving apart, and the sampler must not import torch at
module level.

A missing `nvidia-smi`, a torch import that fails and a full disk all end the
same way: the tick is dropped and the thread keeps going. A broken mirror
leaves the row on disk. The sampler must never stop training.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import time
from pathlib import Path
from threading import Event, Thread
from typing import Any, Callable, Mapping

from .memory import MemoryReport, read_torch_memory

__all__ = ["GPU_ROWS_NAME", "GpuSampler", "read_gpu"]

#: The GPU rows file in the run folder.
GPU_ROWS_NAME = "gpu.jsonl"

#: Where to look for nvidia-smi when PATH does not carry it. A tmux server
#: started from a different shell can have a PATH without the driver tools even
#: though the binary is installed.
NVIDIA_SMI_PATHS: tuple[str, ...] = (
    "/usr/bin/nvidia-smi",
    "/usr/local/nvidia/bin/nvidia-smi",
)

#: One wait in the sampler loop. Short, so `stop()` returns quickly even when
#: the sample interval is long.
WAIT_SLICE_SECONDS = 0.05

_GPU_NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)")


def _gpu_number(value: str) -> float | None:
    """An nvidia-smi CSV field as a number, or None when it is not one.

    nvidia-smi prints text tokens such as N/A, [N/A], [Not Supported] and
    [Unknown Error] for a field it cannot report, so anything that does not
    start with a number becomes None. A number with a unit still parses, so a
    format that keeps `%` or `MiB` does not silently zero the field. Zero is a
    real reading and stays zero.
    """
    text = value.strip()
    if not text:
        return None
    match = _GPU_NUMBER.match(text)
    if match is None:
        return None
    number = float(match.group(0))
    return number if math.isfinite(number) else None


def _nvidia_smi() -> str:
    """The nvidia-smi binary by PATH, then by the usual install paths."""
    found = shutil.which("nvidia-smi")
    if found:
        return found
    for candidate in NVIDIA_SMI_PATHS:
        if Path(candidate).is_file():
            return candidate
    return "nvidia-smi"


def read_gpu() -> dict[str, Any] | None:
    """One nvidia-smi reading for every GPU, or None when nothing was read.

    One CSV row per GPU. Utilisation and temperature take the hottest GPU and
    memory is the sum over all of them, so a multi-GPU host reports the whole
    machine rather than an arbitrary card. A row with a field nvidia-smi cannot
    report keeps that field as None. A reading where every field is None is a
    failure, not a sample, so it returns None.
    """
    command = [
        _nvidia_smi(),
        "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    rows: list[dict[str, float | None]] = []
    for line in completed.stdout.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            continue
        rows.append(
            {
                "utilization": _gpu_number(parts[0]),
                "memory_used": _gpu_number(parts[1]),
                "memory_total": _gpu_number(parts[2]),
                "temperature": _gpu_number(parts[3]),
            }
        )
    if not rows:
        return None

    def total(name: str) -> float | None:
        values = [row[name] for row in rows if row[name] is not None]
        if not values:
            return None
        return float(sum(values))

    def highest(name: str) -> float | None:
        values = [row[name] for row in rows if row[name] is not None]
        if not values:
            return None
        return float(max(values))

    sample = {
        "utilization": highest("utilization"),
        "memory_used": total("memory_used"),
        "memory_total": total("memory_total"),
        "temperature": highest("temperature"),
    }
    if all(value is None for value in sample.values()):
        return None
    return sample


class GpuSampler:
    """Samples the GPU on a background thread and writes one row per tick.

    `path` is the JSONL file. `on_sample` is called with each row after the row
    is on disk, and may be None. `read` and `memory` are injected so the tests
    run without a GPU. `rows` keeps the rows this sampler wrote, for tests.

    A failed tick is dropped. The thread catches every exception so a host
    without `nvidia-smi` cannot stop training.
    """

    def __init__(
        self,
        path: Path,
        on_sample: Callable[[Mapping[str, Any]], None] | None,
        interval: float = 1.0,
        read: Callable[[], Mapping[str, Any] | None] = read_gpu,
        memory: Callable[[], MemoryReport] = read_torch_memory,
    ) -> None:
        self.path = Path(path)
        self.on_sample = on_sample
        self.interval = interval
        self.read = read
        self.memory = memory
        self.rows: list[dict[str, Any]] = []
        self._stop = Event()
        self._thread: Thread | None = None

    def start(self) -> None:
        """Start the sampling thread. A second call does nothing."""
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = Thread(
            target=self._run,
            name="rlvr-gpu-sampler",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Ask the thread to stop and wait for the current tick to finish."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                row = self._sample()
                self._write(row)
                self.rows.append(row)
                if self.on_sample is not None:
                    self.on_sample(row)
            except Exception:  # noqa: BLE001 - the sampler must not stop training
                pass
            self._wait()

    def _wait(self) -> None:
        """Wait one interval in short slices so `stop()` returns quickly."""
        remaining = self.interval
        while remaining > 0 and not self._stop.is_set():
            slice_seconds = min(remaining, WAIT_SLICE_SECONDS)
            self._stop.wait(slice_seconds)
            remaining -= slice_seconds

    def _sample(self) -> dict[str, Any]:
        """One row: the nvidia-smi reading, the allocator split and the time."""
        sample = self.read() or {}
        report = self.memory()
        return {
            "t": time.time(),
            "utilization": sample.get("utilization"),
            "memory_used": sample.get("memory_used"),
            "memory_total": sample.get("memory_total"),
            "temperature": sample.get("temperature"),
            "live_mib": report.live_mib,
            "cached_mib": report.cached_mib,
            "fragmented_mib": report.fragmented_mib,
            "unaccounted_mib": report.unaccounted_mib,
            "total_mib": report.total_mib,
        }

    def _write(self, row: Mapping[str, Any]) -> None:
        """Append one JSON line. The handle is closed, so the row is flushed."""
        line = json.dumps(row, separators=(",", ":"))
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
