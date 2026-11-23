"""Tests for the one second GPU sampler that runs beside training."""

from __future__ import annotations

import json
import time
from pathlib import Path

from rlvr.evaluation.fake import fake_memory
from rlvr.gpu import GpuSampler
from rlvr.core.memory import MemoryReport

ROW_KEYS = {
    "t",
    "utilization",
    "memory_used",
    "memory_total",
    "temperature",
    "live_mib",
    "cached_mib",
    "fragmented_mib",
    "unaccounted_mib",
    "total_mib",
    "idle",
    "count",
    "age",
}


def read_rows(path: Path) -> list[dict]:
    """The rows written so far, one JSON object per line."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_the_sampler_writes_one_row_per_tick(tmp_path: Path) -> None:
    ticks = iter(
        [
            {
                "utilization": 50.0,
                "memory_used": 1.0,
                "memory_total": 2.0,
                "temperature": 60.0,
            }
        ]
        * 3
    )
    sampler = GpuSampler(
        tmp_path / "gpu.jsonl",
        None,
        interval=0.01,
        read=lambda: next(ticks),
        memory=fake_memory,
    )
    sampler.start()
    time.sleep(0.05)
    sampler.stop()
    rows = read_rows(tmp_path / "gpu.jsonl")
    assert len(rows) >= 2
    assert rows[0]["utilization"] == 50.0


def test_the_row_has_the_documented_keys(tmp_path: Path) -> None:
    report = MemoryReport(
        live_mib=100.0, cached_mib=20.0, fragmented_mib=5.0, total_mib=1000.0
    )
    sampler = GpuSampler(
        tmp_path / "gpu.jsonl",
        None,
        interval=0.01,
        read=lambda: {
            "utilization": 50.0,
            "memory_used": 1.0,
            "memory_total": 2.0,
            "temperature": 60.0,
        },
        memory=lambda: report,
    )
    sampler.start()
    time.sleep(0.05)
    sampler.stop()
    row = read_rows(tmp_path / "gpu.jsonl")[0]
    assert set(row) == ROW_KEYS
    assert row["live_mib"] == 100.0
    assert row["cached_mib"] == 20.0
    assert row["fragmented_mib"] == 5.0
    assert row["unaccounted_mib"] == 875.0
    assert row["total_mib"] == 1000.0


def test_a_missing_nvidia_smi_reading_still_writes_the_memory_row(
    tmp_path: Path,
) -> None:
    sampler = GpuSampler(
        tmp_path / "gpu.jsonl",
        None,
        interval=0.01,
        read=lambda: None,
        memory=fake_memory,
    )
    sampler.start()
    time.sleep(0.05)
    sampler.stop()
    rows = read_rows(tmp_path / "gpu.jsonl")
    assert len(rows) >= 2
    assert rows[0]["utilization"] is None
    assert rows[0]["memory_used"] is None
    assert rows[0]["total_mib"] == 0.0


def test_a_failed_read_does_not_stop_the_sampler(tmp_path: Path) -> None:
    calls = {"count": 0}

    def read() -> dict:
        calls["count"] += 1
        if calls["count"] == 1:
            raise OSError("nvidia-smi is missing")
        return {
            "utilization": 10.0,
            "memory_used": 1.0,
            "memory_total": 2.0,
            "temperature": 30.0,
        }

    sampler = GpuSampler(
        tmp_path / "gpu.jsonl",
        None,
        interval=0.01,
        read=read,
        memory=fake_memory,
    )
    sampler.start()
    time.sleep(0.1)
    sampler.stop()
    rows = read_rows(tmp_path / "gpu.jsonl")
    assert rows
    assert rows[0]["utilization"] == 10.0


def test_the_sample_is_passed_to_the_callback(tmp_path: Path) -> None:
    seen: list[dict] = []
    sampler = GpuSampler(
        tmp_path / "gpu.jsonl",
        seen.append,
        interval=0.01,
        read=lambda: {
            "utilization": 50.0,
            "memory_used": 1.0,
            "memory_total": 2.0,
            "temperature": 60.0,
        },
        memory=fake_memory,
    )
    sampler.start()
    time.sleep(0.05)
    sampler.stop()
    row = read_rows(tmp_path / "gpu.jsonl")[0]
    assert seen
    assert seen[0] == row


def test_a_failed_callback_does_not_stop_the_sampler(tmp_path: Path) -> None:
    calls = {"count": 0}

    def on_sample(row: dict) -> None:
        calls["count"] += 1
        raise RuntimeError("the mirror is down")

    sampler = GpuSampler(
        tmp_path / "gpu.jsonl",
        on_sample,
        interval=0.01,
        read=lambda: {
            "utilization": 50.0,
            "memory_used": 1.0,
            "memory_total": 2.0,
            "temperature": 60.0,
        },
        memory=fake_memory,
    )
    sampler.start()
    time.sleep(0.05)
    sampler.stop()
    assert calls["count"] >= 2
    assert len(read_rows(tmp_path / "gpu.jsonl")) >= 2


def test_stop_ends_the_sampling(tmp_path: Path) -> None:
    sampler = GpuSampler(
        tmp_path / "gpu.jsonl",
        None,
        interval=0.01,
        read=lambda: {
            "utilization": 50.0,
            "memory_used": 1.0,
            "memory_total": 2.0,
            "temperature": 60.0,
        },
        memory=fake_memory,
    )
    sampler.start()
    time.sleep(0.03)
    sampler.stop()
    written = len(read_rows(tmp_path / "gpu.jsonl"))
    assert written >= 1
    time.sleep(0.05)
    assert len(read_rows(tmp_path / "gpu.jsonl")) == written


def test_read_gpu_parses_the_csv_output(monkeypatch) -> None:
    from rlvr import gpu as gpu_module

    def fake_run(command, **kwargs):
        return gpu_module.subprocess.CompletedProcess(
            args=command, returncode=0, stdout="20, 19610, 49140, 41\n", stderr=""
        )

    monkeypatch.setattr(gpu_module.subprocess, "run", fake_run)
    sample = gpu_module.read_gpu()

    assert sample == {
        "utilization": 20.0,
        "memory_used": 19610.0,
        "memory_total": 49140.0,
        "temperature": 41.0,
    }


def test_the_sample_carries_idle_count_and_age(tmp_path: Path) -> None:
    readings = iter(
        [None, None, {"utilization": 0.0}, {"utilization": 50.0}, None]
    )
    sampler = GpuSampler(
        tmp_path / "gpu.jsonl",
        None,
        read=lambda: next(readings),
        memory=fake_memory,
    )

    first = sampler._sample()
    assert first["idle"] == 1
    assert first["count"] == 1
    assert first["age"] is None

    second = sampler._sample()
    assert second["idle"] == 1
    assert second["count"] == 2
    assert second["age"] is None

    third = sampler._sample()
    assert third["idle"] == 1
    assert third["count"] == 3
    assert third["age"] == 0.0

    fourth = sampler._sample()
    assert fourth["idle"] == 0
    assert fourth["count"] == 4
    assert fourth["age"] == 0.0

    fifth = sampler._sample()
    assert fifth["idle"] == 1
    assert fifth["count"] == 5
    assert fifth["age"] is not None
    assert fifth["age"] >= 0.0


def test_read_gpu_returns_none_when_the_binary_is_missing(monkeypatch) -> None:
    from rlvr import gpu as gpu_module

    def fake_run(command, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(gpu_module.subprocess, "run", fake_run)
    assert gpu_module.read_gpu() is None
