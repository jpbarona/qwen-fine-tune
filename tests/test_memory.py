"""Tests for the allocator memory report and the allocator setting.

No torch and no GPU are needed. The allocator split is a pure function of a
stats mapping, and the reader is exercised against a stub torch module. The part
that talks to a real CUDA device is review-only on this machine.
"""

from __future__ import annotations

import os
import sys
import types
from types import SimpleNamespace

import pytest

from rlvr.memory import (
    ALLOCATOR_ENV,
    EXPANDABLE_SEGMENTS,
    MEMORY_METRIC_FIELDS,
    MemoryReport,
    configure_expandable_segments,
    read_torch_memory,
    report_from_stats,
)

MIB = 1024 * 1024


def test_zeros_report_every_field_as_zero() -> None:
    report = MemoryReport.zeros()

    assert report.as_metrics() == {
        "gpu_live_mib": 0.0,
        "gpu_cached_mib": 0.0,
        "gpu_fragmented_mib": 0.0,
        "gpu_total_mib": 0.0,
        "gpu_unaccounted_mib": 0.0,
    }


def test_the_metric_fields_are_the_contract_order() -> None:
    report = MemoryReport(live_mib=1.0, cached_mib=2.0, fragmented_mib=3.0, total_mib=10.0)
    assert list(report.as_metrics()) == list(MEMORY_METRIC_FIELDS)


def test_from_bytes_converts_to_mib() -> None:
    report = MemoryReport.from_bytes(
        live=2 * MIB,
        cached=MIB // 2,
        fragmented=MIB // 4,
        total=24 * 1024 * MIB,
    )

    assert report.live_mib == pytest.approx(2.0)
    assert report.cached_mib == pytest.approx(0.5)
    assert report.fragmented_mib == pytest.approx(0.25)
    assert report.total_mib == pytest.approx(24576.0)


def test_the_three_figures_split_the_reserved_bytes() -> None:
    # Cached is reserved minus active. Fragmented is active minus allocated.
    # Together with live they must equal the reserved total.
    stats = {
        "allocated_bytes.all.current": 100 * MIB,
        "active_bytes.all.current": 120 * MIB,
        "reserved_bytes.all.current": 200 * MIB,
    }

    report = report_from_stats(stats, 1000 * MIB)

    assert report.live_mib == pytest.approx(100.0)
    assert report.fragmented_mib == pytest.approx(20.0)
    assert report.cached_mib == pytest.approx(80.0)
    assert report.reserved_mib == pytest.approx(200.0)
    assert report.total_mib == pytest.approx(1000.0)


def test_the_unaccounted_gap_is_the_device_minus_the_reserved_total() -> None:
    stats = {
        "allocated_bytes.all.current": 100 * MIB,
        "active_bytes.all.current": 120 * MIB,
        "reserved_bytes.all.current": 200 * MIB,
    }

    report = report_from_stats(stats, 1000 * MIB)

    assert report.unaccounted_mib == pytest.approx(800.0)


def test_a_reserved_total_above_the_device_shows_a_negative_gap() -> None:
    # The mismatch must be visible rather than clamped away.
    report = MemoryReport(
        live_mib=300.0, cached_mib=20.0, fragmented_mib=5.0, total_mib=200.0
    )

    assert report.unaccounted_mib == pytest.approx(-125.0)


def test_missing_stats_count_as_zero() -> None:
    report = report_from_stats({}, 512 * MIB)

    assert report.live_mib == 0.0
    assert report.cached_mib == 0.0
    assert report.fragmented_mib == 0.0
    assert report.total_mib == pytest.approx(512.0)


def test_inconsistent_stats_never_produce_negative_figures() -> None:
    stats = {
        "allocated_bytes.all.current": 300 * MIB,
        "active_bytes.all.current": 200 * MIB,
        "reserved_bytes.all.current": 100 * MIB,
    }

    report = report_from_stats(stats, 1000 * MIB)

    assert report.cached_mib == 0.0
    assert report.fragmented_mib == 0.0


def test_expandable_segments_are_set_when_the_variable_is_unset(monkeypatch) -> None:
    monkeypatch.delenv(ALLOCATOR_ENV, raising=False)

    assert configure_expandable_segments() is True
    assert os.environ[ALLOCATOR_ENV] == EXPANDABLE_SEGMENTS


def test_a_user_value_is_not_overwritten() -> None:
    environ = {ALLOCATOR_ENV: "max_split_size_mb:64"}

    assert configure_expandable_segments(environ) is False
    assert environ[ALLOCATOR_ENV] == "max_split_size_mb:64"


def test_an_empty_value_counts_as_unset() -> None:
    environ = {ALLOCATOR_ENV: ""}

    assert configure_expandable_segments(environ) is True
    assert environ[ALLOCATOR_ENV] == EXPANDABLE_SEGMENTS


def test_read_returns_zeros_without_torch(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)

    assert read_torch_memory() == MemoryReport.zeros()


def test_read_returns_zeros_when_cuda_is_unavailable(monkeypatch) -> None:
    stub = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", stub)

    assert read_torch_memory() == MemoryReport.zeros()


def test_read_uses_the_allocator_stats_and_the_device_total(monkeypatch) -> None:
    stats = {
        "allocated_bytes.all.current": 100 * MIB,
        "active_bytes.all.current": 120 * MIB,
        "reserved_bytes.all.current": 200 * MIB,
    }
    device = SimpleNamespace(total_memory=1000 * MIB)
    fake_cuda = SimpleNamespace(
        is_available=lambda: True,
        memory_stats=lambda: stats,
        current_device=lambda: 0,
        get_device_properties=lambda index: device,
    )
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=fake_cuda))

    report = read_torch_memory()

    assert report.live_mib == pytest.approx(100.0)
    assert report.cached_mib == pytest.approx(80.0)
    assert report.fragmented_mib == pytest.approx(20.0)
    assert report.total_mib == pytest.approx(1000.0)


def test_read_returns_zeros_when_the_allocator_query_fails(monkeypatch) -> None:
    def fail():
        raise RuntimeError("cuda error")

    fake_cuda = SimpleNamespace(is_available=lambda: True, memory_stats=fail)
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=fake_cuda))

    assert read_torch_memory() == MemoryReport.zeros()
