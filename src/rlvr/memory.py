"""GPU memory reporting from the torch caching allocator.

The device total answers "how big is the card". It does not answer "where is the
memory going". The allocator knows that split:

* live bytes are the tensors the run holds right now.
* cached bytes are reserved blocks the allocator keeps for reuse.
* fragmented bytes are the unused gaps inside reserved segments.
* the rest of the device total is other processes, the CUDA context, and the
  driver, which the allocator cannot see.

The three allocator figures come from `torch.cuda.memory_stats()` as
`allocated_bytes`, `active_bytes`, and `reserved_bytes`:

    live        = allocated_bytes.all.current
    fragmented  = active_bytes.all.current - allocated_bytes.all.current
    cached      = reserved_bytes.all.current - active_bytes.all.current

Live plus cached plus fragmented equals the reserved total. The gap from the
device total is reported separately as unaccounted, so a mismatch is visible
rather than hidden inside one number.

Everything here is MiB. The module imports no torch at module level, so the CPU
suite and the fake path run without it. `read_torch_memory` imports torch
lazily and returns zeros when it, CUDA, or the allocator is unavailable.

The allocator setting is `PYTORCH_CUDA_ALLOC_CONF`. PyTorch reads it once, at
startup, so the entry point calls `configure_expandable_segments` before any
torch import. A value the user already set is left alone.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping

__all__ = [
    "ALLOCATOR_ENV",
    "EXPANDABLE_SEGMENTS",
    "MEMORY_METRIC_FIELDS",
    "MemoryReport",
    "configure_expandable_segments",
    "read_torch_memory",
    "report_from_stats",
]

#: Bytes in one MiB.
MIB = 1024 * 1024

#: The environment variable PyTorch reads at startup to configure the allocator.
ALLOCATOR_ENV = "PYTORCH_CUDA_ALLOC_CONF"

#: The setting that lets segments grow instead of fragmenting on new sizes.
EXPANDABLE_SEGMENTS = "expandable_segments:True"

#: Metric row fields, in the order they are written. The first three are the
#: allocator split. `gpu_total_mib` is the device total. `gpu_unaccounted_mib`
#: is the part of the device the allocator does not hold, so a mismatch shows.
MEMORY_METRIC_FIELDS: tuple[str, ...] = (
    "gpu_live_mib",
    "gpu_cached_mib",
    "gpu_fragmented_mib",
    "gpu_total_mib",
    "gpu_unaccounted_mib",
)


def _mib(value: float) -> float:
    """Bytes as MiB, rounded so the JSON row stays readable."""
    return round(value / MIB, 3)


@dataclass(frozen=True)
class MemoryReport:
    """One step's GPU memory split, in MiB.

    `total_mib` is the device capacity. It is zero when no GPU is present, as
    on the fake path. `unaccounted_mib` may be negative if the allocator reports
    more than the device total, which is a real mismatch worth seeing.
    """

    live_mib: float
    cached_mib: float
    fragmented_mib: float
    total_mib: float

    @classmethod
    def zeros(cls) -> "MemoryReport":
        """Every field zero, used by the fake path and the CPU suite."""
        return cls(live_mib=0.0, cached_mib=0.0, fragmented_mib=0.0, total_mib=0.0)

    @classmethod
    def from_bytes(
        cls,
        *,
        live: float,
        cached: float,
        fragmented: float,
        total: float,
    ) -> "MemoryReport":
        """Build a report from byte counts. Every argument is in bytes."""
        return cls(
            live_mib=_mib(live),
            cached_mib=_mib(cached),
            fragmented_mib=_mib(fragmented),
            total_mib=_mib(total),
        )

    @property
    def reserved_mib(self) -> float:
        """The allocator's reserved total: live plus cached plus fragmented."""
        return round(self.live_mib + self.cached_mib + self.fragmented_mib, 3)

    @property
    def unaccounted_mib(self) -> float:
        """The part of the device the allocator does not hold.

        This covers other processes, the CUDA context, and the driver. Negative
        means the three figures exceed the device total, which is a mismatch.
        """
        return round(self.total_mib - self.reserved_mib, 3)

    def as_metrics(self) -> dict[str, float]:
        """The five metric row fields, in contract order."""
        values = {
            "gpu_live_mib": self.live_mib,
            "gpu_cached_mib": self.cached_mib,
            "gpu_fragmented_mib": self.fragmented_mib,
            "gpu_total_mib": self.total_mib,
            "gpu_unaccounted_mib": self.unaccounted_mib,
        }
        return {name: values[name] for name in MEMORY_METRIC_FIELDS}


def _stat(stats: Mapping[str, Any], name: str) -> float:
    """One allocator stat as a float, or zero when the key is absent."""
    value = stats.get(name, 0)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return float(value)


def report_from_stats(
    stats: Mapping[str, Any], total_bytes: int
) -> MemoryReport:
    """Split `torch.cuda.memory_stats()` output into live, cached, fragmented.

    Pure, so it is tested without torch. A stat that is absent counts as zero.
    Cached and fragmented are clamped at zero, so a stale or partial stats
    snapshot cannot produce a negative figure.
    """
    allocated = _stat(stats, "allocated_bytes.all.current")
    active = _stat(stats, "active_bytes.all.current")
    reserved = _stat(stats, "reserved_bytes.all.current")
    return MemoryReport.from_bytes(
        live=allocated,
        cached=max(reserved - active, 0.0),
        fragmented=max(active - allocated, 0.0),
        total=float(total_bytes),
    )


def read_torch_memory() -> MemoryReport:
    """Read the allocator inside this process. Zeros when there is no GPU.

    A diagnostic must never stop a training run, so a failed CUDA query also
    returns zeros. The dashboard still shows the panel, and a zero total says
    the allocator was not available rather than pretending to have data.
    """
    try:
        import torch  # lazy
    except ImportError:
        return MemoryReport.zeros()

    cuda = torch.cuda
    try:
        if not cuda.is_available():
            return MemoryReport.zeros()
        stats = cuda.memory_stats()
        total = int(cuda.get_device_properties(cuda.current_device()).total_memory)
    except (RuntimeError, AssertionError):
        return MemoryReport.zeros()
    return report_from_stats(stats, total)


def configure_expandable_segments(
    environ: MutableMapping[str, str] | None = None,
) -> bool:
    """Turn on expandable segments unless the user already set the variable.

    Returns whether this call set the value. An empty value counts as unset. The
    caller must run this before torch is imported, because PyTorch reads the
    variable once at allocator startup.
    """
    env = os.environ if environ is None else environ
    if env.get(ALLOCATOR_ENV):
        return False
    env[ALLOCATOR_ENV] = EXPANDABLE_SEGMENTS
    return True
