"""How many prompts one training step takes.

A step runs one group of samples for each prompt it takes. More prompts per step
keep the card busy while the Python side runs tests and judge calls.

The size is chosen from memory, not from config. The batch starts at
`START_PROMPTS`. After each step the free memory is read. While free memory
stays at or above the effective floor the next step takes one more prompt. The
first measurement below the floor settles the size at the largest batch that
fit, and it stays there. The size never drops below `MIN_PROMPTS`.

The effective floor is the larger of the configured `floor_mib` and the measured
footprint of one prompt group from the previous step. A fixed 512 MiB floor is
smaller than one group at the 2,048 token cap can need, so growing on the floor
alone can run out of memory. The footprint is measured as the growth in the
allocator reserved total divided by the growth in prompt count between two
measurements. With no growth measurement yet, the configured floor is used.

A CUDA out-of-memory error calls `settle`, which keeps the largest batch that
fit and blocks further growth. The guard lives with the caller, not with torch,
so this module stays pure.

Free memory is the device total minus the allocator reserved total. That is the
part the allocator does not hold, so it covers free space, the CUDA context, and
other processes. A report with a zero total means the allocator was not
available. That is the fake path and the CPU suite. In that case the policy
holds its current size instead of treating "unknown" as "full".

This module is pure Python and imports no torch.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .memory import MemoryReport

__all__ = [
    "DEFAULT_MEMORY_FLOOR_MIB",
    "MIN_PROMPTS",
    "START_PROMPTS",
    "BatchSizePolicy",
    "free_memory_mib",
    "is_cuda_out_of_memory",
]

#: The first step takes two prompts, so batching is on without a measurement.
START_PROMPTS = 2

#: A step always takes at least one prompt.
MIN_PROMPTS = 1

#: Minimum free memory after a step, in MiB, before the batch grows. The
#: effective floor is this figure or one group's measured footprint, whichever
#: is larger.
DEFAULT_MEMORY_FLOOR_MIB = 512


def free_memory_mib(report: MemoryReport) -> float:
    """The memory the allocator does not hold, in MiB, clamped at zero.

    A negative value would mean the allocator reported more than the device can
    hold. That is a measurement mismatch, not free memory, so it counts as zero.
    """
    return max(report.total_mib - report.reserved_mib, 0.0)


def is_cuda_out_of_memory(error: BaseException) -> bool:
    """Whether an exception is a torch CUDA out-of-memory error.

    The check uses the class name, the module, and the message, so it needs no
    torch import and stays testable on CPU. A wrapped error whose text says
    "CUDA out of memory" is also caught.
    """
    kind = type(error)
    if kind.__name__ == "OutOfMemoryError" and "torch" in kind.__module__.lower():
        return True
    return "cuda out of memory" in str(error).lower()


@dataclass
class BatchSizePolicy:
    """The prompt count for the next step, grown by one while memory allows.

    `current` is the size the next step asks for. `largest_fit` is the largest
    size that finished a step with free memory at or above the floor. `settled`
    turns true after the first measurement below the floor, or after a CUDA
    out-of-memory error, and from then on the policy keeps `largest_fit` instead
    of probing upward again.

    `group_footprint_mib` is the largest measured cost of one prompt group, in
    MiB, from the growth between two steps. The effective floor is the larger of
    `floor_mib` and this figure, so one more group must actually fit.

    `max_prompts` is a test and debugging pin. The run leaves it None.
    """

    floor_mib: int = DEFAULT_MEMORY_FLOOR_MIB
    current: int = START_PROMPTS
    largest_fit: int = 0
    settled: bool = False
    max_prompts: int | None = None
    group_footprint_mib: float = 0.0
    _last_reserved_mib: float | None = field(default=None, repr=False, init=False)
    _last_prompts: int = field(default=0, repr=False, init=False)

    def __post_init__(self) -> None:
        if self.floor_mib < 0:
            raise ValueError(f"memory floor must not be negative, got {self.floor_mib}")
        if self.current < MIN_PROMPTS:
            raise ValueError(
                f"prompt count must be at least {MIN_PROMPTS}, got {self.current}"
            )
        if self.max_prompts is not None and self.max_prompts < MIN_PROMPTS:
            raise ValueError(
                f"max_prompts must be at least {MIN_PROMPTS}, got {self.max_prompts}"
            )
        if self.group_footprint_mib < 0.0:
            raise ValueError(
                f"group footprint must not be negative, got {self.group_footprint_mib}"
            )

    @property
    def effective_floor_mib(self) -> float:
        """The free memory that must remain before the batch grows."""
        return max(float(self.floor_mib), self.group_footprint_mib)

    def count(self, available: int) -> int:
        """The prompt count for the next step, clamped to the pool size.

        `available` is the number of prompts the step may draw from. A short pool
        caps the batch, so a stage with one question still trains. The result is
        never below `MIN_PROMPTS`.
        """
        if available < 1:
            raise ValueError(f"available prompts must be at least 1, got {available}")
        limit = available
        if self.max_prompts is not None:
            limit = min(limit, self.max_prompts)
        return max(MIN_PROMPTS, min(self.current, limit))

    def observe(self, report: MemoryReport, *, prompts: int) -> None:
        """Update the next size from the memory left after a step.

        `prompts` is the count the step actually ran, after the pool clamp. A
        zero device total means no measurement, so the size is left alone. The
        effective floor is inclusive, so free memory exactly equal to it still
        allows growth.

        The step also measures the group footprint when the prompt count grew
        since the previous measurement. The reserved total includes the model,
        so the difference between two steps is the extra prompts plus allocator
        noise. Dividing by the extra prompt count gives one group's cost.
        """
        if prompts < MIN_PROMPTS:
            raise ValueError(f"observed prompts must be at least 1, got {prompts}")
        if report.total_mib <= 0.0:
            return

        if (
            self._last_reserved_mib is not None
            and self._last_prompts >= MIN_PROMPTS
            and prompts > self._last_prompts
        ):
            extra = (report.reserved_mib - self._last_reserved_mib) / (
                prompts - self._last_prompts
            )
            self.group_footprint_mib = max(self.group_footprint_mib, extra, 0.0)

        self._last_reserved_mib = report.reserved_mib
        self._last_prompts = prompts

        if free_memory_mib(report) >= self.effective_floor_mib:
            self.largest_fit = max(self.largest_fit, prompts)
            if not self.settled:
                # A short pool caps the step without lowering the target. The
                # next larger pool still gets the size memory allowed.
                self.current = max(self.current, prompts + 1)
            return
        self.settled = True
        self.current = max(self.largest_fit, MIN_PROMPTS)

    def settle(self) -> None:
        """Stop growing and keep the largest batch that fit.

        Called after a CUDA out-of-memory error. The failed size was too large,
        so the policy settles at `largest_fit`, or one prompt when no size has
        finished a step yet.
        """
        self.settled = True
        self.current = max(self.largest_fit, MIN_PROMPTS)
