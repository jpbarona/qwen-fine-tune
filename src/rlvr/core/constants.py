"""Constants that more than one layer names.

A config, a reward and a report all need the same default or the same arm list.
Keeping those values here lets a base module read one without importing the
domain that owns it, and keeps the import graph acyclic.

Values that only one module uses stay in that module.
"""

from __future__ import annotations

#: Arm names in the order used by the CLI and the run folders.
ARMS: tuple[str, ...] = ("code-score", "hybrid", "gated-judge", "gated-process")

#: Defaults for the code-arm length term.
DEFAULT_LENGTH_SOFT_LIMIT = 900
DEFAULT_LENGTH_PENALTY_PER_TOKEN = 0.001
DEFAULT_LENGTH_PENALTY_FLOOR = 0.2

#: Tokens allowed for the reasoning block, the same on every step.
DEFAULT_THINKING_BUDGET = 1024

#: Minimum free memory after a step, in MiB, before the batch grows.
DEFAULT_MEMORY_FLOOR_MIB = 512

#: The adapter placements a training config may choose.
LORA_TARGETS: tuple[str, ...] = ("attention", "all-linear")

#: The gap between sweeps, as a multiple of the sweep length, and its range.
DEFAULT_SWEEP_GAP_RATIO = 3
MIN_SWEEP_GAP_RATIO = 3
MAX_SWEEP_GAP_RATIO = 10
