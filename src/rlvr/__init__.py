"""Small-model code RLVR reward study.

Public API: `Experiment` and the four backends it injects, the record schema,
the run store, config loading, and the command line entry point.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .experiment import Experiment, RunSummary
from rlvr.core.values import Generation, JudgeResult, Outcome, Prompt
from .reward import ARMS
from .runs import RunStore
from rlvr.core.schema import RolloutRecord

__all__ = [
    "ARMS",
    "Experiment",
    "Generation",
    "JudgeResult",
    "Prompt",
    "RolloutRecord",
    "RunStore",
    "RunSummary",
    "Outcome",
    "__version__",
]
