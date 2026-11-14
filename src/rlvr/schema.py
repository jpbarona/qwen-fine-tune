"""The rollout record schema.

The field list and types in this module are a frozen contract. Every run folder
writes `rollouts.jsonl` with one record per line in exactly this order. Later
tasks read these records, so change a field here only together with every
consumer.

Notes on the contract:

* `error_class` is `"pass"` when every task test passed. Any other string names
  the failure. Task 2 defines the exact classes the Docker runner produces. Do
  not use an empty string for success, and do not treat the current names as a
  closed set.
* `judge_score` and `process_score` are null when the arm does not use them, or
  when the rollout failed its tests and the judge was never called.
* `cache_hit` is true when any judge call for this rollout was served from the
  judge cache. It is always false for the code-score arm, which never judges.
* `reward` may be negative. The code arm clips it to [-1, 1], so the schema
  checks that range rather than [0, 1]. The three score fields stay in [0, 1].
* `stage` is the curriculum band active when the rollout ran. Runs without a
  curriculum use the fixed string `"unstaged"`.
* Timestamps and truncation flags are deliberately not part of this record.
* Entropy is not a rollout field. It is measured on the update pass, at the
  `entropy_every_steps` cadence, so it belongs to the step metric row and may be
  absent there on a step without a measurement.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

__all__ = [
    "FIELD_NAMES",
    "FIELD_TYPES",
    "RolloutRecord",
    "SchemaError",
]

# Type names used in the contract. "null" means the value may be JSON null.
STR = "str"
INT = "int"
FLOAT = "float"
BOOL = "bool"
FLOAT_OR_NULL = ("float", "null")

#: Field names in the order they are written.
FIELD_NAMES: tuple[str, ...] = (
    "arm",
    "step",
    "stage",
    "prompt_id",
    "reasoning",
    "code",
    "prompt_tokens",
    "completion_tokens",
    "tests_passed",
    "tests_total",
    "error_class",
    "code_score",
    "judge_score",
    "process_score",
    "reward",
    "cache_hit",
)

#: Field name to allowed type names.
FIELD_TYPES: dict[str, tuple[str, ...]] = {
    "arm": (STR,),
    "step": (INT,),
    "stage": (STR,),
    "prompt_id": (STR,),
    "reasoning": (STR,),
    "code": (STR,),
    "prompt_tokens": (INT,),
    "completion_tokens": (INT,),
    "tests_passed": (INT,),
    "tests_total": (INT,),
    "error_class": (STR,),
    "code_score": (FLOAT,),
    "judge_score": FLOAT_OR_NULL,
    "process_score": FLOAT_OR_NULL,
    "reward": (FLOAT,),
    "cache_hit": (BOOL,),
}

#: Fields that must be a fraction of the tests, so they stay inside [0, 1].
BOUNDED_FIELDS: tuple[str, ...] = ("code_score", "judge_score", "process_score")

#: The reward may be negative, so it is clipped to this range instead.
REWARD_MIN = -1.0
REWARD_MAX = 1.0

#: Fields that are counts and must not be negative.
COUNT_FIELDS: tuple[str, ...] = (
    "step",
    "prompt_tokens",
    "completion_tokens",
    "tests_passed",
    "tests_total",
)


class SchemaError(ValueError):
    """A record does not match the rollout schema."""


def _type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    return type(value).__name__


def _matches(kind: str, value: Any) -> bool:
    if kind == STR:
        return isinstance(value, str)
    if kind == INT:
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == FLOAT:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if kind == BOOL:
        return isinstance(value, bool)
    if kind == "null":
        return value is None
    raise AssertionError(f"unknown schema type kind {kind!r}")


def _describe(types: tuple[str, ...]) -> str:
    return " or ".join(types)


def _value_problems(payload: dict[str, Any]) -> list[str]:
    """Type and range problems for the fields that are present."""
    problems: list[str] = []

    for name, types in FIELD_TYPES.items():
        if name not in payload:
            continue
        value = payload[name]
        if not any(_matches(kind, value) for kind in types):
            problems.append(
                f"field {name!r} must be {_describe(types)}, got {_type_name(value)}"
            )

    for name in BOUNDED_FIELDS:
        value = payload.get(name)
        if _matches(FLOAT, value) and not 0.0 <= float(value) <= 1.0:
            problems.append(f"field {name!r} must be between 0 and 1, got {value!r}")

    reward = payload.get("reward")
    if _matches(FLOAT, reward) and not REWARD_MIN <= float(reward) <= REWARD_MAX:
        problems.append(
            f"field 'reward' must be between {REWARD_MIN:g} and {REWARD_MAX:g}, got {reward!r}"
        )

    for name in COUNT_FIELDS:
        value = payload.get(name)
        if _matches(INT, value) and value < 0:
            problems.append(f"field {name!r} must not be negative, got {value!r}")

    passed, total = payload.get("tests_passed"), payload.get("tests_total")
    if _matches(INT, passed) and _matches(INT, total) and passed > total:
        problems.append(
            f"field 'tests_passed' must not exceed 'tests_total', got {passed} > {total}"
        )

    return problems


@dataclass(frozen=True)
class RolloutRecord:
    """One rollout, as written to `rollouts.jsonl`."""

    arm: str
    step: int
    stage: str
    prompt_id: str
    reasoning: str
    code: str
    prompt_tokens: int
    completion_tokens: int
    tests_passed: int
    tests_total: int
    error_class: str
    code_score: float
    judge_score: float | None
    process_score: float | None
    reward: float
    cache_hit: bool

    def __post_init__(self) -> None:
        """Validate types and ranges on construction."""
        problems = _value_problems(self.to_dict())
        if problems:
            raise SchemaError("; ".join(problems))

    def to_dict(self) -> dict[str, Any]:
        """The record as a mapping, in contract field order."""
        return {name: getattr(self, name) for name in FIELD_NAMES}

    def to_json(self) -> str:
        """One JSONL line. Values are written in contract field order."""
        return json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RolloutRecord":
        """Build a record from a mapping, raising SchemaError if it does not fit.

        Presence is checked here. Types and ranges are checked by the
        constructor, so a record can never exist in an invalid state.
        """
        if not isinstance(payload, dict):
            raise SchemaError(f"record must be a mapping, got {_type_name(payload)}")

        problems: list[str] = []
        missing = [name for name in FIELD_NAMES if name not in payload]
        extra = sorted(name for name in payload if name not in FIELD_NAMES)
        if missing:
            problems.append(f"missing field(s): {', '.join(repr(name) for name in missing)}")
        if extra:
            problems.append(f"unexpected field(s): {', '.join(repr(name) for name in extra)}")
        if problems:
            raise SchemaError("; ".join(problems))

        return cls(**{name: payload[name] for name in FIELD_NAMES})

    @classmethod
    def from_json(cls, line: str) -> "RolloutRecord":
        """Build a record from one JSONL line."""
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise SchemaError(f"record is not valid JSON: {error}") from error
        return cls.from_dict(payload)
