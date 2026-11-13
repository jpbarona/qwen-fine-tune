"""Tests for the frozen rollout record schema.

The field list and types below are a contract. Later tasks read these records.
Do not change them without changing every consumer.
"""

from __future__ import annotations

import json

import pytest

from rlvr.schema import FIELD_NAMES, FIELD_TYPES, RolloutRecord, SchemaError

# The contract, verbatim. Order matters: records are written in this order.
CONTRACT: list[tuple[str, tuple[str, ...]]] = [
    ("arm", ("str",)),
    ("step", ("int",)),
    ("prompt_id", ("str",)),
    ("reasoning", ("str",)),
    ("code", ("str",)),
    ("prompt_tokens", ("int",)),
    ("completion_tokens", ("int",)),
    ("mean_entropy", ("float",)),
    ("tests_passed", ("int",)),
    ("tests_total", ("int",)),
    ("error_class", ("str",)),
    ("code_score", ("float",)),
    ("judge_score", ("float", "null")),
    ("process_score", ("float", "null")),
    ("reward", ("float",)),
    ("cache_hit", ("bool",)),
]


def valid_record(arm: str = "code-score", step: int = 1) -> dict:
    return {
        "arm": arm,
        "step": step,
        "prompt_id": "mbpp-0001",
        "reasoning": "think",
        "code": "def f():\n    return 1",
        "prompt_tokens": 42,
        "completion_tokens": 17,
        "mean_entropy": 0.35,
        "tests_passed": 3,
        "tests_total": 3,
        "error_class": "pass",
        "code_score": 1.0,
        "judge_score": None,
        "process_score": None,
        "reward": 1.0,
        "cache_hit": False,
    }


def test_field_names_and_order_match_contract() -> None:
    assert list(FIELD_NAMES) == [name for name, _ in CONTRACT]


def test_field_types_match_contract() -> None:
    assert {name: types for name, types in FIELD_TYPES.items()} == {
        name: types for name, types in CONTRACT
    }


def test_record_has_exactly_the_contract_fields() -> None:
    record = RolloutRecord(**valid_record())
    assert set(record.to_dict()) == set(FIELD_NAMES)
    assert list(record.to_dict()) == list(FIELD_NAMES)


def test_valid_record_round_trips_through_json() -> None:
    record = RolloutRecord(**valid_record())
    assert RolloutRecord.from_json(record.to_json()).to_dict() == record.to_dict()


def test_to_json_is_one_line() -> None:
    record = RolloutRecord(**valid_record())
    assert "\n" not in record.to_json()


def test_no_timestamp_or_truncation_fields_in_record() -> None:
    for banned in ("timestamp", "truncated", "created_at", "problem", "tests"):
        assert banned not in FIELD_NAMES


def test_missing_field_is_rejected() -> None:
    payload = valid_record()
    del payload["reward"]
    with pytest.raises(SchemaError, match="reward"):
        RolloutRecord.from_dict(payload)


def test_extra_field_is_rejected() -> None:
    payload = valid_record()
    payload["timestamp"] = "2026-11-12T00:00:00Z"
    with pytest.raises(SchemaError, match="timestamp"):
        RolloutRecord.from_dict(payload)


def test_wrong_string_field_is_rejected() -> None:
    payload = valid_record()
    payload["arm"] = 7
    with pytest.raises(SchemaError, match="arm"):
        RolloutRecord.from_dict(payload)


def test_bool_is_not_an_int() -> None:
    payload = valid_record()
    payload["step"] = True
    with pytest.raises(SchemaError, match="step"):
        RolloutRecord.from_dict(payload)


def test_int_is_not_a_bool() -> None:
    payload = valid_record()
    payload["cache_hit"] = 1
    with pytest.raises(SchemaError, match="cache_hit"):
        RolloutRecord.from_dict(payload)


def test_int_is_accepted_for_a_float_field() -> None:
    payload = valid_record()
    payload["reward"] = 1
    assert RolloutRecord.from_dict(payload).reward == 1.0


def test_string_is_not_a_float() -> None:
    payload = valid_record()
    payload["mean_entropy"] = "0.35"
    with pytest.raises(SchemaError, match="mean_entropy"):
        RolloutRecord.from_dict(payload)


def test_null_scores_are_allowed() -> None:
    payload = valid_record()
    payload["judge_score"] = None
    payload["process_score"] = None
    record = RolloutRecord.from_dict(payload)
    assert record.judge_score is None
    assert record.process_score is None


def test_scores_must_be_in_unit_interval() -> None:
    # The three score fields stay in [0, 1]. The reward has its own clip range.
    for field in ("code_score", "judge_score", "process_score"):
        payload = valid_record()
        payload[field] = 1.5
        with pytest.raises(SchemaError, match=field):
            RolloutRecord.from_dict(payload)


def test_reward_stays_in_the_clip_range_and_may_be_negative() -> None:
    for reward in (-1.0, -0.5, -0.2, 0.0, 1.0):
        payload = valid_record()
        payload["reward"] = reward
        assert RolloutRecord.from_dict(payload).reward == reward
    for reward in (-1.5, 1.5):
        payload = valid_record()
        payload["reward"] = reward
        with pytest.raises(SchemaError, match="reward"):
            RolloutRecord.from_dict(payload)


def test_mean_entropy_is_not_bounded_to_one() -> None:
    # Entropy is logged, not rewarded. Real values can exceed 1.
    payload = valid_record()
    payload["mean_entropy"] = 2.75
    assert RolloutRecord.from_dict(payload).mean_entropy == 2.75


def test_counts_must_not_be_negative() -> None:
    for field in ("prompt_tokens", "completion_tokens", "tests_passed", "tests_total", "step"):
        payload = valid_record()
        payload[field] = -1
        with pytest.raises(SchemaError, match=field):
            RolloutRecord.from_dict(payload)


def test_tests_passed_must_not_exceed_tests_total() -> None:
    payload = valid_record()
    payload["tests_passed"] = 4
    payload["tests_total"] = 3
    with pytest.raises(SchemaError, match="tests_passed"):
        RolloutRecord.from_dict(payload)


def test_error_class_is_a_string_with_pass_for_success() -> None:
    # Task 2 defines the exact classes. Success is "pass", never "".
    assert valid_record()["error_class"] == "pass"
    for value in ("pass", "", "timeout", "assertion", "syntax", "runtime", "future-class"):
        payload = valid_record()
        payload["error_class"] = value
        assert RolloutRecord.from_dict(payload).error_class == value
    payload = valid_record()
    payload["error_class"] = None  # type: ignore[assignment]
    with pytest.raises(SchemaError, match="error_class"):
        RolloutRecord.from_dict(payload)


def test_schema_error_lists_every_problem() -> None:
    payload = valid_record()
    payload["step"] = "one"
    payload["cache_hit"] = "yes"
    with pytest.raises(SchemaError) as excinfo:
        RolloutRecord.from_dict(payload)
    message = str(excinfo.value)
    assert "step" in message and "cache_hit" in message


def test_record_dataclass_is_frozen() -> None:
    import dataclasses

    record = RolloutRecord(**valid_record())
    with pytest.raises(dataclasses.FrozenInstanceError):
        record.reward = 0.0  # type: ignore[misc]


def test_jsonl_dump_is_utf8_and_key_order_stable() -> None:
    payload = valid_record()
    payload["code"] = "print('héllo')"
    line = RolloutRecord(**payload).to_json()
    assert json.loads(line)["code"] == "print('héllo')"
    assert list(json.loads(line)) == list(FIELD_NAMES)
