"""Tests for the merged MBPP dataset, the fixed split, and the prompt format."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rlvr.data import (
    DEFAULT_DATA_PATH,
    PROMPT_TEMPLATE,
    TOTAL_QUESTIONS,
    TRAIN_SIZE,
    TEST_SIZE,
    DatasetError,
    format_prompt,
    load_dataset,
    load_split,
    prompt_for_question,
    prompts_for_questions,
    split_questions,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_committed_dataset_merges_the_three_real_splits() -> None:
    questions = load_dataset(DEFAULT_DATA_PATH)
    assert len(questions) == TOTAL_QUESTIONS == 964
    counts: dict[str, int] = {}
    for question in questions:
        counts[question.source_split] = counts.get(question.source_split, 0) + 1
    assert counts == {"train": 374, "validation": 90, "test": 500}


def test_committed_dataset_has_unique_ids_and_tests() -> None:
    questions = load_dataset(DEFAULT_DATA_PATH)
    assert len({question.task_id for question in questions}) == 964
    for question in questions:
        assert question.test_list, question.task_id
        assert all(isinstance(test, str) for test in question.test_list)


def test_dataset_file_lives_under_data() -> None:
    assert DEFAULT_DATA_PATH == REPO_ROOT / "data" / "mbpp_full.jsonl"
    assert DEFAULT_DATA_PATH.is_file()


def test_loader_skips_the_few_shot_prompt_rows(tmp_path: Path) -> None:
    rows = [
        {
            "task_id": 1,
            "text": "one",
            "test_list": ["assert f() == 1"],
            "test_setup_code": "",
            "source_split": "train",
        },
        {
            "task_id": 2,
            "text": "two",
            "test_list": ["assert f() == 2"],
            "test_setup_code": "",
            "source_split": "prompt",
        },
    ]
    path = tmp_path / "tiny.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    questions = load_dataset(path)

    assert [question.task_id for question in questions] == [1]


def test_loader_rejects_duplicate_ids(tmp_path: Path) -> None:
    row = {
        "task_id": 1,
        "text": "one",
        "test_list": ["assert f() == 1"],
        "test_setup_code": "",
        "source_split": "train",
    }
    path = tmp_path / "dupes.jsonl"
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(DatasetError, match="duplicate task_id"):
        load_dataset(path)


def test_split_is_714_train_and_250_test() -> None:
    questions = load_dataset(DEFAULT_DATA_PATH)
    split = split_questions(questions, seed=1234)
    assert len(split.train) == TRAIN_SIZE == 714
    assert len(split.test) == TEST_SIZE == 250
    assert TRAIN_SIZE + TEST_SIZE == len(questions)


def test_split_is_disjoint_and_covers_every_question() -> None:
    questions = load_dataset(DEFAULT_DATA_PATH)
    split = split_questions(questions, seed=1234)
    train_ids = {question.task_id for question in split.train}
    test_ids = {question.task_id for question in split.test}
    assert train_ids.isdisjoint(test_ids)
    assert train_ids | test_ids == {question.task_id for question in questions}


def test_split_is_deterministic_across_calls() -> None:
    questions = load_dataset(DEFAULT_DATA_PATH)
    first = split_questions(questions, seed=1234)
    second = split_questions(list(reversed(questions)), seed=1234)
    assert [question.task_id for question in first.train] == [
        question.task_id for question in second.train
    ]
    assert [question.task_id for question in first.test] == [
        question.task_id for question in second.test
    ]


def test_a_different_seed_gives_a_different_split() -> None:
    questions = load_dataset(DEFAULT_DATA_PATH)
    first = split_questions(questions, seed=1234)
    second = split_questions(questions, seed=99)
    assert [question.task_id for question in first.train] != [
        question.task_id for question in second.train
    ]


def test_split_rejects_the_wrong_number_of_questions() -> None:
    questions = load_dataset(DEFAULT_DATA_PATH)[:10]
    with pytest.raises(DatasetError, match="expected"):
        split_questions(questions, seed=1)


def test_every_arm_reads_the_same_split_for_the_same_seed() -> None:
    first = load_split(DEFAULT_DATA_PATH, seed=1234)
    second = load_split(DEFAULT_DATA_PATH, seed=1234)
    assert [q.task_id for q in first.train] == [q.task_id for q in second.train]
    assert [q.task_id for q in first.test] == [q.task_id for q in second.test]


def test_prompt_format_is_fixed_and_contains_the_problem() -> None:
    questions = load_dataset(DEFAULT_DATA_PATH)
    question = questions[0]
    prompt = format_prompt(question)
    assert prompt == PROMPT_TEMPLATE.format(text=question.text)
    assert question.text in prompt
    assert "Python function" in prompt


def test_prompt_for_question_uses_a_stable_id() -> None:
    questions = load_dataset(DEFAULT_DATA_PATH)
    prompt = prompt_for_question(questions[0])
    assert prompt.prompt_id == f"mbpp-{questions[0].task_id}"
    assert prompt.problem == format_prompt(questions[0])


def test_prompts_for_questions_is_the_one_formatting_path() -> None:
    questions = load_dataset(DEFAULT_DATA_PATH)[:3]
    prompts = prompts_for_questions(questions)
    assert [prompt.prompt_id for prompt in prompts] == [
        f"mbpp-{question.task_id}" for question in questions
    ]
    assert [prompt.problem for prompt in prompts] == [
        format_prompt(question) for question in questions
    ]
