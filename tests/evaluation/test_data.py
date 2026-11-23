"""Tests for the merged MBPP dataset, the fixed split, and the prompt format."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rlvr.evaluation.data import (
    DEFAULT_DATA_PATH,
    DEFAULT_SPLIT_PATH,
    ARITY_PROMPT_TEMPLATE,
    NAMED_PROMPT_TEMPLATE,
    PROMPT_TEMPLATE,
    TOTAL_QUESTIONS,
    TRAIN_SIZE,
    TEST_SIZE,
    DatasetError,
    MBPPQuestion,
    expected_call_arity,
    expected_function_name,
    format_prompt,
    load_dataset,
    load_split,
    pinned_split,
    prompt_id_for,
    prompt_for_question,
    prompts_for_questions,
    split_questions,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _tiny_questions(*prompt_ids: str) -> list[MBPPQuestion]:
    return [
        MBPPQuestion(
            task_id=9000 + index,
            text="Write a function to do a thing.",
            test_list=("assert do_thing() == 1",),
            prompt_id=prompt_id,
        )
        for index, prompt_id in enumerate(prompt_ids)
    ]


def test_committed_dataset_merges_the_three_real_splits() -> None:
    questions = load_dataset(DEFAULT_DATA_PATH)
    assert len(questions) == TOTAL_QUESTIONS == 961
    counts: dict[str, int] = {}
    for question in questions:
        counts[question.source_split] = counts.get(question.source_split, 0) + 1
    assert counts == {"train": 373, "validation": 88, "test": 500}


def test_committed_dataset_has_unique_ids_and_tests() -> None:
    questions = load_dataset(DEFAULT_DATA_PATH)
    assert len({question.task_id for question in questions}) == 961
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


def test_split_is_711_train_and_250_test() -> None:
    questions = load_dataset(DEFAULT_DATA_PATH)
    split = split_questions(questions, seed=1234)
    assert len(split.train) == TRAIN_SIZE == 711
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


def test_pinned_split_matches_the_stored_id_lists() -> None:
    questions = load_dataset(DEFAULT_DATA_PATH)
    stored = json.loads(DEFAULT_SPLIT_PATH.read_text(encoding="utf-8"))
    split = pinned_split(questions, DEFAULT_SPLIT_PATH)
    assert [prompt_id_for(q) for q in split.train] == stored["train"]
    assert [prompt_id_for(q) for q in split.test] == stored["test"]


def test_pinned_split_rejects_a_missing_file(tmp_path: Path) -> None:
    questions = load_dataset(DEFAULT_DATA_PATH)
    with pytest.raises(DatasetError, match="cannot read split file"):
        pinned_split(questions, tmp_path / "missing.json")


def test_pinned_split_rejects_malformed_json(tmp_path: Path) -> None:
    path = tmp_path / "split.json"
    path.write_text("{not json", encoding="utf-8")
    questions = load_dataset(DEFAULT_DATA_PATH)
    with pytest.raises(DatasetError, match="cannot read split file"):
        pinned_split(questions, path)


def test_pinned_split_rejects_a_duplicate_id(tmp_path: Path) -> None:
    path = tmp_path / "split.json"
    path.write_text(json.dumps({"train": ["a", "a"], "test": ["c"]}), encoding="utf-8")
    questions = _tiny_questions("a", "b", "c")
    with pytest.raises(DatasetError, match="repeats an id"):
        pinned_split(questions, path, train_size=2, test_size=1)


def test_pinned_split_rejects_an_overlap_between_the_halves(tmp_path: Path) -> None:
    path = tmp_path / "split.json"
    path.write_text(json.dumps({"train": ["a", "b"], "test": ["b"]}), encoding="utf-8")
    questions = _tiny_questions("a", "b", "c")
    with pytest.raises(DatasetError, match="both halves"):
        pinned_split(questions, path, train_size=2, test_size=1)


def test_pinned_split_rejects_an_id_missing_from_the_dataset() -> None:
    questions = load_dataset(DEFAULT_DATA_PATH)
    stored = json.loads(DEFAULT_SPLIT_PATH.read_text(encoding="utf-8"))
    missing_id = stored["train"][0]
    trimmed = [q for q in questions if prompt_id_for(q) != missing_id]
    with pytest.raises(DatasetError, match="absent from the dataset"):
        pinned_split(trimmed, DEFAULT_SPLIT_PATH)


def test_pinned_split_rejects_wrong_counts(tmp_path: Path) -> None:
    path = tmp_path / "split.json"
    path.write_text(json.dumps({"train": ["a"], "test": ["c"]}), encoding="utf-8")
    questions = _tiny_questions("a", "b", "c")
    with pytest.raises(DatasetError, match="split file must hold"):
        pinned_split(questions, path)


def test_pinned_split_rejects_dataset_rows_outside_the_lists() -> None:
    questions = list(load_dataset(DEFAULT_DATA_PATH))
    questions.append(
        MBPPQuestion(
            task_id=999999,
            text="Write a function to do a thing.",
            test_list=("assert do_thing() == 1",),
            prompt_id="mbpp-999999",
        )
    )
    with pytest.raises(DatasetError, match="dataset ids are absent from the split file"):
        pinned_split(questions, DEFAULT_SPLIT_PATH)


def test_load_split_rejects_a_missing_pinned_file(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="split file not found"):
        load_split(DEFAULT_DATA_PATH, seed=1234, split_path=tmp_path / "missing.json")


def test_prompt_format_is_fixed_and_contains_the_problem() -> None:
    questions = load_dataset(DEFAULT_DATA_PATH)
    question = questions[0]
    prompt = format_prompt(question)
    name = expected_function_name(question)
    arity = expected_call_arity(question)
    noun = "argument" if arity == 1 else "arguments"
    assert prompt == ARITY_PROMPT_TEMPLATE.format(
        name=name, arity=arity, noun=noun, text=question.text
    )
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


def test_expected_function_name_comes_from_the_first_test() -> None:
    question = MBPPQuestion(
        task_id=9001,
        text="Write a function to do a thing.",
        test_list=("assert do_thing(1) == 2", "assert do_thing(3) == 4"),
    )
    assert expected_function_name(question) == "do_thing"


def test_expected_function_name_is_none_when_no_test_calls_a_function() -> None:
    question = MBPPQuestion(
        task_id=9002,
        text="Write a function to do a thing.",
        test_list=("assert 1 == 1",),
    )
    assert expected_function_name(question) is None


def test_prompt_names_the_function_that_the_tests_call() -> None:
    question = MBPPQuestion(
        task_id=9003,
        text="Write a function to do a thing.",
        test_list=("assert do_thing(1) == 2",),
    )
    prompt = format_prompt(question)
    assert "`do_thing`" in prompt
    assert question.text in prompt
    assert "Output only the function definition" in prompt


def test_prompt_omits_a_name_when_the_tests_name_nothing() -> None:
    question = MBPPQuestion(
        task_id=9004,
        text="Write a function to do a thing.",
        test_list=("assert 1 == 1",),
    )
    assert format_prompt(question) == PROMPT_TEMPLATE.format(text=question.text)


def test_almost_every_committed_question_prompt_names_its_function() -> None:
    questions = load_dataset(DEFAULT_DATA_PATH)
    named = [q for q in questions if expected_function_name(q) is not None]
    assert len(named) >= 960
    for question in named:
        assert expected_function_name(question) in format_prompt(question)


def test_name_skips_a_builtin_wrapped_around_the_real_call() -> None:
    question = MBPPQuestion(
        task_id=9005,
        text="Write a python function to return the area of a circle.",
        test_list=(
            "assert abs(circle_area(1) - 3.141592653589793) < 1e-9",
            "assert circle_area(0) == 0",
            "assert abs(circle_area(2) - 12.566370614359172) < 1e-9",
        ),
    )
    assert expected_function_name(question) == "circle_area"


def test_name_keeps_a_builtin_when_the_tests_call_nothing_else() -> None:
    question = MBPPQuestion(
        task_id=9006,
        text="Write a function to find the sum of common divisors.",
        test_list=("assert sum(10, 15) == 6", "assert sum(100, 150) == 93"),
    )
    assert expected_function_name(question) == "sum"


def test_name_ignores_a_helper_that_only_one_test_calls() -> None:
    question = MBPPQuestion(
        task_id=9007,
        text="Write a function to do a thing.",
        test_list=(
            "assert do_thing(1) == 2",
            "assert check(do_thing(3)) == 4",
        ),
    )
    assert expected_function_name(question) == "do_thing"


def test_name_ignores_a_module_attribute() -> None:
    question = MBPPQuestion(
        task_id=9008,
        text="Write a function to do a thing.",
        test_list=("assert math.isclose(do_thing(1), 2)", "assert do_thing(3) == 4"),
    )
    assert expected_function_name(question) == "do_thing"


def test_every_committed_row_names_exactly_one_common_function() -> None:
    """The cold check for the prompt contract, over every row we train on."""
    from rlvr.cli import load_synthetic_rows

    for question in list(load_dataset(DEFAULT_DATA_PATH)) + list(load_synthetic_rows()):
        name = expected_function_name(question)
        assert name is not None, question.task_id
        assert f"`{name}`" in format_prompt(question), question.task_id
        # The same name must be called in every test of the row, otherwise the
        # prompt names a helper and the task cannot pass.
        for test in question.test_list:
            assert name + "(" in test.replace(" ", ""), (question.task_id, test)


#: Rows whose tests build an object the prompt never provides. Task 601 uses a
#: `Pair` class in its tests and no setup code, so the model has to invent the
#: class name and its attributes. Recorded so a new row like it fails the suite.
KNOWN_MISSING_HELPERS = {601: {"Pair"}}


def test_no_row_needs_a_helper_the_prompt_never_provides() -> None:
    import builtins
    import re as _re

    from rlvr.cli import load_synthetic_rows

    call = _re.compile(r"([A-Za-z_]\w*)\s*\(")
    define = _re.compile(r"(?:def|class)\s+(\w+)")
    missing: dict[int, set[str]] = {}
    for question in list(load_dataset(DEFAULT_DATA_PATH)) + list(load_synthetic_rows()):
        name = expected_function_name(question)
        provided = set(define.findall(question.test_setup_code or ""))
        for test in question.test_list:
            clean = _re.sub(r"(\"[^\"]*\"|'[^']*')", '""', test)
            for match in call.finditer(clean):
                called = match.group(1)
                if clean[match.start() - 1 : match.start()] == ".":
                    continue
                if called in (name, "assert") or called in provided:
                    continue
                if hasattr(builtins, called):
                    continue
                missing.setdefault(question.task_id, set()).add(called)
    assert missing == KNOWN_MISSING_HELPERS, missing


def test_every_test_string_compiles() -> None:
    """A test that does not compile can never pass, whatever the model writes."""
    from rlvr.cli import load_synthetic_rows

    for question in list(load_dataset(DEFAULT_DATA_PATH)) + list(load_synthetic_rows()):
        for index, test in enumerate(question.test_list):
            try:
                compile(test, f"test_{index}.py", "exec")
            except SyntaxError as error:
                raise AssertionError(
                    f"task {question.task_id} test {index} does not compile: {error}"
                ) from error


#: Rows whose source data repeats an assertion. Harmless, but recorded so a new
#: one is noticed.
KNOWN_DUPLICATE_TESTS = {37, 337, 508, 713, 862}


def test_no_row_has_blank_tests() -> None:
    from rlvr.cli import load_synthetic_rows

    for question in list(load_dataset(DEFAULT_DATA_PATH)) + list(load_synthetic_rows()):
        assert all(test.strip() for test in question.test_list), question.task_id


def test_only_the_known_rows_repeat_a_test() -> None:
    from rlvr.cli import load_synthetic_rows

    repeated = set()
    for question in list(load_dataset(DEFAULT_DATA_PATH)) + list(load_synthetic_rows()):
        tests = list(question.test_list)
        if len(set(tests)) != len(tests):
            repeated.add(question.task_id)
    assert repeated == KNOWN_DUPLICATE_TESTS


def test_the_prompt_never_carries_a_test_or_an_expected_value() -> None:
    """The model sees the name, never the assertion it has to satisfy."""
    from rlvr.cli import load_synthetic_rows

    for question in list(load_dataset(DEFAULT_DATA_PATH)) + list(load_synthetic_rows()):
        prompt = format_prompt(question)
        for test in question.test_list:
            assert test.strip() not in prompt, question.task_id
        assert "assert" not in prompt.lower(), question.task_id


def test_prompt_ids_are_unique_across_both_sets() -> None:
    from rlvr.cli import load_synthetic_rows

    ids = [
        prompt_id_for(question)
        for question in list(load_dataset(DEFAULT_DATA_PATH)) + list(load_synthetic_rows())
    ]
    assert len(set(ids)) == len(ids)


def test_the_prompt_says_how_many_arguments_the_tests_pass() -> None:
    question = MBPPQuestion(
        task_id=9009,
        text="Write a function to find the top or bottom surface area of a cylinder.",
        test_list=("assert topbottom_surfacearea(10) == 314.15",),
    )
    prompt = format_prompt(question)
    assert "call it with exactly 1 argument" in prompt, prompt


def test_the_arity_counts_keyword_arguments() -> None:
    question = MBPPQuestion(
        task_id=9010,
        text="Write a function to do a thing.",
        test_list=("assert do_thing(1, scale=2) == 3",),
    )
    assert expected_call_arity(question) == 2


def test_the_arity_is_none_when_the_test_does_not_call_the_function() -> None:
    question = MBPPQuestion(
        task_id=9011,
        text="Write a function to do a thing.",
        test_list=("assert 1 == 1",),
    )
    assert expected_call_arity(question) is None


def test_every_row_states_the_arity_of_its_first_test_call() -> None:
    import ast as _ast

    from rlvr.cli import load_synthetic_rows

    for question in list(load_dataset(DEFAULT_DATA_PATH)) + list(load_synthetic_rows()):
        name = expected_function_name(question)
        arity = expected_call_arity(question)
        assert arity is not None, question.task_id
        assert f"call it with exactly {arity} argument" in format_prompt(question), question.task_id
        tree = _ast.parse(question.test_list[0])
        calls = [
            node
            for node in _ast.walk(tree)
            if isinstance(node, _ast.Call)
            and isinstance(node.func, _ast.Name)
            and node.func.id == name
        ]
        assert calls, question.task_id
        assert len(calls[0].args) + len(calls[0].keywords) == arity, question.task_id


def test_row_644_uses_the_name_the_model_writes() -> None:
    rows = {question.task_id: question for question in load_dataset(DEFAULT_DATA_PATH)}
    question = rows[644]
    assert all("reverse_array_up_to_k(" in test for test in question.test_list)
    assert expected_function_name(question) == "reverse_array_up_to_k"


def test_every_prompt_ends_with_one_output_rule() -> None:
    questions = list(load_dataset(DEFAULT_DATA_PATH))
    for question in questions[:5]:
        prompt = format_prompt(question)
        assert prompt.count("Output only the function definition") == 1, prompt
        assert "code fences" in prompt, prompt
        assert "code block" not in prompt.lower(), prompt
