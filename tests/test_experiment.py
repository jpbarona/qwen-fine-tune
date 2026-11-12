"""Tests for the Experiment seam: injected Generator, Judge, TestRunner, Store.

All four arms are covered end to end. The process arm gates on the tests, then
scores the reasoning trace and the code with separate judge calls.
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import pytest

from rlvr.config import load_config
from rlvr.experiment import (
    Experiment,
    Generation,
    JudgeResult,
    Outcome,
    Prompt,
    TestResult,
)
from rlvr.judge import UnrecoverableJudgeError
from rlvr.reward import RewardError
from rlvr.schema import RolloutRecord

CONFIG_TEXT = """\
[model]
name = "Qwen/Qwen3-0.6B"
revision = "r"

[judge]
model = "fake-judge-model"

[data]
split_seed = 1

[fake]
steps = 2
prompts_per_step = 2
"""


class StubGenerator:
    def __init__(self, codes: dict[str, str] | None = None) -> None:
        self.codes = codes or {}
        self.calls: list[tuple[str, float]] = []

    def generate(self, prompt: Prompt, *, n: int, temperature: float) -> list[Generation]:
        self.calls.append((prompt.prompt_id, temperature))
        code = self.codes.get(prompt.prompt_id, f"# code for {prompt.prompt_id}")
        return [
            Generation(
                reasoning=f"trace for {prompt.prompt_id}",
                code=code,
                prompt_tokens=10,
                completion_tokens=20,
                mean_entropy=0.5,
            )
            for _ in range(n)
        ]


class StubTestRunner:
    """Passes every test unless the code contains FAIL.

    A fixed `passes` count overrides that, so partial and full pass counts can
    be driven from a test.
    """

    def __init__(
        self,
        total: int = 2,
        passes: int | None = None,
        error_class: str = "assertion",
    ) -> None:
        self.total = total
        self.passes = passes
        self.error_class = error_class
        self.calls: list[str] = []

    def run(self, prompt: Prompt, code: str) -> Outcome:
        self.calls.append(code)
        if self.passes is not None:
            return Outcome(
                tests_passed=self.passes,
                tests_total=self.total,
                error_class=(
                    "pass"
                    if self.total > 0 and self.passes == self.total
                    else self.error_class
                ),
            )
        if "FAIL" in code:
            return Outcome(tests_passed=0, tests_total=self.total, error_class=self.error_class)
        return Outcome(tests_passed=self.total, tests_total=self.total, error_class="pass")


class ExplodingJudge:
    """A judge that fails the test if the code-score arm calls it at all."""

    def __init__(self) -> None:
        self.calls = 0

    def score_output(self, prompt: Prompt, code: str) -> JudgeResult:
        self.calls += 1
        raise AssertionError("code-score must never call the output judge")

    def score_process(self, prompt: Prompt, reasoning: str) -> JudgeResult:
        self.calls += 1
        raise AssertionError("code-score must never call the process judge")


class SpyJudge:
    """Records every argument it is given, so a test can inspect the call."""

    def __init__(
        self,
        score: float = 0.5,
        cache_hit: bool = False,
        process_score: float | None = None,
    ) -> None:
        self.score = score
        self.cache_hit = cache_hit
        self.process_score = score if process_score is None else process_score
        self.output_calls: list[tuple[Prompt, str]] = []
        self.process_calls: list[tuple[Prompt, str]] = []

    def score_output(self, prompt: Prompt, code: str) -> JudgeResult:
        self.output_calls.append((prompt, code))
        return JudgeResult(score=self.score, cache_hit=self.cache_hit, reason="stub")

    def score_process(self, prompt: Prompt, reasoning: str) -> JudgeResult:
        self.process_calls.append((prompt, reasoning))
        return JudgeResult(
            score=self.process_score, cache_hit=self.cache_hit, reason="stub"
        )


class FailingJudge:
    """Every judge call raises, as an unrecoverable judge failure would."""

    def score_output(self, prompt: Prompt, code: str) -> JudgeResult:
        raise UnrecoverableJudgeError("judge rejected the request")

    def score_process(self, prompt: Prompt, reasoning: str) -> JudgeResult:
        raise UnrecoverableJudgeError("judge rejected the request")


class LeakyRunner:
    """Passes every test and puts a secret marker in the per-test detail."""

    __test__ = False

    def __init__(self, secret: str) -> None:
        self.secret = secret

    def run(self, prompt: Prompt, code: str) -> Outcome:
        return Outcome(
            tests_passed=1,
            tests_total=1,
            error_class="pass",
            results=(
                TestResult(index=0, passed=True, error_class="pass", detail=self.secret),
            ),
        )


class RecordingStore:
    def __init__(self) -> None:
        self.rollouts: list[RolloutRecord] = []
        self.metrics: list[dict] = []
        self.eval: dict | None = None
        self.meta: dict | None = None
        self.timestamp = "20261112T000000Z"
        self.run_dir = Path("runs/test/20261112T000000Z")

    def write_meta(self, meta: dict) -> None:
        self.meta = meta

    def append_rollout(self, record: RolloutRecord) -> None:
        self.rollouts.append(record)

    def append_metric(self, row: dict) -> None:
        self.metrics.append(row)

    def write_eval(self, payload: dict) -> None:
        self.eval = payload


def make_prompts(count: int = 2) -> list[Prompt]:
    return [Prompt(prompt_id=f"mbpp-{i:04d}", problem=f"problem {i}") for i in range(count)]


def build(arm: str, tmp_path, *, generator=None, judge=None, runner=None, store=None, prompts=None):
    config = load_config(_write_config(tmp_path))
    return Experiment(
        arm=arm,
        config=config,
        generator=generator or StubGenerator(),
        judge=judge or ExplodingJudge(),
        test_runner=runner or StubTestRunner(),
        store=store or RecordingStore(),
        prompts=prompts if prompts is not None else make_prompts(),
    )


def _write_config(tmp_path) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(CONFIG_TEXT, encoding="utf-8")
    return path


def test_run_covers_the_injected_backends_and_returns_counts(tmp_path) -> None:
    store = RecordingStore()
    result = build("code-score", tmp_path, store=store).run()
    assert result.steps == 2
    assert result.rollouts == 4
    assert len(store.rollouts) == 4
    assert len(store.metrics) == 2
    assert store.meta is not None and store.meta["arm"] == "code-score"
    assert store.eval is not None


def test_records_are_built_from_generator_runner_and_judge(tmp_path) -> None:
    store = RecordingStore()
    build("code-score", tmp_path, store=store).run()
    record = store.rollouts[0]
    assert record.arm == "code-score"
    assert record.step == 1
    assert record.prompt_id == "mbpp-0000"
    assert record.reasoning == "trace for mbpp-0000"
    assert record.code == "# code for mbpp-0000"
    assert record.prompt_tokens == 10
    assert record.completion_tokens == 20
    assert record.mean_entropy == 0.5
    assert record.tests_passed == 2 and record.tests_total == 2
    assert record.error_class == "pass"
    assert record.code_score == 1.0


def test_code_score_arm_never_calls_the_judge(tmp_path) -> None:
    judge = ExplodingJudge()
    store = RecordingStore()
    build("code-score", tmp_path, judge=judge, store=store).run()
    assert judge.calls == 0
    for record in store.rollouts:
        assert record.judge_score is None
        assert record.process_score is None
        assert record.cache_hit is False
        assert record.reward == record.code_score


def test_reward_is_zero_when_every_test_fails(tmp_path) -> None:
    generator = StubGenerator(codes={"mbpp-0000": "FAIL"})
    store = RecordingStore()
    build("code-score", tmp_path, generator=generator, store=store).run()
    failed = [r for r in store.rollouts if r.prompt_id == "mbpp-0000"]
    assert {r.reward for r in failed} == {0.0}
    assert {r.code_score for r in failed} == {0.0}
    assert failed[0].error_class == "assertion"


def test_reward_is_partial_when_some_tests_fail(tmp_path) -> None:
    runner = StubTestRunner(total=4, passes=1)
    store = RecordingStore()
    build("code-score", tmp_path, runner=runner, store=store).run()
    assert [r.tests_passed for r in store.rollouts] == [1, 1, 1, 1]
    assert [r.tests_total for r in store.rollouts] == [4, 4, 4, 4]
    assert [r.reward for r in store.rollouts] == pytest.approx([0.25] * 4)


def test_reward_is_one_when_every_test_passes(tmp_path) -> None:
    runner = StubTestRunner(total=3, passes=3)
    store = RecordingStore()
    build("code-score", tmp_path, runner=runner, store=store).run()
    assert [r.reward for r in store.rollouts] == [1.0] * 4


def test_a_zero_test_rollout_fails_loudly(tmp_path) -> None:
    runner = StubTestRunner(total=0, passes=0)
    store = RecordingStore()
    with pytest.raises(RewardError, match="zero tests"):
        build("code-score", tmp_path, runner=runner, store=store).run()
    assert store.rollouts == []


@pytest.mark.parametrize(
    "arm,expected",
    [("hybrid", 0.75), ("gated-judge", 0.5)],
)
def test_judge_arms_on_a_full_pass_use_the_judge_score(arm, expected, tmp_path) -> None:
    judge = SpyJudge(score=0.5)
    store = RecordingStore()
    result = build(arm, tmp_path, judge=judge, store=store).run()
    assert result.rollouts == 4
    assert len(judge.output_calls) == 4
    assert judge.process_calls == []
    assert [record.reward for record in store.rollouts] == pytest.approx(
        [expected] * 4
    )
    assert [record.judge_score for record in store.rollouts] == [0.5] * 4
    assert [record.process_score for record in store.rollouts] == [None] * 4
    assert [record.code_score for record in store.rollouts] == [1.0] * 4


@pytest.mark.parametrize(
    "arm,expected",
    [("hybrid", 1.0), ("gated-judge", 1.0)],
)
def test_a_judge_score_of_one_gives_the_top_reward(arm, expected, tmp_path) -> None:
    judge = SpyJudge(score=1.0)
    store = RecordingStore()
    build(arm, tmp_path, judge=judge, store=store).run()
    assert [record.reward for record in store.rollouts] == [expected] * 4


@pytest.mark.parametrize("arm", ["hybrid", "gated-judge"])
@pytest.mark.parametrize("passed,total", [(0, 2), (1, 2)])
@pytest.mark.parametrize(
    "error_class", ["wrong_answer", "syntax_error", "runtime_error", "runner_error"]
)
def test_gated_arms_score_zero_and_skip_the_judge_on_a_failed_test(
    arm, passed, total, error_class, tmp_path
) -> None:
    judge = SpyJudge()
    runner = StubTestRunner(total=total, passes=passed, error_class=error_class)
    store = RecordingStore()
    build(arm, tmp_path, judge=judge, runner=runner, store=store).run()
    assert judge.output_calls == []
    assert {record.error_class for record in store.rollouts} == {error_class}
    assert {record.reward for record in store.rollouts} == {0.0}
    assert {record.judge_score for record in store.rollouts} == {None}
    assert {record.code_score for record in store.rollouts} == {passed / total}


@pytest.mark.parametrize("arm", ["hybrid", "gated-judge"])
def test_gated_arms_score_zero_and_skip_the_judge_on_a_timeout(arm, tmp_path) -> None:
    judge = SpyJudge()
    runner = StubTestRunner(total=3, passes=0, error_class="timeout")
    store = RecordingStore()
    build(arm, tmp_path, judge=judge, runner=runner, store=store).run()
    assert judge.output_calls == []
    assert {record.error_class for record in store.rollouts} == {"timeout"}
    assert {record.reward for record in store.rollouts} == {0.0}
    assert {record.judge_score for record in store.rollouts} == {None}


@pytest.mark.parametrize("arm", ["hybrid", "gated-judge"])
def test_gated_arms_score_zero_and_skip_the_judge_when_there_are_no_tests(
    arm, tmp_path
) -> None:
    judge = SpyJudge()
    runner = StubTestRunner(total=0, passes=0, error_class="no_tests")
    store = RecordingStore()
    build(arm, tmp_path, judge=judge, runner=runner, store=store).run()
    assert judge.output_calls == []
    assert {record.error_class for record in store.rollouts} == {"no_tests"}
    assert {record.reward for record in store.rollouts} == {0.0}
    assert {record.code_score for record in store.rollouts} == {0.0}
    assert {record.judge_score for record in store.rollouts} == {None}


def test_the_output_judge_sees_the_problem_and_code_only(tmp_path) -> None:
    secret = "SECRET_TEST_CODE_NEVER_PASS_THIS_TO_THE_JUDGE"
    judge = SpyJudge(score=1.0)
    prompts = make_prompts(1)
    store = RecordingStore()
    build(
        "hybrid",
        tmp_path,
        judge=judge,
        runner=LeakyRunner(secret),
        store=store,
        prompts=prompts,
    ).run()

    # The judge is called once per training rollout with the prompt and the
    # generated code, and with nothing else. The eval pass does not judge.
    assert judge.output_calls == [(prompts[0], "# code for mbpp-0000")] * 2
    assert judge.process_calls == []
    assert secret not in repr(judge.output_calls)
    assert {field.name for field in fields(Prompt)} == {"prompt_id", "problem"}


@pytest.mark.parametrize("arm", ["hybrid", "gated-judge", "gated-process"])
def test_a_cached_judge_score_is_recorded(arm, tmp_path) -> None:
    judge = SpyJudge(score=0.5, cache_hit=True)
    store = RecordingStore()
    build(arm, tmp_path, judge=judge, store=store).run()
    assert [record.cache_hit for record in store.rollouts] == [True] * 4


@pytest.mark.parametrize("arm", ["hybrid", "gated-judge", "gated-process"])
def test_an_unrecoverable_judge_error_stops_the_run(arm, tmp_path) -> None:
    store = RecordingStore()
    with pytest.raises(UnrecoverableJudgeError, match="judge rejected"):
        build(arm, tmp_path, judge=FailingJudge(), store=store).run()
    assert store.rollouts == []


def test_gated_process_on_a_full_pass_uses_both_judges(tmp_path) -> None:
    judge = SpyJudge(score=0.5, process_score=1.0)
    store = RecordingStore()
    result = build("gated-process", tmp_path, judge=judge, store=store).run()

    assert result.rollouts == 4
    assert len(judge.output_calls) == 4
    assert len(judge.process_calls) == 4
    assert [record.judge_score for record in store.rollouts] == [0.5] * 4
    assert [record.process_score for record in store.rollouts] == [1.0] * 4
    assert [record.code_score for record in store.rollouts] == [1.0] * 4
    assert [record.reward for record in store.rollouts] == pytest.approx([0.75] * 4)


def test_the_process_judge_sees_the_problem_and_trace_only(tmp_path) -> None:
    secret = "SECRET_TEST_CODE_NEVER_PASS_THIS_TO_THE_JUDGE"
    judge = SpyJudge(score=1.0, process_score=1.0)
    prompts = make_prompts(1)
    store = RecordingStore()
    build(
        "gated-process",
        tmp_path,
        judge=judge,
        runner=LeakyRunner(secret),
        store=store,
        prompts=prompts,
    ).run()

    assert judge.process_calls == [(prompts[0], "trace for mbpp-0000")] * 2
    assert judge.output_calls == [(prompts[0], "# code for mbpp-0000")] * 2
    assert secret not in repr(judge.process_calls)
    assert secret not in repr(judge.output_calls)


@pytest.mark.parametrize("passed,total", [(0, 2), (1, 2)])
@pytest.mark.parametrize(
    "error_class", ["wrong_answer", "syntax_error", "runtime_error", "runner_error"]
)
def test_gated_process_scores_zero_and_skips_both_judges_on_a_failed_test(
    passed, total, error_class, tmp_path
) -> None:
    judge = SpyJudge()
    runner = StubTestRunner(total=total, passes=passed, error_class=error_class)
    store = RecordingStore()
    build("gated-process", tmp_path, judge=judge, runner=runner, store=store).run()

    assert judge.output_calls == []
    assert judge.process_calls == []
    assert {record.error_class for record in store.rollouts} == {error_class}
    assert {record.reward for record in store.rollouts} == {0.0}
    assert {record.judge_score for record in store.rollouts} == {None}
    assert {record.process_score for record in store.rollouts} == {None}
    assert {record.code_score for record in store.rollouts} == {passed / total}


def test_gated_process_scores_zero_and_skips_both_judges_on_a_timeout(
    tmp_path,
) -> None:
    judge = SpyJudge()
    runner = StubTestRunner(total=3, passes=0, error_class="timeout")
    store = RecordingStore()
    build("gated-process", tmp_path, judge=judge, runner=runner, store=store).run()

    assert judge.output_calls == []
    assert judge.process_calls == []
    assert {record.error_class for record in store.rollouts} == {"timeout"}
    assert {record.reward for record in store.rollouts} == {0.0}
    assert {record.judge_score for record in store.rollouts} == {None}
    assert {record.process_score for record in store.rollouts} == {None}


def test_gated_process_scores_zero_and_skips_both_judges_when_there_are_no_tests(
    tmp_path,
) -> None:
    judge = SpyJudge()
    runner = StubTestRunner(total=0, passes=0, error_class="no_tests")
    store = RecordingStore()
    build("gated-process", tmp_path, judge=judge, runner=runner, store=store).run()

    assert judge.output_calls == []
    assert judge.process_calls == []
    assert {record.error_class for record in store.rollouts} == {"no_tests"}
    assert {record.reward for record in store.rollouts} == {0.0}
    assert {record.code_score for record in store.rollouts} == {0.0}
    assert {record.judge_score for record in store.rollouts} == {None}
    assert {record.process_score for record in store.rollouts} == {None}


class EmptyTraceGenerator(StubGenerator):
    """Returns code with an empty reasoning trace."""

    def generate(self, prompt: Prompt, *, n: int, temperature: float) -> list[Generation]:
        return [
            Generation(
                reasoning="",
                code=f"# code for {prompt.prompt_id}",
                prompt_tokens=10,
                completion_tokens=20,
                mean_entropy=0.5,
            )
            for _ in range(n)
        ]


def test_an_empty_trace_is_a_valid_judge_input(tmp_path) -> None:
    judge = SpyJudge(score=0.5, process_score=0.25)
    store = RecordingStore()
    build(
        "gated-process",
        tmp_path,
        generator=EmptyTraceGenerator(),
        judge=judge,
        store=store,
    ).run()

    assert judge.process_calls == [
        (prompt, "") for prompt in make_prompts()
    ] * 2
    assert [record.reasoning for record in store.rollouts] == [""] * 4
    assert [record.process_score for record in store.rollouts] == [0.25] * 4
    assert [record.reward for record in store.rollouts] == pytest.approx([0.375] * 4)


def test_a_fake_gated_process_run_populates_every_record_field(tmp_path) -> None:
    from rlvr.fake import FakeGenerator, FakeJudge, FakeTestRunner, fake_prompts

    prompts = fake_prompts(6)
    store = RecordingStore()
    build(
        "gated-process",
        tmp_path,
        generator=FakeGenerator(),
        judge=FakeJudge(),
        runner=FakeTestRunner(),
        store=store,
        prompts=prompts,
    ).run()

    passing = [
        record
        for record in store.rollouts
        if record.tests_total > 0 and record.tests_passed == record.tests_total
    ]
    assert passing, "the fake run needs at least one full pass"
    for record in store.rollouts:
        assert record.arm == "gated-process"
        assert record.reasoning
        assert record.code
        assert isinstance(record.mean_entropy, float)
        assert record.tests_passed <= record.tests_total
        if record.tests_passed == record.tests_total and record.tests_total > 0:
            assert isinstance(record.process_score, float)
            assert isinstance(record.judge_score, float)
            assert record.reward == pytest.approx(
                0.5 * record.process_score + 0.5 * record.judge_score
            )
        else:
            assert record.process_score is None
            assert record.judge_score is None
            assert record.reward == 0.0


def test_metrics_rows_summarize_the_step(tmp_path) -> None:
    store = RecordingStore()
    generator = StubGenerator(codes={"mbpp-0000": "FAIL"})
    build("code-score", tmp_path, generator=generator, store=store).run()
    row = store.metrics[0]
    assert row["step"] == 1
    assert row["rollouts"] == 2
    assert row["mean_reward"] == pytest.approx(0.5)
    assert row["pass_rate"] == pytest.approx(0.5)
    assert row["mean_entropy"] == pytest.approx(0.5)
    assert row["mean_completion_tokens"] == pytest.approx(20.0)


def test_eval_runs_one_greedy_generation_per_prompt(tmp_path) -> None:
    generator = StubGenerator()
    build("code-score", tmp_path, generator=generator, store=RecordingStore()).run()
    greedy = [call for call in generator.calls if call[1] == 0.0]
    assert greedy == [("mbpp-0000", 0.0), ("mbpp-0001", 0.0)]
    assert len([call for call in generator.calls if call[1] != 0.0]) == 4


def test_eval_pass_at_1_counts_prompts_that_pass_every_test(tmp_path) -> None:
    generator = StubGenerator(codes={"mbpp-0001": "FAIL"})
    store = RecordingStore()
    result = build("code-score", tmp_path, generator=generator, store=store).run()
    assert store.eval["pass_at_1"] == pytest.approx(0.5)
    assert store.eval["evaluations"][0]["pass_at_1"] == pytest.approx(0.5)
    assert result.pass_at_1 == pytest.approx(0.5)


def test_meta_carries_the_config_and_run_identity(tmp_path) -> None:
    store = RecordingStore()
    build("code-score", tmp_path, store=store).run()
    assert store.meta["arm"] == "code-score"
    assert store.meta["timestamp"] == "20261112T000000Z"
    assert store.meta["judge_model"] == "fake-judge-model"
    assert store.meta["split_seed"] == 1
    assert store.meta["config"]["fake"]["steps"] == 2


def test_unknown_arm_fails_before_any_backend_is_called(tmp_path) -> None:
    generator = StubGenerator()
    with pytest.raises(ValueError, match="arm"):
        build("reward-shaping", tmp_path, generator=generator).run()
    assert generator.calls == []


def test_steps_and_prompt_count_come_from_the_config(tmp_path) -> None:
    store = RecordingStore()
    build("code-score", tmp_path, prompts=make_prompts(3), store=store).run()
    assert len(store.rollouts) == 6
    assert len(store.metrics) == 2


def test_run_is_reusable_and_writes_only_new_records(tmp_path) -> None:
    store = RecordingStore()
    experiment = build("code-score", tmp_path, store=store)
    experiment.run()
    experiment.run()
    assert len(store.rollouts) == 8
