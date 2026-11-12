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

EVAL_CONFIG_TEXT = """\
[model]
name = "Qwen/Qwen3-0.6B"
revision = "r"

[judge]
model = "fake-judge-model"

[data]
split_seed = 1

[fake]
steps = 4
prompts_per_step = 1

[run]
eval_every_steps = 2
eval_questions = 3
eval_subset_seed = 77
"""

FINAL_EVAL_CONFIG_TEXT = """\
[model]
name = "Qwen/Qwen3-0.6B"
revision = "r"

[judge]
model = "fake-judge-model"

[data]
split_seed = 1

[fake]
steps = 1
prompts_per_step = 1

[run]
eval_every_steps = 100
eval_questions = 100
eval_subset_seed = 77
"""

SIX_STEP_CONFIG_TEXT = """\
[model]
name = "Qwen/Qwen3-0.6B"
revision = "r"

[judge]
model = "fake-judge-model"

[data]
split_seed = 1

[fake]
steps = 6
prompts_per_step = 6
"""


class StubGenerator:
    def __init__(self, codes: dict[str, str] | None = None) -> None:
        self.codes = codes or {}
        self.calls: list[tuple[str, float]] = []
        self.update_calls: list[tuple[str, tuple[float, ...]]] = []

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

    def update(
        self,
        prompt: Prompt,
        generations: list[Generation],
        rewards: list[float],
    ) -> None:
        self.update_calls.append((prompt.prompt_id, tuple(rewards)))

    def save_checkpoint(self, directory) -> None:
        pass

    def load_checkpoint(self, directory) -> None:
        pass


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
        self.checkpoints_dir = self.run_dir / "checkpoints"

    def write_meta(self, meta: dict) -> None:
        self.meta = meta

    def append_rollout(self, record: RolloutRecord) -> None:
        self.rollouts.append(record)

    def append_metric(self, row: dict) -> None:
        self.metrics.append(row)

    def write_eval(self, payload: dict) -> None:
        self.eval = payload


class MultiEvalStore(RecordingStore):
    """Keeps every eval payload, to show eval.json is rewritten per evaluation."""

    def __init__(self) -> None:
        super().__init__()
        self.evals: list[dict] = []

    def write_eval(self, payload: dict) -> None:
        super().write_eval(payload)
        self.evals.append(payload)


def make_prompts(count: int = 2) -> list[Prompt]:
    return [Prompt(prompt_id=f"mbpp-{i:04d}", problem=f"problem {i}") for i in range(count)]


def build(
    arm: str,
    tmp_path,
    *,
    generator=None,
    judge=None,
    runner=None,
    store=None,
    prompts=None,
    eval_prompts=None,
    config_text: str = CONFIG_TEXT,
    resume_step: int = 0,
    prior_evaluations=None,
):
    config = load_config(_write_config(tmp_path, config_text))
    training = prompts if prompts is not None else make_prompts()
    return Experiment(
        arm=arm,
        config=config,
        generator=generator or StubGenerator(),
        judge=judge or ExplodingJudge(),
        test_runner=runner or StubTestRunner(),
        store=store or RecordingStore(),
        prompts=training,
        # Unit tests default the eval pool to the training prompts. The
        # Experiment itself has no such fallback, and the CLI passes the real
        # held-out split.
        eval_prompts=training if eval_prompts is None else eval_prompts,
        resume_step=resume_step,
        prior_evaluations=prior_evaluations or (),
    )


def _write_config(tmp_path, text: str = CONFIG_TEXT) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_run_covers_the_injected_backends_and_returns_counts(tmp_path) -> None:
    store = RecordingStore()
    result = build("code-score", tmp_path, store=store).run()
    assert result.steps == 2
    assert result.rollouts == 16
    assert len(store.rollouts) == 16
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
    assert [r.tests_passed for r in store.rollouts] == [1] * 16
    assert [r.tests_total for r in store.rollouts] == [4] * 16
    assert [r.reward for r in store.rollouts] == pytest.approx([0.25] * 16)


def test_reward_is_one_when_every_test_passes(tmp_path) -> None:
    runner = StubTestRunner(total=3, passes=3)
    store = RecordingStore()
    build("code-score", tmp_path, runner=runner, store=store).run()
    assert [r.reward for r in store.rollouts] == [1.0] * 16


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
    assert result.rollouts == 16
    assert len(judge.output_calls) == 16
    assert judge.process_calls == []
    assert [record.reward for record in store.rollouts] == pytest.approx(
        [expected] * 16
    )
    assert [record.judge_score for record in store.rollouts] == [0.5] * 16
    assert [record.process_score for record in store.rollouts] == [None] * 16
    assert [record.code_score for record in store.rollouts] == [1.0] * 16


@pytest.mark.parametrize(
    "arm,expected",
    [("hybrid", 1.0), ("gated-judge", 1.0)],
)
def test_a_judge_score_of_one_gives_the_top_reward(arm, expected, tmp_path) -> None:
    judge = SpyJudge(score=1.0)
    store = RecordingStore()
    build(arm, tmp_path, judge=judge, store=store).run()
    assert [record.reward for record in store.rollouts] == [expected] * 16


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
    assert judge.output_calls == [(prompts[0], "# code for mbpp-0000")] * 16
    assert judge.process_calls == []
    assert secret not in repr(judge.output_calls)
    assert {field.name for field in fields(Prompt)} == {"prompt_id", "problem"}


@pytest.mark.parametrize("arm", ["hybrid", "gated-judge", "gated-process"])
def test_a_cached_judge_score_is_recorded(arm, tmp_path) -> None:
    judge = SpyJudge(score=0.5, cache_hit=True)
    store = RecordingStore()
    build(arm, tmp_path, judge=judge, store=store).run()
    assert [record.cache_hit for record in store.rollouts] == [True] * 16


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

    assert result.rollouts == 16
    assert len(judge.output_calls) == 16
    assert len(judge.process_calls) == 16
    assert [record.judge_score for record in store.rollouts] == [0.5] * 16
    assert [record.process_score for record in store.rollouts] == [1.0] * 16
    assert [record.code_score for record in store.rollouts] == [1.0] * 16
    assert [record.reward for record in store.rollouts] == pytest.approx([0.75] * 16)


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

    assert judge.process_calls == [(prompts[0], "trace for mbpp-0000")] * 16
    assert judge.output_calls == [(prompts[0], "# code for mbpp-0000")] * 16
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

    prompts = make_prompts()
    assert judge.process_calls == [(prompts[0], "")] * 8 + [(prompts[1], "")] * 8
    assert [record.reasoning for record in store.rollouts] == [""] * 16
    assert [record.process_score for record in store.rollouts] == [0.25] * 16
    assert [record.reward for record in store.rollouts] == pytest.approx([0.375] * 16)


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
        config_text=SIX_STEP_CONFIG_TEXT,
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
    runner = StubTestRunner(total=4, passes=1)
    build("code-score", tmp_path, runner=runner, store=store).run()
    row = store.metrics[0]
    assert row["step"] == 1
    assert row["rollouts"] == 8
    assert row["mean_reward"] == pytest.approx(0.25)
    assert row["pass_rate"] == pytest.approx(0.0)
    assert row["mean_entropy"] == pytest.approx(0.5)
    assert row["mean_completion_tokens"] == pytest.approx(20.0)


def test_eval_runs_one_greedy_generation_per_prompt(tmp_path) -> None:
    generator = StubGenerator()
    build("code-score", tmp_path, generator=generator, store=RecordingStore()).run()
    greedy = [call for call in generator.calls if call[1] == 0.0]
    assert greedy == [("mbpp-0000", 0.0), ("mbpp-0001", 0.0)]
    assert len([call for call in generator.calls if call[1] != 0.0]) == 2


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


def test_meta_carries_the_trainer_settings_and_checkpoint_dir(tmp_path) -> None:
    store = RecordingStore()
    build("code-score", tmp_path, store=store).run()
    trainer = store.meta["trainer"]
    assert trainer["method"] == "grpo"
    assert trainer["generations_per_prompt"] == 8
    assert trainer["lora_rank"] == 32
    assert trainer["lora_alpha"] == 64
    assert trainer["learning_rate"] == 1e-5
    assert trainer["clip_epsilon"] == 0.2
    assert trainer["kl_beta"] == 0.0
    assert trainer["batch_size"] == 1
    assert trainer["gradient_accumulation"] == 1
    assert store.meta["checkpoint_dir"].endswith("checkpoints")


def test_unknown_arm_fails_before_any_backend_is_called(tmp_path) -> None:
    generator = StubGenerator()
    with pytest.raises(ValueError, match="arm"):
        build("reward-shaping", tmp_path, generator=generator).run()
    assert generator.calls == []


def test_steps_and_generations_per_prompt_come_from_the_config(tmp_path) -> None:
    store = RecordingStore()
    build("code-score", tmp_path, prompts=make_prompts(3), store=store).run()
    assert len(store.rollouts) == 16
    assert len(store.metrics) == 2


def test_run_is_reusable_and_writes_only_new_records(tmp_path) -> None:
    store = RecordingStore()
    experiment = build("code-score", tmp_path, store=store)
    experiment.run()
    experiment.run()
    assert len(store.rollouts) == 32


def test_every_rollout_records_entropy_and_completion_length(tmp_path) -> None:
    store = RecordingStore()
    build("code-score", tmp_path, store=store).run()
    assert [record.mean_entropy for record in store.rollouts] == [0.5] * 16
    assert [record.completion_tokens for record in store.rollouts] == [20] * 16


def test_a_checkpoint_eval_runs_every_configured_step_and_the_final_eval_covers_the_pool(
    tmp_path,
) -> None:
    store = MultiEvalStore()
    eval_prompts = make_prompts(5)
    build(
        "code-score",
        tmp_path,
        store=store,
        prompts=make_prompts(1),
        eval_prompts=eval_prompts,
        config_text=EVAL_CONFIG_TEXT,
    ).run()

    evaluations = store.eval["evaluations"]
    assert [(point["step"], point["kind"]) for point in evaluations] == [
        (2, "checkpoint"),
        (4, "checkpoint"),
        (4, "final"),
    ]
    assert [point["count"] for point in evaluations] == [3, 3, 5]
    assert evaluations[-1]["prompt_ids"] == [prompt.prompt_id for prompt in eval_prompts]


def test_the_checkpoint_subset_is_fixed_by_its_own_seed(tmp_path) -> None:
    store = MultiEvalStore()
    eval_prompts = make_prompts(5)
    build(
        "code-score",
        tmp_path,
        store=store,
        prompts=make_prompts(1),
        eval_prompts=eval_prompts,
        config_text=EVAL_CONFIG_TEXT,
    ).run()

    evaluations = store.eval["evaluations"]
    checkpoint_ids = evaluations[0]["prompt_ids"]
    assert checkpoint_ids == evaluations[1]["prompt_ids"]
    assert checkpoint_ids != [prompt.prompt_id for prompt in eval_prompts[:3]]
    assert store.eval["subset"] == {
        "method": "seeded_sample",
        "seed": 77,
        "requested": 3,
        "selected": 3,
        "pool_size": 5,
    }


def test_eval_json_has_step_pass_at_1_count_and_timestamp(tmp_path) -> None:
    store = MultiEvalStore()
    build(
        "code-score",
        tmp_path,
        store=store,
        prompts=make_prompts(1),
        eval_prompts=make_prompts(5),
        config_text=EVAL_CONFIG_TEXT,
    ).run()
    assert store.eval["step"] == 4
    assert store.eval["count"] == 5
    assert store.eval["timestamp"] == "20261112T000000Z"
    assert store.eval["pass_at_1"] == pytest.approx(1.0)


def test_eval_json_is_rewritten_after_every_evaluation(tmp_path) -> None:
    store = MultiEvalStore()
    build(
        "code-score",
        tmp_path,
        store=store,
        prompts=make_prompts(1),
        eval_prompts=make_prompts(5),
        config_text=EVAL_CONFIG_TEXT,
    ).run()
    assert [len(payload["evaluations"]) for payload in store.evals] == [1, 2, 3]
    assert [payload["step"] for payload in store.evals] == [2, 4, 4]


def test_the_final_eval_covers_all_250_test_questions(tmp_path) -> None:
    generator = StubGenerator()
    store = MultiEvalStore()
    eval_prompts = make_prompts(250)
    build(
        "code-score",
        tmp_path,
        generator=generator,
        store=store,
        prompts=make_prompts(1),
        eval_prompts=eval_prompts,
        config_text=FINAL_EVAL_CONFIG_TEXT,
    ).run()

    assert store.eval["count"] == 250
    assert store.eval["evaluations"][-1]["pass_at_1"] == pytest.approx(1.0)
    greedy = [call for call in generator.calls if call[1] == 0.0]
    assert len(greedy) == 250
    assert {call[0] for call in greedy} == {prompt.prompt_id for prompt in eval_prompts}


def test_the_experiment_refuses_to_evaluate_without_eval_prompts(tmp_path) -> None:
    generator = StubGenerator()
    store = MultiEvalStore()
    with pytest.raises(ValueError, match="evaluation prompt"):
        build(
            "code-score",
            tmp_path,
            generator=generator,
            store=store,
            eval_prompts=(),
        ).run()
    assert generator.calls == []
    assert store.rollouts == []
    assert store.meta is None


BUDGET_STEPS_CONFIG = """\
[model]
name = "Qwen/Qwen3-0.6B"
revision = "r"

[judge]
model = "fake-judge-model"

[data]
split_seed = 1

[fake]
steps = 4
prompts_per_step = 1

[budget]
max_steps = 2
"""

BUDGET_JUDGE_CONFIG = """\
[model]
name = "Qwen/Qwen3-0.6B"
revision = "r"

[judge]
model = "fake-judge-model"

[data]
split_seed = 1

[fake]
steps = 4
prompts_per_step = 1

[budget]
max_judge_calls = 5
"""


def test_one_update_call_per_step_with_eight_rewards(tmp_path) -> None:
    generator = StubGenerator()
    build("code-score", tmp_path, generator=generator, store=RecordingStore()).run()
    assert [prompt_id for prompt_id, _ in generator.update_calls] == [
        "mbpp-0000",
        "mbpp-0001",
    ]
    assert all(len(rewards) == 8 for _, rewards in generator.update_calls)


def test_update_receives_the_rewards_of_its_group(tmp_path) -> None:
    generator = StubGenerator()
    runner = StubTestRunner(total=4, passes=1)
    build("code-score", tmp_path, generator=generator, runner=runner, store=RecordingStore()).run()
    _, rewards = generator.update_calls[0]
    assert rewards == pytest.approx((0.25,) * 8)


def test_max_steps_budget_stops_the_run_early(tmp_path) -> None:
    store = RecordingStore()
    result = build(
        "code-score", tmp_path, store=store, config_text=BUDGET_STEPS_CONFIG
    ).run()
    assert result.stop_reason == "max_steps"
    assert result.steps == 2
    assert len(store.rollouts) == 16
    assert store.eval["stop_reason"] == "max_steps"


def test_max_judge_calls_budget_stops_the_run_early(tmp_path) -> None:
    judge = SpyJudge(score=0.5)
    store = RecordingStore()
    result = build(
        "hybrid", tmp_path, judge=judge, store=store, config_text=BUDGET_JUDGE_CONFIG
    ).run()
    assert result.stop_reason == "max_judge_calls"
    assert result.steps == 1
    # One full-pass group makes 8 judge calls, crossing the cap of 5.
    assert len(judge.output_calls) == 8


def test_resume_starts_after_the_checkpoint_step_and_skips_meta(tmp_path) -> None:
    store = RecordingStore()
    prior = [
        {
            "step": 1,
            "kind": "checkpoint",
            "pass_at_1": 0.5,
            "count": 2,
            "prompt_ids": ["mbpp-0000", "mbpp-0001"],
            "subset": None,
        }
    ]
    result = build("code-score", tmp_path, store=store, resume_step=1, prior_evaluations=prior).run()
    assert result.steps == 2
    assert len(store.rollouts) == 8
    assert {record.step for record in store.rollouts} == {2}
    # A resume does not rewrite meta.json and carries the prior evaluation over.
    assert store.meta is None
    assert [point["kind"] for point in store.eval["evaluations"]] == [
        "checkpoint",
        "final",
    ]


def test_a_checkpoint_manifest_is_saved_every_configured_steps(tmp_path) -> None:
    from rlvr.checkpoint import find_latest_checkpoint
    from rlvr.fake import FakeGenerator, FakeTestRunner
    from rlvr.runs import RunStore

    config_text = (
        "[model]\nname = \"Qwen/Qwen3-0.6B\"\nrevision = \"r\"\n"
        "[judge]\nmodel = \"fake-judge-model\"\n"
        "[data]\nsplit_seed = 1\n"
        "[fake]\nsteps = 3\nprompts_per_step = 1\n"
        "[run]\ncheckpoint_every_steps = 2\n"
    )
    config = load_config(_write_config(tmp_path, config_text))
    generator = FakeGenerator()
    store = RunStore(tmp_path / "runs", "code-score", timestamp="20261112T000000Z")
    Experiment(
        arm="code-score",
        config=config,
        generator=generator,
        judge=ExplodingJudge(),
        test_runner=FakeTestRunner(),
        store=store,
        prompts=make_prompts(1),
        eval_prompts=make_prompts(1),
    ).run()

    latest = find_latest_checkpoint(store.checkpoints_dir)
    assert latest is not None and latest.step == 2
    assert len(generator.save_calls) == 1
    assert generator.save_calls[0].endswith("step-00000002")
