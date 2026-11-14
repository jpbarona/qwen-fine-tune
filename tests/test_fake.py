"""Tests for the fake backends. The fake path is deterministic and offline."""

from __future__ import annotations

import json

import pytest

from rlvr.config import load_config
from rlvr.experiment import Experiment, Prompt
from rlvr.fake import FakeGenerator, FakeJudge, FakeTestRunner, fake_prompts
from rlvr.runs import RunStore


def test_prompt_ids_are_stable() -> None:
    prompts = fake_prompts(4)
    assert [p.prompt_id for p in prompts] == ["mbpp-0000", "mbpp-0001", "mbpp-0002", "mbpp-0003"]
    assert all(p.problem for p in prompts)


def test_generator_is_deterministic_per_prompt() -> None:
    prompt = Prompt(prompt_id="mbpp-0002", problem="p")
    first = FakeGenerator().generate(prompt, n=1, temperature=0.9)
    second = FakeGenerator().generate(prompt, n=1, temperature=0.9)
    assert first == second


def test_generator_separates_reasoning_from_code() -> None:
    generation = FakeGenerator().generate(Prompt("mbpp-0001", "p"), n=1, temperature=0.9)[0]
    assert generation.reasoning.strip()
    assert "def " in generation.code
    assert generation.reasoning != generation.code


def test_generator_token_counts_are_in_range() -> None:
    generation = FakeGenerator().generate(Prompt("mbpp-0003", "p"), n=1, temperature=0.9)[0]
    assert generation.prompt_tokens > 0
    assert generation.completion_tokens > 0


def test_generator_carries_no_entropy() -> None:
    # Entropy is measured on the update pass, not at generation time.
    generation = FakeGenerator().generate(Prompt("mbpp-0003", "p"), n=1, temperature=0.9)[0]
    assert not hasattr(generation, "mean_entropy")


def test_update_returns_a_deterministic_entropy_only_on_a_measurement_step() -> None:
    prompt = Prompt("mbpp-0003", "p")
    generation = FakeGenerator().generate(prompt, n=1, temperature=0.9)

    plain = FakeGenerator()
    assert plain.update(prompt, generation, [0.0]) is None
    assert plain.update(prompt, generation, [0.0], measure_entropy=False) is None
    assert plain.step == 2

    measured = FakeGenerator()
    first = measured.update(prompt, generation, [0.0], measure_entropy=True)
    assert first is not None
    assert 0.0 <= first < 2.0
    assert FakeGenerator().update(prompt, generation, [0.0], measure_entropy=True) == first


def test_generate_batch_returns_one_group_per_prompt_in_order() -> None:
    prompts = fake_prompts(3)
    groups = FakeGenerator().generate_batch(prompts, n=2, temperature=0.9)

    assert len(groups) == 3
    assert all(len(group) == 2 for group in groups)
    for prompt, group in zip(prompts, groups):
        assert group == FakeGenerator().generate(prompt, n=2, temperature=0.9)
        assert prompt.prompt_id in group[0].code


def test_generate_batch_rejects_an_empty_prompt_list() -> None:
    with pytest.raises(ValueError, match="at least one prompt"):
        FakeGenerator().generate_batch([], n=1, temperature=0.9)


def test_apply_update_counts_one_weight_update_per_call() -> None:
    generator = FakeGenerator()
    prompts = fake_prompts(3)
    for prompt in prompts:
        generations = generator.generate(prompt, n=2, temperature=0.9)
        generator.accumulate(prompt, generations, [0.0, 1.0])
    assert generator.step == 0

    generator.apply_update()

    assert generator.step == 1
    assert generator.update_groups == [3]


def test_apply_update_without_an_accumulated_group_fails() -> None:
    with pytest.raises(RuntimeError, match="no accumulated groups"):
        FakeGenerator().apply_update()


def test_generator_asks_for_n_candidates() -> None:
    generations = FakeGenerator().generate(Prompt("mbpp-0000", "p"), n=3, temperature=0.9)
    assert len(generations) == 3


def test_greedy_and_sampled_generations_differ() -> None:
    generator = FakeGenerator()
    prompt = Prompt("mbpp-0000", "p")
    greedy = generator.generate(prompt, n=1, temperature=0.0)[0]
    sampled = generator.generate(prompt, n=1, temperature=0.9)[0]
    assert greedy.code != sampled.code


def test_thinking_off_drops_the_reasoning_trace_but_keeps_the_code() -> None:
    prompt = Prompt("mbpp-0000", "p")
    generator = FakeGenerator()
    with_thinking = generator.generate(prompt, n=1, temperature=0.9)[0]
    without_thinking = generator.generate(
        prompt, n=1, temperature=0.9, thinking=False
    )[0]
    assert with_thinking.reasoning
    assert without_thinking.reasoning == ""
    assert without_thinking.code == with_thinking.code


def test_test_runner_is_deterministic_and_bounded() -> None:
    runner = FakeTestRunner()
    prompt = Prompt("mbpp-0000", "p")
    code = FakeGenerator().generate(prompt, n=1, temperature=0.9)[0].code
    first = runner.run(prompt, code)
    second = runner.run(prompt, code)
    assert first == second
    assert 0 <= first.tests_passed <= first.tests_total
    assert first.tests_total > 0


def test_test_runner_uses_pass_on_success_and_a_class_on_failure() -> None:
    prompt = Prompt("mbpp-0000", "p")
    results = []
    for index in range(6):
        code = f"def solution():\n    return {index}"
        results.append(FakeTestRunner().run(prompt, code))
    for result in results:
        if result.tests_passed == result.tests_total:
            assert result.error_class == "pass"
        else:
            assert result.error_class in {
                "wrong_answer",
                "syntax_error",
                "runtime_error",
                "timeout",
            }


def test_fake_run_shows_a_spread_of_rewards_across_prompts() -> None:
    runner, generator = FakeTestRunner(), FakeGenerator()
    rewards = set()
    for prompt in fake_prompts(6):
        code = generator.generate(prompt, n=1, temperature=0.9)[0].code
        result = runner.run(prompt, code)
        rewards.add(result.tests_passed / result.tests_total)
    assert len(rewards) > 1


def test_fake_code_score_run_never_judges(run_fake) -> None:
    run_dir, _ = run_fake("code-score")
    records = [json.loads(line) for line in (run_dir / "rollouts.jsonl").read_text().splitlines()]
    assert all(record["judge_score"] is None for record in records)
    assert all(record["process_score"] is None for record in records)
    assert all(record["cache_hit"] is False for record in records)
    for record in records:
        if record["error_class"] == "syntax_error":
            assert record["reward"] == pytest.approx(-0.5)
        else:
            assert record["reward"] == pytest.approx(record["code_score"])


@pytest.mark.parametrize("arm", ["hybrid", "gated-judge"])
def test_fake_judge_arm_run_populates_scores(run_fake, arm: str) -> None:
    run_dir, _ = run_fake(arm)
    records = [
        json.loads(line)
        for line in (run_dir / "rollouts.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert all(record["process_score"] is None for record in records)
    for record in records:
        passed = record["tests_total"] > 0 and record["tests_passed"] == record["tests_total"]
        if passed:
            assert isinstance(record["judge_score"], float)
            if arm == "gated-judge":
                expected = record["judge_score"]
            else:
                expected = 0.5 + 0.5 * record["judge_score"]
            assert record["reward"] == pytest.approx(expected)
        else:
            assert record["judge_score"] is None
            assert record["reward"] == 0.0
    assert any(record["judge_score"] is not None for record in records)
    assert any(record["judge_score"] is None for record in records)


def test_fake_code_score_run_has_a_spread_of_rewards(run_fake) -> None:
    run_dir, _ = run_fake("code-score")
    rewards = [json.loads(line)["reward"] for line in (run_dir / "rollouts.jsonl").read_text().splitlines()]
    assert all(-1.0 <= reward <= 1.0 for reward in rewards)
    assert len(set(rewards)) > 1


def test_fake_code_score_run_writes_a_syntax_penalty(tmp_path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        '[model]\nname = "Qwen/Qwen3-0.6B"\nrevision = "r"\n'
        '[judge]\nmodel = "fake-judge-model"\n'
        "[data]\nsplit_seed = 1\n"
        "[fake]\nsteps = 1\n",
        encoding="utf-8",
    )
    # This prompt's hash maps to the syntax_error class in the fake runner.
    prompt = fake_prompts(7)[6]
    store = RunStore(tmp_path / "runs", "code-score", timestamp="20261112T000000Z")
    Experiment(
        arm="code-score",
        config=load_config(config_path),
        generator=FakeGenerator(),
        judge=FakeJudge(),
        test_runner=FakeTestRunner(),
        store=store,
        prompts=[prompt],
        eval_prompts=[prompt],
    ).run()
    records = [
        json.loads(line)
        for line in (store.run_dir / "rollouts.jsonl").read_text().splitlines()
    ]
    assert {record["error_class"] for record in records} == {"syntax_error"}
    assert all(record["reward"] == pytest.approx(-0.5) for record in records)


def test_judge_scores_are_deterministic_and_bounded() -> None:
    prompt = Prompt("mbpp-0000", "p")
    judge = FakeJudge()
    first = judge.score_output(prompt, "code")
    second = judge.score_output(prompt, "code")
    assert first.score == second.score
    assert 0.0 <= first.score <= 1.0


def test_judge_repeats_hit_the_cache() -> None:
    prompt = Prompt("mbpp-0000", "p")
    judge = FakeJudge()
    assert judge.score_output(prompt, "code").cache_hit is False
    assert judge.score_output(prompt, "code").cache_hit is True
    assert judge.score_output(prompt, "other code").cache_hit is False


def test_process_and_output_judge_caches_are_separate() -> None:
    prompt = Prompt("mbpp-0000", "p")
    judge = FakeJudge()
    judge.score_output(prompt, "same text")
    assert judge.score_process(prompt, "same text").cache_hit is False


def test_judge_scores_differ_between_prompts() -> None:
    judge = FakeJudge()
    scores = {judge.score_output(Prompt(f"mbpp-{i:04d}", "p"), "code").score for i in range(8)}
    assert len(scores) > 1


def test_fakes_touch_no_network(monkeypatch, tmp_path) -> None:
    import socket

    from rlvr.cli import main

    def fail(*args, **kwargs):
        raise AssertionError("network use is not allowed in the fake path")

    monkeypatch.setattr(socket.socket, "connect", fail)
    monkeypatch.setattr(socket, "create_connection", fail)
    code = main(
        ["run", "--arm", "code-score", "--fake", "--runs-root", str(tmp_path / "runs")]
    )
    assert code == 0
