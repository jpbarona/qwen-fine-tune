"""Tests for the fake backends. The fake path is deterministic and offline."""

from __future__ import annotations

import json

from rlvr.experiment import Prompt
from rlvr.fake import FakeGenerator, FakeJudge, FakeTestRunner, fake_prompts


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


def test_generator_token_counts_and_entropy_are_in_range() -> None:
    generation = FakeGenerator().generate(Prompt("mbpp-0003", "p"), n=1, temperature=0.9)[0]
    assert generation.prompt_tokens > 0
    assert generation.completion_tokens > 0
    assert 0.0 <= generation.mean_entropy < 2.0


def test_generator_asks_for_n_candidates() -> None:
    generations = FakeGenerator().generate(Prompt("mbpp-0000", "p"), n=3, temperature=0.9)
    assert len(generations) == 3


def test_greedy_and_sampled_generations_differ() -> None:
    generator = FakeGenerator()
    prompt = Prompt("mbpp-0000", "p")
    greedy = generator.generate(prompt, n=1, temperature=0.0)[0]
    sampled = generator.generate(prompt, n=1, temperature=0.9)[0]
    assert greedy.code != sampled.code


def test_test_runner_is_deterministic_and_bounded() -> None:
    runner = FakeTestRunner()
    prompt = Prompt("mbpp-0000", "p")
    code = FakeGenerator().generate(prompt, n=1, temperature=0.9)[0].code
    first = runner.run(prompt, code)
    second = runner.run(prompt, code)
    assert first == second
    assert 0 <= first.tests_passed <= first.tests_total
    assert first.tests_total > 0


def test_test_runner_reports_an_error_class_only_on_failure() -> None:
    prompt = Prompt("mbpp-0000", "p")
    results = []
    for index in range(6):
        code = f"def solution():\n    return {index}"
        results.append(FakeTestRunner().run(prompt, code))
    for result in results:
        if result.tests_passed == result.tests_total:
            assert result.error_class == ""
        else:
            assert result.error_class in {"assertion", "syntax", "runtime", "timeout"}


def test_fake_run_shows_a_spread_of_rewards_across_prompts() -> None:
    runner, generator = FakeTestRunner(), FakeGenerator()
    rewards = set()
    for prompt in fake_prompts(6):
        code = generator.generate(prompt, n=1, temperature=0.9)[0].code
        result = runner.run(prompt, code)
        rewards.add(result.tests_passed / result.tests_total)
    assert len(rewards) > 1


def test_fake_gated_run_records_cache_hits_after_the_first_step(run_fake) -> None:
    run_dir, _ = run_fake("gated-judge")
    records = [json.loads(line) for line in (run_dir / "rollouts.jsonl").read_text().splitlines()]
    judged = [record for record in records if record["judge_score"] is not None]
    assert judged, "expected at least one rollout judged"
    first = [record for record in judged if record["step"] == 1]
    later = [record for record in judged if record["step"] > 1]
    assert first and later, "expected judged records in the first step and in a later step"
    assert all(record["cache_hit"] is False for record in first)
    assert all(record["cache_hit"] is True for record in later)


def test_fake_code_score_run_never_judges(run_fake) -> None:
    run_dir, _ = run_fake("code-score")
    records = [json.loads(line) for line in (run_dir / "rollouts.jsonl").read_text().splitlines()]
    assert all(record["judge_score"] is None for record in records)
    assert all(record["cache_hit"] is False for record in records)
    assert all(record["reward"] == record["code_score"] for record in records)


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
        ["run", "--arm", "gated-process", "--fake", "--runs-root", str(tmp_path / "runs")]
    )
    assert code == 0
