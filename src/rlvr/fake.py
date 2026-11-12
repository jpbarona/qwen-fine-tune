"""Fake backends for the offline end-to-end path.

Everything here is deterministic and local. No network, no GPU, no Docker, no
model download. The same command on the same machine produces the same records.

Determinism rule: the fake model output depends on the prompt id and the
decoding mode, and the fake test outcome depends on the prompt id and the code.
The judge cache key is the judge kind plus the prompt id plus the text it scores.
Because the fake code for a prompt does not change between steps, a gated arm
records cache_hit false on the first step and true on later steps.

The values are derived from SHA-256, so they are stable across Python versions
and machines. They carry no meaning. They exist to move data through the seam.
"""

from __future__ import annotations

import hashlib

from .experiment import Generation, JudgeResult, Prompt, Outcome, TRAINING_TEMPERATURE

__all__ = ["FakeGenerator", "FakeJudge", "FakeTestRunner", "fake_prompts"]

#: Error classes the fake runner reports on failure. Success is "pass".
FAKE_ERROR_CLASSES: tuple[str, ...] = ("assertion", "syntax", "runtime", "timeout")

#: Test groups the fake runner uses, so partial passes exist.
FAKE_TEST_COUNTS: tuple[int, ...] = (1, 2, 3, 4)


def _digest(*parts: str) -> bytes:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).digest()


def _value(*parts: str) -> int:
    return int.from_bytes(_digest(*parts)[:8], "big")


def fake_prompts(count: int) -> list[Prompt]:
    """A stable list of fake MBPP-style prompts."""
    if count < 1:
        raise ValueError(f"count must be at least 1, got {count}")
    return [
        Prompt(
            prompt_id=f"mbpp-{index:04d}",
            problem=(
                f"Fake task {index}: return the sum of the integers below {index + 3}. "
                "The tests are hidden from the model."
            ),
        )
        for index in range(count)
    ]


class FakeGenerator:
    """Stands in for the model. Reasoning is kept apart from code.

    With n greater than 1 the fake returns n identical candidates. Task 8 adds
    the group update seam: `update` is a no-op that counts calls, and the
    checkpoint methods are no-ops too, because the fake has no weights.
    """

    def __init__(self) -> None:
        self.update_calls: list[tuple[str, tuple[float, ...]]] = []
        self.save_calls: list[str] = []
        self.load_calls: list[str] = []

    def generate(
        self,
        prompt: Prompt,
        *,
        n: int = 1,
        temperature: float = TRAINING_TEMPERATURE,
    ) -> list[Generation]:
        if n < 1:
            raise ValueError(f"n must be at least 1, got {n}")
        mode = "greedy" if temperature == 0.0 else "sample"
        seed = _value(prompt.prompt_id, mode)
        answer = seed % 100
        reasoning = (
            f"Read the problem. Plan a loop over the range. "
            f"Sum the values and return {answer}. Check the empty case."
        )
        code = (
            f"def solution():\n    # {prompt.prompt_id} {mode}\n"
            f"    total = 0\n    return {answer}\n"
        )
        generation = Generation(
            reasoning=reasoning,
            code=code,
            prompt_tokens=32 + (seed >> 8) % 64,
            completion_tokens=48 + (seed >> 16) % 96,
            mean_entropy=round(((seed >> 24) % 180) / 100 + 0.05, 4),
        )
        return [generation] * n

    def update(
        self,
        prompt: Prompt,
        generations: list[Generation],
        rewards: list[float],
    ) -> None:
        """No-op GRPO update. Records the call so tests can assert the group shape."""
        self.update_calls.append((prompt.prompt_id, tuple(rewards)))

    def save_checkpoint(self, directory) -> None:
        """No-op. The fake has no weights, only the manifest is written."""
        self.save_calls.append(str(directory))

    def load_checkpoint(self, directory) -> None:
        """No-op. The fake has no weights to restore."""
        self.load_calls.append(str(directory))


class FakeTestRunner:
    """Stands in for the Docker runner. Deterministic pass counts per code."""

    def run(self, prompt: Prompt, code: str) -> Outcome:
        seed = _value(prompt.prompt_id, code)
        total = FAKE_TEST_COUNTS[seed % len(FAKE_TEST_COUNTS)]
        passed = (seed >> 8) % (total + 1)
        if passed == total:
            error_class = "pass"
        else:
            error_class = FAKE_ERROR_CLASSES[(seed >> 16) % len(FAKE_ERROR_CLASSES)]
        return Outcome(tests_passed=passed, tests_total=total, error_class=error_class)


class FakeJudge:
    """Stands in for the LLM judge, with a cache that only records hits."""

    def __init__(self) -> None:
        self._seen: set[str] = set()

    def score_output(self, prompt: Prompt, code: str) -> JudgeResult:
        return self._score("output", prompt.prompt_id, code)

    def score_process(self, prompt: Prompt, reasoning: str) -> JudgeResult:
        return self._score("process", prompt.prompt_id, reasoning)

    def _score(self, kind: str, prompt_id: str, text: str) -> JudgeResult:
        key = "\x1f".join((kind, prompt_id, text))
        cache_hit = key in self._seen
        self._seen.add(key)
        return JudgeResult(score=(_value(key) % 1001) / 1000, cache_hit=cache_hit)
