"""Fake backends for the offline end-to-end path.

Everything here is deterministic and local. No network, no GPU, no Docker, no
model download. The same command on the same machine produces the same records.

Determinism rule: the fake model output depends on the prompt id and the
decoding mode, and the fake test outcome depends on the prompt id and the code.
The judge cache key is the judge kind plus the prompt id plus the text it scores.
Because the fake code for a prompt does not change between steps, a gated arm
records cache_hit false on the first step and true on later steps.

Entropy is returned by `update` only when the caller asks for a measurement, so
the fake follows the same `entropy_every_steps` cadence as the real trainer.

The values are derived from SHA-256, so they are stable across Python versions
and machines. They carry no meaning. They exist to move data through the seam.
"""

from __future__ import annotations

import hashlib

from .experiment import Generation, JudgeResult, Prompt, Outcome
from .memory import MemoryReport
from .stages import BANDS

__all__ = [
    "FakeGenerator",
    "FakeJudge",
    "FakeTestRunner",
    "fake_band_labels",
    "fake_memory",
    "fake_prompts",
]

#: Default sampling temperature used by the fake generator's signature. The real
#: generator reads this from config. The fake keeps its own default.
TRAINING_TEMPERATURE = 0.9

#: Error classes the fake runner reports on failure. Success is "pass". The
#: names match the Docker runner so the reward path sees the same classes.
FAKE_ERROR_CLASSES: tuple[str, ...] = (
    "wrong_answer",
    "syntax_error",
    "runtime_error",
    "timeout",
)

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


def fake_band_labels(prompts: list[Prompt] | tuple[Prompt, ...]) -> dict[str, str]:
    """Deterministic bands for a fake run.

    The fake run has to show stage one exhausting and stage two starting. The
    first prompt, in id order, that the fake runner fully passes goes alone
    into the first band. One training step then masters it, the pool empties,
    and the next step is in the second band. The remaining prompts are spread
    round robin over the later bands.

    The result depends only on the prompt ids and the fake code path, so the
    same command always moves stage at the same step.
    """
    ordered = sorted(prompts, key=lambda prompt: prompt.prompt_id)
    if not ordered:
        return {}

    generator, runner = FakeGenerator(), FakeTestRunner()
    first = ordered[0]
    for prompt in ordered:
        generation = generator.generate(
            prompt, n=1, temperature=TRAINING_TEMPERATURE
        )[0]
        outcome = runner.run(prompt, generation.code)
        if outcome.tests_total > 0 and outcome.tests_passed == outcome.tests_total:
            first = prompt
            break

    later_bands = BANDS[1:] or BANDS
    labels = {first.prompt_id: BANDS[0]}
    rest = [prompt for prompt in ordered if prompt.prompt_id != first.prompt_id]
    for index, prompt in enumerate(rest):
        labels[prompt.prompt_id] = later_bands[index % len(later_bands)]
    return labels


def fake_memory() -> MemoryReport:
    """Zeros for the fake path, so the metric row keeps the same shape.

    A machine with no GPU would read the same values. The fake run must not
    import torch, so it supplies the zeros directly rather than asking the
    allocator.
    """
    return MemoryReport.zeros()


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
        thinking: bool = True,
    ) -> list[Generation]:
        if n < 1:
            raise ValueError(f"n must be at least 1, got {n}")
        mode = "greedy" if temperature == 0.0 else "sample"
        seed = _value(prompt.prompt_id, mode)
        answer = seed % 100
        reasoning = (
            f"Read the problem. Plan a loop over the range. "
            f"Sum the values and return {answer}. Check the empty case."
            if thinking
            else ""
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
        )
        return [generation] * n

    def update(
        self,
        prompt: Prompt,
        generations: list[Generation],
        rewards: list[float],
        *,
        measure_entropy: bool = False,
    ) -> float | None:
        """No-op GRPO update. Records the call so tests can assert the group shape.

        On a measurement step the fake returns one deterministic entropy value
        for the prompt, in the same 0.05 to 1.85 band the old per-rollout fake
        values used. Otherwise it returns None and writes no entropy.
        """
        self.update_calls.append((prompt.prompt_id, tuple(rewards)))
        if not measure_entropy:
            return None
        seed = _value(prompt.prompt_id, "entropy")
        return round(((seed >> 24) % 180) / 100 + 0.05, 4)

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

    def score_outputs(self, items: list[tuple[Prompt, str]]) -> list[JudgeResult]:
        return [self.score_output(prompt, text) for prompt, text in items]

    def score_processes(self, items: list[tuple[Prompt, str]]) -> list[JudgeResult]:
        return [self.score_process(prompt, text) for prompt, text in items]

    def _score(self, kind: str, prompt_id: str, text: str) -> JudgeResult:
        key = "\x1f".join((kind, prompt_id, text))
        cache_hit = key in self._seen
        self._seen.add(key)
        return JudgeResult(score=(_value(key) % 1001) / 1000, cache_hit=cache_hit)
