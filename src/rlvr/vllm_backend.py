"""The vLLM inference backend.

Rollouts are sampled by vLLM. Training stays on the Hugging Face model with the
LoRA adapter, so the two copies of the weights have to agree after every
optimizer step. `VLLMGenerator` is the seam: it implements the same `Generator`
protocol as `QwenGenerator`, samples token ids with the engine, and hands those
ids back to the trainer to build the same `Generation` objects and cache the
same group data the update pass needs.

Weight sync is the risk. The engine exposes one documented method, `sync`, and
the trainer exposes one documented view, `engine_state_dict`. A future vLLM
version only changes `VLLMEngine`, never the training loop.

vLLM is optional. The import is inside `VLLMEngine.__init__`, so importing this
module on a CPU box is safe. Selecting the engine without the package installed
is a startup error with one message, raised by the CLI.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from .experiment import Generation, Prompt

__all__ = ["VLLMEngine", "VLLMGenerator"]


class VLLMEngine:
    """A thin wrapper over one `vllm.LLM` instance.

    `generate` samples `n` completions per prompt in one call and returns token
    ids only. Logprobs are never trusted from the engine. The update pass
    recomputes the old logprobs on the training model, so the engine can change
    its sampling stack without touching the ratio.

    `sync` replaces the engine weights with the trainer's current weights. It
    uses vLLM's `collective_rpc` hook, which reaches every worker process behind
    the engine. A build without the hook raises instead of silently keeping
    stale weights.
    """

    def __init__(
        self,
        *,
        model_name: str,
        revision: str,
        gpu_memory_utilisation: float,
        max_model_len: int,
        max_lora_rank: int,
    ) -> None:
        import vllm  # lazy: the package is optional on a CPU host

        self.model_name = model_name
        self.revision = revision
        self.gpu_memory_utilisation = float(gpu_memory_utilisation)
        self.max_model_len = int(max_model_len)
        self.max_lora_rank = int(max_lora_rank)
        self._vllm = vllm
        self._llm: Any = vllm.LLM(
            model=model_name,
            revision=revision,
            gpu_memory_utilization=self.gpu_memory_utilisation,
            max_model_len=self.max_model_len,
            max_lora_rank=self.max_lora_rank,
            enable_lora=True,
        )

    def generate(
        self,
        prompt_token_ids: Sequence[Sequence[int]],
        *,
        n: int,
        temperature: float,
        max_tokens: int,
    ) -> list[list[list[int]]]:
        """Sample `n` completions for each prompt, one call for the batch.

        Returns one list of `n` token id lists per prompt, in prompt order. The
        prompt tokens are not part of the returned rows, which is the shape the
        trainer's group builder expects.
        """
        if n < 1:
            raise ValueError(f"n must be at least 1, got {n}")
        prompts = [
            self._vllm.TokensPrompt(prompt_token_ids=[int(token) for token in ids])
            for ids in prompt_token_ids
        ]
        parameters = self._vllm.SamplingParams(
            n=n, temperature=temperature, max_tokens=max_tokens
        )
        outputs = self._llm.generate(prompts, parameters, use_tqdm=False)
        return [
            [[int(token) for token in completion.token_ids] for completion in request.outputs]
            for request in outputs
        ]

    def sync(self, state_dict: Iterable[tuple[str, Any]]) -> None:
        """Replace the engine weights with the trainer's current weights.

        `state_dict` is an iterable of name and tensor pairs. The pairs are
        materialised once, so a generator is accepted. Every worker gets the
        same list through one `collective_rpc` call, which is the documented
        route for a weight update that never runs per prompt.
        """
        pairs = [(name, tensor) for name, tensor in state_dict]
        collective_rpc = getattr(self._llm, "collective_rpc", None)
        if collective_rpc is None:
            raise RuntimeError(
                "this vLLM build has no collective_rpc, so the engine weights "
                "cannot be synced after an optimizer step. Install a vLLM build "
                'with collective_rpc, or set engine = "hf" under [training].'
            )
        collective_rpc("update_weights", args=(pairs,))

    def close(self) -> None:
        """Release the engine and its worker processes."""
        engine = getattr(self, "_llm", None)
        self._llm = None
        if engine is None:
            return
        shutdown = getattr(engine, "shutdown", None)
        if shutdown is not None:
            shutdown()


class VLLMGenerator:
    """The `Generator` seam over a vLLM engine and the Hugging Face trainer.

    The trainer is used for encoding, parsing, logprobs, backward and
    checkpoints. The engine is used for sampling only. `generate_batch` samples
    token ids in one engine call, then asks the trainer to build the generation
    objects and cache the group data for the update pass.

    `apply_update` runs the one optimizer step for the batch, then syncs the new
    weights to the engine. That is one sync per optimizer step and never per
    prompt.
    """

    def __init__(self, *, trainer: Any, engine: Any) -> None:
        self._trainer = trainer
        self._engine = engine

    def generate(
        self, prompt: Prompt, *, n: int, temperature: float
    ) -> list[Generation]:
        """Sample `n` generations for one prompt, the single prompt shortcut."""
        return self.generate_batch([prompt], n=n, temperature=temperature)[0]

    def generate_batch(
        self, prompts: Sequence[Prompt], *, n: int, temperature: float
    ) -> list[list[Generation]]:
        """Sample every prompt group in one engine call.

        The trainer encodes each prompt to token ids and truncates it to the
        configured prompt cap. The engine returns `n` completion token id lists
        per prompt. The trainer turns those rows into the same `Generation`
        objects the Hugging Face path builds and caches the old logprobs, so the
        update pass does not depend on engine logprobs.
        """
        prompt_list = list(prompts)
        if not prompt_list:
            raise ValueError("generate_batch needs at least one prompt")
        prompt_token_ids: list[list[int]] = []
        prompt_truncated: list[bool] = []
        for prompt in prompt_list:
            ids, truncated = self._trainer.encode_prompt(prompt)
            prompt_token_ids.append(list(ids))
            prompt_truncated.append(bool(truncated))

        sampled = self._engine.generate(
            prompt_token_ids,
            n=n,
            temperature=temperature,
            max_tokens=self._trainer.max_completion_tokens,
        )
        cache = temperature > 0.0 and getattr(self._trainer, "trainable", True)
        return self._trainer.build_generation_batch(
            prompt_list,
            prompt_token_ids,
            prompt_truncated,
            sampled,
            cache=cache,
        )

    def accumulate(
        self,
        prompt: Prompt,
        generations: Sequence[Generation],
        rewards: Sequence[float],
        *,
        measure_entropy: bool = False,
    ) -> float | None:
        """Delegate the group's backward pass to the trainer."""
        return self._trainer.accumulate(
            prompt,
            generations,
            rewards,
            measure_entropy=measure_entropy,
        )

    def apply_update(self) -> None:
        """Step the optimizer once, then sync the new weights to the engine."""
        self._trainer.apply_update()
        self._engine.sync(self._trainer.engine_state_dict())

    def save_checkpoint(self, directory) -> None:
        """Delegate the checkpoint write to the trainer."""
        self._trainer.save_checkpoint(directory)

    def load_checkpoint(self, directory) -> None:
        """Delegate the checkpoint load to the trainer."""
        self._trainer.load_checkpoint(directory)
