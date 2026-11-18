"""The vLLM inference backend.

Rollouts are sampled by vLLM. Training stays on the Hugging Face model with the
LoRA adapter, so the two copies of the weights have to agree after every
optimizer step. `VLLMGenerator` is the seam: it implements the same `Generator`
protocol as `QwenGenerator`, samples token ids with the engine, and hands those
ids back to the trainer to build the same `Generation` objects and cache the
same group data the update pass needs.

Weights move as a LoRA adapter. The trainer writes the adapter to disk after
each optimizer step and the engine gets a new `LoRARequest` for it. A future
vLLM version only changes `VLLMEngine`, never the training loop.

vLLM is optional. The import is inside `VLLMEngine.__init__`, so importing this
module on a CPU box is safe. Selecting the engine without the package installed
is a startup error with one message, raised by the CLI.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any, Sequence

from .experiment import Generation, Prompt

__all__ = ["VLLMEngine", "VLLMGenerator"]


class VLLMEngine:
    """A thin wrapper over one `vllm.LLM` instance.

    `generate` samples `n` completions per prompt in one call and returns token
    ids only. Logprobs are never trusted from the engine. The update pass
    recomputes the old logprobs on the training model, so the engine can change
    its sampling stack without touching the ratio.

    Weights move as a LoRA adapter, not as a full model. The trainer writes the
    adapter to disk after each optimizer step and `set_adapter` hands the engine
    a new `LoRARequest` pointing at it. This is vLLM's supported route for
    dynamic adapters. Each request uses a new integer id, so the engine cannot
    keep serving the previous adapter.
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
        self._lora_request: Any = None
        self._adapter_id = 0
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
        outputs = self._llm.generate(
            prompts,
            parameters,
            lora_request=self._lora_request,
            use_tqdm=False,
        )
        return [
            [[int(token) for token in completion.token_ids] for completion in request.outputs]
            for request in outputs
        ]

    def set_adapter(self, path: str | Path) -> None:
        """Point the engine at the trainer's current LoRA adapter.

        The adapter is a PEFT directory on disk. vLLM loads it on first use of
        the new `LoRARequest` and keeps it under a new integer id. The previous
        adapter is dropped so a long run does not accumulate them in GPU memory.
        """
        engine = getattr(self._llm, "llm_engine", None)
        if engine is None:
            raise RuntimeError(
                "this vLLM build has no llm_engine, so the adapter cannot be "
                "handed over after an optimizer step."
            )

        from vllm.lora.request import LoRARequest  # lazy: optional package

        previous = self._adapter_id
        self._adapter_id += 1
        request = LoRARequest(
            f"rlvr-policy-{self._adapter_id}", self._adapter_id, str(path)
        )
        if previous:
            remove = getattr(engine, "remove_lora", None)
            if remove is not None:
                remove(previous)
        add = getattr(engine, "add_lora", None)
        if add is not None:
            add(request)
        self._lora_request = request

    def close(self) -> None:
        """Release the engine and its worker processes."""
        engine = getattr(self, "_llm", None)
        self._llm = None
        self._lora_request = None
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

    `apply_update` runs the one optimizer step for the batch, then writes the
    adapter and hands it to the engine. That is one handoff per optimizer step
    and never per prompt.
    """

    #: Saved adapter directories kept on disk. The engine needs the current one,
    #: the previous one covers a step still in flight.
    KEEP_ADAPTERS = 2

    def __init__(
        self, *, trainer: Any, engine: Any, adapter_dir: str | Path | None = None
    ) -> None:
        self._trainer = trainer
        self._engine = engine
        if adapter_dir is None:
            adapter_dir = tempfile.mkdtemp(prefix="rlvr-lora-")
        self._adapter_root = Path(adapter_dir)
        self._adapter_root.mkdir(parents=True, exist_ok=True)
        self._sync_adapter()

    def _sync_adapter(self) -> None:
        """Write the current adapter and hand a fresh one to the engine."""
        step = int(getattr(self._trainer, "step", 0))
        folder = self._adapter_root / f"step-{step:08d}"
        path = self._trainer.save_lora_adapter(folder)
        self._engine.set_adapter(str(path))
        self._prune_adapters()

    def _prune_adapters(self) -> None:
        """Keep only the newest adapter directories."""
        folders = sorted(path for path in self._adapter_root.iterdir() if path.is_dir())
        for folder in folders[: -self.KEEP_ADAPTERS]:
            shutil.rmtree(folder, ignore_errors=True)

    def generate(
        self, prompt: Prompt, *, n: int, temperature: float
    ) -> list[Generation]:
        """Sample `n` generations for one prompt, the single prompt shortcut."""
        return self.generate_batch([prompt], n=n, temperature=temperature)[0]

    def generate_batch(
        self,
        prompts: Sequence[Prompt],
        *,
        n: int,
        temperature: float,
        thinking_budget: int | None = None,
    ) -> list[list[Generation]]:
        """Sample every prompt group in one engine call.

        The trainer encodes each prompt to token ids and truncates it to the
        configured prompt cap. The engine returns `n` completion token id lists
        per prompt. The trainer turns those rows into the same `Generation`
        objects the Hugging Face path builds and caches the old logprobs, so the
        update pass does not depend on engine logprobs.

        `thinking_budget` is enforced by the Hugging Face path only. vLLM stays
        opt-in until its weight copy is proven, so it is accepted for the
        protocol and unused here.
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

    def apply_update(self) -> float | None:
        """Step the optimizer once, then hand the new adapter to the engine.

        The trainer owns the gradients, so its norm is passed straight back to
        the caller.
        """
        grad_norm = self._trainer.apply_update()
        self._sync_adapter()
        return grad_norm

    def save_checkpoint(self, directory) -> None:
        """Delegate the checkpoint write to the trainer."""
        self._trainer.save_checkpoint(directory)

    def load_checkpoint(self, directory) -> None:
        """Delegate the checkpoint load to the trainer, then resync the engine.

        A resume must not sample the first group with the adapter from before
        the checkpoint, or the ratios would use different weights.
        """
        self._trainer.load_checkpoint(directory)
        self._sync_adapter()
