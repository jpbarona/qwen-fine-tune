"""Serve LFM2.5-2.6B with vLLM, load a LoRA adapter, and sample a few tokens.

Run this on the GPU host with the vLLM environment:

    ~/env-vllm/bin/python scripts/smoke_lfm_vllm.py

The script builds a small PEFT-format LoRA adapter from the model's own linear
weight shapes, so it needs no PEFT package. Sampling is greedy, so repeated
samples of the same prompt are comparable. It then:

1. loads the adapter into the engine and samples a few greedy tokens,
2. samples the same prompt without the adapter and compares the chosen token
   logprob, which proves the adapter changed the weights,
3. samples adapter one a second time and requires an identical first token and
   logprob, which proves sampling is deterministic,
4. writes a second adapter and hands it over, then requires its first token or
   logprob to differ from adapter one, which proves the engine reloaded the
   adapter rather than reusing the first one.

Limit: the script drives `VLLMEngine.set_adapter` directly with a synthetic
adapter. It does not call `VLLMGenerator.apply_update`, does not save an adapter
from a trained model, and does not run an optimizer step. The per-step handoff
from the trainer is unproven by this script. Proving it needs a trainer-driven
two-step run on the GPU host.

The adapter is deliberately strong so the logprob difference is visible. It is
a loading test, not a quality test. GPU memory is read before the engine is
built and the utilisation is capped below the free memory, so a training job
already on the card is not disturbed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

MODEL_NAME = "LiquidAI/LFM2.5-2.6B"
REVISION = "7141b56cd4650af95c7c7b86983443aed050c96a"

#: Leave this much free for the running training job and CUDA context growth.
MEMORY_HEADROOM_BYTES = 768 * 1024 * 1024


def linear_shapes(snapshot: Path) -> dict[str, tuple[int, int]]:
    """Every two-dimensional weight in the checkpoint, keyed without `.weight`."""
    from safetensors import safe_open

    index_path = snapshot / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        files = sorted({snapshot / name for name in weight_map.values()})
    else:
        files = [snapshot / "model.safetensors"]
    shapes: dict[str, tuple[int, int]] = {}
    for path in files:
        with safe_open(str(path), framework="pt") as handle:
            for key in handle.keys():
                if not key.endswith(".weight"):
                    continue
                shape = handle.get_slice(key).get_shape()
                if len(shape) != 2:
                    continue
                if "embed_tokens" in key or "lm_head" in key:
                    continue
                shapes[key[: -len(".weight")]] = (int(shape[0]), int(shape[1]))
    return shapes


def build_adapter(
    snapshot: Path,
    folder: Path,
    *,
    rank: int,
    alpha: int,
    seed: int,
    attention_only: bool,
) -> list[str]:
    """Write a PEFT-format adapter and return its target module names."""
    import torch
    from safetensors.torch import save_file

    shapes = linear_shapes(snapshot)
    if attention_only:
        shapes = {
            name: shape
            for name, shape in shapes.items()
            if name.startswith("model.layers.") and ".self_attn." in name
        }
    if not shapes:
        raise RuntimeError("the checkpoint holds no linear weights to adapt")

    generator = torch.Generator().manual_seed(seed)
    tensors = {}
    for name, (out_features, in_features) in shapes.items():
        a = torch.randn(rank, in_features, generator=generator) * 0.1
        b = torch.randn(out_features, rank, generator=generator) * 0.5
        tensors[f"base_model.model.{name}.lora_A.weight"] = a.to(torch.bfloat16)
        tensors[f"base_model.model.{name}.lora_B.weight"] = b.to(torch.bfloat16)

    folder.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(folder / "adapter_model.safetensors"))
    config = {
        "base_model_name_or_path": MODEL_NAME,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "lora_alpha": alpha,
        "lora_dropout": 0.0,
        "peft_type": "LORA",
        "r": rank,
        "target_modules": sorted({name.split(".")[-1] for name in shapes}),
        "task_type": "CAUSAL_LM",
        "use_dora": False,
        "use_rslora": False,
    }
    (folder / "adapter_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    return sorted(shapes)


def snapshot_folder() -> Path:
    """The cached snapshot directory for the pinned model revision."""
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(MODEL_NAME, revision=REVISION))


def sample(llm, prompt_ids, *, lora_request, max_tokens: int):
    """Greedy sample and return the first token logprob and the token ids."""
    from vllm import SamplingParams, TokensPrompt

    params = SamplingParams(
        n=1, temperature=0.0, max_tokens=max_tokens, logprobs=1
    )
    request = llm.generate(
        [TokensPrompt(prompt_token_ids=list(prompt_ids))],
        params,
        lora_request=lora_request,
        use_tqdm=False,
    )[0]
    output = request.outputs[0]
    first = output.logprobs[0] if output.logprobs else {}
    if not first:
        return None, list(output.token_ids)
    token, logprob = next(iter(first.items()))
    return {"token": int(token), "logprob": float(logprob.logprob)}, list(output.token_ids)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--attention-only",
        action="store_true",
        help="adapt attention projections only instead of every linear layer",
    )
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/rlvr-smoke-lora"))
    args = parser.parse_args()

    started = time.monotonic()
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "src"))

    import torch

    from rlvr.engine.vllm_backend import VLLMEngine

    free_bytes, total_bytes = torch.cuda.mem_get_info()
    budget = free_bytes - MEMORY_HEADROOM_BYTES
    if budget <= 0:
        print(
            f"only {free_bytes / 1024**3:.2f} GiB free, headroom is "
            f"{MEMORY_HEADROOM_BYTES / 1024**3:.2f} GiB, refusing to start",
            file=sys.stderr,
        )
        return 2
    utilisation = min(0.40, budget / total_bytes)
    print(
        f"gpu free {free_bytes / 1024**3:.2f} GiB of {total_bytes / 1024**3:.2f} GiB, "
        f"requesting {utilisation:.3f} ({total_bytes * utilisation / 1024**3:.2f} GiB)",
        flush=True,
    )

    snapshot = snapshot_folder()
    adapter_one = args.work_dir / "adapter-one"
    adapter_two = args.work_dir / "adapter-two"
    targets = build_adapter(
        snapshot,
        adapter_one,
        rank=args.rank,
        alpha=args.alpha,
        seed=1,
        attention_only=args.attention_only,
    )
    build_adapter(
        snapshot,
        adapter_two,
        rank=args.rank,
        alpha=args.alpha,
        seed=2,
        attention_only=args.attention_only,
    )
    print(f"adapter targets: {len(targets)} modules", flush=True)

    engine = VLLMEngine(
        model_name=MODEL_NAME,
        revision=REVISION,
        gpu_memory_utilisation=utilisation,
        max_model_len=args.max_model_len,
        max_lora_rank=args.rank,
        seed=1,
        max_num_seqs=1,
        enforce_eager=True,
    )
    engine_ready = time.monotonic()
    print(f"engine ready at {engine_ready - started:.1f}s", flush=True)

    tokenizer = engine._llm.get_tokenizer()
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Write one line of Python that prints hello."}],
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_ids = tokenizer.encode(text, add_special_tokens=False)

    base, base_tokens = sample(
        engine._llm, prompt_ids, lora_request=None, max_tokens=args.max_tokens
    )

    engine.set_adapter(adapter_one)
    one, one_tokens = sample(
        engine._llm,
        prompt_ids,
        lora_request=engine._lora_request,
        max_tokens=args.max_tokens,
    )
    repeat, repeat_tokens = sample(
        engine._llm,
        prompt_ids,
        lora_request=engine._lora_request,
        max_tokens=args.max_tokens,
    )

    engine.set_adapter(adapter_two)
    two, two_tokens = sample(
        engine._llm,
        prompt_ids,
        lora_request=engine._lora_request,
        max_tokens=args.max_tokens,
    )

    finished = time.monotonic()
    if base is None or one is None or two is None or repeat is None:
        print("sampling returned no logprobs", file=sys.stderr)
        return 3
    if base["logprob"] == one["logprob"] and base_tokens == one_tokens:
        print(
            "adapter one did not change the chosen token logprob, so the engine "
            "may have ignored it",
            file=sys.stderr,
        )
        return 4
    if one != repeat or one_tokens != repeat_tokens:
        print(
            "sampling adapter one twice gave different output, so greedy "
            "sampling is not deterministic here",
            file=sys.stderr,
        )
        return 5
    if one["logprob"] == two["logprob"] and one_tokens == two_tokens:
        print(
            "adapter two produced the same output as adapter one, so the engine "
            "did not reload the adapter",
            file=sys.stderr,
        )
        return 6

    result = {
        "gpu_utilisation": round(utilisation, 3),
        "target_modules": len(targets),
        "base_first_token": base,
        "adapter_one_first_token": one,
        "adapter_one_repeat_first_token": repeat,
        "adapter_two_first_token": two,
        "adapter_one_id": 1,
        "adapter_two_id": 2,
        "base_tokens": base_tokens,
        "adapter_one_tokens": one_tokens,
        "adapter_one_repeat_tokens": repeat_tokens,
        "adapter_two_tokens": two_tokens,
        "engine_startup_seconds": round(engine_ready - started, 1),
        "total_seconds": round(finished - started, 1),
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
