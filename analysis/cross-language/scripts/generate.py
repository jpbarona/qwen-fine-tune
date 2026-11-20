"""Generate greedy completions for the 20-problem cross-language benchmark.

Runs on the GPU box. Builds the repository's QwenGenerator with attention LoRA
(rank 32, alpha 64), optionally loads the run 17 step 300 adapter, then writes
one JSON file per language with the raw completed source for each problem.

The prompt is fed as raw text with no chat template, the standard completion
protocol for HumanEval style benchmarks. One attempt per problem, greedy
decoding. Stop strings come from the MultiPL-E row, plus HumanEval's stops for
Python. Nothing else is cleaned except a single markdown fence.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from transformers import StoppingCriteria, StoppingCriteriaList

REPO = "/home/Ubuntu/rl-grpo/src"
sys.path.insert(0, REPO)

from rlvr.qwen import QwenGenerator  # noqa: E402

MODEL = "Qwen/Qwen3-0.6B"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
ADAPTER = "/home/Ubuntu/qwen-bench/adapter/step-00000300"

PYTHON_STOPS = ["\ndef ", "\nclass ", "\nif __name__", "\nprint(", "\n#", "\n```"]
TEXT = {"python": "Python", "js": "JavaScript", "java": "Java", "cpp": "C++"}


class StopOnText(StoppingCriteria):
    def __init__(self, tokenizer, prompt_length: int, stops: list[str]) -> None:
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.stops = [s for s in stops if s]

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        generated = input_ids[0][self.prompt_length :][-24:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return any(stop in text for stop in self.stops)


def strip_fence(text: str) -> str:
    """Drop a markdown fence but keep leading indentation on the first line."""
    text = text.rstrip()
    if text.lstrip().startswith("```"):
        start = text.find("\n")
        text = text[start + 1 :] if start != -1 else ""
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    return text.rstrip()


def cut_at_stop(text: str, stops: list[str]) -> str:
    positions = [text.find(stop) for stop in stops if stop and stop in text]
    if positions:
        text = text[: min(positions)]
    return text.rstrip()


def build_generator(adapter: bool) -> QwenGenerator:
    generator = QwenGenerator(
        model_name=MODEL,
        revision=REVISION,
        lora_rank=32,
        lora_alpha=64,
        lora_targets="attention",
        learning_rate=1e-5,
        max_prompt_tokens=2048,
        max_completion_tokens=512,
        clip_epsilon=0.2,
        kl_beta=0.0,
        seed=0,
        trainable=adapter,
    )
    if adapter:
        generator.load_checkpoint(ADAPTER)
    generator.model.eval()
    return generator


def completion_for(generator: QwenGenerator, prompt: str, stops: list[str]) -> str:
    tokenizer = generator.tokenizer
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].to(generator.model.device)
    attention = encoded["attention_mask"].to(generator.model.device)
    with torch.no_grad():
        output = generator.model.generate(
            input_ids=input_ids,
            attention_mask=attention,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            stopping_criteria=StoppingCriteriaList(
                [StopOnText(tokenizer, input_ids.shape[1], stops)]
            ),
        )
    new_ids = output[0][input_ids.shape[1] :]
    raw = tokenizer.decode(new_ids, skip_special_tokens=True)
    return cut_at_stop(strip_fence(raw), stops)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model", choices=["base", "adapter"], required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())["problems"]
    generator = build_generator(args.model == "adapter")
    print(f"model ready: {args.model}", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    languages = ("python", "js", "java", "cpp")
    for language in languages:
        rows = []
        for entry in manifest:
            row = entry[language]
            prompt = row["prompt"] if language != "python" else row["prompt"]
            stops = list(row.get("stop_tokens") or [])
            if language == "python":
                stops = PYTHON_STOPS
            completion = completion_for(generator, prompt, stops)
            full = prompt + completion
            rows.append(
                {
                    "number": entry["number"],
                    "name": row.get("name") or row.get("task_id"),
                    "prompt": prompt,
                    "completion": completion,
                    "full": full,
                    "stop_tokens": stops,
                }
            )
            print(
                f"{args.model} {language} {entry['number']} chars={len(completion)}",
                flush=True,
            )
        path = out_dir / f"{args.model}-{language}.json"
        path.write_text(json.dumps({"model": args.model, "language": language, "rows": rows}, indent=1))
        print("wrote", path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
