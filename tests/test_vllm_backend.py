"""Tests for the vLLM inference backend.

The engine is faked, so the suite needs no vllm package, no GPU, and no network.
The fake trainer mirrors the small surface `VLLMGenerator` uses: prompt
encoding, the group builder, the update, and the weight view.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from rlvr.cli import UsageError, build_generator
from rlvr.config import ConfigError, load_config
from rlvr.experiment import Generation
from rlvr.vllm_backend import VLLMEngine, VLLMGenerator

CONFIG = """\
[model]
name = "Qwen/Qwen3-0.6B"
revision = "rev"

[judge]
model = "fake-judge-model"

[data]
split_seed = 1234

[fake]
steps = 1
"""


def prompt(index: int):
    from rlvr.experiment import Prompt

    return Prompt(prompt_id=f"p-{index}", problem=f"problem {index}")


class FakeEngine:
    """Records sampling calls and syncs. Returns one n-row group per prompt."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.syncs: list[list[tuple[str, str]]] = []

    def generate(self, prompt_token_ids, *, n, temperature, max_tokens):
        ids = [list(row) for row in prompt_token_ids]
        self.calls.append(
            {
                "prompt_token_ids": ids,
                "n": n,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        return [
            [[index * 10 + sample] for sample in range(n)]
            for index in range(len(ids))
        ]

    def sync(self, state_dict):
        self.syncs.append([(name, value) for name, value in state_dict])

    def close(self) -> None:
        pass


class FakeTrainer:
    """The trainer surface the wrapper needs, with no torch."""

    trainable = True
    max_completion_tokens = 32

    def __init__(self) -> None:
        self.encoded: list[str] = []
        self.built: list[dict] = []
        self.accumulated: list[tuple[str, int]] = []
        self.updates = 0
        self.checkpoint_calls: list[tuple[str, object]] = []

    def encode_prompt(self, prompt):
        self.encoded.append(prompt.prompt_id)
        return [100, 101], False

    def build_generation_batch(
        self, prompts, prompt_ids, prompt_truncated, completion_token_ids, *, cache
    ):
        self.built.append(
            {
                "prompts": [item.prompt_id for item in prompts],
                "prompt_ids": [list(ids) for ids in prompt_ids],
                "truncated": list(prompt_truncated),
                "rows": [[list(row) for row in group] for group in completion_token_ids],
                "cache": cache,
            }
        )
        return [
            [
                Generation(
                    reasoning="",
                    code=f"{item.prompt_id}:{row[0]}",
                    prompt_tokens=len(prompt_ids[index]),
                    completion_tokens=len(row),
                )
                for row in group
            ]
            for index, (item, group) in enumerate(zip(prompts, completion_token_ids))
        ]

    def accumulate(self, prompt, generations, rewards, *, measure_entropy=False):
        self.accumulated.append((prompt.prompt_id, len(generations)))
        return None

    def apply_update(self) -> None:
        self.updates += 1

    def engine_state_dict(self):
        return [("adapter.weight", "tensor")]

    def save_checkpoint(self, directory) -> None:
        self.checkpoint_calls.append(("save", directory))

    def load_checkpoint(self, directory) -> None:
        self.checkpoint_calls.append(("load", directory))


def test_n_answers_per_prompt_stay_grouped_in_prompt_order() -> None:
    engine, trainer = FakeEngine(), FakeTrainer()
    generator = VLLMGenerator(trainer=trainer, engine=engine)

    batches = generator.generate_batch([prompt(0), prompt(1)], n=3, temperature=0.9)

    assert [len(group) for group in batches] == [3, 3]
    assert [[item.code for item in group] for group in batches] == [
        ["p-0:0", "p-0:1", "p-0:2"],
        ["p-1:10", "p-1:11", "p-1:12"],
    ]
    assert trainer.encoded == ["p-0", "p-1"]
    assert len(engine.calls) == 1
    assert engine.calls[0]["n"] == 3
    assert engine.calls[0]["temperature"] == 0.9
    assert engine.calls[0]["max_tokens"] == 32
    assert engine.calls[0]["prompt_token_ids"] == [[100, 101], [100, 101]]
    assert trainer.built[0]["cache"] is True


def test_generate_single_prompt_is_one_group() -> None:
    generator = VLLMGenerator(trainer=FakeTrainer(), engine=FakeEngine())

    group = generator.generate(prompt(0), n=2, temperature=0.9)

    assert [item.code for item in group] == ["p-0:0", "p-0:1"]


def test_one_sync_per_optimizer_step_not_per_prompt() -> None:
    engine, trainer = FakeEngine(), FakeTrainer()
    generator = VLLMGenerator(trainer=trainer, engine=engine)

    batches = generator.generate_batch([prompt(0), prompt(1)], n=2, temperature=0.9)
    for item, group in zip([prompt(0), prompt(1)], batches):
        generator.accumulate(item, group, [0.0, 1.0])
    generator.apply_update()

    assert engine.syncs == [[("adapter.weight", "tensor")]]
    assert trainer.updates == 1
    assert trainer.accumulated == [("p-0", 2), ("p-1", 2)]


def test_second_update_syncs_again() -> None:
    engine, trainer = FakeEngine(), FakeTrainer()
    generator = VLLMGenerator(trainer=trainer, engine=engine)

    generator.apply_update()
    generator.apply_update()

    assert len(engine.syncs) == 2
    assert trainer.updates == 2


def test_checkpoints_delegate_to_the_trainer(tmp_path) -> None:
    trainer = FakeTrainer()
    generator = VLLMGenerator(trainer=trainer, engine=FakeEngine())

    generator.save_checkpoint(tmp_path)
    generator.load_checkpoint(tmp_path)

    assert trainer.checkpoint_calls == [("save", tmp_path), ("load", tmp_path)]


def test_missing_vllm_package_is_a_usage_error_naming_vllm(
    write_config, monkeypatch
) -> None:
    path = write_config(CONFIG + '\n[training]\nengine = "vllm"\n')
    config = load_config(path)
    monkeypatch.setitem(sys.modules, "vllm", None)

    with pytest.raises(UsageError) as raised:
        build_generator(config, fake=False)

    message = str(raised.value)
    assert "vllm package" in message
    assert 'engine = "hf"' in message


def test_engine_config_rejects_an_unknown_value(write_config) -> None:
    path = write_config(CONFIG + '\n[training]\nengine = "tensorrt"\n')

    with pytest.raises(ConfigError, match="training.engine"):
        load_config(path)


def test_engine_defaults_to_hf(write_config) -> None:
    config = load_config(write_config(CONFIG))

    assert config.engine == "hf"


def test_engine_accepts_vllm(write_config) -> None:
    path = write_config(CONFIG + '\n[training]\nengine = "vllm"\n')

    assert load_config(path).engine == "vllm"


def test_engine_accepts_the_inference_engine_spelling(write_config) -> None:
    path = write_config(CONFIG + '\n[training]\ninference_engine = "vllm"\n')

    assert load_config(path).engine == "vllm"


def test_shipped_real_config_stays_on_hf() -> None:
    path = Path(__file__).resolve().parents[1] / "configs" / "real.toml"

    assert load_config(path).engine == "hf"


def test_sync_without_collective_rpc_raises_a_clear_error() -> None:
    engine = object.__new__(VLLMEngine)
    engine._llm = object()

    with pytest.raises(RuntimeError, match="collective_rpc"):
        engine.sync([("adapter.weight", "tensor")])
