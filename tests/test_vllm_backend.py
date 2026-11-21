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
    """Records sampling calls and adapter handoffs."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.adapters: list[str] = []

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

    def set_adapter(self, path):
        self.adapters.append(str(path))

    def close(self) -> None:
        pass


class FakeTrainer:
    """The trainer surface the wrapper needs, with no torch."""

    trainable = True
    max_completion_tokens = 32

    def __init__(self) -> None:
        self.encoded: list[str] = []
        self.thinking: list[bool] = []
        self.built: list[dict] = []
        self.accumulated: list[tuple[str, int]] = []
        self.updates = 0
        self.step = 0
        self.last_update_norm: float | None = None
        self.adapter_saves: list[str] = []
        self.checkpoint_calls: list[tuple[str, object]] = []

    def encode_prompt(self, prompt, *, thinking=True):
        self.encoded.append(prompt.prompt_id)
        self.thinking.append(thinking)
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
        self.step += 1
        self.last_update_norm = 0.75

    def save_lora_adapter(self, directory):
        folder = Path(directory)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "adapter_model.safetensors").write_text("fake", encoding="utf-8")
        self.adapter_saves.append(str(folder))
        return folder

    def save_checkpoint(self, directory) -> None:
        self.checkpoint_calls.append(("save", directory))

    def load_checkpoint(self, directory) -> None:
        self.checkpoint_calls.append(("load", directory))


class CapEngine:
    """An engine that returns a completion of exactly the token cap."""

    def __init__(self) -> None:
        self.calls: list[int] = []

    def generate(self, prompt_token_ids, *, n, temperature, max_tokens):
        self.calls.append(max_tokens)
        return [
            [[7] * max_tokens for _ in range(n)] for _ in prompt_token_ids
        ]


class RealDecodeTrainer:
    """The trainer surface with the production completion decode.

    The real decode runs without torch, because only the caching path needs the
    model. That keeps the cap, the truncated flag, and the reward on one path
    without a model or a GPU.
    """

    trainable = False

    def __init__(self, max_completion_tokens: int) -> None:
        from types import SimpleNamespace

        from rlvr.qwen import QwenGenerator

        self.max_completion_tokens = max_completion_tokens
        self.decoder = object.__new__(QwenGenerator)
        self.decoder._torch = None
        self.decoder._device = None
        self.decoder.think_end_token_id = 99
        self.decoder.profile = SimpleNamespace(model_type="qwen3")
        self.decoder.tokenizer = SimpleNamespace(
            eos_token_id=1,
            decode=lambda ids, skip_special_tokens=True: "def f():\n    return 1",
        )

    def encode_prompt(self, prompt, *, thinking=True):
        return [100, 101], False

    def build_generation_batch(
        self, prompts, prompt_ids, prompt_truncated, completion_token_ids, *, cache
    ):
        groups = []
        for index, prompt in enumerate(prompts):
            ids = list(prompt_ids[index])
            group, _ = self.decoder._decode_group(
                prompt,
                ids,
                [1] * len(ids),
                len(ids),
                prompt_truncated=prompt_truncated[index],
                completion_rows=completion_token_ids[index],
                caching=False,
            )
            groups.append(group)
        return groups


def test_a_completion_that_reaches_the_cap_is_truncated_and_scores_minus_one(
    tmp_path,
) -> None:
    from types import SimpleNamespace

    from rlvr.config import load_config
    from rlvr.experiment import Experiment, Outcome

    config = load_config(
        Path(__file__).resolve().parents[1] / "configs" / "lfm-spread.toml"
    )
    engine = CapEngine()
    generator = VLLMGenerator(
        trainer=RealDecodeTrainer(config.max_completion_tokens),
        engine=engine,
        adapter_dir=tmp_path,
    )

    [group] = generator.generate_batch([prompt(0)], n=1, temperature=0.9)
    generation = group[0]

    assert engine.calls == [1024]
    assert generation.completion_tokens == 1024
    assert generation.truncated is True

    experiment = object.__new__(Experiment)
    experiment.arm = "code-score"
    experiment.config = SimpleNamespace(
        length_soft_limit=config.length_soft_limit,
        length_penalty_per_token=config.length_penalty_per_token,
        length_penalty_floor=config.length_penalty_floor,
    )
    record = experiment._build_record(
        prompt(0),
        generation,
        Outcome(tests_passed=4, tests_total=4, error_class="pass"),
        1,
        judge_score=None,
        process_score=None,
        cache_hit=False,
    )

    assert record.reward == -1.0


def test_n_answers_per_prompt_stay_grouped_in_prompt_order(tmp_path) -> None:
    engine, trainer = FakeEngine(), FakeTrainer()
    generator = VLLMGenerator(trainer=trainer, engine=engine, adapter_dir=tmp_path)

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


def test_the_screen_can_turn_thinking_off_through_the_same_generator(tmp_path) -> None:
    trainer = FakeTrainer()
    generator = VLLMGenerator(
        trainer=trainer, engine=FakeEngine(), adapter_dir=tmp_path
    )

    generator.generate(prompt(0), n=2, temperature=0.9, thinking=False)

    assert trainer.thinking == [False]


def test_generate_single_prompt_is_one_group(tmp_path) -> None:
    generator = VLLMGenerator(
        trainer=FakeTrainer(), engine=FakeEngine(), adapter_dir=tmp_path
    )

    group = generator.generate(prompt(0), n=2, temperature=0.9)

    assert [item.code for item in group] == ["p-0:0", "p-0:1"]


def test_one_adapter_handoff_per_optimizer_step_not_per_prompt(tmp_path) -> None:
    engine, trainer = FakeEngine(), FakeTrainer()
    generator = VLLMGenerator(trainer=trainer, engine=engine, adapter_dir=tmp_path)

    batches = generator.generate_batch([prompt(0), prompt(1)], n=2, temperature=0.9)
    for item, group in zip([prompt(0), prompt(1)], batches):
        generator.accumulate(item, group, [0.0, 1.0])
    generator.apply_update()

    assert len(engine.adapters) == 2
    assert trainer.updates == 1
    assert generator.last_update_norm == pytest.approx(0.75)
    assert trainer.accumulated == [("p-0", 2), ("p-1", 2)]


def test_second_update_hands_over_again(tmp_path) -> None:
    engine, trainer = FakeEngine(), FakeTrainer()
    generator = VLLMGenerator(trainer=trainer, engine=engine, adapter_dir=tmp_path)

    generator.apply_update()
    generator.apply_update()

    assert len(engine.adapters) == 3
    assert trainer.updates == 2


def test_an_untrainable_trainer_samples_the_base_weights_with_no_adapter(
    tmp_path,
) -> None:
    trainer = FakeTrainer()
    trainer.trainable = False
    engine = FakeEngine()
    generator = VLLMGenerator(trainer=trainer, engine=engine, adapter_dir=tmp_path)

    batch = generator.generate(prompt(0), n=2, temperature=0.9)

    assert len(batch) == 2
    assert engine.adapters == []
    assert trainer.adapter_saves == []
    assert trainer.built[0]["cache"] is False


def test_the_initial_adapter_is_handed_to_the_engine(tmp_path) -> None:
    engine, trainer = FakeEngine(), FakeTrainer()

    VLLMGenerator(trainer=trainer, engine=engine, adapter_dir=tmp_path)

    assert len(engine.adapters) == 1
    assert (tmp_path / "step-00000000").is_dir()


def test_only_the_two_newest_adapters_stay_on_disk(tmp_path) -> None:
    engine, trainer = FakeEngine(), FakeTrainer()
    generator = VLLMGenerator(trainer=trainer, engine=engine, adapter_dir=tmp_path)

    for _ in range(4):
        generator.apply_update()

    folders = sorted(path.name for path in tmp_path.iterdir() if path.is_dir())
    assert folders == ["step-00000003", "step-00000004"]
    assert len(engine.adapters) == 5


def test_checkpoints_delegate_to_the_trainer_and_resync(tmp_path) -> None:
    trainer = FakeTrainer()
    engine = FakeEngine()
    generator = VLLMGenerator(trainer=trainer, engine=engine, adapter_dir=tmp_path)

    generator.save_checkpoint(tmp_path)
    generator.load_checkpoint(tmp_path)

    assert trainer.checkpoint_calls == [("save", tmp_path), ("load", tmp_path)]
    assert len(engine.adapters) == 2


def test_missing_vllm_package_is_a_usage_error_naming_vllm(
    write_config, monkeypatch
) -> None:
    config = load_config(write_config(CONFIG))
    monkeypatch.setitem(sys.modules, "vllm", None)

    with pytest.raises(UsageError) as raised:
        build_generator(config, fake=False)

    message = str(raised.value)
    assert "vllm package" in message
    assert "pip install -e '.[vllm]'" in message


def test_the_removed_engine_key_is_a_config_error(write_config) -> None:
    path = write_config(CONFIG + '\n[training]\nengine = "vllm"\n')

    with pytest.raises(ConfigError, match="training.engine"):
        load_config(path)


def test_the_removed_inference_engine_spelling_is_a_config_error(write_config) -> None:
    path = write_config(CONFIG + '\n[training]\ninference_engine = "vllm"\n')

    with pytest.raises(ConfigError, match="training.inference_engine"):
        load_config(path)


def test_no_committed_config_sets_an_engine_key() -> None:
    root = Path(__file__).resolve().parents[1] / "configs"

    for path in sorted(root.glob("*.toml")):
        training = load_config(path).raw.get("training", {})
        assert "engine" not in training, path.name
        assert "inference_engine" not in training, path.name


def test_set_adapter_needs_an_engine_to_hand_the_adapter_to() -> None:
    engine = object.__new__(VLLMEngine)
    engine._llm = object()
    engine._adapter_id = 0
    engine._lora_request = None

    with pytest.raises(RuntimeError, match="llm_engine"):
        engine.set_adapter("/tmp/adapter")


def test_lfm_vllm_needs_the_engine_that_supports_its_lora_layers() -> None:
    from rlvr.model_profiles import validate_inference_engine

    validate_inference_engine("LiquidAI/LFM2.5-2.6B", vllm_version="0.17.0")
    validate_inference_engine("LiquidAI/LFM2.5-2.6B", vllm_version="0.34.0")
    with pytest.raises(ValueError, match="vllm >= 0.17.0"):
        validate_inference_engine("LiquidAI/LFM2.5-2.6B", vllm_version="0.10.1")


def test_lfm_vllm_rejects_the_release_before_the_lora_support() -> None:
    from rlvr.model_profiles import validate_inference_engine

    with pytest.raises(ValueError, match="vllm >= 0.17.0"):
        validate_inference_engine("LiquidAI/LFM2.5-2.6B", vllm_version="0.16.9")


def test_lfm_vllm_accepts_a_development_build_at_the_floor() -> None:
    from rlvr.model_profiles import validate_inference_engine

    validate_inference_engine("LiquidAI/LFM2.5-2.6B", vllm_version="0.17.0.dev0")


def test_qwen_keeps_no_vllm_floor() -> None:
    from rlvr.model_profiles import validate_inference_engine

    for version in ("0.9.0", "0.10.1", "0.17.0"):
        validate_inference_engine("Qwen/Qwen3-0.6B", vllm_version=version)


def test_a_source_checkout_version_comes_from_the_module(monkeypatch) -> None:
    import types
    from importlib.metadata import PackageNotFoundError

    import rlvr.model_profiles as profiles

    def missing(name):
        raise PackageNotFoundError(name)

    monkeypatch.setattr("importlib.metadata.version", missing)
    module = types.ModuleType("vllm")
    module.__version__ = "0.17.0"
    monkeypatch.setitem(sys.modules, "vllm", module)

    profiles.validate_inference_engine("LiquidAI/LFM2.5-2.6B")

    module.__version__ = "0.16.9"
    with pytest.raises(ValueError, match="vllm >= 0.17.0"):
        profiles.validate_inference_engine("LiquidAI/LFM2.5-2.6B")


def test_an_importable_vllm_with_no_version_is_an_error(monkeypatch) -> None:
    import types
    from importlib.metadata import PackageNotFoundError

    import rlvr.model_profiles as profiles

    def missing(name):
        raise PackageNotFoundError(name)

    monkeypatch.setattr("importlib.metadata.version", missing)
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))

    with pytest.raises(ValueError, match="cannot read the vllm version"):
        profiles.validate_inference_engine("LiquidAI/LFM2.5-2.6B")


def test_a_missing_vllm_package_is_left_to_the_engine_builder(monkeypatch) -> None:
    import types
    from importlib.metadata import PackageNotFoundError

    import rlvr.model_profiles as profiles

    def missing(name):
        raise PackageNotFoundError(name)

    monkeypatch.setattr("importlib.metadata.version", missing)
    monkeypatch.setitem(sys.modules, "vllm", None)

    profiles.validate_inference_engine("LiquidAI/LFM2.5-2.6B")


def test_a_refused_adapter_load_is_an_error(tmp_path, monkeypatch) -> None:
    import types

    vllm = types.ModuleType("vllm")
    lora = types.ModuleType("vllm.lora")
    request_module = types.ModuleType("vllm.lora.request")

    class LoRARequest:
        def __init__(self, name, adapter_id, path):
            self.name = name
            self.adapter_id = adapter_id
            self.path = path

    request_module.LoRARequest = LoRARequest
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.lora", lora)
    monkeypatch.setitem(sys.modules, "vllm.lora.request", request_module)

    class Engine:
        llm_engine = types.SimpleNamespace(add_lora=lambda request: False)

    engine = object.__new__(VLLMEngine)
    engine._llm = Engine()
    engine._adapter_id = 0
    engine._lora_request = None

    with pytest.raises(RuntimeError, match="refused to load"):
        engine.set_adapter(tmp_path)


def _lora_modules(monkeypatch):
    import types

    vllm = types.ModuleType("vllm")
    lora = types.ModuleType("vllm.lora")
    request_module = types.ModuleType("vllm.lora.request")

    class LoRARequest:
        def __init__(self, name, adapter_id, path):
            self.name = name
            self.adapter_id = adapter_id
            self.path = path

    request_module.LoRARequest = LoRARequest
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.lora", lora)
    monkeypatch.setitem(sys.modules, "vllm.lora.request", request_module)


def test_a_new_adapter_is_added_before_the_previous_one_is_removed(
    tmp_path, monkeypatch
) -> None:
    import types

    _lora_modules(monkeypatch)
    events: list[tuple[str, int]] = []

    def add(request):
        events.append(("add", request.adapter_id))
        return True

    def remove(adapter_id):
        events.append(("remove", adapter_id))

    class Engine:
        llm_engine = types.SimpleNamespace(
            add_lora=add, remove_lora=remove
        )

    engine = object.__new__(VLLMEngine)
    engine._llm = Engine()
    engine._adapter_id = 0
    engine._lora_request = None

    engine.set_adapter(tmp_path)
    engine.set_adapter(tmp_path)

    assert events == [("add", 1), ("add", 2), ("remove", 1)]
    assert engine._lora_request.adapter_id == 2


def test_a_refused_adapter_keeps_the_previous_one_loaded(tmp_path, monkeypatch) -> None:
    import types

    _lora_modules(monkeypatch)
    events: list[tuple[str, int]] = []
    answer = [True, False]

    def add(request):
        events.append(("add", request.adapter_id))
        return answer.pop(0)

    def remove(adapter_id):
        events.append(("remove", adapter_id))

    class Engine:
        llm_engine = types.SimpleNamespace(add_lora=add, remove_lora=remove)

    engine = object.__new__(VLLMEngine)
    engine._llm = Engine()
    engine._adapter_id = 0
    engine._lora_request = None

    engine.set_adapter(tmp_path)
    with pytest.raises(RuntimeError, match="previous adapter stays loaded"):
        engine.set_adapter(tmp_path)

    assert events == [("add", 1), ("add", 2)]
    assert engine._lora_request.adapter_id == 1
