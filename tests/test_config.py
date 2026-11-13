"""Tests for config loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from rlvr.config import Config, ConfigError, default_config_path, load_config

GOOD = """\
[model]
name = "Qwen/Qwen3-0.6B"
revision = "abc123"

[judge]
model = "deepseek-v4.1-flash"

[data]
split_seed = 1234

[fake]
steps = 3
prompts_per_step = 2
"""


def test_load_reads_every_declared_value(write_config) -> None:
    path = write_config(GOOD)
    config = load_config(path)
    assert isinstance(config, Config)
    assert config.model_name == "Qwen/Qwen3-0.6B"
    assert config.model_revision == "abc123"
    assert config.judge_model == "deepseek-v4.1-flash"
    assert config.split_seed == 1234
    assert config.steps == 3


def test_load_keeps_the_raw_mapping_for_meta(tmp_path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(GOOD, encoding="utf-8")
    config = load_config(path)
    assert config.raw["model"]["name"] == "Qwen/Qwen3-0.6B"
    assert config.path == path


def test_missing_file_names_the_path(tmp_path) -> None:
    missing = tmp_path / "nope" / "config.toml"
    with pytest.raises(ConfigError, match="config file not found"):
        load_config(missing)


def test_unparseable_file_is_reported(tmp_path) -> None:
    path = tmp_path / "config.toml"
    path.write_text("[model\nname = ", encoding="utf-8")
    with pytest.raises(ConfigError, match="config"):
        load_config(path)


def test_directory_instead_of_file_is_reported(tmp_path) -> None:
    with pytest.raises(ConfigError, match="config"):
        load_config(tmp_path)


@pytest.mark.parametrize(
    "text, key",
    [
        (GOOD.replace('[model]\nname = "Qwen/Qwen3-0.6B"\n', "[model]\n"), "model.name"),
        (GOOD.replace('revision = "abc123"\n', ""), "model.revision"),
        (GOOD.replace("model = \"deepseek-v4.1-flash\"", "other = 1"), "judge.model"),
        (GOOD.replace("split_seed = 1234", "other = 1"), "data.split_seed"),
    ],
)
def test_missing_required_key_is_named(tmp_path, text: str, key: str) -> None:
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match=key):
        load_config(path)


@pytest.mark.parametrize(
    "text, key",
    [
        (GOOD.replace('name = "Qwen/Qwen3-0.6B"', "name = 3"), "model.name"),
        (GOOD.replace('model = "deepseek-v4.1-flash"', "model = []"), "judge.model"),
        (GOOD.replace("split_seed = 1234", 'split_seed = "twelve"'), "data.split_seed"),
        (GOOD.replace("steps = 3", "steps = 3.5"), "fake.steps"),
        (GOOD.replace("steps = 3", "steps = 0"), "fake.steps"),
    ],
)
def test_wrong_value_type_is_named(tmp_path, text: str, key: str) -> None:
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match=key):
        load_config(path)


def test_empty_string_name_is_invalid(tmp_path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(GOOD.replace('name = "Qwen/Qwen3-0.6B"', 'name = ""'), encoding="utf-8")
    with pytest.raises(ConfigError, match="model.name"):
        load_config(path)


def test_top_level_not_a_table_is_invalid(tmp_path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('name = "only a scalar"\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="config"):
        load_config(path)


def test_extra_keys_are_kept_and_tolerated(tmp_path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(GOOD + "\n[training]\nlora_rank = 32\n", encoding="utf-8")
    config = load_config(path)
    assert config.raw["training"]["lora_rank"] == 32


def test_default_config_ships_with_the_package() -> None:
    path = default_config_path()
    assert Path(path).is_file()
    config = load_config(path)
    assert config.model_name == "Qwen/Qwen3-0.6B"
    assert config.judge_model == "deepseek/deepseek-v4.1-flash-1017"


def test_default_config_names_openrouter_as_the_judge_provider() -> None:
    judge = load_config(default_config_path()).raw["judge"]
    assert judge["provider"] == "openrouter"
    assert judge["base_url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert judge["api_key_env"] == "OPENROUTER_API_KEY"


def test_default_config_records_the_split_seed(tmp_path) -> None:
    config = load_config(default_config_path())
    assert isinstance(config.split_seed, int)


def test_config_path_accepts_str_and_path(tmp_path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(GOOD, encoding="utf-8")
    assert load_config(str(path)).steps == 3
    assert load_config(path).steps == 3


def test_eval_settings_default_when_the_run_table_is_absent(tmp_path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(GOOD, encoding="utf-8")
    config = load_config(path)
    assert config.eval_every_steps == 100
    assert config.eval_questions == 100
    assert config.eval_subset_seed == 2026


def test_eval_settings_are_read_from_the_run_table(tmp_path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        GOOD + "\n[run]\neval_every_steps = 50\neval_questions = 25\neval_subset_seed = 7\n",
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.eval_every_steps == 50
    assert config.eval_questions == 25
    assert config.eval_subset_seed == 7


@pytest.mark.parametrize(
    "line, key",
    [
        ("eval_every_steps = 0", "run.eval_every_steps"),
        ("eval_every_steps = 2.5", "run.eval_every_steps"),
        ("eval_questions = -1", "run.eval_questions"),
        ("eval_subset_seed = -2", "run.eval_subset_seed"),
        ("eval_subset_seed = true", "run.eval_subset_seed"),
    ],
)
def test_a_bad_eval_setting_is_named(tmp_path, line: str, key: str) -> None:
    path = tmp_path / "config.toml"
    path.write_text(GOOD + f"\n[run]\n{line}\n", encoding="utf-8")
    with pytest.raises(ConfigError, match=key):
        load_config(path)


def test_the_default_config_sets_the_eval_schedule() -> None:
    config = load_config(default_config_path())
    assert config.eval_every_steps == 100
    assert config.eval_questions == 100
    assert isinstance(config.eval_subset_seed, int)


def test_training_settings_default_and_are_read(tmp_path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(GOOD, encoding="utf-8")
    config = load_config(path)
    assert config.generations_per_prompt == 8
    assert config.epochs == 5
    assert config.temperature == pytest.approx(0.9)
    assert config.learning_rate == pytest.approx(1e-5)
    assert config.lora_rank == 32
    assert config.lora_alpha == 64
    assert config.max_prompt_tokens == 512
    assert config.max_completion_tokens == 768
    assert config.clip_epsilon == pytest.approx(0.2)
    assert config.kl_beta == pytest.approx(0.0)


def test_training_settings_are_read_from_the_table(tmp_path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        GOOD
        + "\n[training]\ngenerations_per_prompt = 4\nepochs = 2\n"
        "temperature = 0.7\nlearning_rate = 2e-5\nlora_rank = 16\nlora_alpha = 32\n"
        "max_prompt_tokens = 256\nmax_completion_tokens = 384\n"
        "clip_epsilon = 0.1\nkl_beta = 0.05\n",
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.generations_per_prompt == 4
    assert config.epochs == 2
    assert config.temperature == pytest.approx(0.7)
    assert config.learning_rate == pytest.approx(2e-5)
    assert config.lora_rank == 16
    assert config.lora_alpha == 32
    assert config.max_prompt_tokens == 256
    assert config.max_completion_tokens == 384
    assert config.clip_epsilon == pytest.approx(0.1)
    assert config.kl_beta == pytest.approx(0.05)


def test_checkpoint_and_budget_defaults(write_config) -> None:
    config = load_config(write_config(GOOD))
    assert config.checkpoint_every_steps == 50
    assert config.eval_final_questions is None
    assert config.max_steps is None
    assert config.max_judge_calls is None


def test_budget_and_checkpoint_are_read_from_config(tmp_path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        GOOD
        + "\n[run]\ncheckpoint_every_steps = 10\neval_final_questions = 25\n"
        + "[budget]\nmax_steps = 100\nmax_judge_calls = 5000\n",
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.checkpoint_every_steps == 10
    assert config.eval_final_questions == 25
    assert config.max_steps == 100
    assert config.max_judge_calls == 5000


def test_a_config_without_the_fake_table_loads(tmp_path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        GOOD.replace("\n[fake]\nsteps = 3\nprompts_per_step = 2\n", ""),
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.steps == 1


@pytest.mark.parametrize(
    "line, key",
    [
        ("clip_epsilon = 0.0", "training.clip_epsilon"),
        ("kl_beta = -0.1", "training.kl_beta"),
        ("temperature = 0", "training.temperature"),
        ("learning_rate = 0", "training.learning_rate"),
    ],
)
def test_a_bad_training_setting_is_named(tmp_path, line: str, key: str) -> None:
    path = tmp_path / "config.toml"
    path.write_text(GOOD + f"\n[training]\n{line}\n", encoding="utf-8")
    with pytest.raises(ConfigError, match=key):
        load_config(path)


def test_the_shipped_real_config_loads() -> None:
    config = load_config(Path(__file__).resolve().parents[1] / "configs" / "real.toml")
    assert config.model_name == "Qwen/Qwen3-0.6B"
    assert config.model_revision == "c1899de289a04d12100db370d81485cdf75e47ca"
    assert config.epochs == 5
    assert config.generations_per_prompt == 8
    assert config.lora_rank == 32 and config.lora_alpha == 64
    assert config.checkpoint_every_steps == 50


def test_the_shipped_smoke_config_loads() -> None:
    config = load_config(Path(__file__).resolve().parents[1] / "configs" / "smoke.toml")
    assert config.max_steps == 8
    assert config.max_judge_calls == 64
    assert config.eval_questions == 4
    assert config.eval_final_questions == 8


def test_seed_defaults_to_zero_and_is_read(write_config) -> None:
    assert load_config(write_config(GOOD)).seed == 0
    path = write_config(GOOD + "\n[training]\nseed = 7\n", name="seed.toml")
    assert load_config(path).seed == 7
