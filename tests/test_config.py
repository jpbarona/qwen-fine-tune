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
    assert config.prompts_per_step == 2


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
        (GOOD.replace("steps = 3\n", ""), "fake.steps"),
        (GOOD.replace("prompts_per_step = 2\n", ""), "fake.prompts_per_step"),
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
        (GOOD.replace("prompts_per_step = 2", "prompts_per_step = -1"), "fake.prompts_per_step"),
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
    assert config.judge_model == "deepseek-v4.1-flash"


def test_default_config_records_the_split_seed(tmp_path) -> None:
    config = load_config(default_config_path())
    assert isinstance(config.split_seed, int)


def test_config_path_accepts_str_and_path(tmp_path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(GOOD, encoding="utf-8")
    assert load_config(str(path)).steps == 3
    assert load_config(path).steps == 3
