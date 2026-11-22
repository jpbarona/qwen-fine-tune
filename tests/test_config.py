"""Tests for config loading and validation."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from rlvr.config import Config, ConfigError, default_config_path, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]

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
    assert config.eval_every_steps == 20
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
    assert config.eval_every_steps == 20
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
    assert config.lora_targets == "attention"
    assert config.max_prompt_tokens == 512
    assert config.max_completion_tokens == 2048
    assert config.entropy_every_steps == 500
    assert config.clip_epsilon == pytest.approx(0.2)
    assert config.kl_beta == pytest.approx(0.0)
    assert config.length_soft_limit == 900
    assert config.length_penalty_per_token == pytest.approx(0.001)
    assert config.length_penalty_floor == pytest.approx(0.2)


def test_training_settings_are_read_from_the_table(tmp_path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        GOOD
        + "\n[training]\ngenerations_per_prompt = 4\nepochs = 2\n"
        "temperature = 0.7\nlearning_rate = 2e-5\nlora_rank = 16\nlora_alpha = 32\n"
        "max_prompt_tokens = 256\nmax_completion_tokens = 384\n"
        "entropy_every_steps = 25\n"
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
    assert config.entropy_every_steps == 25
    assert config.clip_epsilon == pytest.approx(0.1)
    assert config.kl_beta == pytest.approx(0.05)


def test_lora_targets_reads_all_linear(tmp_path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        GOOD + '\n[training]\nlora_targets = "all-linear"\n', encoding="utf-8"
    )
    assert load_config(path).lora_targets == "all-linear"


def test_an_unknown_lora_targets_value_is_rejected(tmp_path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        GOOD + '\n[training]\nlora_targets = "every-layer"\n', encoding="utf-8"
    )
    with pytest.raises(ConfigError, match="training.lora_targets.*all-linear"):
        load_config(path)


def test_length_settings_are_read_from_the_training_table(tmp_path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        GOOD
        + "\n[training]\nlength_soft_limit = 700\n"
        "length_penalty_per_token = 0.002\nlength_penalty_floor = 0.5\n",
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.length_soft_limit == 700
    assert config.length_penalty_per_token == pytest.approx(0.002)
    assert config.length_penalty_floor == pytest.approx(0.5)


def test_the_thinking_budget_is_read_from_the_training_table(tmp_path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        GOOD + "\n[training]\nthinking_budget = 512\n", encoding="utf-8"
    )
    config = load_config(path)
    assert config.thinking_budget == 512


def test_a_zero_thinking_budget_is_named(tmp_path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        GOOD + "\n[training]\nthinking_budget = 0\n", encoding="utf-8"
    )
    with pytest.raises(ConfigError, match="training.thinking_budget"):
        load_config(path)


@pytest.mark.parametrize(
    "line, key",
    [
        ("length_soft_limit = -1", "training.length_soft_limit"),
        ("length_soft_limit = 1.5", "training.length_soft_limit"),
        ("length_penalty_per_token = -0.001", "training.length_penalty_per_token"),
        ("length_penalty_per_token = true", "training.length_penalty_per_token"),
        ("length_penalty_floor = -0.1", "training.length_penalty_floor"),
        ("length_penalty_floor = 1.0", "training.length_penalty_floor"),
        ("length_penalty_floor = 1.1", "training.length_penalty_floor"),
    ],
)
def test_a_bad_length_setting_is_named(tmp_path, line: str, key: str) -> None:
    path = tmp_path / "config.toml"
    path.write_text(GOOD + f"\n[training]\n{line}\n", encoding="utf-8")
    with pytest.raises(ConfigError, match=key):
        load_config(path)


def test_checkpoint_and_budget_defaults(write_config) -> None:
    config = load_config(write_config(GOOD))
    assert config.checkpoint_every_steps == 20
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
        GOOD.replace("\n[fake]\nsteps = 3\n", ""),
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
        ("entropy_every_steps = -1", "training.entropy_every_steps"),
        ("entropy_every_steps = 1.5", "training.entropy_every_steps"),
    ],
)
def test_a_bad_training_setting_is_named(tmp_path, line: str, key: str) -> None:
    path = tmp_path / "config.toml"
    path.write_text(GOOD + f"\n[training]\n{line}\n", encoding="utf-8")
    with pytest.raises(ConfigError, match=key):
        load_config(path)


def test_entropy_every_steps_zero_disables_the_measurement(tmp_path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(GOOD + "\n[training]\nentropy_every_steps = 0\n", encoding="utf-8")
    assert load_config(path).entropy_every_steps == 0


def test_the_memory_floor_defaults_to_512_mib(write_config) -> None:
    assert load_config(write_config(GOOD)).memory_floor_mib == 512


def test_the_batch_cap_defaults_to_none(write_config) -> None:
    assert load_config(write_config(GOOD)).max_prompts is None


def test_the_batch_cap_is_read_from_config(tmp_path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(GOOD + "\n[training]\nmax_prompts = 20\n", encoding="utf-8")
    assert load_config(path).max_prompts == 20


@pytest.mark.parametrize("value", ["0", "-3", "1.5", '"20"'])
def test_a_bad_batch_cap_is_named(tmp_path, value: str) -> None:
    path = tmp_path / "config.toml"
    path.write_text(GOOD + f"\n[training]\nmax_prompts = {value}\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="training.max_prompts"):
        load_config(path)


def test_the_memory_floor_is_read_from_config(tmp_path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(GOOD + "\n[training]\nmemory_floor_mib = 1024\n", encoding="utf-8")
    assert load_config(path).memory_floor_mib == 1024


def test_a_negative_memory_floor_is_named(tmp_path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(GOOD + "\n[training]\nmemory_floor_mib = -1\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="training.memory_floor_mib"):
        load_config(path)


def test_the_shipped_real_config_loads() -> None:
    config = load_config(Path(__file__).resolve().parents[1] / "configs" / "real.toml")
    assert config.model_name == "Qwen/Qwen3-0.6B"
    assert config.model_revision == "c1899de289a04d12100db370d81485cdf75e47ca"
    assert config.epochs == 5
    assert config.generations_per_prompt == 8
    assert config.max_completion_tokens == 2048
    assert config.entropy_every_steps == 500
    assert config.lora_rank == 32 and config.lora_alpha == 64
    assert config.checkpoint_every_steps == 10
    assert config.eval_every_steps == 20
    assert config.length_soft_limit == 900
    assert config.length_penalty_per_token == pytest.approx(0.001)
    assert config.length_penalty_floor == pytest.approx(0.2)


def test_the_shipped_smoke_config_loads() -> None:
    config = load_config(Path(__file__).resolve().parents[1] / "configs" / "smoke.toml")
    assert config.max_steps == 8
    assert config.max_judge_calls == 128
    assert config.max_completion_tokens == 2048
    assert config.eval_questions == 4
    assert config.eval_final_questions == 8


def test_seed_defaults_to_zero_and_is_read(write_config) -> None:
    assert load_config(write_config(GOOD)).seed == 0
    path = write_config(GOOD + "\n[training]\nseed = 7\n", name="seed.toml")
    assert load_config(path).seed == 7


def test_thinking_budget_defaults_to_one_value(write_config) -> None:
    config = load_config(write_config(GOOD))
    assert config.thinking_budget == 1024


def test_the_thinking_budget_is_read_from_the_training_table(write_config) -> None:
    path = write_config(
        GOOD + "\n[training]\nthinking_budget = 512\n", name="budget.toml"
    )
    assert load_config(path).thinking_budget == 512


def test_a_zero_thinking_budget_is_named(write_config) -> None:
    path = write_config(
        GOOD + "\n[training]\nthinking_budget = 0\n", name="zero-budget.toml"
    )
    with pytest.raises(ConfigError, match="training.thinking_budget"):
        load_config(path)


def test_the_removed_curriculum_keys_are_ignored(write_config) -> None:
    path = write_config(
        GOOD + '\n[curriculum]\nstage = "hard"\ncarry_over_ratio = 0.35\n',
        name="curriculum.toml",
    )
    config = load_config(path)
    assert "curriculum" in config.raw
    assert not hasattr(config, "stage")
    assert not hasattr(config, "carry_over_ratio")


def test_throughput_settings_have_bounded_defaults(write_config) -> None:
    config = load_config(write_config(GOOD))
    assert config.test_workers == 4
    assert config.eval_batch_size == 8
    assert config.prefetch_scoring is True


@pytest.mark.parametrize(
    "setting",
    ["test_workers = 0", "test_workers = true", "eval_batch_size = -1",
     "eval_batch_size = 1.5", "prefetch_scoring = 1", 'prefetch_scoring = "true"'],
)
def test_throughput_settings_reject_invalid_values(write_config, setting) -> None:
    with pytest.raises(ConfigError, match=setting.split()[0]):
        load_config(write_config(GOOD + "\n[training]\n" + setting))


def test_throughput_settings_can_be_overridden(write_config) -> None:
    config = load_config(write_config(
        GOOD + "\n[training]\ntest_workers = 2\neval_batch_size = 3\nprefetch_scoring = false\n"
    ))
    assert (config.test_workers, config.eval_batch_size, config.prefetch_scoring) == (2, 3, False)


def test_pool_settings_default(write_config) -> None:
    config = load_config(write_config(GOOD))

    assert config.pool_policy is None
    assert config.screen_rows is None
    assert config.rollouts_per_step == 608
    assert config.sweep_gap_ratio == 3


def test_pool_settings_are_read(tmp_path) -> None:
    path = tmp_path / "pool.toml"
    path.write_text(
        GOOD
        + "\n[training]\n"
        'pool_policy = "spread"\n'
        'screen_rows = "screens/latest/rows.jsonl"\n'
        "rollouts_per_step = 320\n",
        encoding="utf-8",
    )
    config = load_config(path)

    assert config.pool_policy == "spread"
    assert config.screen_rows == "screens/latest/rows.jsonl"
    assert config.rollouts_per_step == 320


def test_the_default_config_sets_the_rollout_budget() -> None:
    config = load_config(default_config_path())

    assert config.rollouts_per_step == 608
    assert config.pool_policy is None


@pytest.mark.parametrize(
    "line, key",
    [
        ('pool_policy = "random"', "training.pool_policy"),
        ('pool_policy = ""', "training.pool_policy"),
        ("pool_policy = 3", "training.pool_policy"),
        ("screen_rows = 3", "training.screen_rows"),
        ('screen_rows = ""', "training.screen_rows"),
        ("rollouts_per_step = 0", "training.rollouts_per_step"),
        ("rollouts_per_step = 1.5", "training.rollouts_per_step"),
        ("sweep_gap_ratio = 2", "training.sweep_gap_ratio"),
        ("sweep_gap_ratio = 11", "training.sweep_gap_ratio"),
        ("sweep_gap_ratio = 3.5", "training.sweep_gap_ratio"),
        ("sweep_gap_ratio = true", "training.sweep_gap_ratio"),
    ],
)
def test_a_bad_pool_setting_is_named(tmp_path, line: str, key: str) -> None:
    path = tmp_path / "bad-pool.toml"
    path.write_text(GOOD + "\n[training]\n" + line + "\n", encoding="utf-8")

    with pytest.raises(ConfigError, match=key):
        load_config(path)


def test_the_sweep_policy_and_gap_ratio_are_read(tmp_path) -> None:
    path = tmp_path / "sweep.toml"
    path.write_text(
        GOOD
        + "\n[training]\n"
        'pool_policy = "sweep"\n'
        'screen_rows = "screens/latest/rows.jsonl"\n'
        "sweep_gap_ratio = 5\n",
        encoding="utf-8",
    )
    config = load_config(path)

    assert config.pool_policy == "sweep"
    assert config.sweep_gap_ratio == 5


def test_the_sweep_gap_ratio_defaults_to_three(tmp_path) -> None:
    path = tmp_path / "sweep-default.toml"
    path.write_text(
        GOOD
        + "\n[training]\n"
        'pool_policy = "sweep"\n'
        'screen_rows = "screens/latest/rows.jsonl"\n',
        encoding="utf-8",
    )

    assert load_config(path).sweep_gap_ratio == 3


def test_a_pool_policy_cannot_be_combined_with_max_prompts(tmp_path) -> None:
    path = tmp_path / "conflict.toml"
    path.write_text(
        GOOD
        + '\n[training]\npool_policy = "spread"\n'
        'screen_rows = "rows.jsonl"\nmax_prompts = 10\n',
        encoding="utf-8",
    )

    with pytest.raises(ConfigError) as error:
        load_config(path)

    message = str(error.value)
    assert "training.pool_policy" in message
    assert "training.max_prompts" in message


def test_a_pool_policy_needs_screen_rows(tmp_path) -> None:
    path = tmp_path / "no-screen.toml"
    path.write_text(
        GOOD + '\n[training]\npool_policy = "spread"\n', encoding="utf-8"
    )

    with pytest.raises(ConfigError) as error:
        load_config(path)

    message = str(error.value)
    assert "training.pool_policy" in message
    assert "training.screen_rows" in message


def test_refresh_steps_is_read_and_defaults_to_forty(tmp_path) -> None:
    configured = tmp_path / "configured.toml"
    configured.write_text(
        GOOD
        + "\n[training]\n"
        'pool_policy = "spread"\n'
        'screen_rows = "rows.jsonl"\n'
        "refresh_steps = 25\n",
        encoding="utf-8",
    )
    default = tmp_path / "default.toml"
    default.write_text(
        GOOD
        + "\n[training]\n"
        'pool_policy = "spread"\n'
        'screen_rows = "rows.jsonl"\n',
        encoding="utf-8",
    )

    assert load_config(configured).refresh_steps == 25
    assert load_config(default).refresh_steps == 40

    bad = tmp_path / "bad.toml"
    bad.write_text(
        GOOD
        + "\n[training]\n"
        'pool_policy = "spread"\n'
        'screen_rows = "rows.jsonl"\n'
        "refresh_steps = 0\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="refresh_steps"):
        load_config(bad)


@pytest.mark.parametrize(
    ("name", "policy"),
    (("lfm-spread.toml", "spread"), ("lfm-sweep.toml", "sweep")),
)
def test_the_parked_pool_arm_configs_load(name: str, policy: str) -> None:
    config = load_config(REPO_ROOT / "configs" / name)

    assert config.model_name == "LiquidAI/LFM2.5-2.6B"
    assert config.pool_policy == policy
    assert config.generations_per_prompt == 16
    assert config.rollouts_per_step == 304
    assert config.max_completion_tokens == 1024
    assert config.learning_rate == 3e-5
    assert config.lora_targets == "all-linear"
    assert config.lora_rank == 16
    assert config.lora_alpha == 32
    assert config.thinking_budget == 1024
    assert config.max_steps == 100
    assert config.max_prompts is None
    assert config.screen_rows == "screens/mbpp-base/rows.jsonl"
    assert config.eval_every_steps == 5
    assert config.checkpoint_every_steps == 10


@pytest.mark.parametrize("name", ("lfm-spread.toml", "lfm-sweep.toml"))
def test_the_parked_pool_arm_configs_checkpoint_every_ten_steps(name: str) -> None:
    config = load_config(REPO_ROOT / "configs" / name)

    assert config.checkpoint_every_steps == 10


def test_the_half_size_step_keeps_the_planned_parked_arithmetic() -> None:
    from rlvr.schedule import Spread, Sweep

    spread = load_config(REPO_ROOT / "configs" / "lfm-spread.toml")
    sweep = load_config(REPO_ROOT / "configs" / "lfm-sweep.toml")
    group = spread.generations_per_prompt
    prompts_per_step = spread.rollouts_per_step // group

    assert prompts_per_step == 19
    assert spread.rollouts_per_step // 4 // group == 4
    assert (
        Spread(
            prompts_per_step=prompts_per_step,
            cap=4,
            refresh_steps=spread.refresh_steps,
        ).parked_count(267)
        == 4
    )
    sweep_schedule = Sweep(prompts_per_step=sweep.rollouts_per_step // group)
    assert sweep_schedule.sweep_length(267) == 15
    assert sweep_schedule.gap_length(267) == 45


def test_the_arm_config_accepts_a_screen_for_every_trained_prompt(
    tmp_path,
) -> None:
    from dataclasses import replace

    from rlvr.cli import build_pools, build_prompts
    from rlvr.data import load_split, prompt_id_for
    from rlvr.screening import ScreenRow, ScreenStore

    config = load_config(REPO_ROOT / "configs" / "lfm-spread.toml")
    split = load_split(seed=config.split_seed)
    directory = tmp_path / "screen"
    store = ScreenStore(directory)
    for question in split.train:
        store.append_row(
            ScreenRow(
                prompt_id=prompt_id_for(question),
                samples=16,
                passed=8,
                pass_rate=0.5,
            )
        )
    measured = replace(config, screen_rows=str(directory))

    train, _ = build_prompts(measured)
    pools = build_pools(measured, train)

    assert [prompt.prompt_id for prompt in train] == [
        prompt_id_for(question) for question in split.train
    ]
    assert len(train) == 711
    assert pools is not None
    assert pools.samples == 16
    assert len(pools.train) == 711


def test_the_parked_pool_arm_configs_differ_only_in_policy_keys() -> None:
    def read(name: str) -> dict:
        path = REPO_ROOT / "configs" / name
        return tomllib.loads(path.read_text(encoding="utf-8"))

    spread = read("lfm-spread.toml")
    sweep = read("lfm-sweep.toml")
    assert spread["training"].pop("pool_policy") == "spread"
    assert sweep["training"].pop("pool_policy") == "sweep"
    assert spread["training"].pop("refresh_steps") == 30
    assert "refresh_steps" not in sweep["training"]
    assert sweep["training"].pop("sweep_gap_ratio") == 3
    assert "sweep_gap_ratio" not in spread["training"]

    assert spread == sweep
