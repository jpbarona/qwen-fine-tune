"""Config loading and validation.

The config is TOML, read with the standard library `tomllib`. A run reads one
config file and records it verbatim in `meta.json`, so a run folder describes
itself.

Required keys:

    [model] name, revision
    [judge] model
    [data]  split_seed

Optional keys, with the defaults shown:

    [fake]  steps = 1

    [run] eval_every_steps = 100
    [run] eval_questions   = 100
    [run] eval_subset_seed = 2026
    [run] checkpoint_every_steps = 50
    [run] eval_final_questions   = (unset: the whole pool)

    [training] generations_per_prompt = 8
    [training] epochs = 5
    [training] temperature = 0.9
    [training] learning_rate = 1e-5
    [training] lora_rank = 32
    [training] lora_alpha = 64
    [training] max_prompt_tokens = 512
    [training] max_completion_tokens = 768
    [training] clip_epsilon = 0.2
    [training] kl_beta = 0.0
    [training] seed = 0

    [budget] max_steps = (unset: run all of fake.steps)
    [budget] max_judge_calls = (unset: no judge call cap)

`eval_subset_seed` chooses the fixed checkpoint subset from the test pool. It is
deliberately separate from `data.split_seed`, so the eval subset is not tied to
the train/test cut.

Extra keys and tables are kept and ignored here. Later tasks add the training,
data, and judge settings they need without changing this loader's contract.

The judge API key is never part of the config. It comes from the environment in
the task that calls the judge.
"""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["Config", "ConfigError", "default_config_path", "load_config"]

#: Resource path of the config used when --config is not given.
DEFAULT_CONFIG_RESOURCE = "resources/default_config.toml"

#: Defaults for the optional [run] table. The spec's schedule is one checkpoint
#: evaluation every 100 steps over 100 fixed questions.
DEFAULT_EVAL_EVERY_STEPS = 100
DEFAULT_EVAL_QUESTIONS = 100
DEFAULT_EVAL_SUBSET_SEED = 2026
DEFAULT_CHECKPOINT_EVERY_STEPS = 50

#: Defaults for the optional [fake] table, used only by `--fake`.
DEFAULT_STEPS = 1

#: Defaults for the optional [training] table. These match SPEC.md and are only
#: read by the real Qwen trainer. The fake path ignores them.
DEFAULT_GENERATIONS_PER_PROMPT = 8
DEFAULT_EPOCHS = 5
DEFAULT_TEMPERATURE = 0.9
DEFAULT_LEARNING_RATE = 1e-5
DEFAULT_LORA_RANK = 32
DEFAULT_LORA_ALPHA = 64
DEFAULT_MAX_PROMPT_TOKENS = 512
DEFAULT_MAX_COMPLETION_TOKENS = 768
DEFAULT_CLIP_EPSILON = 0.2
DEFAULT_KL_BETA = 0.0
DEFAULT_SEED = 0


class ConfigError(ValueError):
    """The config file is missing, unreadable, or invalid."""


@dataclass(frozen=True)
class Config:
    path: Path
    raw: dict[str, Any]
    model_name: str
    model_revision: str
    judge_model: str
    split_seed: int
    steps: int
    eval_every_steps: int = DEFAULT_EVAL_EVERY_STEPS
    eval_questions: int = DEFAULT_EVAL_QUESTIONS
    eval_subset_seed: int = DEFAULT_EVAL_SUBSET_SEED
    checkpoint_every_steps: int = DEFAULT_CHECKPOINT_EVERY_STEPS
    eval_final_questions: int | None = None
    generations_per_prompt: int = DEFAULT_GENERATIONS_PER_PROMPT
    epochs: int = DEFAULT_EPOCHS
    temperature: float = DEFAULT_TEMPERATURE
    learning_rate: float = DEFAULT_LEARNING_RATE
    lora_rank: int = DEFAULT_LORA_RANK
    lora_alpha: int = DEFAULT_LORA_ALPHA
    max_prompt_tokens: int = DEFAULT_MAX_PROMPT_TOKENS
    max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS
    clip_epsilon: float = DEFAULT_CLIP_EPSILON
    kl_beta: float = DEFAULT_KL_BETA
    seed: int = DEFAULT_SEED
    max_steps: int | None = None
    max_judge_calls: int | None = None


def default_config_path() -> Path:
    """The packaged default config, as a filesystem path."""
    from importlib.resources import files

    return Path(str(files("rlvr") / DEFAULT_CONFIG_RESOURCE))


def _lookup(raw: dict[str, Any], dotted: str) -> Any:
    node: Any = raw
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            raise ConfigError(f"config: missing required key '{dotted}'")
        node = node[part]
    return node


def _string(raw: dict[str, Any], dotted: str) -> str:
    value = _lookup(raw, dotted)
    if not isinstance(value, str):
        raise ConfigError(
            f"config: '{dotted}' must be a non-empty string, got {_shown(value)}"
        )
    if not value.strip():
        raise ConfigError(f"config: '{dotted}' must be a non-empty string, got an empty value")
    return value


def _positive_int(raw: dict[str, Any], dotted: str) -> int:
    value = _lookup(raw, dotted)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"config: '{dotted}' must be an integer, got {_shown(value)}")
    if value < 1:
        raise ConfigError(f"config: '{dotted}' must be at least 1, got {value}")
    return value


def _non_negative_int(raw: dict[str, Any], dotted: str) -> int:
    value = _lookup(raw, dotted)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"config: '{dotted}' must be an integer, got {_shown(value)}")
    if value < 0:
        raise ConfigError(f"config: '{dotted}' must not be negative, got {value}")
    return value


def _shown(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return f"bool {value!r}"
    return f"{type(value).__name__} {value!r}"


def _optional(raw: dict[str, Any], dotted: str, default: Any) -> Any:
    """The value at `dotted`, or `default` when the key or its table is absent."""
    node: Any = raw
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def _optional_positive_int(raw: dict[str, Any], dotted: str, default: int) -> int:
    if _optional(raw, dotted, None) is None:
        return default
    return _positive_int(raw, dotted)


def _optional_non_negative_int(raw: dict[str, Any], dotted: str, default: int) -> int:
    if _optional(raw, dotted, None) is None:
        return default
    return _non_negative_int(raw, dotted)


def _optional_float(raw: dict[str, Any], dotted: str, default: float, *, positive: bool) -> float:
    """A finite float from `dotted`, or `default`. `positive` rejects 0 and negatives."""
    value = _optional(raw, dotted, None)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"config: '{dotted}' must be a number, got {_shown(value)}")
    number = float(value)
    if not math.isfinite(number):
        raise ConfigError(f"config: '{dotted}' must be finite, got {value!r}")
    if positive and number <= 0.0:
        raise ConfigError(f"config: '{dotted}' must be positive, got {value!r}")
    if not positive and number < 0.0:
        raise ConfigError(f"config: '{dotted}' must not be negative, got {value!r}")
    return number


def _optional_int_none(raw: dict[str, Any], dotted: str) -> int | None:
    """An optional positive integer that may be absent (None), never zero."""
    value = _optional(raw, dotted, None)
    if value is None:
        return None
    return _positive_int(raw, dotted)


def load_config(path: str | Path) -> Config:
    """Read and validate a config file. Raises ConfigError with a clear message."""
    file_path = Path(path)

    if file_path.is_dir():
        raise ConfigError(f"config: expected a file, got a directory {file_path}")
    if not file_path.is_file():
        raise ConfigError(f"config file not found: {file_path}")

    try:
        with file_path.open("rb") as handle:
            raw = tomllib.load(handle)
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f"config: {file_path} is not valid TOML: {error}") from error
    except OSError as error:
        raise ConfigError(f"config: cannot read {file_path}: {error}") from error

    if not isinstance(raw, dict):
        raise ConfigError(f"config: {file_path} must contain a table of settings")

    return Config(
        path=file_path,
        raw=raw,
        model_name=_string(raw, "model.name"),
        model_revision=_string(raw, "model.revision"),
        judge_model=_string(raw, "judge.model"),
        split_seed=_non_negative_int(raw, "data.split_seed"),
        steps=_optional_positive_int(raw, "fake.steps", DEFAULT_STEPS),
        eval_every_steps=_optional_positive_int(
            raw, "run.eval_every_steps", DEFAULT_EVAL_EVERY_STEPS
        ),
        eval_questions=_optional_positive_int(
            raw, "run.eval_questions", DEFAULT_EVAL_QUESTIONS
        ),
        eval_subset_seed=_optional_non_negative_int(
            raw, "run.eval_subset_seed", DEFAULT_EVAL_SUBSET_SEED
        ),
        checkpoint_every_steps=_optional_positive_int(
            raw, "run.checkpoint_every_steps", DEFAULT_CHECKPOINT_EVERY_STEPS
        ),
        eval_final_questions=_optional_int_none(raw, "run.eval_final_questions"),
        generations_per_prompt=_optional_positive_int(
            raw, "training.generations_per_prompt", DEFAULT_GENERATIONS_PER_PROMPT
        ),
        epochs=_optional_positive_int(raw, "training.epochs", DEFAULT_EPOCHS),
        temperature=_optional_float(
            raw, "training.temperature", DEFAULT_TEMPERATURE, positive=True
        ),
        learning_rate=_optional_float(
            raw, "training.learning_rate", DEFAULT_LEARNING_RATE, positive=True
        ),
        lora_rank=_optional_positive_int(
            raw, "training.lora_rank", DEFAULT_LORA_RANK
        ),
        lora_alpha=_optional_positive_int(
            raw, "training.lora_alpha", DEFAULT_LORA_ALPHA
        ),
        max_prompt_tokens=_optional_positive_int(
            raw, "training.max_prompt_tokens", DEFAULT_MAX_PROMPT_TOKENS
        ),
        max_completion_tokens=_optional_positive_int(
            raw, "training.max_completion_tokens", DEFAULT_MAX_COMPLETION_TOKENS
        ),
        clip_epsilon=_optional_float(
            raw, "training.clip_epsilon", DEFAULT_CLIP_EPSILON, positive=True
        ),
        kl_beta=_optional_float(
            raw, "training.kl_beta", DEFAULT_KL_BETA, positive=False
        ),
        seed=_optional_non_negative_int(raw, "training.seed", DEFAULT_SEED),
        max_steps=_optional_int_none(raw, "budget.max_steps"),
        max_judge_calls=_optional_int_none(raw, "budget.max_judge_calls"),
    )
