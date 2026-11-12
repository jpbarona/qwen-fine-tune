"""Config loading and validation.

The config is TOML, read with the standard library `tomllib`. A run reads one
config file and records it verbatim in `meta.json`, so a run folder describes
itself.

Required keys:

    [model] name, revision
    [judge] model
    [data]  split_seed
    [fake]  steps, prompts_per_step

Extra keys and tables are kept and ignored here. Later tasks add the training,
data, and judge settings they need without changing this loader's contract.

The judge API key is never part of the config. It comes from the environment in
the task that calls the judge.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["Config", "ConfigError", "default_config_path", "load_config"]

#: Resource path of the config used when --config is not given.
DEFAULT_CONFIG_RESOURCE = "resources/default_config.toml"


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
    prompts_per_step: int


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
        steps=_positive_int(raw, "fake.steps"),
        prompts_per_step=_positive_int(raw, "fake.prompts_per_step"),
    )
