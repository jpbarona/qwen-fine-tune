"""Tests for the packaging: the pinned train extra and the shipped configs."""

from __future__ import annotations

import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_train_extra_pins_the_gpu_stack() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = pyproject["project"]["optional-dependencies"]
    assert "train" in extras
    pins = {}
    for requirement in extras["train"]:
        name, _, version = requirement.partition("==")
        assert version, f"{requirement!r} must pin an exact version"
        pins[name] = version
    for name in ("torch", "transformers", "peft", "accelerate", "safetensors"):
        assert name in pins, f"{name} must be pinned in the train extra"


def test_train_extra_does_not_pin_torch_via_a_range() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    train = pyproject["project"]["optional-dependencies"]["train"]
    assert all(">=" not in requirement and "<" not in requirement for requirement in train)


def test_requires_python_is_3_12_or_newer() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["requires-python"] == ">=3.12"
