"""Architecture details shared by the causal language model training paths.

Keep GRPO, reward scoring and checkpoint handling independent of model family.
Only attention projection names, chat template options and runtime constraints
belong here. Profiles are selected from the loaded model configuration, with
the repository name available for startup checks before downloading weights.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelProfile:
    model_type: str
    attention_targets: tuple[str, ...]
    supports_thinking_toggle: bool
    minimum_transformers_major: int = 4


QWEN3 = ModelProfile("qwen3", ("q_proj", "k_proj", "v_proj", "o_proj"), True)
LFM2 = ModelProfile(
    "lfm2",
    (
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.out_proj",
    ),
    False,
    5,
)


def model_profile(model_name: str, *, model_type: str | None = None) -> ModelProfile:
    """Resolve a supported architecture, including locally stored checkpoints."""
    if model_type == "lfm2" or (model_type is None and "lfm2" in model_name.lower()):
        return LFM2
    if model_type in (None, "qwen3"):
        return QWEN3
    raise ValueError(
        f"unsupported model architecture {model_type!r}; supported architectures "
        "are qwen3 and lfm2"
    )


def validate_transformers_version(profile: ModelProfile, version: str) -> None:
    """Fail before model loading when the tokenizer needs a newer runtime."""
    major = int(version.split(".", 1)[0])
    if major < profile.minimum_transformers_major:
        raise ValueError(
            f"{profile.model_type} requires transformers >= "
            f"{profile.minimum_transformers_major}.0.0; installed version is {version}. "
            "Install the LFM environment with pip install -e '.[lfm]'."
        )


def validate_inference_engine(model_name: str, engine: str) -> None:
    """The pinned vLLM stack serves Qwen adapters. LFM uses Hugging Face."""
    if model_profile(model_name) == LFM2 and engine == "vllm":
        raise ValueError(
            'LFM training requires training.engine = "hf" and the [lfm] extra. '
            "The pinned Qwen vLLM environment does not support this LFM tokenizer."
        )
