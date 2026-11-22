"""Architecture details shared by the causal language model training paths.

Keep GRPO, reward scoring and checkpoint handling independent of model family.
Only attention projection names, chat template options and runtime constraints
belong here. Profiles are selected from the loaded model configuration, with
the repository name available for startup checks before downloading weights.
"""

from __future__ import annotations

from dataclasses import dataclass

#: The adapter placements a training config may choose.
LORA_TARGETS: tuple[str, ...] = ("attention", "all-linear")


@dataclass(frozen=True)
class ModelProfile:
    model_type: str
    attention_targets: tuple[str, ...]
    supports_thinking_toggle: bool
    minimum_transformers_major: int = 4
    #: Oldest vLLM release that can serve this architecture with LoRA, or None
    #: when any release works. LFM2 landed in 0.10.2, and LoRA over its fused
    #: MLP and short convolution landed in 0.17.0. Qwen keeps no floor, so its
    #: pinned 0.10.1 stack and older releases behave as before.
    minimum_vllm: tuple[int, int] | None = None


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
    (0, 17),
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
            "Install the runtime with pip install -e '.[vllm]'."
        )


def validate_inference_engine(
    model_name: str, *, vllm_version: str | None = None
) -> None:
    """Check the installed vLLM can serve this architecture with LoRA.

    Qwen keeps no version floor. LFM2 needs 0.17.0 or newer, because earlier
    releases cannot place LoRA on the fused MLP and short convolution.
    `vllm_version` lets a test pin the version without the package. A package
    that is not installed at all is left to the engine builder, which reports
    the install command.
    """
    profile = model_profile(model_name)
    if profile.minimum_vllm is None:
        return
    if vllm_version is None:
        vllm_version = _installed_vllm_version()
        if vllm_version is None:
            return
    installed = _major_minor(vllm_version)
    if installed < profile.minimum_vllm:
        minimum = ".".join(str(part) for part in profile.minimum_vllm) + ".0"
        raise ValueError(
            f"{profile.model_type} rollouts on vLLM require vllm >= {minimum}, "
            f"but version {vllm_version} is installed. Install it with "
            "pip install -e '.[vllm]'."
        )


def _installed_vllm_version() -> str | None:
    """The installed vLLM version, or None when the package is not installed.

    A wheel records its version in the package metadata. A source checkout on
    PYTHONPATH often does not, so fall back to `vllm.__version__`. A package
    that imports but reports no version fails here instead of passing the
    version gate unchecked.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("vllm")
    except PackageNotFoundError:
        pass
    try:
        import vllm
    except ImportError:
        return None
    found = getattr(vllm, "__version__", None)
    if not isinstance(found, str) or not found:
        raise ValueError(
            "cannot read the vllm version: the package metadata and "
            "vllm.__version__ are both missing. Reinstall vllm."
        )
    return found


def _major_minor(version: str) -> tuple[int, int]:
    """The first two numeric parts of a version string, for a floor check."""
    parts = version.split(".")
    try:
        return int(parts[0]), int(parts[1])
    except (IndexError, ValueError) as error:
        raise ValueError(f"cannot read a version from {version!r}") from error
