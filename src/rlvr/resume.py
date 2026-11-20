"""Checkpoint resume state and compatibility checks, without model imports."""

from __future__ import annotations

import json
from pathlib import Path

from .batching import BatchSizePolicy
from .config import Config


class ResumeError(ValueError):
    """A checkpoint cannot be safely resumed with the requested settings."""


def resume_batch_policy(config: Config, manifest: dict) -> BatchSizePolicy | None:
    """The saved batch policy, or None for an older manifest.

    A manifest written before batching has no `batch` table. A corrupt table is
    ignored so the run still resumes from the step, which is the part that
    matters most.
    """
    batch = manifest.get("batch")
    if not isinstance(batch, dict):
        return None
    current = batch.get("current")
    largest_fit = batch.get("largest_fit", 0)
    settled = batch.get("settled", False)
    if (
        isinstance(current, bool)
        or not isinstance(current, int)
        or current < 1
        or isinstance(largest_fit, bool)
        or not isinstance(largest_fit, int)
        or largest_fit < 0
        or not isinstance(settled, bool)
    ):
        return None
    return BatchSizePolicy(
        floor_mib=config.memory_floor_mib,
        max_prompts=config.max_prompts,
        current=current,
        largest_fit=largest_fit,
        settled=settled,
    )


def resume_stage(manifest: dict) -> tuple[int | None, int, list[str] | None]:
    """The saved curriculum position, or defaults for an older manifest.

    `None` for the index and the pool means "start the curriculum from the
    beginning", which is what a pre-batching manifest gets.
    """
    stage = manifest.get("stage")
    if not isinstance(stage, dict):
        return None, 0, None
    index = stage.get("index")
    step = stage.get("step", 0)
    pool = stage.get("pool")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        return None, 0, None
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        return None, 0, None
    if not isinstance(pool, list) or not all(isinstance(item, str) for item in pool):
        return index, step, None
    return index, step, list(pool)


def manifest_count(manifest: dict, name: str, default: int) -> int:
    """A non-negative checkpoint manifest field, or the default.

    A field absent from an older manifest falls back to the default. A corrupt
    value does the same rather than stopping the resume, because the step and
    arm are the only fields a checkpoint needs to be usable.
    """
    value = manifest.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return default
    return value


def validate_resume(meta_path: Path, config: Config, *, arm: str, fake: bool) -> None:
    """Check saved identity before loading weights or changing any run files.

    Older metadata may omit individual fields. Validate the fields it records
    while allowing those older runs to retain their existing resume behavior.
    """
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ResumeError(f"cannot read resume metadata {meta_path}: {error}") from error
    if not isinstance(meta, dict):
        raise ResumeError(f"resume metadata must be an object: {meta_path}")
    saved_config = meta.get("config")
    saved_config = saved_config if isinstance(saved_config, dict) else {}
    model = saved_config.get("model")
    model = model if isinstance(model, dict) else {}
    trainer = meta.get("trainer")
    trainer = trainer if isinstance(trainer, dict) else {}
    saved = {
        "arm": meta.get("arm"),
        "mode": meta.get("mode"),
        "model.name": model.get("name", trainer.get("model_name")),
        "model.revision": model.get("revision", meta.get("model_revision")),
        "data.split_seed": meta.get("split_seed"),
    }
    requested = {
        "arm": arm,
        "mode": "fake" if fake else "real",
        "model.name": config.model_name,
        "model.revision": config.model_revision,
        "data.split_seed": config.split_seed,
    }
    for key, value in saved.items():
        if value is not None and value != requested[key]:
            raise ResumeError(
                f"resume {key} mismatch: saved {value!r}, requested {requested[key]!r}. "
                "Use the original config or a separate --runs-root for a new model."
            )
