"""Reproducibility data recorded in every meta.json.

git_commit returns None when git is unavailable or the tree is not a repository,
so a run started from a source tarball still completes.
"""

from __future__ import annotations

import platform
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

__all__ = ["PACKAGE_NAMES", "git_commit", "package_versions"]

#: Packages whose versions are recorded. Absent packages are recorded as null.
PACKAGE_NAMES: tuple[str, ...] = (
    "rlvr",
    "torch",
    "transformers",
    "peft",
    "datasets",
    "trl",
)


def package_versions() -> dict[str, str | None]:
    """The interpreter version plus the version of each tracked package."""
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for name in PACKAGE_NAMES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def git_commit(cwd: Path | None = None) -> str | None:
    """The current git commit, or None when it cannot be determined."""
    workdir = str(cwd) if cwd is not None else None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    commit = result.stdout.strip()
    return commit or None
