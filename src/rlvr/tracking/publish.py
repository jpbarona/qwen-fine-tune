"""Publish the static report to a GitHub Pages branch.

The report is not committed to the study branch. A throwaway repository is made
in a temporary folder, the report is copied in as `index.html`, an empty
`.nojekyll` file is added, one commit is made, and that commit is force-pushed
to the Pages branch. The branch is generated output, so replacing it wholesale
is intended.

The remote is whatever `git push` accepts: an SSH URL, an HTTPS URL, or a path
to a local repository. Credentials come from the caller's git setup. This
module never reads or writes a token.

Real invocation:

    rlvr publish --report report.html \\
        --remote git@github.com:<user>/<repo>.git --branch gh-pages

Use a deploy key with write access to that one repository.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path

__all__ = ["PublishError", "publish_report"]

#: One path segment. A slash would create a nested ref the caller did not ask
#: for, and `..` is never valid in a branch name.
_BRANCH = re.compile(r"^[A-Za-z0-9._-]+$")

_DEFAULT_MESSAGE = "Publish the RLVR study report"


class PublishError(RuntimeError):
    """The report could not be pushed to the Pages branch."""


def _git(cwd: Path, *args: str) -> str:
    command = [
        "git",
        "-c",
        "user.name=rlvr",
        "-c",
        "user.email=rlvr@localhost",
        "-c",
        "commit.gpgsign=false",
        *args,
    ]
    environment = dict(os.environ)
    environment["GIT_TERMINAL_PROMPT"] = "0"
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
            env=environment,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise PublishError(f"cannot run git: {error}") from error
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no output"
        raise PublishError(f"git {args[0]} failed: {detail}")
    return result.stdout.strip()


def publish_report(
    report: str | Path,
    *,
    remote: str,
    branch: str = "gh-pages",
    message: str | None = None,
) -> str:
    """Push `report` to `branch` on `remote` as `index.html`.

    Returns the pushed commit. Raises PublishError for a missing report, a bad
    branch name, or a git failure. The remote is never contacted before the
    report has been read.
    """
    report_path = Path(report)
    if not report_path.is_file():
        raise PublishError(f"report not found: {report_path}")
    if not branch or not _BRANCH.fullmatch(branch) or branch in (".", ".."):
        raise PublishError(f"branch must be one simple name, got {branch!r}")
    if not remote:
        raise PublishError("remote must not be empty")

    content = report_path.read_bytes()
    with tempfile.TemporaryDirectory(prefix="rlvr-pages-") as temporary:
        workdir = Path(temporary)
        _git(workdir, "init", "-q")
        _git(workdir, "symbolic-ref", "HEAD", f"refs/heads/{branch}")
        (workdir / "index.html").write_bytes(content)
        (workdir / ".nojekyll").write_text("", encoding="utf-8")
        _git(workdir, "add", "index.html", ".nojekyll")
        _git(workdir, "commit", "-q", "-m", message or _DEFAULT_MESSAGE)
        _git(workdir, "push", "--force", remote, f"HEAD:refs/heads/{branch}")
        return _git(workdir, "rev-parse", "HEAD")
