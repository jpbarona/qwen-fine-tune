"""Tests for the GitHub Pages publish command against a local bare repo.

Nothing here touches a network or real remote. The remote is a bare repository
in tmp_path, which is the closest local stand-in for a Pages branch.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from rlvr.publish import PublishError, publish_report

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


def git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def make_bare_repo(tmp_path: Path) -> Path:
    bare = tmp_path / "pages.git"
    subprocess.run(
        ["git", "init", "--bare", str(bare)],
        capture_output=True,
        text=True,
        check=True,
    )
    return bare


def test_publish_pushes_the_report_to_the_pages_branch(tmp_path) -> None:
    report = tmp_path / "report.html"
    report.write_text("<!DOCTYPE html><h1>study</h1>\n", encoding="utf-8")
    bare = make_bare_repo(tmp_path)

    commit = publish_report(report, remote=str(bare), branch="gh-pages")

    assert len(commit) == 40
    assert git(bare, "show", "gh-pages:index.html") == "<!DOCTYPE html><h1>study</h1>"
    assert git(bare, "show", "gh-pages:.nojekyll") == ""
    assert git(bare, "rev-parse", "gh-pages") == commit


def test_publish_replaces_an_existing_pages_branch(tmp_path) -> None:
    report = tmp_path / "report.html"
    report.write_text("first", encoding="utf-8")
    bare = make_bare_repo(tmp_path)
    publish_report(report, remote=str(bare), branch="gh-pages")

    report.write_text("second", encoding="utf-8")
    publish_report(report, remote=str(bare), branch="gh-pages")

    assert git(bare, "show", "gh-pages:index.html") == "second"
    assert len(git(bare, "rev-list", "gh-pages").splitlines()) == 1


def test_publish_accepts_another_branch_name(tmp_path) -> None:
    report = tmp_path / "report.html"
    report.write_text("report", encoding="utf-8")
    bare = make_bare_repo(tmp_path)
    publish_report(report, remote=str(bare), branch="pages")
    assert git(bare, "show", "pages:index.html") == "report"


def test_publish_uses_the_given_commit_message(tmp_path) -> None:
    report = tmp_path / "report.html"
    report.write_text("report", encoding="utf-8")
    bare = make_bare_repo(tmp_path)
    publish_report(report, remote=str(bare), branch="gh-pages", message="study report")
    assert git(bare, "log", "-1", "--format=%s", "gh-pages") == "study report"


def test_a_missing_report_is_named(tmp_path) -> None:
    with pytest.raises(PublishError, match="report not found"):
        publish_report(tmp_path / "absent.html", remote=str(tmp_path / "bare.git"))


@pytest.mark.parametrize("branch", ["", "a/b", "..", "with space"])
def test_a_bad_branch_name_is_rejected(tmp_path, branch: str) -> None:
    report = tmp_path / "report.html"
    report.write_text("report", encoding="utf-8")
    with pytest.raises(PublishError, match="branch"):
        publish_report(report, remote=str(tmp_path / "bare.git"), branch=branch)


def test_a_remote_that_is_not_a_repository_fails_with_git_output(tmp_path) -> None:
    report = tmp_path / "report.html"
    report.write_text("report", encoding="utf-8")
    with pytest.raises(PublishError, match="git push"):
        publish_report(report, remote=str(tmp_path / "not-a-repo"), branch="gh-pages")
