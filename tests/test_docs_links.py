"""Every relative Markdown link in the shared docs resolves.

A dead link is a doc that silently points at a deleted file. The README, the
indexes, the agent notes and the docs folder are the files a person or an agent
reads first, so they are the ones checked.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FILES = [
    ROOT / "README.md",
    ROOT / "INDEX.md",
    ROOT / "AGENTS.md",
    ROOT / "deploy" / "README.md",
    *sorted((ROOT / "docs").rglob("*.md")),
]

LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
FENCE = re.compile(r"```.*?```", re.DOTALL)


def _links(path: Path) -> list[str]:
    text = FENCE.sub("", path.read_text(encoding="utf-8"))
    return LINK.findall(text)


def test_relative_markdown_links_resolve() -> None:
    missing: list[str] = []
    for path in FILES:
        if not path.exists():
            continue
        for target in _links(path):
            if target.startswith(("http://", "https://", "#", "mailto:")):
                continue
            local = target.split("#", 1)[0]
            if not local:
                continue
            if not (path.parent / local).exists():
                missing.append(f"{path.relative_to(ROOT)}: {target}")
    assert not missing, "\n".join(missing)
