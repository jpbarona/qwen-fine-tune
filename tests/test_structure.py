"""The package shape we rely on.

Two properties are cheap to lose and expensive to notice late:

- the values that cross the experiment seam live in one module, so a caller
  does not have to import the training loop to name a prompt
- the module import graph has no cycle

Both are read from the source with the standard library, so the check needs no
imports of the package itself.
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "rlvr"

SEAM_VALUES = frozenset({"Prompt", "Generation", "TestResult", "Outcome", "JudgeResult"})


def _sources() -> dict[str, Path]:
    return {path.stem: path for path in sorted(SRC.glob("*.py"))}


def _local_imports(path: Path) -> dict[str, set[str]]:
    """Map each sibling module this file imports to the names it takes."""
    found: dict[str, set[str]] = {}
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.ImportFrom) or node.level != 1 or not node.module:
            continue
        root = node.module.split(".")[0]
        names = {alias.name for alias in node.names}
        found.setdefault(root, set()).update(names)
    return found


def test_the_seam_values_are_not_imported_from_the_training_loop() -> None:
    offenders: list[str] = []
    for name, path in _sources().items():
        if name in {"experiment", "values"}:
            continue
        taken = _local_imports(path).get("experiment", set()) & SEAM_VALUES
        if taken:
            offenders.append(f"{name} imports {', '.join(sorted(taken))} from experiment")
    assert not offenders, "\n".join(offenders)


def test_the_module_graph_has_no_cycle() -> None:
    sources = _sources()
    graph = {
        name: {root for root in _local_imports(path) if root in sources and root != name}
        for name, path in sources.items()
    }

    state: dict[str, int] = {}
    stack: list[str] = []
    cycles: list[str] = []

    def visit(node: str) -> None:
        state[node] = 1
        stack.append(node)
        for child in sorted(graph[node]):
            if state.get(child) == 1:
                start = stack.index(child)
                cycles.append(" -> ".join(stack[start:] + [child]))
            elif state.get(child, 0) == 0:
                visit(child)
        stack.pop()
        state[node] = 2

    for node in sorted(graph):
        if state.get(node, 0) == 0:
            visit(node)

    assert not cycles, "\n".join(sorted(set(cycles)))
