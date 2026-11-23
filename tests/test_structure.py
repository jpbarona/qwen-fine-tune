"""The package shape we rely on.

Two properties are cheap to lose and expensive to notice late:

- the values that cross the experiment seam live in one module, so a caller
  does not have to import the training loop to name a prompt
- the module import graph has no cycle, and a base module such as the config
  loader does not reach into a domain

All of it is read from the source with the standard library, so the check never
imports the package itself.
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "rlvr"

SEAM_VALUES = frozenset({"Prompt", "Generation", "TestResult", "Outcome", "JudgeResult"})


def _sources() -> dict[str, Path]:
    found: dict[str, Path] = {}
    for path in sorted(SRC.rglob("*.py")):
        parts = list(path.relative_to(SRC).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        found[".".join(parts)] = path
    return found


def _resolve(module: str, node: ast.ImportFrom) -> list[str]:
    """The dotted modules an ImportFrom points at, without the rlvr prefix."""
    if node.level == 0:
        target = node.module or ""
        return [target[len("rlvr.") :]] if target.startswith("rlvr.") else [target]
    parts = module.split(".")
    base = parts[:-node.level]
    if node.module:
        return [".".join([*base, *node.module.split(".")])]
    return [".".join([*base, alias.name]) for alias in node.names]


def _imports(module: str, path: Path) -> dict[str, set[str]]:
    """Map each module this file imports to the names it takes from it."""
    found: dict[str, set[str]] = {}
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.ImportFrom):
            continue
        names = {alias.name for alias in node.names} if node.module else set()
        for target in _resolve(module, node):
            found.setdefault(target, set()).update(names)
    return found


def test_the_seam_values_are_not_imported_from_the_training_loop() -> None:
    offenders: list[str] = []
    for name, path in _sources().items():
        if name in {"training.experiment", "core.values"}:
            continue
        taken = _imports(name, path).get("training.experiment", set()) & SEAM_VALUES
        if taken:
            offenders.append(f"{name} imports {', '.join(sorted(taken))} from experiment")
    assert not offenders, "\n".join(offenders)


def test_the_config_loader_stays_in_the_base_layer() -> None:
    module = "core.config"
    sources = _sources()
    targets = {
        target for target in _imports(module, SRC / "core" / "config.py") if target in sources
    }
    outside = sorted(target for target in targets if not target.startswith("core."))
    assert not outside, f"config imports {', '.join(outside)}"


def test_the_module_graph_has_no_cycle() -> None:
    sources = _sources()
    graph = {
        name: {target for target in _imports(name, path) if target in sources and target != name}
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
