# Agent notes

Read `README.md` for the study and `docs/INDEX.md` for every design, plan and
report. This file holds rules, not facts.

## Rules

- One home per fact. Tool paths, credentials and the machine setup live in
  [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md). Do not copy them here.
- Never print a token or copy one into a tracked file.
- The package is layered. `.importlinter` enforces the order:

  ```text
  app -> report -> tracking -> training -> engine | evaluation -> core
  ```

  `rlvr.training.experiment` is the only orchestration point. Seam values live
  in `rlvr.core.values`. Do not import a seam value from `experiment`.
- Tests mirror the packages.
- Before you report work as done, run:

  ```bash
  .venv/bin/lint-imports
  .venv/bin/python -m pytest -q
  ```

  A fake run with the default config needs the test caveat in
  [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md).
- Open a pull request with `~/.local/bin/gh pr create`. The full path matters.
- CI is manual on purpose. Nothing runs on push or pull request. Start it with
  `~/.local/bin/gh workflow run Checks` only when the owner asks for it.
