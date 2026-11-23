# Plan: MLflow saved view

Date: 23 November 2026.
Spec: `docs/superpowers/specs/2026-11-23-mlflow-saved-view-design.md`.

Two tickets. Ticket 2 blocks on ticket 1.

## Ticket 1: the view module and three unit tests

Build the new module with the builder, the encoder and the writer. Add the
three unit tests. No CLI yet.

Files:

- `src/rlvr/mlflow_view.py`
- `tests/test_mlflow_view.py`

Acceptance:

- The builder returns eight sections in order, the exact membership the spec
  names, the three core cards in order and the three core ranges. Other cards
  have no range.
- The filter is ``tags.`rlvr/run_dir` != ''``.
- The encoder returns the envelope with a `deflate;` state that zlib can
  inflate, and raises above 20000 characters.
- The writer sets `mlflow.sharedViewState.rlvr-study` on the `rlvr` experiment
  in a temporary SQLite store, and a second write replaces the tag.
- The three unit tests pass.

## Ticket 2: the CLI and the integration test

Add the `saved-view` subcommand, the command handler and the integration test.
Update the README and the docs index.

Files:

- `src/rlvr/cli.py`
- `tests/test_mlflow_view.py`
- `README.md`
- `docs/INDEX.md`
- `docs/DECISIONS.md`
- `docs/journal.md`

Acceptance:

- `rlvr saved-view --runs-root <dir>` writes the tag and prints the hash route.
- The integration test runs the CLI against a temporary store, reads the tag
  with `MlflowClient`, decodes it and checks the route.
- A second CLI run leaves one tag.
- The full suite passes.
- The docs name the command, the route and the fixed ranges.

## Ticket 3: review

Read the spec and the diff. Run the suite. Check the tag shape against the
MLflow 3.18 source. Fix findings in one pass.
