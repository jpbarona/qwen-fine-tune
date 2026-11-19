# Band evaluation plan

2026-11-19. Add a way to score a model on one difficulty band, so the medium
questions can be measured without waiting for the curriculum to reach them.

Context: the pinned 100-question cohort comes from the test split, and the test
split has no difficulty labels. The labels cover the 711 training rows. The
existing `run` path only ever evaluates that pinned cohort, so a medium number
is not reachable today.

## Global Constraints

- No behaviour change to `rlvr run`, the split, the labels or the reward.
- No new dependencies.
- CPU-testable: the selection logic must be exercised without a model or GPU.
- Doc style: plain English, no semicolons, no em dashes.
- Suite must pass, currently 961 passed and 5 skipped.

## Task 1: Add the band selection and the command

Files:

- `src/rlvr/evaluation.py`. Add `band_eval_prompts(prompts, labels, band,
  count, seed)` returning a tuple of prompts. It keeps only prompts whose id is
  labelled with the requested band, then applies the existing
  `select_eval_subset`. Raise a clear error when the band holds fewer prompts
  than `count`, and when `band` is not one of `BANDS`.
- `src/rlvr/cli.py`. Add a subcommand `eval-band` with `--config`, `--band`
  (required), `--count` (default 100), `--seed` (default 2026), `--checkpoint`
  (optional directory), `--out` (optional JSON path). It builds the split, loads
  the labels, selects the prompts, builds the generator with
  `_qwen_generator(config, trainable=False)`, loads the checkpoint when given,
  scores greedily with `build_test_runner`, and prints one JSON line with
  `model`, `band`, `pass_at_1`, `count`, `passed`, `seconds`.
- Reuse the scoring pattern in `Experiments._evaluate`: chunk by
  `config.eval_batch_size`, greedy temperature, `all_tests_passed`.

Tests in `tests/test_evaluation.py` and `tests/test_cli.py`:

- selection keeps only the requested band and is deterministic for a seed
- a band with too few prompts raises
- an unknown band raises
- the parser accepts `eval-band` and requires `--band`

Report file: `task-1-report.md`.

## Task 2: Document the command

Files:

- `README.md`: a short subsection under the evaluation material with one example
  invocation and the note that band prompts come from the training split, so
  they stop being held out once training uses them.
- `docs/INDEX.md`: add the command to the repo items if commands are listed.
- `docs/DECISIONS.md`: D27, why a band command exists and its contamination
  caveat.
- `docs/journal.md`: one line.

Report file: `task-2-report.md`.
