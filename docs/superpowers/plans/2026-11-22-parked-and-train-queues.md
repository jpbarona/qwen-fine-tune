# Plan: parked and train queues

Spec: `docs/superpowers/specs/2026-11-22-parked-and-train-queues-design.md`.
Status: ready for implementation.
Branch: `feat/parked-train-queues`, base `0be0851`.

## Why

The spread schedule computes each block's start from the step number. A change
in block size moves the start backwards, and a pool change reshuffles the list.
The 22 November run missed 52 of 267 parked prompts and repeated 101 of 316
slots. The train block has the same fault. The rehearsal pool holds 174 solved
prompts that training never draws. The demotion rule needs more steps than a run
has.

This plan replaces the step arithmetic with a saved queue per pool, retires
solved prompts, and lowers the demotion threshold.

## Global constraints

- Tests: `.venv/bin/python -m pytest -q` must pass. Baseline is 1118 passed, 5
  skipped. Dashboard tests are untouched.
- No GPU work and no network in any task.
- The reward, the optimizer, the learning rate, the eval cohort and the split
  are untouched.
- Commits use `Juan Pablo Barona <141630180+jpbarona@users.noreply.github.com>`
  for both author and committer.
- Exact values below are binding.
- Doc style: plain English, no semicolons, no em dashes.

## Binding values

    generations_per_prompt = 16
    rollouts_per_step      = 304
    prompts_per_step       = 304 // 16 = 19
    parked block           = max(4, 19 // 2) = 9
    train block            = 19 - 9 = 10
    promotion bar          = 2 of 16
    demotion               = 3 zero-pass groups across the run
    retired                = 16 of 16 passes
    refresh_steps          = removed

The parked block is `max(4, prompts_per_step // 2)`. If that leaves no train
prompt, use `prompts_per_step - 1`.

## Task 1: Retire rehearsal

Files: `src/rlvr/pools.py`, `src/rlvr/cli.py`, `src/rlvr/experiment.py`,
`tests/test_pools.py`, `tests/test_cli.py`, `tests/test_experiment.py`,
`tests/test_screening.py`, `tests/test_dashboard.py`, `README.md`.

- Rename the `Pools.rehearsal` field to `Pools.retired`.
- In `from_rows`, a row with every sample passing goes to `retired`.
- `_pools_meta` writes `retired` in place of `rehearsal`.
- `_pool_prompts` still holds only train and parked ids.
- `build_pools` includes retired ids in the coverage check.
- Update every test fixture and README sentence that says rehearsal.
- Training behaviour does not change, because rehearsal was never drawn.

Tests: a 16-of-16 prompt is retired. Meta records the retired count. Retired ids
are absent from the train and parked prompts.

## Task 2: Queue helper and saved state

Files: new `src/rlvr/queues.py`, new `tests/test_queues.py`,
`src/rlvr/experiment.py`.

- Add `PromptQueue` with `build(ids, seed)`, `take(n)`, `keep(ids)`,
  `append(id)`, `remove(id)` and `as_list()`.
- `build` shuffles the ids with the run seed once.
- `take(n)` returns the first `n` ids and moves those ids to the back in the
  same order. If `n` is at least the queue length, return every id once.
- `keep(ids)` drops ids that left the pool. `append` ignores an id already in
  the queue.
- The run writes one `queues` row after each step:
  `{"step": n, "parked_queue": [...], "train_queue": [...]}`.
- The replay drops saved ids that are not in the current pool and appends pool
  ids missing from the saved queue, in sorted order.

Tests: coverage before a repeat, cycle order, `keep` and `append`, and a saved
queue that restores with new and removed ids.

## Task 3: Spread queue and share

Files: `src/rlvr/schedule.py`, `src/rlvr/experiment.py`, `src/rlvr/cli.py`,
`src/rlvr/config.py`, `configs/lfm-spread.toml`, `configs/lfm-sweep.toml`,
`tests/test_schedule.py`, `tests/test_config.py`, `tests/test_experiment.py`,
`README.md`.

- Remove `Spread.cap` and `Spread.refresh_steps`.
- `parked_count` becomes `max(4, prompts_per_step // 2)`, capped at
  `prompts_per_step - 1`.
- `plan` takes the parked block from the parked queue and the train block from
  the train queue. It no longer uses the step number.
- A promoted parked id leaves the parked queue and joins the back of the train
  queue. A demoted train id leaves the train queue and joins the back of the
  parked queue.
- Remove `training.refresh_steps` from `config.py` and both arm configs. An old
  config that still sets it keeps the key in the raw table and ignores it, like
  the removed curriculum keys.
- Replace `spread_parked_warning` with a warning when the parked block is below
  its planned share. The old 321-prompt warning no longer applies.
- `Spread` gets `state_row` and `restore_state`, so the run saves both queues.

Tests: 19 prompts give 9 parked and 10 train. A small step keeps one train
prompt. Every parked prompt is shown before a repeat. A promotion and a demotion
land at the correct end of the other queue. A resume continues the same order.

## Task 4: Sweep train queue

Files: `src/rlvr/schedule.py`, `tests/test_schedule.py`.

- Keep the sweep order and position for parked steps. They already visit each
  prompt once.
- Add a train queue for the gap steps. A gap step takes its train block from
  that queue.
- A promoted id joins the back of the train queue.
- A demoted id leaves the train queue and joins the end of the sweep order.
- Save the train queue in `Sweep.state_row` and restore it with the position and
  order.

Tests: a gap step covers the train pool before a repeat. A resume mid-gap
continues the same order.

## Task 5: Demotion at three

Files: `src/rlvr/schedule.py`, `src/rlvr/experiment.py`,
`tests/test_schedule.py`.

- `DEFAULT_DEMOTE_AFTER` becomes 3.
- `_observe_outcome` adds one to the count on a zero-pass group. A pass does not
  clear the count. A demotion resets it to 0.
- The saved `streak` rows keep the same shape, so old runs still replay.
- Record a step row when the count changes.

Tests: three zero-pass groups demote. A pass between them does not reset the
count. A demotion resets the count to 0.

## Task 6: Resume rejects a changed step size

Files: `src/rlvr/resume.py`, `tests/test_resume.py`.

- Read `saved_config["training"]["rollouts_per_step"]` and
  `saved_config["training"]["generations_per_prompt"]` when present.
- Raise `ResumeError` when either differs from the requested config.
- Keep the existing allowance for older metadata that omits the fields.

Tests: a changed `rollouts_per_step` is rejected. A changed
`generations_per_prompt` is rejected. Older metadata without the fields still
resumes.

## Task 7: Docs and comments

Files: `README.md`, `docs/DECISIONS.md`, `docs/IDEAS.md`, `docs/INDEX.md`,
config comments.

- Add D41 to `docs/DECISIONS.md` with the queue, share, retirement and demotion
  decisions, and the 22 November evidence.
- In `docs/IDEAS.md`, mark parked coverage, demotion and retirement as chosen.
  Remove the second-passing-group idea and the old rehearsal-set idea. Rewrite
  the "None is chosen" line.
- Update the README pool section, the spread arithmetic and the demotion rule.
- Add the spec and plan to `docs/INDEX.md`.

Tests: none. The documentation must match the code.

## Order

1. Task 1
2. Task 2
3. Task 3
4. Task 4
5. Task 5
6. Task 6
7. Task 7
