# Plan: reward and checkpoint fixes before the next run

Status: ready for implementation.
Branch: `fix/reward-checkpoints`, base `cb5970d`.

## Why

Run two collapsed at step 22 because the model stopped closing its reasoning
block. The reward gave it no reason to close, the truncation penalty was barely
worse than a syntax error, and the first checkpoint was due at step 50. The
shipped config also selects vLLM, which is not installed.

This plan fixes the reward, the budget, the checkpoint cadence and the shipped
engine. It does not touch the optimizer, the loss, or the vLLM weight copy.

## Global constraints

- Tests: `.venv/bin/pytest -q -W error` must pass. Torch is not installed in
  `.venv`, so real-trainer paths stay covered by pure helpers only, matching the
  existing style.
- Never write the literal Qwen reasoning-tag strings in any message, code
  comment, commit message, report or new document. Refer to them as "pufflink"
  (open) and "pufflink close" (close). Existing code may contain them, do not
  add or quote more.
- Do not change the optimizer, the learning rate, or the loss function.
- Do not remove or rewrite the vLLM backend. Only the shipped config changes.
- Exact values below are binding.

## Task 1. Truncation reward

File: `src/rlvr/reward.py`.

- Change `TRUNCATION_REWARD` from `-0.6` to `-1.0`.
- Keep `SYNTAX_ERROR_REWARD` at `-0.5`.
- Update every test that expects `-0.6` for a truncated rollout. Search the
  whole `tests/` tree, for example `grep -rn -- "-0.6" tests`.
- Update the README reward table row for a truncated rollout to `-1.0`.

Tests: a truncated rollout scores `-1.0`, and it stays at `-1.0` after the
length term because of the clip.

## Task 2. Close bonus

Files: `src/rlvr/qwen.py`, `src/rlvr/reward.py`, `src/rlvr/experiment.py`.

- In `reward.py`, add `CLOSE_REWARD = 0.4` and
  `close_reward(tests_total) = min(CLOSE_REWARD, 0.9 / tests_total)`.
- Add a keyword-only parameter `closed: bool = False` to
  `reward_code_score` and to `compose_reward`. Forward it unchanged.
- In `reward_code_score`, closing pays only when no test passed:
  `base = close_reward(tests_total)`. Partial answers are unchanged, so one
  test always beats closing alone, and a full pass always beats a partial
  answer. Truncated and syntax-error rollouts get no close reward.
- Return `min(MAX_REWARD, max(MIN_REWARD, base - penalty))`.
- The other three arm functions accept `closed` and ignore it, so every arm
  keeps one call shape.
- In `qwen.py`, find the `Generation` dataclass and add `closed: bool = False`.
- Add a module-level pure helper
  `thinking_marker_index(token_ids: Sequence[int], marker_id: int) -> int`
  that returns the index just past the last marker token, or `0` when absent.
  Rewrite `_split_at_think_end` to call it.
- In `_decode_group`, set `closed=split > 0` on the `Generation`.
- In `experiment.py`, pass `closed=generation.closed` into `compose_reward`.
- `FakeGenerator` leaves `closed` at its default, so fake-run rewards do not
  change.

Tests: with `closed=True`, a full pass clips at `1.0`, a zero-test failure with
four tests scores `0.225`, a 3-of-4 answer stays `0.75`, and a truncated
rollout stays `-1.0`. Every test count scores strictly above the one below it.
A closed partial answer never beats a full pass, so `0.75` against `0.8`.
Add a pure test for `thinking_marker_index` covering present, absent and
repeated markers.

## Task 3. Thinking budget

Files: new `src/rlvr/budget.py`, `src/rlvr/qwen.py`, `src/rlvr/config.py`,
`src/rlvr/experiment.py`, `configs/real.toml`, `src/rlvr/resources/default_config.toml`.

- New module `budget.py`, pure Python, no torch:
  - `thinking_budget_for_stage(start: int, stage_index: int, cap: int) -> int`
    returns `min(cap, start * 2 ** stage_index)`, and raises `ValueError` for
    `start < 1`, `stage_index < 0` or `cap < 1`.
  - `row_should_stop_thinking(generated_len: int, budget: int, has_marker: bool) -> bool`
    returns `generated_len >= budget and not has_marker`.
- Config: `DEFAULT_THINKING_BUDGET_START = 768`. Add field
  `thinking_budget_start: int = DEFAULT_THINKING_BUDGET_START`. Read
  `training.thinking_budget_start` as a positive int. Add the key to
  `configs/real.toml` with the comment "Thinking is capped at this many tokens
  at the first stage, doubling each stage up to max_completion_tokens".
- `qwen.py`: `generate_batch` accepts `thinking_budget: int | None = None`.
  When it is not `None`, pass a `StoppingCriteria` to `model.generate` that, per
  row, returns `row_should_stop_thinking(new_tokens, thinking_budget, row has
  the marker token)`. Keep the class thin. The decision stays in the pure
  helper.
- `experiment.py` passes
  `thinking_budget_for_stage(config.thinking_budget_start, stage_index, config.max_completion_tokens)`
  to `generate_batch`. Stage index is the curriculum position already known to
  the step loop.
- `Generator` implementations that do not support it (`FakeGenerator`,
  `VLLMGenerator`) accept the keyword and ignore it. Note the limitation in a
  one-line comment.
- `configs/real.toml` value: `thinking_budget_start = 768`.

Tests: schedule gives `768, 1536, 2048, 2048` for stages 0 to 3 with cap 2048.
The predicate stops at the budget only while the marker is absent. Config
loader reads the key and rejects zero.

## Task 4. Checkpoint cadence and retention

Files: `configs/real.toml`, `src/rlvr/config.py`, `src/rlvr/checkpoint.py`,
`src/rlvr/experiment.py`, `src/rlvr/resources/default_config.toml`.

- Set `checkpoint_every_steps = 20` and `eval_every_steps = 20` in
  `configs/real.toml`, and the same defaults in `config.py`. Update the header
  comment in `real.toml`.
- In `checkpoint.py`, add:
  - `select_checkpoints_to_keep(steps: Sequence[int], *, recent: int = 5,
    total: int = 10) -> list[int]`. Sort ascending, keep the newest `recent`,
    then step further back at offsets `2, 4, 8, 16, 32` from the newest
    (cumulative offsets `6, 10, 18, 34, 66`), until `total` is reached.
  - `prune_checkpoints(directory: str | Path, keep: Sequence[int]) -> list[Path]`
    deletes `step-XXXXXXXX` directories whose step is not in `keep`. Never
    touches other files or directories. Returns what it deleted.
- `Experiment.save_checkpoint` calls `prune_checkpoints` with the kept steps
  after the manifest is written, for example 10 checkpoints retained.
- Retention only affects the local run directory, never persistent storage.

Tests: 3 checkpoints keeps all 3. With 40 checkpoints it keeps exactly 10, the
5 newest consecutive plus 5 older at doubling gaps. Prune removes only
`step-` directories that are not kept, and leaves other files alone.

## Task 5. Engine and docs

Files: `configs/real.toml`, `README.md`, `docs/DECISIONS.md`, `docs/INDEX.md`,
`issues/09-spec-single-arm-curriculum.md`.

- `configs/real.toml`: `engine = "hf"`. Comment that vLLM stays opt-in until
  its weight copy is proven.
- `README.md`: reward table row for truncation `-1.0`, one row for the close
  bonus `+0.4`, note the thinking budget. Fix the claim that completion tokens
  are only logged, since the length term uses them.
- `docs/DECISIONS.md`:
  - Delete the claim that medium and hard are cut by measured pass rate, and
    the D5 sentence that band assignment follows the measurement. Labels come
    from agents reading problem statements. The screening command is a separate
    measurement tool and does not re-band anything.
  - Delete the two sentences claiming the gradient division makes the update a
    mean. The line stays in code, the claim goes.
  - Give the close bonus and thinking budget entries a status line saying what
    is implemented, with the exact values 0.4 and 768.
  - Give the checkpoint cadence entry a status line saying 20 and 20, retention
    10 with 5 recent.
- `issues/09-spec-single-arm-curriculum.md`: remove the band-assignment
  criterion that says bands follow measured pass rate.
- `docs/INDEX.md`: test count matches the suite after the change.

Tests: full suite passes. No code change in this task beyond the config.
