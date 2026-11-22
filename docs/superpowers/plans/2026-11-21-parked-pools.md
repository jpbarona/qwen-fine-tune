# Plan: pools from pass counts and two parked re-screen policies

Spec: `issues/19-parked-pools-spec.md`.
Tickets: `issues/20` to `issues/26`.
Status: ready for implementation.
Branch: `experimental/parked-pools`, base `a6faf68`.

## Why

Two runs spent most of their steps on prompt groups with no reward spread. The
fix is to build the training pool from measured pass counts and to re-screen
parked prompts inside training. Two re-screen policies are compared by a real
100-step arm each.

The plan also removes the Hugging Face sampling path, so the repository has one
engine.

## Global constraints

- Tests: `.venv/bin/python -m pytest -q` must pass. Baseline is 989 passed, 5
  skipped.
- No GPU work, no network, in any task. The screen and the arms run later, by
  hand.
- The reward, the optimizer and the learning rate are untouched.
- The pinned 100-question eval cohort and the 250-question split are untouched.
- vLLM is the only sampling engine after Task 1. The base model still loads in
  process for the optimizer.
- Exact values below are binding.
- Doc style: plain English, no semicolons, no em dashes.

## Binding values

    generations_per_prompt = 16
    rollouts_per_step      = 608
    prompts_per_step       = 608 // 16 = 38
    parked cap             = 608 // 4 // 16 = 9 prompts
    refresh_steps          = 40
    promotion_bar          = 2 of 16
    demote_after           = 10 consecutive full-failure groups
    sweep gap ratio        = 3, allowed 3 to 10
    thinking_budget        = 1024

The 608 is a config default taken from the 19 November run, where 608 sequences
at group size 8 took 236 seconds. It is not a constant in code.

## Task 1: vLLM is the only sampling engine

Ticket `20`. Files: `src/rlvr/qwen.py`, `src/rlvr/vllm_backend.py`,
`src/rlvr/model_profiles.py`, `src/rlvr/cli.py`, `src/rlvr/config.py`,
`pyproject.toml`, `README.md`, `tests/`.

- Delete the Hugging Face sampling methods and every branch that chooses between
  engines. Sampling always goes through the vLLM backend.
- Delete the `[training] engine` key. An unknown key is a config error, so a
  config that still sets it fails loudly with the key name.
- Keep the vLLM version gate. vLLM must be 0.17 or newer for LFM and unchanged
  for Qwen.
- One optional extra named `vllm` with vllm 0.34.0, torch 2.14, transformers
  5.22.2, peft 0.18.1 and accelerate 1.8.1. Delete the `lfm` and `vllm-lfm`
  extras.
- The screen uses the same backend as training, so its `trainable=False` path
  builds a vLLM generator.
- Delete tests that only cover the HF engine. Keep the version-gate tests.
- README: one environment, one install command.

Tests: a config setting `engine` raises a config error naming the key. The
generator used by the screen and by training is the same class.

## Task 2: pools from screen rows

Ticket `21`. Files: new `src/rlvr/pools.py`, new `tests/test_pools.py`.

    @dataclass(frozen=True)
    class Pools:
        train: tuple[str, ...]
        rehearsal: tuple[str, ...]
        parked: tuple[str, ...]
        samples: int

    def from_rows(rows, *, samples: int, promotion_bar: int = 2) -> Pools

Rules: zero passes parked, all passes rehearsal, otherwise train. A row with a
sample count different from `samples` raises an error naming the row id. Lists are
sorted. Empty input gives three empty tuples.

Tests: the three boundaries, a short-sample row, an empty input, deterministic
ordering.

## Task 3: one pool, no stages

Ticket `22`. Files: `src/rlvr/experiment.py`, `src/rlvr/resume.py`,
`src/rlvr/config.py`, `src/rlvr/budget.py`, `src/rlvr/summary.py`,
`src/rlvr/dashboard.py`, `src/rlvr/web/index.html`, `tests/`.

- Delete `stages` from the experiment and the stage build from the CLI.
- The experiment takes `Pools` and a label file path instead.
- Delete `thinking_budget_for_stage`. One `[training] thinking_budget = 1024` is
  used for every step.
- Every step and eval record carries `stage: "unstaged"`, the existing constant.
- Old checkpoints still resume. The stage position fields become optional and are
  ignored when present.
- The dashboard shows the stage field as it is, so an unstaged run renders.
- `eval-band` and the band column keep working.

Tests: a run with no stages completes, a resume with no stage position works, a
record carries `unstaged`.

## Task 4: spread schedule

Ticket `23`. Files: new `src/rlvr/schedule.py`, `src/rlvr/experiment.py`,
`src/rlvr/config.py`, new `tests/test_schedule.py`, `tests/test_experiment.py`.

    @dataclass(frozen=True)
    class StepPlan:
        train: tuple[Prompt, ...]
        parked: tuple[Prompt, ...]

    class Schedule(Protocol):
        def plan(self, step: int, train_pool, parked_pool) -> StepPlan
        def observe(self, step: int, outcomes) -> tuple[LabelUpdate, ...]

`Spread(prompts_per_step, cap, refresh_steps)` takes
`k = min(cap, ceil(len(parked) / refresh_steps))` parked prompts per step, in a
seeded rotation. Train prompts fill the rest of `prompts_per_step`.

Parked rollouts enter the same gradient update. After the step, `observe` writes
label updates: promote at 2 of 16 or more, demote a train prompt after ten
consecutive full-failure groups.

Config: `[training] pool_policy = "spread"`, `[training] screen_rows = <path>`,
`[training] rollouts_per_step = 608`.

Step row gains `pool_policy`, `train_prompts`, `parked_prompts` and
`advantage_share`.

Tests: slice arithmetic at P of 0, 9, 300 and 1000, the cap, the seeded rotation
being stable across a resume, promotion and demotion boundaries.

## Task 5: sweep schedule

Ticket `24`. Files: `src/rlvr/schedule.py`, `src/rlvr/config.py`,
`tests/test_schedule.py`.

`Sweep(prompts_per_step, gap_ratio)` covers all parked prompts in
`m = ceil(P / prompts_per_step)` consecutive steps, then trains for
`gap_ratio * m` steps. During a sweep, every prompt in the plan is parked.
Labels update at the end of the sweep.

Config: `pool_policy = "sweep"`, `sweep_gap_ratio = 3`, validated to 3 to 10.

Tests: length and gap arithmetic, a pool smaller than one step, a resume landing
mid-sweep, and both policies producing the same parked share within one prompt
over a full cycle.

## Task 6: screen on vLLM at sixteen samples

Ticket `25`. Files: `src/rlvr/screening.py`, `src/rlvr/cli.py`,
`tests/test_screening.py`.

The screen samples through vLLM, defaults to 16 samples, and writes the same row
schema. Restart behaviour and the torn last line stay as they are.

Tests: the default sample count, the row schema, the restart path, the summary
counts.

## Task 7: two arms and the runbook

Ticket `26`. Files: `configs/lfm-spread.toml`, `configs/lfm-sweep.toml`,
`docs/DECISIONS.md`, `docs/journal.md`, `docs/INDEX.md`, `README.md`.

Both configs are identical except `pool_policy`. 100 steps, base weights, group
size 16, rollout budget 608, rate 1e-4, all-linear LoRA rank 16 and alpha 32,
thinking budget 1024.

The runbook covers install, screen, both arms, the files to pull, and the four
measurements.

Tests: the suite passes and both configs load.

## Order

Tasks 1, 2 and 3 have no blockers. Task 4 needs 2 and 3. Task 5 needs 4. Task 6
needs 1. Task 7 needs 5 and 6.
