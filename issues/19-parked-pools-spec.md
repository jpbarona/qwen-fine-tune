# Spec: pools from measured pass rates, with two parked re-screen policies

Status: ready-for-agent.
Branch: `experimental/parked-pools`.

## Problem Statement

Two runs show the same failure. Run 17 and the LFM run both spent most steps on
prompt groups that could not produce an advantage: 61 percent of groups failed
all eight rollouts. A group with one reward value gives GRPO zero gradient, so
two thirds of the compute taught nothing.

The cause is the pool, not the model. Difficulty bands are a guess made before
the model runs, and retirement removes exactly the prompts the model can solve.
What stays behind is the set it cannot solve at all, so the pool drifts toward
prompts with no signal.

Separately, the repository carries two sampling engines. The Hugging Face path is
slow, and it is the only path the screen can use, which turns a 16-sample screen
into a three hour job instead of a twenty five minute one.

## Solution

Build the training pool from measured pass counts instead of bands. A screen
runs the model on each training prompt and records how many of 16 rollouts pass.
Prompts with a partial count go to train. Prompts that pass all 16 go to
rehearsal. Prompts that pass none go to parked, and are re-screened inside
training steps so their rollouts still produce gradients.

Two ways to schedule the parked re-screen exist, and the question of which is
better is itself the experiment:

- Spread: a fixed slice of parked prompts joins every step.
- Sweep: all parked prompts are covered in consecutive steps, with a gap of
  training steps in between.

Both arms start from the base model and run 100 steps, so the comparison is a
real training run, not a simulation.

The repository uses vLLM for every rollout. The Hugging Face sampling path is
deleted.

## User Stories

1. As a researcher, I want the training pool built from measured pass counts, so
   that most steps carry a usable advantage.
2. As a researcher, I want a prompt that passes zero of sixteen to leave the
   pool, so that it stops consuming steps.
3. As a researcher, I want parked prompts re-screened inside training, so that
   the rollouts produce gradients instead of being thrown away.
4. As a researcher, I want the re-screen policy chosen by one config key, so that
   the two policies can be compared without a code change.
5. As a researcher, I want both policies to spend the same share of rollouts on
   parked work, so that the comparison is equal compute.
6. As a researcher, I want the parked share capped at a quarter of the rollouts
   per step, so that training keeps most of the batch.
7. As a researcher, I want the sweep to leave a gap of three to ten times its own
   length, so that training dominates each cycle.
8. As a researcher, I want every policy number computed from the parked count and
   the rollout budget, so that nothing is hardcoded to today's pool size.
9. As a researcher, I want a prompt promoted only at two of sixteen or more, so
   that one lucky pass does not move it.
10. As a researcher, I want labels to update only from screen measurements, so
    that selection effects do not feed back into the pool.
11. As a researcher, I want a prompt demoted after repeated full failures in
    training, so that stale labels do not keep a dead prompt in the pool.
12. As a researcher, I want the stage machinery gone from training, so that the
    pass count is the only difficulty signal.
13. As a researcher, I want the band label kept in the data for reporting, so
    that results can still be read per difficulty band.
14. As a researcher, I want one fixed thinking budget per run, so that the pool
    policy is the only variable.
15. As a researcher, I want the screen to use vLLM, so that sixteen samples over
    all training prompts is minutes, not hours.
16. As a researcher, I want the Hugging Face sampling path deleted, so that there
    is one engine to install, test and reason about.
17. As a researcher, I want the whole training step to use vLLM, sampling and the
    adapter handoff alike.
18. As a researcher, I want the run record to say which pool policy produced it,
    so that two arms are distinguishable afterwards.
19. As a researcher, I want each step to record the parked prompt count, the
    train prompt count and the non-zero advantage share, so that the policy can
    be judged from the run files.
20. As a researcher, I want a 100-step arm per policy from the base model, so
    that the checkpoint comparison is a real training result.
21. As a researcher, I want the old LFM checkpoints kept, so that the new arms
    can be compared with the previous method.
22. As an operator, I want resume to work with a single pool, so that a long arm
    survives a restart.
23. As an operator, I want an invalid pool config to fail at load, so that a bad
    arm does not waste GPU hours.
24. As a maintainer, I want the pool and schedule code behind small interfaces,
    so that the two policies share one seam.
25. As a maintainer, I want the dashboard to keep working with the stage field
    gone, so that no view breaks on an unstaged run.

## Implementation Decisions

**One pool module.** A new module returns the pools from screen rows: the train
list, the parked list, the rehearsal list, and the counts. Its interface is a
constructor and one call. The bands are not an input.

**One schedule seam.** The step draw asks a schedule for a plan:

    plan(step_index) -> StepPlan(train=..., parked=...)

Two adapters satisfy it, `Spread` and `Sweep`, selected by a config key. Both
compute their numbers from the parked count and the rollout budget:

    spread: k = min(cap, ceil(P / refresh_steps))
    sweep:  m = ceil(P / prompts_per_step), gap = ratio * m

The cap is a quarter of the rollouts per step, divided by the group size, which
is 9 prompts at 38 prompts and group size 16. The sweep ratio defaults to 3,
which makes both policies spend about a quarter of their rollouts on parked
work. Nothing is written as a fixed 608 anywhere in the code.

**Rollout budget.** A new config key sets the rollout budget per step, with a
default taken from the measured 19 November run. Prompts per step is the budget
divided by the group size.

**Label updates.** After each step, the schedule reads the group outcomes for
parked prompts and writes label updates. Promotion needs a measured 2 of 16 or
more. Demotion after ten consecutive full-failure groups is allowed, promotion
from training counts is not.

**Stages are deleted from training.** The existing unstaged path becomes the only
path. Records carry the existing unstaged stage value. The difficulty band stays
in the data file and in `eval-band`, and no training code reads it.

**Thinking budget.** One fixed budget per run, from config. The per-stage ladder
is deleted with the stages.

**vLLM only.** The engine gate, the engine config key and the Hugging Face
sampling path are deleted. Both training and the screen use vLLM. The optional
dependency extra becomes a single vLLM extra. The base model is still loaded in
process for the optimizer, and sampling always goes through the vLLM backend.

**Records.** Each step row gains the pool policy name, the train prompt count,
the parked prompt count, and the share of groups with a non-zero advantage. The
run metadata gains the policy and the screen path.

## Testing Decisions

Good tests cross the interface, not the implementation. The pools module is
tested from rows to pool membership, including the promotion bar and the
boundary counts. The schedule is tested from a parked count and a rollout budget
to a plan, for both adapters, including the cap, the gap and an empty parked
pool. The step loop is covered by the existing fake-generator tests, extended
with a policy key.

Prior art: `tests/test_stages.py` for pool-shaped tests, `tests/test_batching.py`
for quota arithmetic, and `tests/test_experiment.py` for the fake step loop.

Engine deletion is covered by the suite running with no Hugging Face import in
the sampling path, and by the removed engine tests.

## Out of Scope

- The cross-language MultiPL-E run.
- The Qwen arm. The two policies are compared on LFM first.
- Changes to the reward.
- Changes to the optimizer.
- mHumanEval.
- Automatic pool sizing from a scheduler.

## Further Notes

The arms: LFM2.5-2.6B, 100 steps each, base weights, all-linear LoRA at rank 16
and alpha 32, rate 1e-4, group size 16, rollout budget equal in both arms. The
measurements are the non-zero advantage share, the mean absolute advantage, the
promotions per thousand parked rollouts, and pass@1 on the pinned 100.

The comparison against the old method uses the LFM run checkpoints already
pulled to `rl-grpo-data/2026-11-20-lfm-run100`, base 0.51.
