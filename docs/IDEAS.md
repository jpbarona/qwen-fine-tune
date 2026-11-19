# Ideas

Not built. Each entry says what it is, why it might be worth doing, and what it
costs. Move an entry into `DECISIONS.md` when it is chosen.

## Rehearsal set for retired prompts

The problem. A prompt leaves the stage pool as soon as all eight rollouts pass.
The pool then holds only the prompts the model fails, so most groups reward
nothing and a tenth of them carry an identical reward to all eight rollouts.

The idea. Retire a prompt into a second list instead of deleting it. Each step
draws most prompts from the fresh pool and a fixed share, 20 to 30 percent, from
the rehearsal list, with a seeded rotation so the mix is reproducible.

Why it might work. Mixed batches keep rewards varied, so advantages stay nonzero,
and the model keeps seeing prompts it already solves. The stage still ends
because retirement counts only fresh prompts.

Cost. New state in the step loop, and the training reward becomes easier to read
because easy prompts are always present. Roughly half a day including tests.

Where. The prompt draw in `src/rlvr/experiment.py`, near the stage pool selection,
plus the retirement path.

## Judge arms that shape instead of gate

Two cheap judges, one on the reasoning trace and one on the code with the test
output in hand. Reward becomes the test fraction plus a small judge term smaller
than half of one test, so a full pass always outranks a partial one.

Why it might work. Tests give zero gradient to a group where every rollout scores
zero. A judge breaks those ties. With the rehearsal set in place there may be
little left to fix, so measure first.

Cost. Two model calls per rollout, a new failure mode where a plausible looking
answer scores well, and a reward that is no longer the thing the tests measure.

## Adaptive generations per prompt

Raise generations per prompt from 8 as the pool shrinks, so a step keeps roughly
constant work and the advantage estimate improves with group size.

Blocked by D16. If solved means every rollout passed, a larger group is harder to
pass, so prompts retire more slowly. The workable form judges mastery on the
first eight and lets the rest feed only the advantage.

## Say `evaluating` on the dashboard

Evaluations run every 20 steps for about ten minutes. During that time the page
shows the last training step and the GPU looks busy for no stated reason.

## Move the base evaluation into the runner

D22 is currently a procedure, not code. The runner could score the base model on
its own cohort before step 1 and write the row into `eval.json`.
