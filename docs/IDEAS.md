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

## Cross-language evaluation

The problem. Every result so far is one language. The question the papers leave
open is whether a gain on Python arrives on other languages, and whether it
costs anything there.

The idea. Score the same adapter on two public translations of the same
questions. MultiPL-E rewrites HumanEval and MBPP into 18 programming languages
with compiled tests. mHumanEval translates the prompts into more than 200
natural languages and keeps the Python tests, so the code stays the same and
only the reading changes.

Why it might work. A search of the 2025 and 2026 literature found no study of
reinforcement learning transfer across programming languages. The closest work
is cross-lingual RLVR in maths, and it reports stronger transfer than
supervised tuning.

Cost. MultiPL-E needs a runtime and grader per language, roughly a day of
setup. mHumanEval reuses the Python docker runner and needs nothing new. Each
pass over 164 questions is minutes on one card. The expert-translated subset is
the one to use, since machine translation quality varies by language.

Where. A new `eval-languages` path beside `eval-band`, or a script under
`scripts/` while the question is open.

## Upgrade vLLM past 0.17

The problem. The repository pins vLLM 0.10.1, which predates LFM2 entirely, so
the LFM path falls back to Hugging Face generation. A 20 step probe took an
hour. The same work on vLLM is minutes.

The facts. `lfm2` landed in PR 22845 three days after 0.10.1 and first shipped
in 0.10.2. LoRA on LFM's fused MLP and short convolution needed PR 34921, first
shipped in 0.17.0. Current vLLM is 0.34.

Why it might work. Every future run gets the speedup, including the two-model
comparison and every cross-language pass.

Cost. A version bump, a reinstall, and a check that a runtime adapter swap
produces the same logits as the base model. About an hour, and it changes a
validated path.

Where. `pyproject.toml` vllm extra, `validate_inference_engine` in
`model_profiles.py`, and the adapter update path in `qwen.py`.

## Label the 250 test questions by band

The problem. The pinned 100 is a mixed sample, so a gain can hide inside it.
The difficulty labels cover the 711 training rows only, which is why
`eval-band` reads from the training split and carries a contamination caveat.

The idea. Read the 250 test questions and label them into the four bands. Then
the pinned cohort reports per band, and `eval-band` can draw from genuinely
unseen questions.

Cost. Reading 250 short problems, an hour or two. No code change beyond letting
the label loader accept the test split.

Where. `data/difficulty_labels.jsonl`, or a second file for the test side, plus
`eval-band`.

## Stopping rule and rate schedule

The problem. A run at the wrong rate spends hours learning nothing, and the
only lever today is stopping by hand.

The idea. Two parts. First, a written stop rule per run, for example stop at
step 40 if the pinned 100 has not moved. Second, a rate schedule, either a
decay after a fraction of the budget or a drop on the plateau. D28 makes a
manual drop possible today, since a resumed run now takes the configured rate.

Cost. A stop rule is a convention. A scheduler is a small change in the update
loop and one test.

Where. The run notes, and `src/rlvr/experiment.py` for a scheduler.

## Re-screening parked prompts inside training

The problem. Parked prompts, the ones the model fails 8 of 8, are currently left
alone. Some become solvable as training moves, and nothing notices.

The idea. Two ways to keep the parked labels fresh. Both keep the rollouts in the
gradient instead of throwing them away as evaluation work.

Option 1, spread. Every step, include `k` parked prompts beside the training
pool, each with 16 rollouts. The results update the label. `k` is chosen so the
pool cycles once every `N` steps.

    k = min(19, ceil(P / N))

`P` is the parked count and `N` the refresh interval in steps. The cap of 19 is
half of a 38 prompt step, so parked work never passes half the batch. If the pool
grows, raise `N` rather than the share.

Option 2, dedicated batches. Every `N` steps, run `m` steps through the whole
parked pool, group size 16, keeping 608 rollouts per step.

    m = ceil(P / 38)

For `P` of 300 that is 8 dedicated steps.

Why option 1. A prompt's 16 rollouts happen in one step, so its label is already
a snapshot of one adapter. Spreading prompts across steps costs nothing in label
quality. Option 2 spends `m` consecutive steps on a batch that is mostly all-fail
groups, which carry no advantage, and it shifts the batch distribution abruptly.
Same compute, worse shape.

Promotion bar. 2 of 16 or more, so one lucky pass does not promote a prompt that
still fails.

Why it might work. It keeps a pool of prompts that can produce advantage, which
is the failure mode of run 17 and the LFM run, where 61 percent of groups failed
all eight.

Cost. A screen job, a label file, a slice in the step loop, and a test for the
slice cap. About a day.

Where. The prompt draw in `src/rlvr/experiment.py`, a new `src/rlvr/screening.py`,
and `data/` for the label file.

Status. Option 1 chosen on 21 November 2026. Not built.

## Arm: spread against sweep for parked re-screening

The question. At equal compute, does spreading parked re-screens across training
steps produce more gradient than sweeping them in dedicated batches?

Why it might be worth doing. D32 chose spreading on the argument that dedicated
sweeps spend several consecutive steps on mostly all-fail groups. That argument is
reasoning, not a measurement. The two policies differ only in scheduling, so they
make a clean pair with one variable.

Design.

- Both arms start from the same checkpoint, same seed, same pool, same group size
  16, and the same screen labels.
- Arm A, spread. Every step, `k = min(19, ceil(P / N))` parked prompts join the
  training pool.
- Arm B, sweep. Every `N` steps, `m = ceil(P / 38)` consecutive steps cover the
  whole parked pool at 608 rollouts each.
- Match total sequences and step count between arms. If they differ, the
  comparison is unreadable.

Measurements, in order of usefulness.

1. Share of groups with a nonzero advantage.
2. Mean absolute advantage across groups.
3. Promotions per thousand parked sequences.
4. Pass@1 on the pinned 100, as a secondary. It will sit inside five points over
   60 steps, so it cannot decide the question alone.

Decision rule, written before the runs. For example, A wins if the
nonzero-advantage share is higher by at least five points on both cohorts.

Cost. Two boxes, 60 steps each, about four hours and five dollars. Needs the
screen and the pool code first, so it runs after the first screened run.

Risk. The difference may be smaller than the measurement noise. A null result is
still worth writing down, since it settles the scheduling argument.

Where. Same files as the entry above, plus a config key for the policy choice.
