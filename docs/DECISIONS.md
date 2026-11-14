# Decisions

Study decisions, newest first. Each one says what was chosen and why.

## D9. Several prompts per step, sized by free memory

A step takes several prompt groups, one group of 8 samples each. The batch starts
at two prompts. Free memory is read after each step, and the batch grows by one
while free memory stays at or above the larger of `[training] memory_floor_mib`
and one prompt group's measured footprint from the previous step. The first
step below that floor settles the batch at the largest size that fit, and it
stays there. A CUDA out-of-memory error does the same instead of crashing. It
never goes below one prompt. There is no percentage rule.

Advantage is normalised inside one prompt group. The backward pass runs once per
generation and gradients accumulate across the group, so only one autograd graph
is alive at a time. This is what makes the 2,048 token cap fit at a larger batch.

A checkpoint records the prompt cursor, the batch policy state, and the
curriculum position. A resume continues on the next batch boundary with the same
prompt position, batch size, and stage pool.

## D1. One arm only for now

The four-arm comparison is paused. The judge and process arms, hybrid and
gated-process, all gate on unit tests. If the model almost never passes tests,
all four arms score zero and the study cannot answer its own question.

Code arm only until the model passes tests at a usable rate. Revisit the arms
after that.

## D2. Thinking on, 2,048 token cap

Thinking stays on, since the target behaviour is concise thinking plus correct
code. The completion cap rises from 768 to 2,048, counted over the whole
output, reasoning and code together.

Prerequisite: drop `output_scores` before raising the cap. Stored logits cost
about 10 GB at 2,048 and 34 GB at 7,000, which does not fit beside the rest.
This is now decided, see D8.

## D3. Reward

| Case | Reward |
|---|---|
| Truncated at the cap | -0.6 |
| No code parsed, or syntax error | -0.5 |
| Compiles, tests fail | 0.0 |
| Partial tests | passed / total, from 0 to 1 |
| All tests pass | 1.0 |
| Length above 900 tokens | -0.001 per token above 900, floor -0.2 |

Clip the sum to the range -1 to 1.

The length penalty applies to the whole output, thinking included.

Invariant, any correct answer scores above any wrong answer, at any length:

- Best wrong answer is 0.0, compiles and fails every test, and the length floor
  only pushes it lower.
- Worst correct answer is 1.0 less the 0.2 floor, which is 0.8.
- 0.8 is greater than 0.0, so the invariant holds.

Partial answers sit between 0.0 and 0.8 and are still above every wrong answer.

Why partial credit carries the early training: a group of 8 all-zero rollouts
gives zero advantage, so no penalty helps. Fraction of tests passed creates the
variance that GRPO needs.

## D4. Curriculum stages

Four bands, in order: very easy, medium, hard, very hard.

- Very easy: 50 synthetic questions, each with its own tests. Written by an
  agent. Start here.
- Medium and hard: cut from the 714 MBPP training rows by measured pass rate.
- Very hard: last stage only.

Each stage keeps about 20 percent of the previous band, so the model does not
forget it.

Within each stage, 80 percent of questions train and 20 percent are held for
evaluation.

## D5. Difficulty labels

Labels come from subagents, in batches of 20 questions. Labels are accepted as
imperfect. They are a prior. The measured pass rate is the truth, and the band
assignment follows the measurement.

## D6. Skip mastered questions

A run mode where a question leaves the pool once a full pass is observed for
it. Mastered questions give zero advantage, so keeping them burns steps.

## D7. Dashboard

Show the current stage band, the step count, and pass rate per band.

## D8. Entropy is sampled, not recorded every step

`output_scores` comes out of `generate`. It is not used by the loss, only for
the per-step entropy chart.

Entropy is measured every N training steps instead, default 500, set by
`[training] entropy_every_steps` in the config. At that point the update pass
already has logits for the tokens, so the measurement is free. Set to 0 to
disable.

The dashboard entropy chart shows one point per measurement, not per step.

## Open

- Eval frequency and size for the new setup.
- How the synthetic easy set is checked, beyond its own tests.
- Whether the 250 held out MBPP rows stay as the final evaluation set.
- Smallest workable group size. Try GRPO with as few samples per prompt as
  still trains, 4 or 6, as a later experiment. Default stays 8 for now.

## Finding: run two collapsed at step 22

Run `20261114T042144Z` reached reward 0.31 by step 2, then collapsed. From
step 22 every completion ran to the 2,048 token cap without closing the
reasoning block, so no code was parsed, every reward was -0.8, and every group
had zero advantage. No checkpoint existed before step 50, so there was nothing
to resume. The run was stopped at step 35.

Contributing causes, in order:

1. One optimizer update per prompt. A 26 prompt step meant 26 updates, about
   300 updates by step 22, so the policy moved far faster than the step counter
   suggests.
2. No KL anchor, so nothing held the policy near the base model.
3. Checkpoint and eval cadence of 50 and 100 steps, too coarse for a collapse
   at step 22.
4. Batching ran prompt groups one at a time, so the batch per generate call
   stayed at 8 sequences.

## D9. One update per step

Status: implemented. `Experiment._training_step` accumulates every prompt group
and calls `Generator.apply_update` once, and `QwenGenerator.step` counts calls
to that method, so the step number is the weight update count.

Gradients accumulate across every prompt group in a step, and the optimizer
steps once at the end. Reason: per prompt updates multiply the learning rate by
the batch size and caused the collapse. The step counter now matches the number
of weight updates.

The accumulated gradient is divided by the prompt group count before the step,
so the update is the mean over the batch instead of a sum that grows with it.
Advantage normalisation stays inside one prompt group.

## D10. Several prompts in one generate call

Status: implemented. `Generator.generate_batch` takes the prompt list and returns
one group per prompt. `Experiment._training_step` pairs the returned groups with
the prompts by position and checks the shape before any scoring.

Prompts are padded to a common length and sampled together, eight answers each,
in one call. Reason: decode is CPU bound at 152 tokens per second, about 5
percent of the roofline. One call per prompt leaves the card idle between calls.
Gradients, rewards and parsing do not change.

Padding is on the left, so every completion starts at the same column and one
prefix length slices every row. The attention mask marks the pads, is checked
before the call, and is stored with each group so the update pass feeds the same
mask back.

## D11. vLLM for rollouts

Rollouts move to vLLM. Training stays on the Hugging Face model with LoRA
weights synced to the engine. Reason: decode is the bottleneck. This is a
separate ticket and a larger change than D9 or D10.

## D12. Thinking budget schedule

The thinking budget starts short, around 512 to 768 tokens, so that closing the
reasoning block is reachable. Closing pays 0.25, which is below the worst
correct answer of 0.8, so closing never beats correct code. The budget grows in
stages as the close rate rises.

Reason: while every rollout fails to close, every reward is identical and no
shaping term can help. A short budget makes closing reachable and creates the
variance the bonus needs.

## D13. Checkpoint and eval every 20 steps

Reason: the collapse arrived at step 22 and the first checkpoint was at 50.

## D14. Docs live in docs/

Every markdown document moves to `docs/`. `README.md` stays at the root for
GitHub. "Go and update the docs" means update `DECISIONS.md` and `journal.md`,
keep the docs index accurate, and leave the code alone.

## Note on the length penalty

The length term is implemented and on by default: soft limit 900 tokens, 0.001
per token above it, floor 0.2. It is covered by tests. The three keys now sit in
`configs/real.toml` under `[training]`, so the term is visible and tunable.

