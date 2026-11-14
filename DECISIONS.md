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
