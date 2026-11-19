# Run 17 postmortem

Qwen3-0.6B, GRPO on MBPP, arm `code-score`, 313 steps, LoRA rank 32 over the
four attention projections of every layer. Snapshot: the step-313 pull.

## The question

The run's own checkpoint evaluations sat between 0.32 and 0.40 for 300 steps.
That was read as the model not improving.

## What the numbers say

The model improved. The curve was flat because it started at the wrong place.

| copy | greedy pass@1 | reasoning off, step 20 | reasoning off, mean tokens |
| --- | --- | --- | --- |
| untrained | 0.19 | 0.12 | 1,169 |
| step 40 | 0.27 | 0.13 | 775 |
| step 180 | 0.27 | 0.16 | 567 |
| step 300 | 0.37 | 0.24 | 411 |

The untrained model solves 12 of the 100 held-out questions. Step 300 solves
24. Paired on the same questions, 13 questions were gained and 1 lost, exact
McNemar p = 0.002.

The run's own evaluations began at step 20, when the score was already 0.32.
Two thirds of the gain happened before the first data point, so the remaining
300 steps looked flat.

## Sixteen copies, reasoning off

Cohort: the 100 ids the run itself evaluated, seed 2026, from the 250-question
test split. Greedy is temperature 0. Sampled is eight completions at 0.9.

| step | greedy | 8-sample mean | any of 8 | all 8 | tokens |
| --- | --- | --- | --- | --- | --- |
| base | 0.120 | 0.110 | 0.280 | 0.040 | 34 |
| 20 | 0.120 | 0.107 | 0.260 | 0.000 | 32 |
| 40 | 0.130 | 0.139 | 0.310 | 0.010 | 32 |
| 60 | 0.120 | 0.116 | 0.320 | 0.010 | 31 |
| 80 | 0.120 | 0.099 | 0.260 | 0.000 | 30 |
| 100 | 0.120 | 0.120 | 0.310 | 0.000 | 32 |
| 180 | 0.160 | 0.126 | 0.320 | 0.000 | 36 |
| 200 | 0.130 | 0.129 | 0.330 | 0.010 | 35 |
| 220 | 0.150 | 0.115 | 0.280 | 0.010 | 34 |
| 240 | 0.170 | 0.121 | 0.330 | 0.010 | 35 |
| 260 | 0.160 | 0.130 | 0.300 | 0.010 | 34 |
| 270 | 0.200 | 0.185 | 0.360 | 0.050 | 44 |
| 280 | 0.230 | 0.198 | 0.390 | 0.050 | 47 |
| 290 | 0.240 | 0.217 | 0.440 | 0.060 | 47 |
| 300 | 0.240 | 0.214 | 0.420 | 0.080 | 46 |
| 310 | 0.230 | 0.212 | 0.380 | 0.060 | 45 |

Nothing happens for 260 steps. The gain is steps 260 to 290, eight points in
thirty steps. Answers hold near 30 tokens for 250 steps, then grow to 46.
Noise on 100 questions is about five points.

## What the training logs show

Over 132,408 rollouts and 16,551 prompt groups.

- **Zero-variance groups: 10.3 percent.** 1,713 groups gave all eight rollouts
  the same reward, 1,508 of them all zero. Those groups carry no advantage.
- **Duplicate code: 20.6 percent of groups** repeated a code body, with a mean
  of 7.63 distinct bodies out of eight.
- **Adapter norm grew 0.27 percent** from step 20 to step 310, with no interval
  declining. At 1e-5 the updates were real and very small.
- **Retirement bias.** The medium stage entered 446 prompts and retired 109.
  Retired prompts averaged an 85.6 percent pass rate, the ones left behind
  averaged 12.1 percent.
- **Batch.** Median step was 20 prompts and 160 sequences, range 1 to 175
  prompts. 42 steps ran at three prompts or fewer. 69 prompts received exactly
  one gradient update.
- **Error classes in medium.** Wrong answer 49.2 percent, runtime error 28.2,
  pass 15.0, syntax error 7.1, timeout 0.6.
- **Eval integrity.** All 15 checkpoint evaluations used the same 100 ids.
- **Instrumentation gap.** Advantage and gradient norm were never logged.
  Entropy was set to every 500 steps and the run stopped at 313, so it was
  written once at most.

## Learning rate probe

Two twenty step runs, one per card, same config and cohort, the rate the only
change.

| rate | reward over the run | step 20 pass@1 | update norm | outcome |
| --- | --- | --- | --- | --- |
| 1e-5 (run 17) | 0.2 to 0.3 | 0.32 | not logged | flat for 300 steps |
| 1e-4 | 0.44 to 0.91 | 0.31 | 0.08 | stable |
| 1e-3 | collapsed to -0.99 | not reached | 0.7 to 2.2 | destroyed by step 14 |

The 1e-3 run failed the way divergence looks here: reward at -0.99, pass rate
zero, every answer running to the 2,048 token cap. It was stopped at step 16.

The update norm separates the rates by 25 times. The gradient norm did not, 0.11
to 0.33 against 0.05 to 0.59, because AdamW divides the gradient magnitude away.

Ten times the old rate is safe and has bought nothing after twenty steps. The
open question is whether it pulls ahead over sixty.

## Conclusions

The run worked and was measured wrongly. Its first data point was step 20, by
which time most of the gain had happened. A base evaluation on the same cohort
costs five minutes.

The pipeline removes its own signal. Strict mastery retires the prompts the
model can solve, so the pool fills with prompts it cannot solve, and the share
of zero-variance groups rises with it. A tenth of all steps taught nothing.

## What the next run changes

- Evaluate the base model at step 0 on the run's own cohort.
- Keep retired prompts in a rehearsal set instead of deleting them.
- Log advantage spread, gradient norm and update norm every step.
- Raise the learning rate, measured rather than guessed. See D24.

## Method notes

Sampling used vLLM 0.10.1 with the checkpoints loaded as LoRA adapters. A
Hugging Face run of four copies agreed with vLLM on seven of eight questions.
Prompt text came from the repository's own `format_prompt`, and tests ran in the
repository's own docker runner, so nothing was reimplemented.

Two limits. The cohort is the run's own eval set, not a second held-out set. And
reasoning-on numbers are not comparable with reasoning-off numbers in absolute
terms, only in their direction and size.
