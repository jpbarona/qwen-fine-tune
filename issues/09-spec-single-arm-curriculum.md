# 09: Spec, single arm with curriculum

**Status:** ready-for-agent

Supersedes the four-arm design in `../docs/SPEC.md` for now. The arms are deferred, not
deleted.

## Problem Statement

The first real run trained for 851 steps and learned nothing. Two causes.

Thinking is enabled and the completion cap is 768 tokens. The model spends the
whole budget on its reasoning trace and is cut off before it writes code. With
no closing marker the split never happens, the trace is treated as the answer,
and Docker reports a syntax error. Two thirds of rollouts failed this way.

The reward is also too sparse. It is binary at the group level. When all eight
rollouts fail, every reward is zero and GRPO has no advantage to learn from. At
a 2 percent pass rate that is almost every step.

A third problem follows from the second. All four arms gate on unit tests, so
with a model that rarely passes tests, all four arms score zero and the study
cannot answer its own question.

## Solution

Train one arm only, with a reward that teaches concise thinking and correct
code, on a curriculum that starts at the easiest questions.

- Thinking stays on, with a 2,048 token cap.
- Reward gives partial credit for tests passed, and penalises truncation,
  syntax errors and excessive length.
- Questions are staged: very easy, medium, hard, very hard. The model starts on
  the very easy set, which includes 50 synthetic questions.
- A question the model has mastered stops appearing.
- Several prompts are batched per step, with the batch sized for GPU
  utilisation rather than memory.
- Entropy is sampled every 500 updates instead of every step, which removes the
  per-token logit storage and makes 2,048 fit.
- The dashboard shows the stage, the batch size and pass rate per band.

## User Stories

1. As a researcher, I want the model to see very easy questions first, so that it can learn to produce code at all.
2. As a researcher, I want 50 synthetic very easy questions with their own tests, so that the first stage is solvable.
3. As a researcher, I want the 714 MBPP training rows labelled by difficulty, so that the medium and hard stages exist.
4. As a researcher, I want labels assigned in batches of 20, so that labelling is cheap and reviewable.
5. As a researcher, I want band assignment to follow measured pass rate, so that labels cannot mislead the curriculum.
6. As a researcher, I want a very hard band held for the end, so that the model is not thrown at impossible questions early.
7. As a researcher, I want each stage to carry over about 20 percent of the previous band, so that the model does not forget what it learned.
8. As a researcher, I want an 80/20 train and eval split inside each stage, so that progress can be measured on unseen questions from the same band.
9. As a researcher, I want a question to leave the pool once it has been fully passed, so that mastered questions stop wasting steps.
10. As a researcher, I want always-failed questions handled by the band design rather than a separate rule, so that the pool logic stays simple.
11. As a researcher, I want partial credit for tests passed, so that a group of failures still produces a gradient.
12. As a researcher, I want a penalty for truncation, so that overrunning the cap is worse than finishing.
13. As a researcher, I want a penalty for code that does not compile, so that prose in the answer slot is discouraged.
14. As a researcher, I want a length penalty over the whole output, so that thinking stays concise.
15. As a researcher, I want any correct answer to beat any wrong answer at any length, so that length never rewards being wrong.
16. As a researcher, I want the completion cap raised to 2,048, so that thinking and code both fit.
17. As a researcher, I want per-token logits no longer stored, so that 2,048 fits in memory.
18. As a researcher, I want entropy measured every 500 updates, configurable, so that the health signal survives without the cost.
19. As a researcher, I want several prompts batched per step, so that the GPU is busy instead of waiting on one Python thread.
20. As a researcher, I want the batch size chosen per step, so that utilisation is as high as the card allows.
21. As a researcher, I want the chosen batch size logged per step, so that I can see where more batching stops helping.
22. As a researcher, I want the dashboard to show the stage and pass rate per band, so that I can see whether the curriculum is working.
23. As a researcher, I want a screening run over all 714 questions before training, so that bands come from data.
24. As a researcher, I want the four arms left in the code and marked deferred, so that the study can resume later.
25. As an implementing agent, I want a pure reward function with a table of cases, so that the reward is testable without a GPU.
26. As an implementing agent, I want the curriculum pool builder to be a pure function, so that staging is testable without a model.

## Implementation Decisions

- One arm runs for now. The arm dispatch table stays, the code arm is the default, and the other three remain implemented but unused. `../docs/SPEC.md` gets a note that the four-arm comparison is deferred, and `../docs/DECISIONS.md` remains the record of why.
- The completion cap becomes 2,048, counted over reasoning plus code.
- `output_scores` is removed from generation. Entropy is computed during the update pass, every `entropy_every` updates, default 500, zero disables. The entropy chart becomes sparse points rather than one per step.
- The reward is one pure function of a rollout outcome plus token count. Cases: truncated -0.6, syntax error or no code -0.5, compiles but fails every test 0.0, partial passed over total, all passed 1.0. Length term is 0.001 per token above 900, floor 0.2. The sum is clipped to -1 to 1.
- The invariant that any correct answer beats any wrong answer is enforced by the numbers, worst correct 0.8 against best wrong 0.0, and is covered by a test.
- Penalties and length terms act only on within-group differences. Partial credit is what creates the variance, and the spec states that plainly so nobody re-tightens it into a binary.
- Generation takes a list of prompts per step instead of one. Each prompt keeps its own group of 8 samples. Advantage is normalised inside each group.
- Backward runs per generation, so only one autograd graph is alive at a time. Gradients accumulate across the group. This is what makes the larger batch fit at 2,048 tokens.
- Batch size is as high as the card allows. Start at 2 prompts and grow while
  free memory stays above the floor. Stop only at the memory limit, or when
  throughput stops improving. There is no percentage rule.
- A new stage module owns the bands: it takes labelled rows, ordered band definitions and a carry-over ratio, and returns the ordered pools plus the held out slice per stage.
- The run folder gains stage information. Each rollout record gains the stage name. Each metric record gains the stage, the prompt count, the sequence count and an optional entropy value.
- The pool is mutable within a run. A question is removed once a full pass is recorded for it.
- Config gains `entropy_every`, `stage`, `prompts_per_step` or a dynamic alternative, `carry_over_ratio`, `length_soft_limit` and `length_penalty_per_token`.
- The dashboard reads stage and batch size from the run folder and groups pass rate by band. No new data source.
- Memory reporting uses `torch.cuda.memory_stats()` and logs live, cached and fragmented as separate fields. The allocator is configured with expandable segments.

## Testing Decisions

- A good test drives the public behaviour and asserts observable outcomes. No tests of private helpers, no assertions on internal call order beyond what the contract requires.
- The highest existing seam is the `Experiment` loop with injected generator, test runner, judge and store. Batching, pool removal, stage order and entropy cadence are tested there with fakes, in the same style as the existing fake pipeline tests.
- The reward stays a pure function with table driven tests, extending the existing reward tests. One property style test covers the ordering invariant across token counts.
- The stage builder is a pure function and gets its own tests: band order, carry over ratio, 80/20 split, exhaustion of a band.
- Config parsing is tested through the loader, as today, including rejection of bad values for the new keys.
- The dashboard collector is tested against a synthetic run folder, as today, asserting stage and per band pass rate appear in the collected data.
- Prior art: the reward table tests, the fake pipeline tests, the config rejection tests and the dashboard collector tests already in the suite.

## Out of Scope

- The four-arm comparison, the judge client, and the process reward.
- pass@8 or any multi-sample evaluation metric.
- Multiple seeds.
- A 7,000 token cap.
- vLLM or any inference engine change.
- Persistent storage work on the cloud instance.
- Full fine tuning, distributed training, length curricula for the judge arms.

## Further Notes

- The cap raise is only safe after the logit storage is removed. At 2,048 tokens the stored logits alone are about 10 GB.
- At 32 sequences and 2,048 tokens the KV cache alone is about 7 GB, so the dynamic batch should start at 2 prompts and grow from measurement.
- The screening run is expected to take about 8 hours and 5 dollars. It sets the bands, it does not train.
- The previous instance is terminated. Code and run data are backed up locally at
  `/home/opencode/Work/rl-grpo-instance/home-Ubuntu/`.
- Cost reference from the last run: 73 seconds per step at 768 tokens, so any
  long run needs the batching change first.
