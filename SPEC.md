# Spec: small-model code RLVR reward study

## Problem Statement

I want to know whether an LLM judge can train a small code model as well as unit
tests, and whether adding process scoring changes what the model learns. I have a
single rented 24GB GPU. I want the run to produce enough data to study entropy
and length later without retraining.

## Solution

A Python package that trains `Qwen/Qwen3-0.6B` with GRPO on MBPP for four reward
arms. Every arm uses the same model, data split, and judge. The only difference is
the reward. Every rollout is logged. The package runs on a fresh Vast.ai 3090 with
one command per arm.

## User Stories

1. As a researcher, I want to pick the reward arm with a CLI flag, so one codebase runs all four arms.
2. As a researcher, I want one command per arm, so I do not hand-edit configuration.
3. As a researcher, I want the judge model fixed in config, so the arms differ only in reward.
4. As a researcher, I want failed judge calls retried, so a network blip does not end a run.
5. As a researcher, I want judge responses cached, so retries and repeated evaluations do not pay twice.
6. As a researcher, I want every rollout stored with prompt, reasoning, code, test results, scores, and reward, so I can analyze it later.
7. As a researcher, I want token entropy per rollout, so I can study entropy change.
8. As a researcher, I want completion length per rollout, so I can study the length effect.
9. As a researcher, I want per-test results, so I can classify failures.
10. As a researcher, I want checkpoints every 50 steps, so a crash costs minutes.
11. As a researcher, I want to resume from a checkpoint.
12. As a researcher, I want a smoke mode, so I can validate a new GPU in minutes.
13. As a researcher, I want budget guards on steps and judge calls.
14. As a researcher, I want evaluation every 100 steps, so I can plot a curve.
15. As a researcher, I want pass@1 on 250 held-out questions, so I can compare arms.
16. As a researcher, I want run metadata in plain text.
17. As a researcher, I want rollouts in JSONL.
18. As a researcher, I want a manifest of package versions.
19. As a researcher, I want generated code isolated in Docker.
20. As a researcher, I want code separated from reasoning.
21. As a researcher, I want process scoring to count only when tests pass.
22. As a researcher, I want a summary command and a static HTML report.
23. As a researcher, I want the report published to GitHub Pages.
24. As a researcher, I want the package to run on a CUDA 12 image.

## Implementation Decisions

**Data.** Merge the MBPP train, validation, and test splits into 964 questions.
Use a fixed split seed. 714 questions train, 250 test. Every arm uses the same
split and prompt format. The 10 few-shot prompt examples are excluded.

**Model and training.** `Qwen/Qwen3-0.6B` with thinking enabled. LoRA rank 32,
alpha 64, learning rate 1e-5. GRPO with 8 generations per prompt, temperature
0.9, batch size 1, gradient accumulation 1, 5 epochs. Max prompt 512 tokens, max
completion 768. One seed. Log the truncation rate.

**Arms.**

| Arm | Tests fail | Tests pass |
|---|---|---|
| Code score | Fraction of tests passed | 1.0 |
| Hybrid | 0 | `0.5 + 0.5 * judge` |
| Gated judge | 0 | `judge` |
| Gated process | 0 | `0.5 * process + 0.5 * judge` |

The code score is the fraction of the task tests that pass. The judge score is
the LLM output score. The process score is the LLM trace score. All scores are
between 0 and 1. The process judge sees the reasoning trace and the problem only.
The output judge sees the problem and the code. Neither sees the test code.

**Judge.** OpenCode Go OpenAI-compatible endpoint. Model `deepseek-v4.1-flash` at
normal reasoning effort. Temperature 0. JSON output. 32 concurrent requests.
Retries with backoff. Cache keyed by hash of model, prompt, and rubric version.
Missing key fails before training starts.

**Judge credential.** The key is supplied for this short-lived experiment and is
accepted as low-value. It expires 7 days after issue.

```
${OPENCODE_GO_API_KEY}
```

**Execution.** Each candidate runs in a Docker container with no network, a
non-root user, a read-only mount, a 5 second timeout, and no secrets in its
environment. Per-test results and an error class are recorded.

**Logging.** One JSONL record per rollout: arm, step, prompt id, reasoning, code,
token counts, mean token entropy, tests passed, tests total, error class, code
score, judge score, process score, reward, and cache hit.

**Evaluation.** Every 100 steps, evaluate 100 fixed test questions greedily. At
the end, evaluate all 250 test questions. Report pass@1 only.

**Reporting.** `summary` builds reward, pass@1, entropy, and length per step. It
writes a static HTML report with arms side by side. A publish command pushes the
report to GitHub Pages.

**Run layout.** Each run writes to `runs/<arm>/<timestamp>/` with `metrics.jsonl`,
`rollouts.jsonl`, `eval.json`, `checkpoints/`, and `meta.json`.

**Reliability.** Fail fast on bad config, missing data, missing key, and
unrecoverable judge errors. The smoke mode raises any error. Budget guards cap
steps and judge calls. No wall-clock cap.

**Reproducibility.** Record git commit, package versions, model revision, judge
model, split seed, and timestamp in `meta.json`.

## Testing Decisions

A good test checks external behavior through a stable seam.

The single seam is an `Experiment` entry point with injected `Generator`, `Judge`,
`TestRunner`, and `Store`. Tests drive it with fakes and assert on records and
reward values.

Tested modules: reward composition for all four arms, judge parsing and cache,
Docker runner classification, metrics, and the logging schema. No GPU tests. The
GPU check is a manual smoke run.

## Out of Scope

- Cross-language training.
- Length penalties.
- Multiple seeds.
- Full fine-tuning.
- Distributed training.
- Agentic tasks.
- pass@8.
- A live dashboard.

## Further Notes

The 250-question test set is fixed before any arm runs. It is never used for
training or for checkpoint evaluation beyond the fixed 100-question subset.

The process rubric scores planning, debugging, and honest reporting. It must not
reward longer traces by itself. Length is logged, not rewarded.
