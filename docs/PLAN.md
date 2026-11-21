# Plan: small-model code RLVR reward study

## Global Constraints

- Model `Qwen/Qwen3-0.6B` with thinking. LoRA rank 32, alpha 64, learning rate 1e-5. GRPO 8 generations, temperature 0.9, batch 1, gradient accumulation 1, 5 epochs, max prompt 512, max completion 768, one seed. Log truncation rate.
- Data: MBPP train plus validation plus test merged to 961 questions, fixed seed split 711 train and 250 test. Same split and prompt format for every arm. The 10 few-shot examples are excluded.
- Rewards: code-score is passed/total. Hybrid is 0 on any test failure else 0.5 + 0.5*judge. Gated judge is 0 on any test failure else judge. Gated process is 0 on any test failure else 0.5*process + 0.5*judge. All scores in [0,1].
- Judge: OpenCode Go OpenAI-compatible endpoint, `deepseek-v4.1-flash`, normal reasoning effort, temperature 0, JSON output, 32 concurrent, retries with backoff, content-hash cache. Output judge sees problem and code only. Process judge sees problem and trace only. Neither sees tests.
- Execution: Docker per rollout, no network, non-root, read-only mount, 5 second timeout, no secrets in the container. Record per-test results and an error class.
- Logging: one JSONL record per rollout with arm, step, prompt id, reasoning, code, token counts, mean token entropy, tests passed, tests total, error class, code score, judge score, process score, reward, cache hit.
- Run layout `runs/<arm>/<timestamp>/` with `metrics.jsonl`, `rollouts.jsonl`, `eval.json`, `checkpoints/`, `meta.json`. Checkpoints every 10 steps with resume, fifteen kept per run. Evaluation every 20 steps on 100 fixed test questions, final pass@1 on all 250. pass@8 out of scope.
- Reliability: fail fast on bad config, missing data, missing key, unrecoverable judge errors. Smoke mode raises errors. Budget guards cap steps and judge calls. No wall-clock cap.
- Reproducibility: record git commit, package versions, model revision, judge model, split seed, timestamp in `meta.json`.
- The only seam is an `Experiment` entry point with injected `Generator`, `Judge`, `TestRunner`, `Store`. No GPU tests. The GPU check is a manual smoke run.
- Reporting: `summary` builds a static HTML report with arms side by side, published to a GitHub Pages branch.

## Tasks

## Task 1: Fake end-to-end pipeline

**What to build:** A runnable command that executes one arm from start to finish
with fake model, judge, and test-runner backends, and writes the full run folder.
This proves the orchestration seam, config, and record schema before any GPU or
API is involved.

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent

- [ ] `run --arm <code-score|hybrid|gated-judge|gated-process> --fake` completes and exits 0.
- [ ] Run folder contains `meta.json`, `metrics.jsonl`, `rollouts.jsonl`, and `eval.json`.
- [ ] Each rollout record has the agreed fields with correct types.
- [ ] Unknown arm and missing config fail with a clear message.
- [ ] `pytest` passes on CPU with no network and no GPU.

## Task 2: Merged split and Docker test runner

**What to build:** Load MBPP, merge the three real splits into 961 questions, cut
a fixed 711/250 split, and run candidate code against task tests in Docker with
per-test results.

**Blocked by:** 01.

**Status:** ready-for-agent

- [ ] Loader merges train, validation, and test into 961 questions.
- [ ] Split with a fixed seed gives 711 train and 250 test.
- [ ] Every arm reads the same split and prompt format.
- [ ] Candidate code runs in Docker with no network, a non-root user, a read-only mount, and a 5 second timeout.
- [ ] A hung candidate is killed and classified as a timeout.
- [ ] No secret from the parent environment reaches the container.
- [ ] Fixture code produces each expected error class.

## Task 3: Judge client

**What to build:** A frozen judge client for OpenCode Go that scores output code
and, separately, the reasoning trace. This feeds arms 2, 3, and 4.

**Blocked by:** 02.

**Status:** ready-for-agent

- [ ] Requests go to the OpenCode Go OpenAI-compatible endpoint with `deepseek-v4.1-flash` at normal reasoning effort.
- [ ] Output judge prompt contains the problem and code only, never the tests.
- [ ] Process judge prompt contains the problem and reasoning trace only, never the tests.
- [ ] Responses parse into a bounded score and a short reason.
- [ ] Transient failures retry with backoff.
- [ ] Identical requests hit the cache.
- [ ] The key is read from the environment and never written to disk or logs.
- [ ] A missing key fails before training starts.

## Task 4: Code score arm

**What to build:** The code-score arm. Reward is the fraction of tests passed, with
no judge and no gate. This is the dense baseline.

**Blocked by:** 02.

**Status:** ready-for-agent

- [ ] Reward equals passed tests divided by total tests.
- [ ] A rollout that passes all tests gets 1.0.
- [ ] A rollout that passes none gets 0.
- [ ] No judge call is made in this arm.
- [ ] A short GPU run logs a spread of rewards between 0 and 1.
- [ ] Unit tests cover 0, partial, and full pass.

## Task 5: Hybrid and gated judge arms

**What to build:** Two arms that gate on all tests passing. Hybrid gives
`0.5 + 0.5 * judge`. Gated judge gives the judge score.

**Blocked by:** 03, 04.

**Status:** ready-for-agent

- [ ] Any test failure gives reward 0 and no judge call.
- [ ] Hybrid on a pass gives `0.5 + 0.5 * judge`.
- [ ] Gated judge on a pass gives the judge score.
- [ ] The judge sees no test code.
- [ ] Unit tests cover pass, fail, and timeout for both arms.
- [ ] A short GPU run completes for both arms with populated scores.

## Task 6: Gated process arm

**What to build:** The process arm. Any test failure gives 0. On success the reward
is `0.5 * process + 0.5 * judge`, where the process judge reads only the reasoning
trace.

**Blocked by:** 05.

**Status:** ready-for-agent

- [ ] Thinking traces are captured separately from the final code.
- [ ] The process judge receives the trace and problem only.
- [ ] Failing code gets reward 0 and no judge calls.
- [ ] Passing code gets `0.5 * process + 0.5 * judge`.
- [ ] Rollout records carry both scores and the combined reward.
- [ ] A short GPU run completes with all fields populated.

## Task 7: Instrumentation, evaluation, summary, report

**What to build:** Entropy and length on every rollout, checkpoint evaluation every
20 steps, final pass@1 on 250 test questions, a summary command, and a static
report published to GitHub Pages.

**Blocked by:** 03.

**Status:** ready-for-agent

- [ ] Every rollout records token entropy and completion length.
- [ ] Evaluation runs every 20 steps on 100 fixed test questions with greedy decoding.
- [ ] Final evaluation reports pass@1 on all 250 test questions.
- [ ] Records are append-only JSONL with a stable schema.
- [ ] `summary <run-dir>` prints reward, pass@1, entropy, and length per step.
- [ ] `summary` writes a static HTML report with one section per arm.
- [ ] A publish command pushes the report to a GitHub Pages branch.
- [ ] Metrics functions are unit tested on synthetic records.

## Task 8: Massed Compute packaging and full runs

**What to build:** A pinned environment, one-command launch, checkpoint resume, and
budget guards, so all four arms run on a fresh Massed Compute RTX A6000 without
babysitting.

**Blocked by:** 06, 07.

**Status:** ready-for-agent

- [ ] Pinned requirements install cleanly on a CUDA 12 image.
- [ ] `run.sh <arm>` starts a run with no manual edits.
- [ ] Resume continues from the last checkpoint after a restart.
- [ ] Budget guard stops at the configured step or judge-call limit.
- [ ] `meta.json` records package versions, model revision, judge model, split seed, and commit.
- [ ] The smoke profile finishes in under 15 minutes on a Massed Compute RTX A6000.
- [ ] One full arm finishes without manual intervention.
- [ ] The report for a finished arm is reachable at a GitHub Pages URL.

## Study plan, November 2026

The question. Does GRPO on Python code change pass rates on other languages,
and does the answer depend on the base model.

The two models. Qwen3-0.6B, base pass@1 0.19 on the pinned 100. LFM2.5-2.6B,
base 0.51 on the same 100. Both use all-linear LoRA, rank 16, alpha 32.

The three axes.

1. Same language. The pinned 100 test questions, greedy, every 20 steps.
2. Other programming languages. MultiPL-E, HumanEval and MBPP in 18 languages.
3. Other natural languages. mHumanEval prompts in the eight languages LFM
   claims, with the Python tests unchanged.

The metric. Transfer ratio is the change on an untrained axis divided by the
change on Python, reported per language with a paired McNemar test. A ratio
near zero means the gain stayed in Python. A negative ratio means the training
cost the other languages something.

The order. LFM first, because its base is high enough that a fall is visible.
Then the 60 step Qwen all-linear baseline at 1e-4, so placement is constant
across models. Then MultiPL-E, then mHumanEval. The band evaluation fills in
where the gain sits, since the pinned cohort mixes difficulty.

Known risks. LFM at 0.51 may be near its ceiling on easy questions, which
compresses any gain. Machine-translated prompts can fail on comprehension
rather than on transfer, so use the expert subset. MultiPL-E needs a compiler
per language, which is the one piece of setup nobody has done yet.

### Note, 21 November 2026

The LFM run ended flat at 138 steps, 0.53 against a base of 0.51, with 61 percent
all-fail groups at step 60. That repeats the pool problem from run 17, so the
next run fixes the pool before changing the model or the rate.

The plan is a screen of the 711 train prompts at 16 rollouts each, pools built
from the pass counts, and parked prompts re-screened inside training steps so
their rollouts still produce gradients. See D32 and `IDEAS.md`.

