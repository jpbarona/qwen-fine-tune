# rlvr

A study of whether an LLM judge can train a small code model as well as unit
tests. See `SPEC.md` for the study design and `PLAN.md` for the task breakdown.

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -e . && .venv/bin/pip install pytest
```

## Run the fake pipeline

The fake path needs no GPU, no network, no Docker, and no model download. It is
deterministic: the same command produces the same rollout records.

```bash
.venv/bin/rlvr run --arm code-score --fake
```

Arms: `code-score`, `hybrid`, `gated-judge`, `gated-process`. This build
implements `code-score`, `hybrid`, and `gated-judge`. `gated-process` exits 2
with a clear message until Task 6 adds it.
Flags: `--config PATH` (default: the packaged config at
`src/rlvr/resources/default_config.toml`) and `--runs-root PATH` (default
`./runs`). Unknown arms, a missing config, and an invalid config print one
message on stderr and exit 2. Without `--fake` the command says that real
backends are not implemented yet.

## Tests

```bash
.venv/bin/pytest
```

CPU only. No network, no GPU.

## Run folder layout

```
runs/<arm>/<timestamp>/
    meta.json        arm, timestamp, git commit, versions, model revision,
                     judge model, split seed, the full config
    rollouts.jsonl   one record per rollout, schema below
    metrics.jsonl    one row per step: mean reward, pass rate, entropy,
                     completion length, cache hits
    eval.json        greedy pass@1 over the prompts used in the run
    checkpoints/     empty in the fake path, task 8 writes adapters here
```

`<timestamp>` is UTC as `YYYYMMDDTHHMMSSZ`.

## Rollout record schema (frozen)

One JSON object per line, fields in this order. Defined in
`src/rlvr/schema.py`. Later tasks read these records, so field names, order,
and types are a contract.

| Field | Type | Notes |
|---|---|---|
| `arm` | str | one of the four arm names |
| `step` | int | 1-based step within the run |
| `prompt_id` | str | MBPP-style id, stable across arms |
| `reasoning` | str | thinking trace, kept apart from code |
| `code` | str | generated code only |
| `prompt_tokens` | int | |
| `completion_tokens` | int | completion length for the length study |
| `mean_entropy` | float | mean per-token entropy, logged not rewarded, can exceed 1 |
| `tests_passed` | int | |
| `tests_total` | int | |
| `error_class` | str | `pass` when every test passed, otherwise the failure class set by task 2 |
| `code_score` | float | passed / total, in [0, 1] |
| `judge_score` | float or null | null when the arm does not judge or tests failed |
| `process_score` | float or null | null except in `gated-process` on a pass |
| `reward` | float | the arm reward, in [0, 1] |
| `cache_hit` | bool | true when any judge call was served from cache |

Timestamps and truncation flags are deliberately out of this record. Truncation
rate is a step-level metric, added in a later task.

## The seam

`Experiment` (`src/rlvr/experiment.py`) is the only orchestration point. It
takes four injected services and nothing else touches a model, API, container,
or the filesystem during a run.

```python
Generator.generate(prompt, *, n, temperature) -> list[Generation]
TestRunner.run(prompt, code) -> Outcome
Judge.score_output(prompt, code) -> JudgeResult
Judge.score_process(prompt, reasoning) -> JudgeResult
Store.write_meta(meta) / append_rollout(record) / append_metric(row) / write_eval(payload)
Store.arm, Store.timestamp, Store.run_dir
```

`temperature` is `0.0` for greedy evaluation. `n` is the number of generations
per prompt; the fake path uses 1, GRPO will use 8. The judge sees the problem
and the code, or the problem and the trace. It never sees the tests.

Fakes in `src/rlvr/fake.py`: output depends on prompt id and decoding mode, test
outcome on prompt id and code, judge score on judge kind, prompt id, and text.
Judge scores and test outcomes come from SHA-256, so they repeat across
machines. They carry no meaning.

## Judge client

`JudgeClient` in `src/rlvr/judge.py` implements the `Judge` seam against the
OpenCode Go OpenAI-compatible endpoint
`https://opencode.ai/inference/go/openai/v1/chat/completions`, model
`deepseek-v4.1-flash`, temperature 0, JSON output, and `reasoning_effort`
`medium`. Every request sends a concrete `User-Agent` and a stable
`x-opencode-session` header, because the gateway rejects the default urllib
User-Agent.

The key comes from `OPENCODE_GO_API_KEY`. `JudgeClient.from_config` reads it and
fails with `MissingAPIKeyError` before any request. The key is only used in the
Authorization header. It is never written to the cache, an error message, or
`meta.json`.

The output judge sees the problem and code. The process judge sees the problem
and reasoning trace. Neither prompt has a way to receive task tests.

Calls retry on 408, 429, 5xx, timeouts, and connection errors with exponential
backoff plus jitter. `[judge] max_attempts` caps the attempts. Other 4xx
responses raise `UnrecoverableJudgeError` at once.

Scores are cached in `[judge] cache_dir` (default `judge-cache/`). The key is
SHA-256 over model, judge kind, full prompt text, and rubric version. Writes use
`os.replace`. Identical requests return `cache_hit=true`.

`score_batch` runs requests on a bounded pool of `[judge] concurrency` threads
(default 32) and preserves input order.

## Reward table (frozen, from SPEC.md)

| Arm | Tests fail | Tests pass |
|---|---|---|
| `code-score` | fraction of tests passed | 1.0 |
| `hybrid` | 0 | `0.5 + 0.5 * judge` |
| `gated-judge` | 0 | `judge` |
| `gated-process` | 0 | `0.5 * process + 0.5 * judge` |

Any failed test closes the gate, so a gated arm makes no judge call. For the
gated arms zero tests counts as a failure. For `code-score` zero tests is an
error: a rollout with nothing to run must fail loudly, not score 0.

`code-score`, `hybrid`, and `gated-judge` are implemented in this build.
`gated-process` is accepted by the parser but stops before a run folder is
made.

For a real run, the judge client is built at startup only for arms that use the
judge, so a missing `OPENCODE_GO_API_KEY` fails before any rollout. The fake
path uses the fake judge and needs no key.

## Status

Tasks 1 to 5 are done: the fake end-to-end path, config, schema, run folders,
merged MBPP split, Docker test runner, the judge client, and the code-score,
hybrid, and gated-judge reward arms. Tasks 6 to 8 add the process arm, GPU
training, evaluation and reporting, and Vast.ai packaging.
