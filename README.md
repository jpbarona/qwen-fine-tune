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
.venv/bin/rlvr run --arm hybrid --fake
.venv/bin/rlvr run --arm gated-judge --fake
.venv/bin/rlvr run --arm gated-process --fake
```

Arms: `code-score`, `hybrid`, `gated-judge`, `gated-process`.
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
| `error_class` | str | empty when no error, vocabulary set by task 2 |
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

## Reward table (frozen, from SPEC.md)

| Arm | Tests fail | Tests pass |
|---|---|---|
| `code-score` | fraction of tests passed | 1.0 |
| `hybrid` | 0 | `0.5 + 0.5 * judge` |
| `gated-judge` | 0 | `judge` |
| `gated-process` | 0 | `0.5 * process + 0.5 * judge` |

Any failed test closes the gate, so a gated arm makes no judge call. Zero tests
counts as a failure.

## Status

Task 1 is done: the fake end-to-end path, config, schema, and run folders. Tasks
2 to 8 add real data, Docker, the judge client, the arms on GPU, evaluation and
reporting, and Vast.ai packaging.
