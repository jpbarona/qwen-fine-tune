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
deterministic: the same command produces the same rollout records. The CLI
still reads the committed MBPP split, so the run trains on the 714 train
questions and evaluates the 250 held-out test questions. Only the model, judge,
and Docker runner are replaced by fakes.

```bash
.venv/bin/rlvr run --arm code-score --fake
```

Arms: `code-score`, `hybrid`, `gated-judge`, `gated-process`. All four are
implemented on the fake path. The process arm judges the reasoning trace and
the code separately, and only on a full test pass.
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
    eval.json        final pass@1 plus every checkpoint evaluation
    checkpoints/     empty in the fake path, task 8 writes adapters here
```

`<timestamp>` is UTC as `YYYYMMDDTHHMMSSZ`.

## Evaluation

Every `[run] eval_every_steps` steps (default 100) the run evaluates a fixed
subset of the test pool with greedy decoding. The subset is
`[run] eval_questions` questions (default 100), chosen once per run by
`[run] eval_subset_seed` (default 2026). That seed is separate from the data
split seed, and the subset is a seeded sample, not the first questions in file
order. The subset keeps the pool order.

The final evaluation always covers the whole test pool, which is the 250
held-out questions for a real run. pass@1 is the only metric: the fraction of
questions whose greedy generation passed every task test.

`eval.json` has the final result at the top level and every evaluation in a
list. It is replaced atomically after each evaluation, so a crash keeps the
last checkpoint result.

```json
{
  "arm": "hybrid",
  "timestamp": "20261112T000000Z",
  "step": 1000,
  "pass_at_1": 0.42,
  "count": 250,
  "subset": {"method": "seeded_sample", "seed": 2026,
             "requested": 100, "selected": 100, "pool_size": 250},
  "evaluations": [
    {"step": 100, "kind": "checkpoint", "pass_at_1": 0.31, "count": 100,
     "prompt_ids": ["..."], "subset": {"...": "the same selector"}},
    {"step": 1000, "kind": "final", "pass_at_1": 0.42, "count": 250,
     "prompt_ids": ["..."], "subset": null}
  ]
}
```

The CLI loads the fixed MBPP split with `data.load_split`, uses the 714 train
questions as training prompts, and the 250 test questions as eval prompts. Fake
mode uses the same split, so a fake run also evaluates held-out questions and
never evaluates a training question. The `Experiment` itself has no fallback.
It refuses to run when `eval_prompts` is empty. A wrong or empty split stops the
command before a run folder is made.

## The rollouts record: entropy and length

`mean_entropy` is the mean Shannon entropy of the model's next-token
distribution at each generated token position, in nats, averaged over the
generated tokens only. Prompt tokens are excluded. The generator computes it
from the logprobs it already receives, and the Experiment records it unchanged.
It is logged, not rewarded, and it is not bounded by 1. The fake generator
returns a deterministic synthetic value.

`completion_tokens` is the completion length used for the length study. It is
also logged, not rewarded.

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
| `mean_entropy` | float | mean per-token entropy in nats over generated tokens, logs only, can exceed 1 |
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

## Summary and static report

`summary` takes one or more run folders. A `runs/` root, or a single arm folder,
also works and expands to every `<timestamp>` folder below it.

```bash
.venv/bin/rlvr summary runs
.venv/bin/rlvr summary runs/code-score/20261112T000000Z runs/hybrid/20261113T000000Z
.venv/bin/rlvr summary runs --html report.html
.venv/bin/rlvr summary runs --no-html
```

It prints four plain tables, one per metric: reward, pass@1, entropy, and
completion length. Steps are rows and arms are columns, so arms sit side by
side. pass@1 shows only at steps where an evaluation ran.

The HTML report is written to `report.html` in the current folder unless
`--html` says otherwise. It is one static file: inline CSS, no script, no
external font, stylesheet, or image. There is one section per arm with a
per-step table, and a small comparison table at the top.

## Publish to GitHub Pages

`publish` copies the report into a throwaway git repository as `index.html`,
adds `.nojekyll`, commits it, and force-pushes that commit to the Pages branch.
The branch is replaced each time, which is intended for generated output.

```bash
.venv/bin/rlvr publish --report report.html \
    --remote git@github.com:<user>/<repo>.git --branch gh-pages
```

Credentials come from the caller's git setup. This command does not read a
token. Use a deploy key with write access to the repository. It is not run in
the test suite, which pushes to a local bare repository instead.

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

`gated-process` calls the process judge with the problem and the reasoning
trace, then the output judge with the problem and the code. Both scores are
stored in the rollout record. An empty trace is sent to the judge like any
other trace.

For a real run, the judge client is built at startup only for arms that use the
judge, so a missing `OPENCODE_GO_API_KEY` fails before any rollout. The fake
path uses the fake judge and needs no key.

## Status

Tasks 1 to 7 are done: the fake end-to-end path, config, schema, run folders,
merged MBPP split, Docker test runner, the judge client, all four reward arms,
entropy and length logging, the checkpoint and final evaluation schedule, the
summary command, the static HTML report, and the Pages publish command. Task 8
adds the pinned GPU environment, checkpoint resume, budget guards, and the real
runs.
