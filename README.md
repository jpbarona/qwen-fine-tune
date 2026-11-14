# rlvr

A study of whether an LLM judge can train a small code model as well as unit
tests. See `docs/SPEC.md` for the study design and `docs/PLAN.md` for the task breakdown.
`docs/journal.md` holds the running problem and fix notes. `docs/DECISIONS.md` records
the current choices.

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -e . && .venv/bin/pip install pytest
```

The real training path needs the `train` extra, pinned for a CUDA 12, Python
3.12 host:

```bash
.venv/bin/pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
.venv/bin/pip install -e ".[train]"
```

The CUDA 12.4 index keeps the torch build compatible with a 3090 driver. The
`train` extra pins `transformers==4.57.1`, `peft==0.17.1`,
`accelerate==1.8.1`, and `safetensors==0.5.3`.

## Run the fake pipeline

The fake path needs no GPU, no network, no Docker, and no model download. It is
deterministic: the same command produces the same rollout records. The CLI
still reads the committed MBPP split and the synthetic easy file, so the run
trains on the 714 MBPP train questions plus the 50 synthetic easy questions, and
evaluates the 250 held-out test questions. Only the model, judge, and Docker
runner are replaced by fakes.

```bash
.venv/bin/rlvr run --arm code-score --fake
```

Arms: `code-score`, `hybrid`, `gated-judge`, `gated-process`. All four are
implemented on the fake path. The process arm judges the reasoning trace and
the code separately, and only on a full test pass.
Flags: `--config PATH` (default: the packaged config at
`src/rlvr/resources/default_config.toml`) and `--runs-root PATH` (default
`./runs`). Unknown arms, a missing config, and an invalid config print one
message on stderr and exit 2.

## Difficulty screening

The screen samples the base model on a question set and writes a difficulty
table. It never trains, builds no optimizer, writes no checkpoint, and writes
into its own output folder, not a run folder.

```bash
.venv/bin/rlvr screen \
    --questions data/mbpp_full.jsonl \
    --samples 8 \
    --output screening/20261114
```

`--questions` is a JSONL file in the MBPP row schema, so the synthetic easy set
and the MBPP rows both work. `--samples` is the number of model samples per
question. The screen samples with thinking off, at the completion cap from
`[training] max_completion_tokens`, because that is the regime the curriculum
trains in.

The output folder holds two files:

```
rows.jsonl      one row per question: prompt_id, samples, passed, pass_rate
summary.json    the pass rate histogram from 0.0 to 1.0, and the
                zero_pass, partial_pass, and full_pass question counts
```

A sample counts as passed only when every task test passed. A rerun reads the
rows already written and skips those question ids, so an interrupted screen
continues where it stopped. A torn final line is dropped and removed before the
restart appends. Rows must keep the same sample count and belong to the same
question set, so two screens cannot be mixed by accident.

`--fake` replaces the model and Docker with the deterministic fakes and needs no
GPU, network, or Docker. It exercises the same writer, summary, and resume path.

## Real training

One command per arm, no manual edits:

```bash
OPENROUTER_API_KEY=... ./run.sh code-score
```

`run.sh` reads the judge key from the environment, passes every extra argument
to `rlvr run`, and defaults to `configs/real.toml`. The four arms are
`code-score`, `hybrid`, `gated-judge`, `gated-process`. The real path uses the
Qwen generator (`src/rlvr/qwen.py`), the Docker test runner, and the judge
client, all behind the same `Experiment` seam.

A smoke run validates a fresh GPU in minutes:

```bash
OPENROUTER_API_KEY=... ./run.sh code-score --config configs/smoke.toml
```

The smoke profile runs 8 training steps. A step starts with two prompt groups
and grows while free GPU memory stays above the floor, so it runs at least 128
generations. It also runs two checkpoint evaluations on 4 questions and a final
evaluation on 8 questions. Expected runtime is roughly 5 to 10 minutes,
dominated by the Docker test runs. This estimate is unverified: there is no GPU
in this environment.

Each step takes a batch of prompts, one group of 8 samples per prompt, all
sampled in one padded call. The batch starts at two prompts. After each step the
free memory is read. While it stays at or above `[training] memory_floor_mib` the
next step takes one more prompt. The first step below the floor settles the size
at the largest batch that fit, and the run keeps that size. Gradients accumulate
across the prompt groups and the optimizer steps once at the end of the step, so
the step count is the number of weight updates. A group is normalised on its own
rewards, never across the batch.

## Deploy on Massed Compute

Rent one RTX A6000: 48GB VRAM, 96GB system RAM, 300GB disk, $0.57 per hour. Use the
Ubuntu Server 24.04 image and set `OPENROUTER_API_KEY` at deploy time.

After the instance boots, install Docker, then run the install, smoke, and full
arm commands from the sections above:

```bash
sudo apt install -y docker.io
OPENROUTER_API_KEY=... ./run.sh code-score --config configs/smoke.toml
OPENROUTER_API_KEY=... ./run.sh code-score
```

## Checkpoints and resume

A checkpoint is saved every `[run] checkpoint_every_steps` steps (default 50).
Each checkpoint directory holds the LoRA adapter, the optimizer state, the step
number, and a `checkpoint.json` manifest. After a crash, resume from the last
checkpoint:

```bash
OPENROUTER_API_KEY=... ./run.sh code-score --resume
```

Resume opens the latest run folder for the arm, restores the model, optimizer,
and step, truncates any records past the checkpoint, and continues appending to
the same folder. Records are never duplicated.

## Budget guards

`[budget] max_steps` and `[budget] max_judge_calls` cap the run. The loop stops
cleanly at the cap, writes the final evaluation, and records the reason
(`completed`, `max_steps`, `max_judge_calls`, or `out_of_memory`) in
`eval.json`.

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
    metrics.jsonl    one row per step: mean reward, pass rate, completion
                     length, cache hits, the GPU memory split, and entropy
                     on a measurement step
    eval.json        final pass@1 plus every checkpoint evaluation
    checkpoints/     one directory per saved step: LoRA adapter, optimizer
                     state, and a checkpoint.json manifest
```

`<timestamp>` is UTC as `YYYYMMDDTHHMMSSZ`.

## Evaluation

Every `[run] eval_every_steps` steps (default 100) the run evaluates a fixed
subset of the test pool with greedy decoding. The subset is
`[run] eval_questions` questions (default 100), chosen once per run by
`[run] eval_subset_seed` (default 2026). That seed is separate from the data
split seed, and the subset is a seeded sample, not the first questions in file
order. The subset keeps the pool order.

Each curriculum stage also evaluates its own held-out slice. That entry has
`kind` `"stage"` and carries the band name in `stage`, so `eval.json` holds the
pass rate per band. The dashboard shows the band beside the evaluation. A run
that stops inside a stage still evaluates that stage's held-out slice.

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
    {"step": 400, "kind": "stage", "stage": "very easy", "pass_at_1": 0.55,
     "count": 12, "prompt_ids": ["..."], "subset": null},
    {"step": 1000, "kind": "final", "pass_at_1": 0.42, "count": 250,
     "prompt_ids": ["..."], "subset": null}
  ]
}
```

The CLI loads the fixed MBPP split with `data.load_split`, uses the 714 train
questions plus the 50 synthetic easy questions as training prompts, and the 250
test questions as eval prompts. Fake mode uses the same set, so a fake run also
evaluates held-out questions and never evaluates a training question. The
`Experiment` itself has no fallback. It refuses to run when `eval_prompts` is
empty. A wrong or empty split stops the command before a run folder is made.

## Completion length and sampled entropy

`completion_tokens` is the completion length used for the length study. It is
logged, not rewarded.

Entropy is not a rollout field. It is measured on the update pass, on the same
forward logits the policy loss uses, every `[training] entropy_every_steps`
training steps (default 500). Every prompt group updated in that step is
measured. Zero disables it. The value is the mean Shannon entropy in nats
over the generated tokens, estimated from at most 64 evenly spaced positions
when a completion is longer than that. It is logged, not rewarded, and it is
not bounded by 1.

`metrics.jsonl` carries `mean_entropy` only on a step that measured it. A step
without a measurement omits the key. The summary and the dashboard treat the
key as absent, never as zero. The fake generator returns one deterministic
value per prompt on the same cadence.

## GPU memory

Each metric row carries the allocator split in MiB: `gpu_live_mib`,
`gpu_cached_mib`, `gpu_fragmented_mib`, `gpu_total_mib`, and
`gpu_unaccounted_mib`. The first three come from `torch.cuda.memory_stats()`
inside the training process, read after the update. Live plus cached plus
fragmented is the allocator's reserved total. Unaccounted is the device total
minus that, which covers other processes, the CUDA context, and the driver.

A run sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` at the entry
point, before torch is imported, so segments grow instead of fragmenting on a
new size. A value already set by the caller is left alone. The fake path writes
zeros, so the dashboard memory panel renders on a machine with no GPU.

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
side. pass@1 shows only at steps where an evaluation ran. The entropy column
shows a value only on a step that measured it, and a dash elsewhere.

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
Generator.generate_batch(prompts, *, n, temperature) -> list[list[Generation]]
Generator.accumulate(prompt, generations, rewards, *, measure_entropy) -> float | None
Generator.apply_update() -> None
Generator.save_checkpoint(directory) -> None
Generator.load_checkpoint(directory) -> None
TestRunner.run(prompt, code) -> Outcome
Judge.score_output(prompt, code) -> JudgeResult
Judge.score_process(prompt, reasoning) -> JudgeResult
Store.write_meta(meta) / append_rollout(record) / append_metric(row) / write_eval(payload)
Store.arm, Store.timestamp, Store.run_dir, Store.checkpoints_dir
```

`temperature` is `0.0` for greedy evaluation. `n` is the number of generations
per prompt; training samples 8, evaluation samples 1. One training step samples
every prompt group in one `generate_batch` call, runs and judges each group,
logs the records, accumulates the group rewards one group at a time, then calls
`apply_update` once. That is the only weight update in the step. The judge sees
the problem and the code, or the problem and the trace. It never sees the tests.

Fakes in `src/rlvr/fake.py`: output depends on prompt id and decoding mode, test
outcome on prompt id and code, judge score on judge kind, prompt id, and text.
Judge scores and test outcomes come from SHA-256, so they repeat across
machines. They carry no meaning.

## Judge client

`JudgeClient` in `src/rlvr/judge.py` implements the `Judge` seam. It uses
OpenRouter by default: endpoint `https://openrouter.ai/api/v1/chat/completions`,
model `deepseek/deepseek-v4.1-flash-1017`, temperature 0, JSON output, and
`reasoning_effort` `medium`. The model id is the one listed by
`https://openrouter.ai/api/v1/models`, and `reasoning_effort` is one of that
model's supported parameters. OpenRouter requests send `Authorization`,
`Content-Type`, `User-Agent: rlvr/0.1 (research judge)`, and `X-Title: rlvr`.
`HTTP-Referer` is not sent: the docs mark it optional and describe it as app
attribution for the public rankings page.

The `[judge]` table picks the backend:

| Key | Default | Meaning |
|---|---|---|
| `provider` | `openrouter` | `openrouter` or `opencode` |
| `base_url` | the provider endpoint | full chat completions URL |
| `api_key_env` | the provider variable | name of the env var that holds the key |
| `model` | (required) | model id, must match the provider |

With `provider = "opencode"` the client uses
`https://opencode.ai/inference/go/openai/v1/chat/completions`, model
`deepseek-v4.1-flash`, and the same four headers plus `Accept` and a stable
`x-opencode-session` header. That gateway rejects the default urllib User-Agent.
The OpenCode-only session header is never sent to OpenRouter.

So the two run configurations are: `openrouter` with
`model = "deepseek/deepseek-v4.1-flash-1017"` and `OPENROUTER_API_KEY`, or
`opencode` with `model = "deepseek-v4.1-flash"` and `OPENCODE_GO_API_KEY`.
Provider, base URL, and model must be changed together.

The key comes from the variable named by `[judge] api_key_env`, which defaults
to `OPENROUTER_API_KEY`. `JudgeClient.from_config` reads it and fails with
`MissingAPIKeyError` before any request. The key is only used in the
Authorization header. It is never written to the cache, an error message, or
`meta.json`.

The output judge sees the problem and code. The process judge sees the problem
and reasoning trace. Neither prompt has a way to receive task tests.

Calls retry on 408, 429, 5xx, timeouts, and connection errors with exponential
backoff plus jitter. `[judge] max_attempts` caps the attempts. Other 4xx
responses raise `UnrecoverableJudgeError` at once.

Scores are cached in `[judge] cache_dir` (default `judge-cache/`). The key is
SHA-256 over provider, base URL, model, judge kind, full prompt text, and rubric
version. Writes use `os.replace`. Identical requests return `cache_hit=true`.
Changing the provider or the base URL starts a separate cache generation, so a
switched provider cannot reuse the other one's scores.

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
judge, so a missing `OPENROUTER_API_KEY` fails before any rollout. The fake
path uses the fake judge and needs no key.

## Status

Tasks 1 to 8 are done: the fake end-to-end path, config, schema, run folders,
merged MBPP split, Docker test runner, the judge client, all four reward arms,
entropy and length logging, the checkpoint and final evaluation schedule, the
summary command, the static HTML report, and the Pages publish command. Task 8
adds the real Qwen generator and GRPO trainer, the pinned GPU environment, one
command per arm, checkpoint resume, budget guards, and the smoke profile.

The real generator, GRPO update, and the 15-minute smoke runtime are not
verified here because this environment has no GPU. The Pages publish command
exists but is not run live here: it needs a writable remote.

## Remote instance

See `docs/DEPLOY.md` for the live host, user, SSH key, and run commands.
