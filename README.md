# rlvr

A study of whether a small code model can be trained with group relative policy
optimization, and whether gains in Python carry to other languages.

Two models run the same pipeline. Qwen3-0.6B and LFM2.5-2.6B. The data is MBPP,
961 rows, split into 711 train and 250 test by `data/split_seed1234.json`.

## What one run does

1. Read a screen. The screen is the base model, 16 samples on each of the 711
   training questions.
2. Split the questions into three pools from that screen.
3. Train for a fixed number of steps. Each step samples a group per prompt,
   scores the code with hidden tests, and takes one optimizer update.
4. Evaluate on a pinned subset every few steps.
5. Write everything to a run folder.

The screen is a one-time input. Training never re-screens. Promotions come from
live rollouts.

## Before every run

Three things to check. Skipping them costs an hour.

**1. Pull main.**

```bash
git pull origin main
```

The box and this repository must hold the same code, or the results are not
comparable and a resume can be rejected.

**2. The screen file.** It is not in git, because `screens/` is ignored.

```bash
rclone copy hetzner-dumps:opencodeagent1/dumps/vm/rl-grpo-data/2026-11-21-lfm-screen \
  screens/mbpp-base
```

The file is `screens/mbpp-base/rows.jsonl`, 711 rows, with `summary.json` beside
it. The arm configs name it with `screen_rows`.

**3. The image token.** The training image is private. A fresh box needs the
token once, either as `GHCR_TOKEN` or through a manual `docker login`. The setup
script reads `GHCR_TOKEN` and `GHCR_USER`. Without it the pull fails with
`unauthorized`.

To drop the token step, make the package public once at
`https://github.com/users/jpbarona/package/container/rlvr/settings`.

## Start a run in the container

This is the fast path. The image holds Python 3.12, torch, vLLM, CUDA 13.0 and
the compiled kernels. The repository mounts over it, so code changes need no
rebuild.

```bash
./scripts/docker-setup.sh
./scripts/docker-run.sh code-score --config configs/lfm-spread.toml --runs-root runs
```

`docker-setup.sh` installs Docker and the NVIDIA container runtime when missing,
logs in when `GHCR_TOKEN` is set, then pulls the image and the small Python image
the test runner uses. A new box reaches the first step in about five minutes.

`docker-run.sh` starts the container with:

- the GPU,
- host shared memory (`--ipc=host`), because vLLM fails without it,
- the repository at `/app`,
- the host Docker socket, because the test runner starts containers,
- a cache volume for the model and compiled kernels.

The training user must be in the `docker` group. Without it every test records
`runner_error` and the pass rate stays at zero.

Rebuild the image only when a dependency changes:

```bash
docker build -f docker/Dockerfile -t ghcr.io/jpbarona/rlvr:latest .
docker push ghcr.io/jpbarona/rlvr:latest
```

The build takes about 15 minutes. The image is 17.6 GB on disk.

## Bare metal

Use this when a container is not available.

```bash
./setup.sh
./run.sh code-score --config configs/lfm-spread.toml --runs-root runs
```

`setup.sh` installs what is missing and skips what is present. It checks Python
3.12, the headers, `g++`, ninja, the CUDA toolkit that matches torch, and the
screen. It fails with the exact fix command.

`./setup.sh --check` reports without changing anything. `./setup.sh --fast`
skips the optional convolution kernel.

`run.sh` exports `CUDA_HOME` and `CC` for every run, so the kernels build without
a manual export.

## Setup cost

The box costs 0.57 dollars an hour, which is 0.0095 a minute.

| Event | Time | Cost |
| --- | --- | --- |
| Fresh box to first step, bare metal | about 25 min | 0.24 |
| Fresh box to first step, container | about 5 min | 0.05 |
| One hundred steps at 19 groups | about 4.5 h | 2.6 |

The money is small. The cost is your attention and the serial delay across
boxes. Start the run before you tune anything optional.

## The pools

Each prompt goes to one pool from its screen count.

| Screen result | Pool | Trained |
| --- | --- | --- |
| 0 of 16 pass | parked | Re-screened in training steps |
| 1 to 15 pass | train | Every solution step |
| 16 of 16 pass | retired | Never |

The current screen gives 267 parked, 270 train and 174 retired. Retired prompts
have zero reward spread, so a group of them carries no gradient.

A parked prompt promotes to train at 2 passes out of 16. A train prompt demotes
to parked after 3 zero-pass groups across the run. A passing group does not clear
the count, and a demotion resets it.

## The schedules

Both arms use 304 sequences per step. At 16 samples per prompt that is 19
prompts, 9 parked and 10 train in the spread arm.

| Policy | Shape |
| --- | --- |
| `spread` | Joins a parked block to every step |
| `sweep` | Covers the parked pool in consecutive steps, then trains for a gap |

Spread keeps a queue per pool. A step takes a block from the front and moves it
to the back, so every prompt shows once before any repeat. Sweep keeps a saved
parked order and a train queue for the gap steps. Both save their queues in
`labels.jsonl` after each step, so a restart continues the same order.

## Run configuration

`configs/lfm-spread.toml` and `configs/lfm-sweep.toml` hold the arm settings.
They differ only in the policy keys.

| Key | Value |
| --- | --- |
| model | `LiquidAI/LFM2.5-2.6B`, revision `7141b56` |
| learning rate | 3e-5 |
| rollouts per step | 304 |
| generations per prompt | 16 |
| completion cap | 1024 tokens |
| thinking budget | 1024, recorded but not enforced by vLLM |
| LoRA | all linear, rank 16, alpha 32 |
| steps | 100 |
| evaluation | every 5 steps, pinned 100 questions, seed 2026 |
| checkpoint | every 10 steps |

`[budget] max_steps` and `[budget] max_judge_calls` stop a run cleanly. The stop
reason lands in `eval.json`.

## Run folder

```
runs/<arm>/<timestamp>/
    meta.json        arm, commit, versions, model revision, full config, pools
    metrics.jsonl    one row per step
    rollouts.jsonl   one record per rollout
    labels.jsonl     pool changes, zero-pass counts, saved queues
    eval.json        final pass@1 and every checkpoint evaluation
    control.json     live edits, see below
    checkpoints/     adapter, optimizer state, manifest
```

`<timestamp>` is UTC in the form `YYYYMMDDTHHMMSSZ`.

The step rows carry reward, pass rate, completion length, truncation, timings,
GPU memory, gradient and update norms, pool counts, and the advantage share.

Copy a run off the box before you destroy it. The folder exists only there.

## Dashboard

```bash
.venv/bin/rlvr dashboard --runs-root runs --port 8081
```

It serves `http://127.0.0.1:8081` and refreshes twice per second while the tab is
visible. It shows pass rate, pass@1, reward, truncation, step time, GPU memory,
pool composition, group signal, and a questions table.

The page reads process state as well as files. A run whose process is gone turns
`stopped` after two minutes.

Build the page after a UI change:

```bash
cd dashboard && npm run build
```

The build writes `src/rlvr/web/index.html`, which is committed.

## Analysis

```bash
.venv/bin/rlvr summary runs --html report.html
.venv/bin/rlvr eval-band --config configs/lfm-spread.toml --band medium \
    --count 100 --seed 2026 --checkpoint runs/code-score/<timestamp>/checkpoints/step-000010
.venv/bin/rlvr publish --report report.html \
    --remote git@github.com:<user>/<repo>.git --branch gh-pages
```

`summary` prints reward, pass@1, entropy and completion length side by side.
`eval-band` scores one difficulty band from a checkpoint. `publish` replaces a
Pages branch with the report.

The four measurements for the schedule comparison are the advantage share, the
mean absolute advantage, promotions per thousand parked rollouts, and pass@1 on
the pinned 100.

## Editing a run in place

Write `control.json` while a run is live. The change applies at the next step.

```json
{"drop": ["mbpp-679"],
 "tests": {"mbpp-535": ["assert topbottom_surfacearea(10) == 314.15"]}}
```

`drop` removes a prompt from its pool. `tests` replaces hidden tests for one
prompt. A bad edit is rejected and the previous state stays in force. The error
lands in `control-events.jsonl`.

## Checkpoints and resume

```bash
./run.sh code-score --config configs/lfm-spread.toml --runs-root runs --resume
```

Resume restores the adapter, optimizer and step, truncates records past the
checkpoint, and appends to the same folder. Fifteen checkpoints are kept per run.

A resume is rejected when the saved run and the requested config disagree on the
arm, the split seed, the model revision, the screen, the pool policy, the rollout
budget or the group size. Use a new run root for a different arm.

## Code map

`src/rlvr/experiment.py` owns the step loop and is the only orchestration point.
It takes four injected services.

| File | Role |
| --- | --- |
| `experiment.py` | Step loop, pools, queues, labels, metrics |
| `qwen.py` | GRPO trainer, LoRA, advantage, loss |
| `vllm_backend.py` | Sampling and adapter handoff |
| `pools.py` | Build the three pools from screen rows |
| `queues.py` | Saved rotation for each pool |
| `schedule.py` | Spread and sweep policies, promotion and demotion |
| `reward.py` | Error classes, truncation, length term |
| `docker_runner.py` | Runs one code sample against its tests |
| `dashboard.py` | Live page and `data.json` |
| `cli.py` | `run`, `screen`, `dashboard`, `summary`, `eval-band`, `publish` |

The seams are `Generator`, `TestRunner`, `Judge` and `Store`. The fake versions
in `fake.py` use SHA-256, so a fake run repeats on any machine.

## Data contracts

One rollout record per line. Field order and types are fixed.

| Field | Type | Notes |
| --- | --- | --- |
| `arm` | str | one of the four arm names |
| `step` | int | 1 based |
| `prompt_id` | str | stable across arms |
| `reasoning` | str | thinking trace |
| `code` | str | generated code |
| `prompt_tokens` | int | |
| `completion_tokens` | int | |
| `tests_passed` | int | |
| `tests_total` | int | |
| `error_class` | str | `pass`, `syntax_error`, `wrong_answer`, `runtime_error`, `runner_error` |
| `code_score` | float | passed over total |
| `judge_score` | float or null | null when not judged |
| `process_score` | float or null | process arm only |
| `reward` | float | clipped to [-1, 1] |
| `stage` | str | always `unstaged`, kept for older readers |
| `cache_hit` | bool | judge cache |

Each screen row holds `prompt_id`, `samples`, `passed` and `pass_rate`.

## Rewards

| Arm | Tests fail | Tests pass |
| --- | --- | --- |
| `code-score` | 0.45 at one test, up to 0.75 | 1.0 |
| `hybrid` | 0 | `0.5 + 0.5 * judge` |
| `gated-judge` | 0 | `judge` |
| `gated-process` | 0 | `0.5 * process + 0.5 * judge` |

Fixed outcomes in `code-score`:

| Outcome | Reward |
| --- | --- |
| truncated | -1.0 |
| does not compile | -0.3 |
| raises while running | 0.0 |
| closed marker, zero tests | 0.05 |
| clean failure | 0.0 |
| over 900 tokens | -0.001 per token, capped at -0.2 |

The sum is clipped to [-1, 1].

## Judge

`[judge]` picks the backend.

| Key | Default | Meaning |
| --- | --- | --- |
| `provider` | `openrouter` | `openrouter` or `opencode` |
| `base_url` | the provider endpoint | chat completions URL |
| `api_key_env` | `OPENROUTER_API_KEY` | name of the key variable |
| `model` | required | provider model id |

The key is read at startup and fails before the first rollout when missing. It is
never written to a cache, a log or `meta.json`. Scores are cached by content
hash under `judge-cache/`.

## Tests

```bash
.venv/bin/python -m pytest -q
cd dashboard && npm test
```

CPU only, no network, about 40 seconds. Docker tests skip when the image is
missing.

## Known limits

- The thinking budget is recorded but not enforced under vLLM.
- GPU utilisation reads high on any batch, so use throughput to compare sizes.
- A 100-question cohort carries about five points of noise. Do not compare it
  with a 250-question score.
- The container needs the model and kernel caches rebuilt on a new box, about
  eight minutes once.
