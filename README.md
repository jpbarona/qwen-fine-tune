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

The tools and credentials on this VM are in [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md). Three things
to check. Skipping them costs an hour.

**1. Pull main.**

```bash
git pull origin main
```

The box and this repository must hold the same code, or the results are not
comparable and a resume can be rejected.

**2. The screen file.** It is not in git, because `screens/` is ignored.

```bash
rclone copy hetzner-dumps:opencodeagent1/dumps/screens/2026-11-21-lfm-base \
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

For a run that writes to the store on this VM, use
`./scripts/start-remote-run.sh` instead. The Dashboard section covers it.

`docker-setup.sh` installs Docker and the NVIDIA container runtime when missing,
adds the caller to the docker group when the caller is missing from it, logs in
when `GHCR_TOKEN` is set, then pulls the image and the small Python image the
test runner uses. The current SSH session keeps its old groups, so the script
runs the pull through `sg docker` and prints the new session step. A new box
reaches the first step in about ten minutes: three for the pull, one for the
model, and the rest is the first vLLM kernel compile. Pull, never build, on a
run box.

When the caller has no root and no docker access, the script stops before the
pull and prints the `usermod`, `newgrp` and rerun commands.

`docker-run.sh` starts the container with:

- the GPU,
- host shared memory (`--ipc=host`), because vLLM fails without it,
- the repository at `/app`,
- the host Docker socket, because the test runner starts containers,
- a cache volume for the model and compiled kernels.

The training user must be in the `docker` group. Without it every test records
`runner_error` and the pass rate stays at zero. Open a new SSH session after the
group change, because the current one keeps the old groups.

Rebuild the image only when `pyproject.toml` or `docker/Dockerfile` changes:

```bash
GHCR_TOKEN=... ./scripts/build-image.sh
```

It tags the image with the UTC timestamp and `latest`, pushes both, and prints
the digest. The run records the digest in `meta.json`.

The build takes about 15 minutes and the image is 18.5 GB on disk. A fresh box
pulls it in about three minutes.

## Bare metal

Use this when a container is not available.

```bash
./setup.sh
./run.sh code-score --config configs/lfm-spread.toml --runs-root runs
```

`setup.sh` installs what is missing and skips what is present. It checks Python
3.12, the headers, `g++`, ninja, the CUDA toolkit that matches torch, and the
screen. It fails with the exact fix command. The install uses the `vllm` and
`tracking` extras, so `rlvr dashboard` can start MLflow from the same venv.

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
    gpu.jsonl        one row per GPU sample, copied to the child run
    mlflow.json      MLflow run ids and table append counts
    checkpoints/     adapter, optimizer state, manifest
```

`<timestamp>` is UTC in the form `YYYYMMDDTHHMMSSZ`. A store that fails appends
one line to `mlflow-errors.log` and leaves the run folder intact.

The step rows carry reward, pass rate, completion length, truncation, timings,
GPU memory, gradient and update norms, pool counts, and the advantage share.

Copy a run off the box before you destroy it. A run with `[mlflow] bucket` set
also copies its folder to the bucket while it trains, and a run without a bucket
exists only on the box.

## Dashboard

The store lives on this VM, not on the box. One MLflow server serves every box,
so a run keeps its history after the box is destroyed.

The server is a systemd unit in `deploy/`, run as `opencode`. Once installed it
binds `127.0.0.1:8081`, keeps its SQLite store at
`/home/opencode/Work/rl-grpo-data/mlflow/mlflow.db`, and proxies artifacts to
`s3://opencodeagent1/mlflow/v1/artifacts`. `sudo deploy/install.sh` installs the unit and
the backup timer. `deploy/README.md` has the details.

The box reaches the server through an SSH reverse tunnel opened by this VM. The
tunnel binds the Docker bridge gateway on the box, and it needs
`GatewayPorts clientspecified` in the box `sshd_config`. `scripts/docker-setup.sh`
writes that line as a drop-in.

Open the tunnel and start a run:

```bash
./scripts/start-remote-run.sh code-score Ubuntu@<host> \
  --config configs/lfm-spread.toml --runs-root runs
```

The script opens the tunnel, waits for `/health` from the box, then launches the
run detached. It launches nothing when the health check does not answer, because
that run would train with no store. `./scripts/open-tunnel.sh Ubuntu@<host>`
opens the tunnel alone. One tunnel serves every arm on the box, and it reconnects
when the link drops.

`rlvr dashboard` prints the store URL:

```bash
.venv/bin/rlvr dashboard
```

Open the UI through the Mac to VM tunnel on port 8083. That tunnel already
forwards `8083` to `127.0.0.1:8081` on this VM. Then open
`http://127.0.0.1:8083/`.

`rlvr dashboard --local` starts the old local server on port 8099 with the store
under the runs root. It starts a server from the local venv and needs no tunnel.

A run with `[mlflow] bucket` set copies its checkpoints and run folder to
`s3://opencodeagent1/runs/v1/<arm>/<started>/` while it trains. The uploader
reads `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` and `MLFLOW_S3_ENDPOINT_URL`
from the box environment, so keep them in `$HOME/.env`. The key is scoped to the
`runs/` prefix.

After a dropped link, replay one run folder into the store:

```bash
.venv/bin/rlvr sync-mlflow runs/code-score/<timestamp>
```

The command deletes and rebuilds that run's entries, so it is safe to repeat. It
matches the run on `rlvr/run_name`, the arm and start timestamp.

`deploy/mlflow-backup.timer` copies the store nightly and keeps 14 copies under
`s3://opencodeagent1/mlflow/v1/backups/`.

The store starts empty. Old dumps under `dumps/` are untouched.

An old run folder is imported once:

```bash
.venv/bin/rlvr import-mlflow --runs-root runs
```

A folder with no `gpu.jsonl` imports without error. `--arm <arm>` imports one
arm only.

### Saved view

The study page is a saved MLflow view. Write the tag once, then open the link
the command prints:

```bash
.venv/bin/rlvr saved-view --runs-root /home/opencode/Work/rl-grpo-data/mlflow
```

The command writes the tag to the `rlvr` experiment and prints a hash route
such as `#/experiments/1/runs?viewStateShareKey=rlvr-study&compareRunsMode=CHART`.
The runs root is the folder that holds `mlflow.db`, so point it at the store on
this VM. A second run replaces the same tag, so the store keeps one view.

Open the route on the dashboard origin, for example
`http://127.0.0.1:8083/#/experiments/1/runs?viewStateShareKey=rlvr-study&compareRunsMode=CHART`.
The route opens the experiment Chart view.

The view has eight sections in this order: core, quality, optim, speed, data,
gpu, band, stage. Core holds mean reward, pass rate and advantage share. Their
ranges are fixed: -1 to 1.1 for mean reward, and 0 to 1.05 for pass rate and
advantage share. The headroom keeps the top 1.0 label inside the plot. Every
other chart keeps the auto range. A filter keeps the
parent runs and drops the GPU child. The run page stays as it was, so the
tables, the artifacts and the GPU child run are still there.

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

`rlvr.training.experiment` owns the step loop and is the only orchestration
point. It takes four injected services. The packages below point one way, and
`.importlinter` enforces the order in CI.

| Package | Role |
| --- | --- |
| `core` | Seam values, shared constants, run schema, stage labels, memory, batching, checkpoints, control file, config loader |
| `training` | Step loop, pools, queues, schedule, reward, screening, budget |
| `engine` | GRPO trainer, vLLM sampling, judge client, model profiles |
| `evaluation` | Data loading, pass@1, Docker test runner, fakes, synthetic rows, repro helpers |
| `tracking` | Run folder writer, MLflow mirror and importer, saved view, upload, run readers |
| `report` | Static report from a finished run |
| `app` | Command line and resume |

The seams are `Generator`, `TestRunner`, `Judge` and `Store`. The fake versions
in `rlvr.evaluation.fake` use SHA-256, so a fake run repeats on any machine.

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
```

CPU only, no network, about five minutes on this VM. Docker tests skip when the
image is
missing.

## Known limits

- The thinking budget is recorded but not enforced under vLLM.
- GPU utilisation reads high on any batch, so use throughput to compare sizes.
- A 100-question cohort carries about five points of noise. Do not compare it
  with a 250-question score.
- The container needs the model and kernel caches rebuilt on a new box, about
  eight minutes once.
