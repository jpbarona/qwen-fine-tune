# MLflow replaces the custom dashboard

Status: design, revised after review. Branch `feat/mlflow`.

## Why

The React dashboard is our code to maintain. Its charts are hard to read: flat
lines, single point series, empty panels, axes that start at odd values. MLflow
gives a maintained UI, run comparison, and a live metrics page. Delete the React
app and keep every field the dashboard shows today.

## What the dashboard shows today

Per step:

- reward, pass rate, pass@1, entropy
- gradient norm, update norm
- completion tokens, truncation rate, cache hits
- prompt count, sequence count, train and parked split
- group spread, group standard deviation, advantage share
- five GPU memory splits: live, cached, fragmented, unaccounted, total
- four timings: step, generation, scoring wait, update
- throughput: completion tokens per second, sequences per second
- stage progress: solved and total per band

Per run:

- arm, run name, status, stage, last write age, started, elapsed
- step count against the target
- per band pass rate
- pass@1 per checkpoint and at the end
- reward histogram, tests passed histogram
- evaluations table
- pool composition over steps
- a questions table: prompt id, stage, steps, rollouts, full pass steps, pass
  rate, best tests passed, last error, last step, solved, dropped, prompt text
- dropped prompt events
- a system strip: GPU utilization, memory utilization, idle samples, GPU
  memory, temperature, sample count, last sample age

Sources: `metrics.jsonl`, `rollouts.jsonl`, `labels.jsonl`, `eval.json`,
`control-events.jsonl`, `meta.json`, and a live GPU sampler.

## Where each field goes in MLflow

| Today | In MLflow |
| --- | --- |
| Step series | Metrics with the same names and the step number |
| Arm, commit, model revision, screen hash, pool policy, config | Params and tags |
| Step target | Param `max_steps` |
| Pass@1 per checkpoint | Metric `pass_at_1/checkpoint` and `pass_at_1/final` |
| Per band pass rate | Metric `band/<name>/pass_rate` |
| Stage progress | Metric `stage/<name>/solved` and `stage/<name>/total` |
| Reward and tests passed histograms | Logged table `distributions.json`, replaced each eval |
| Evaluations table | Logged table `evaluations.json` and the `eval.json` artifact |
| Questions table | Logged table `questions.json`, replaced each eval |
| Dropped prompts | Logged table `dropped.json` and a tag with the count |
| Label events, promotions, demotions | Logged table `labels.json` |
| Control events | Logged table `control-events.json` |
| Status, started, elapsed | Tags, refreshed each step |
| Last write age | Metric `rlvr/heartbeat_seconds` and tag `rlvr/last_write_utc` |
| Process liveness | The heartbeat, plus a sweep that closes dead runs |
| Run folder path | Tag `rlvr/run_dir` |
| GPU utilization, memory, temperature | Child run metrics, every second |
| Five allocator splits | Child run metrics, every second |
| Idle samples, sample count, sample age | Child run metrics, every second |

## The GPU sampler

MLflow logs GPU utilization, GPU memory, power, CPU, disk and network every ten
seconds. Set `MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL=1` so its own series match
the sampler. MLflow still has no temperature, no allocator split and no idle
count, so the sampler covers those.

Torch holds the five allocator splits, so only the training process can read
them. The sampler in `dashboard.py` reads `nvidia-smi` and runs only when a page
asks. It cannot produce those five series. A thread inside training can.

Design: one sampler thread inside the training process. Every second it
reads `nvidia-smi` for utilization, memory and temperature, and
`torch.cuda.memory_stats` for the five splits. It appends a row to `gpu.jsonl`
and writes the numbers to MLflow.

The samples go to a nested child run, stepped by sample index, not by training
step. A 150 second step would otherwise draw 75 points on one vertical line. The
thread calls `MlflowClient().log_metric` with the child run id, because the
active run is thread local. The interval is `[mlflow] gpu_sample_seconds`,
default 1.

One second over 100 steps gives about 15,000 points per series. SQLite stores
that in a few megabytes.

## What stays out of MLflow

- `rollouts.jsonl`. A hundred steps of 304 rollouts is about thirty thousand
  rows. That slows the UI and adds nothing to a chart. It stays on disk and in
  the bucket.
- `checkpoints/`. Model weights do not belong in the tracking store.

## Approach

Chosen: MLflow only, with the questions table logged at each eval. You accepted
a manual page refresh.

Rejected: MLflow plus a small plain HTML page. It leaves custom code to maintain
for the strip and the questions table.
Rejected: MLflow only, with questions and the strip dropped. It loses
information you curated.

## How it is built

### Store and server

- Backend store is SQLite at `<runs-root>/mlflow.db`. Artifacts are at
  `<runs-root>/mlartifacts`. One experiment, `rlvr`.
- The server runs in the same image and the same mounts as training, so the
  `/app/...` artifact paths resolve. In the container path,
  `./scripts/docker-run.sh dashboard` starts it. In bare metal, `rlvr dashboard`
  starts it. Both bind `0.0.0.0:8081`, so the tunnel and the docs stay the same.
- Training writes as root in the container. Running the server in the same image
  gives it the same user, so it can write the SQLite file.
- `rlvr dashboard` keeps its name and starts `mlflow server` instead of the
  custom page.

### The mirror

- New module `src/rlvr/mlflow_log.py` with one class. Methods: `start`, `step`,
  `eval`, `table`, `tag`, `gpu_sample`, `finish`.
- Training writes to MLflow at the seam where it appends a metric row, so a live
  run updates. The mirror catches its own failures, writes a warning, and
  training continues.
- The run folder is the record. `metrics.jsonl`, `labels.jsonl`, `eval.json` and
  `gpu.jsonl` are written as they are now. Removing MLflow later loses nothing.
- The MLflow run ids live in a new `mlflow.json` in the run folder, so
  `meta.json` stays immutable.

### Resume

MLflow has no metric delete, so a resumed run would get duplicate points past
the last checkpoint. On resume the mirror deletes the parent and child runs and
rebuilds them from the local files, which the store truncated. The rebuild uses
the same code as the importer.

### Liveness

A stale run shows an old heartbeat and an old last write tag. On startup the
trainer scans the experiment for runs in `RUNNING` with an old heartbeat and sets
them to `FAILED`. Without that, a crashed run stays green in MLflow.

### The importer

`rlvr import-mlflow <runs-root>` reads old run folders and writes them into the
store. It rebuilds the step series, evals, labels, questions, histograms and
stage progress from the local files, so the two existing runs look like new
ones. Move the computation out of `dashboard.py` into a plain module and call it
from both the importer and the live mirror before deleting the web layer.

### What gets deleted

- `dashboard/`, `src/rlvr/dashboard.py`, `src/rlvr/web/index.html`, the dashboard
  tests, the npm build step and the `web` package data.
- `summary.py` and `publish` do not change.

### Dependencies

- Add `mlflow`, pinned to one exact version.
- MLflow brings pandas, SQLAlchemy, protobuf and numpy. The image pins torch,
  vLLM, Transformers and PEFT. After the install, run `pip check` and one smoke
  step. If a pinned package moves, stop and report before the push.

## Goals

- Every field in the table above is visible in MLflow or kept on disk.
- No React, no npm, no build step, no custom server.
- One command still opens the live view.
- Old runs can be compared with new ones and show the same panels.

## Non-goals

- No change to training, rewards, pools, schedules or the queue code.
- No remote MLflow server.
- No per-rollout table in MLflow.
- The run folder gains `gpu.jsonl` and `mlflow.json`, and nothing else.

## Anti-goals

- A field lost because MLflow has no chart for it.
- A run folder that only MLflow can read.
- A second dashboard to maintain.

## Constraints

- We set up each box from the container image. Adding MLflow means a rebuild and
  a push from the box.
- MLflow UI refresh is manual. There is no half second poll.
- MLflow metric values are floats. Booleans become 0 and 1. Nulls are skipped.
- Metric names allow letters, digits, `_`, `-`, `.` and `/`.
- MLflow step values are integers.

## Calls made here, open to correction

- Mirror inside the training process, not a later reader, so the live view
  works.
- One experiment for all arms, tags for the arm, so runs compare in one place.
- SQLite, not the file store, so the UI stays fast as runs accumulate.
- Old runs are imported with full fidelity, including their question tables.
- Histograms are logged at each eval, not each step.
- GPU samples go to a nested child run every second, stepped by sample
  index, so the main charts keep a clean step axis.
- A resumed run is rebuilt in MLflow from the local files.
- The dashboard computation moves to a plain module and is reused, rather than
  deleted with the web layer.

## What is left to the builder

- Exact metric names for the allocator splits and the timings.
- Where the mirror lives, in `RunStore` or injected beside it.
- How the questions table is replaced, by deleting the local artifact first or
  by writing one file per eval.
- Thread shutdown and error handling for the sampler.
- The tests and the docs.
