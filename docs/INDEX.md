# Docs index

The study compares GRPO training on a small code model across two parked-prompt
schedules. Code sits in the repository root. This folder holds the reasoning.

## Current work

- [PR #4](https://github.com/jpbarona/qwen-fine-tune/pull/4): the store moves to
  this VM, live metrics over one SSH tunnel, checkpoints to the bucket.

## Start here

- [../README.md](../README.md): install, run, MLflow dashboard, data layout, contracts.
- `DECISIONS.md`: choices as they were made, newest last. D25 onward covers the
  parked-pool study, and D47 onward covers the remote store.
- `IDEAS.md`: proposals, two chunks each, newest first inside a topic.
- `journal.md`: one line per problem and fix, newest at the bottom.

## Designs and plans

- [superpowers/specs/2026-11-23-remote-tracking-store-design.md](superpowers/specs/2026-11-23-remote-tracking-store-design.md):
  one store on the VM, the reverse tunnel, the bucket layout.
- [superpowers/plans/2026-11-23-remote-tracking-store.md](superpowers/plans/2026-11-23-remote-tracking-store.md):
  the task breakdown for that change.
- [superpowers/specs/2026-11-23-mlflow-saved-view-design.md](superpowers/specs/2026-11-23-mlflow-saved-view-design.md):
  the saved view on the study page, the eight sections and the ranges.
- [superpowers/plans/2026-11-23-mlflow-saved-view.md](superpowers/plans/2026-11-23-mlflow-saved-view.md):
  the task breakdown for that change.
- [superpowers/specs/2026-11-23-mlflow-design.md](superpowers/specs/2026-11-23-mlflow-design.md):
  the MLflow store, the parent and child runs, the tables and the importer.
- [superpowers/plans/2026-11-23-mlflow.md](superpowers/plans/2026-11-23-mlflow.md):
  the task breakdown for that change.
- [superpowers/specs/2026-11-22-parked-and-train-queues-design.md](superpowers/specs/2026-11-22-parked-and-train-queues-design.md):
  saved queues, retirement, three-failure demotion.
- [superpowers/plans/2026-11-22-parked-and-train-queues.md](superpowers/plans/2026-11-22-parked-and-train-queues.md):
  the task breakdown for that change.
- [superpowers/plans/2026-11-21-parked-pools.md](superpowers/plans/2026-11-21-parked-pools.md):
  the earlier breakdown for measured pools and the two arms.
- [../issues/19-parked-pools-spec.md](../issues/19-parked-pools-spec.md): the
  pool spec. Tickets `20` to `26` cover that build.
- [superpowers/specs/2026-11-19-lfm-all-linear-lora-design.md](superpowers/specs/2026-11-19-lfm-all-linear-lora-design.md):
  all-linear LoRA on LFM.
- `SPEC.md`: the deferred four-arm sampling study.
- `PLAN.md`: the original ticket breakdown.

## Deploy and scripts

- [../deploy/README.md](../deploy/README.md): the server unit, the nightly
  backup, the install steps and the restore steps.
- `../deploy/mlflow.service`: the server on `127.0.0.1:8081`, one worker,
  sqlite backend, S3 artifacts.
- `../deploy/mlflow-backup.timer` and `mlflow-backup.service`: the nightly
  `sqlite3 .backup` copy to the bucket, 14 kept.
- `../scripts/open-tunnel.sh`: the reverse forward, the retry loop and the
  health probe from the box.
- `../scripts/start-remote-run.sh`: tunnel, health gate, detached launch of one
  arm on the box.
- `../configs/real.toml` and `../configs/lfm-spread.toml`: the `[mlflow]` table
  names the bucket and the artifact root. The tracking URI comes from the
  tunnel, never from a config file.

## Reports

- `postmortem-run17.md` and `postmortem-run17.pdf`: why the 313-step curve looked
  flat.
- `6.1-sol-xhigh-pass.md`: repository layout, LFM support, throughput, and the
  validation record.
- `SMOKE.md`: the smoke run result.
- `DEPLOY.md`: host, user, key, and run commands.
- `JUDGE_MODEL.md`: judge model research, kept local.

## Data

- `../data/mbpp_full.jsonl`: 961 MBPP rows.
- `../data/split_seed1234.json`: the pinned 711 and 250 split.
- `../data/train_split.jsonl`: the 711 training rows.
- `../data/difficulty_labels.jsonl`: one band per training row.
- `../screens/mbpp-base/`: the base-model screen. Ignored by git, restored from
  the bucket. See the README.
- `../runs/`: the box run folders. The live store is
  `/home/opencode/Work/rl-grpo-data/mlflow/mlflow.db` and is rebuildable from
  these folders.
- `../configs/`: arm configs, starting with `lfm-spread.toml` and
  `lfm-sweep.toml`.

## Outside the repository

- `/home/opencode/Work/rl-grpo-data/`: the data index and two reports. The bulk
  of the run data is in the Hetzner bucket under `dumps/`, and new runs upload
  under `runs/v1/`.

To refresh the docs, bring `DECISIONS.md` and `journal.md` up to date, fix this
index, and leave the code alone.
