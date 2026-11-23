# One tracking store, on the VM

Status: design, branch `feat/remote-tracking-store`.

## Why

The MLflow store is SQLite inside the runs root on the rented GPU box. Hetzner
reuses boxes, so each new box starts with an empty store and the previous
history dies with the old box. The bucket holds the run folders, but nothing
reads them back into a view.

The store moves to this VM, which stays up. The box sends metrics to it while a
run is live. One experiment accumulates every run across every box. The run
folder stays the record and the bucket stays the archive.

## What changes

1. An MLflow server runs on this VM as a systemd unit, bound to loopback.
2. The VM opens an SSH reverse tunnel to the box, so the box reaches the server
   with no public port and no second credential.
3. The mirror reads its tracking URI from config instead of hardcoding a local
   SQLite file.
4. An uploader copies checkpoints and the run folder to the bucket while the run
   goes, using a key limited to the `runs/` prefix.
5. A repair command replays a run folder into the store after a dropped link.

## Layout

    GPU box                                 this VM
    -------
    training -> run folder
             -> mlflow over ssh -R  ------> mlflow server -> sqlite
             -> checkpoints to bucket           |
                                                +-> artifacts proxy -> bucket

    ssh connection is opened by the VM, so the box needs no key and no open port.

## The server

- Unit `mlflow.service`, system unit, `User=opencode`.
- `ExecStart=/home/opencode/Work/rl-grpo/.venv/bin/mlflow server`
  `--host 127.0.0.1 --port 8081`
  `--backend-store-uri sqlite:////home/opencode/Work/rl-grpo-data/mlflow/mlflow.db`
  `--artifacts-destination s3://opencodeagent1/mlflow/v1`
  `--serve-artifacts --workers 1`
- Credentials in `/home/opencode/.config/mlflow/env`, mode 0600, read through
  `EnvironmentFile`. Keys for Hetzner object storage: `MLFLOW_S3_ENDPOINT_URL`,
  `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`.
- Health check: `GET /health` returns 200.
- The Mac reaches the UI through the existing Mac to VM tunnel on port 8083.
  MLflow listens on loopback only.
- The experiment is created with an explicit artifact location,
  `s3://opencodeagent1/mlflow/v1/artifacts`, from a new config key
  `[mlflow] artifact_root`. The server passes the same prefix to
  `--artifacts-destination`. Without it, `create_experiment` pins the experiment
  to a `file:` path under the runs root and eval tables land on the box disk.
- `rlvr dashboard` and `docker-run.sh dashboard` stop starting a second server.
  They print the remote URL and exit. `rlvr dashboard --local` keeps the old
  local server for offline use, on a different port.

## The tunnel

- The VM already holds `id_ed25519` for the box, so it opens the connection:
  `ssh -N -R <docker-gateway>:8081:127.0.0.1:8081 Ubuntu@<box>`.
- `<docker-gateway>` is read on the box with
  `docker network inspect bridge --format '{{(index .IPAM.Config 0).Gateway}}'`.
  It is 172.17.0.1 on a default install. Reading it removes the guess.
- `-o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=3`
  and a retry loop, so a dropped link reconnects without a person.
- `GatewayPorts clientspecified` in the box `sshd_config`, set by
  `scripts/docker-setup.sh`. Without it the forward binds loopback only and the
  container cannot reach it.
- The container gets `MLFLOW_TRACKING_URI=http://<docker-gateway>:8081`.
- No token and no TLS, because nothing listens off loopback on either end.

## Four arms on one box

The first real run puts four arms on four GPUs of the same box, at the same
time. The design already covers it, with two settings.

- One tunnel per box, not per run. All four arms point at the same forwarded
  port. Four clients posting to one server is the normal case.
- `--workers 1`. MLflow documents that SQLite permits one concurrent writer, so
  the server serialises writes in one process and threads the reads. More
  workers would collide on the database file.
- Four runs stay distinct because a run's identity is the arm and the start
  timestamp. Same arm twice gets two prefixes, because the timestamps differ.
- Checkpoint uploads cap at two concurrent transfers across the box. Four arms
  at 7.6 GB each saturate the link and the hashing would fight the training
  processes for CPU.

Postgres becomes worth it when more than one server process, or more than one
box writing through separate tunnels, becomes routine. Not for four arms on one
box.

## The mirror

- New config key `[mlflow] tracking_uri`, optional. Absent means the current
  behaviour, `sqlite:///<runs root>/mlflow.db`, so the tests and bare metal runs
  keep working.
- `MlflowMirror` passes that URI to `set_tracking_uri`.
- Everything else in `mlflow_log.py` stays. Run ids still land in
  `mlflow.json`, the heartbeat and the dead-run sweep still work against the
  remote store.

## Checkpoints and the run folder

- New module `src/rlvr/upload.py`. It copies:
  - `checkpoints/` after each checkpoint is written,
  - `meta.json`, `metrics.jsonl`, `eval.json` and `labels.jsonl` on close,
  - a `manifest.json` with the schema version, the config hash and a sha256 per
    file.
- Uploads run on a background thread. A slow or failed upload never blocks a
  step and never stops a run. Failures retry with backoff, then give up and
  leave the file for the next attempt.
- Bucket destination `s3://opencodeagent1/runs/v1/<arm>/<started>/`. The key on
  the box is scoped to the `runs/` prefix, so it cannot read, write or delete
  `dumps/`.
- The box gets `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` and
  `MLFLOW_S3_ENDPOINT_URL` from `~/.env`, which the deploy docs already source
  before a launch. `docker-run.sh` adds three `-e` flags to the existing list.
- The uploader owns the `rlvr/checkpoints_uri` tag. It reads the run id from
  `mlflow.json` and sets the tag through `MlflowClient`, because only the
  uploader knows when the first checkpoint lands.
- Checkpoints are not MLflow artifacts. A 7.6 GB run would stall a step on the
  upload and store a second copy. MLflow records `rlvr/checkpoints_uri` as a tag
  instead, so the run page points at the bucket.
- `boto3==1.44.9` joins the `tracking` extra. MLflow needs it for the artifact
  proxy and the uploader needs it on the box.

## The bucket

- New prefixes only: `runs/v1/...` and `mlflow/v1/...`.
- `dumps/` stays untouched. Moving it under `archive/` is a later, separate
  task, and it must verify byte equality with `rclone check` before anything is
  deleted.
- The `v1` segment lets a later layout change without breaking old prefixes.

## Repair

- `rlvr sync-mlflow <run folder>` replays a folder into the store with the same
  code as `rlvr import-mlflow`. It deletes and rebuilds that run's entries, so
  repeated calls are safe.
- A run's identity is `rlvr/run_name`, the arm and the start timestamp, not the
  folder path. The container path `/app/runs/...` and the VM path differ, so
  filtering on `rlvr/run_dir` would find nothing and add a duplicate entry on
  every call. `_runs_for_folder` filters on `rlvr/run_name`, and the tag is
  written at `start`.
- The tracking URI comes from `--config`, else the `config` snapshot in
  `meta.json`, else the environment.
- Used after a tunnel drop, or on a box that ran with the server down.

## Backup

- `mlflow-backup.timer` runs a nightly `sqlite3 .backup` into
  `s3://opencodeagent1/mlflow/v1/backups/`, keeping 14 copies.
- The store is rebuildable from run folders. The backup saves the rebuild.

## Goals

- Every run from now on appears in one store, whatever box runs it.
- The live view updates during a run over one SSH connection.
- A dead box loses no checkpoints that already uploaded.
- One credential lives on this VM for the artifact proxy. One scoped credential
  lives on the box for checkpoints.

## Non-goals

- No public listener, no ngrok, no tailscale.
- No Postgres. SQLite on this VM is enough for one experiment.
- No import of the existing dumps. The store starts empty.
- No change to training, rewards, pools, schedules or the queue.
- No backwards compatibility for old bucket paths.

## Anti-goals

- Losing a run because the server or the link was down.
- Keys in git, in the image, or in a run folder.
- Checkpoint uploads that stall a training step.
- A store only MLflow can read. The run folder stays readable without it.

## Constraints

- The VM has no sudo from the agent. The install command is written for the
  person to run.
- The box host changes on each pull, so the tunnel command takes the host as an
  argument.
- MLflow has no metric delete. Repair and resume rebuild the run entry instead.
- The box runs the training container on the default bridge network. The
  gateway address is a host address the container can reach.

## Calls made here

- Store on this VM rather than a managed database.
- SSH reverse tunnel rather than a public port or a third-party tunnel.
- Checkpoints direct to the bucket rather than through MLflow.
- Uploads on a background thread, best effort.
- Start empty, no import of old runs.
- Keep every checkpoint. No retention policy yet.
- Explicit artifact root in S3, not the server default.
- `rlvr dashboard` points at the remote store. `--local` keeps the old server.
- Identity for repair is `rlvr/run_name`, the arm and timestamp, not the path.

## What is left to the builder

- The exact systemd unit text and the one-line install command.
- Whether the tunnel is opened by `scripts/start-remote-run.sh` or by
  `docker-run.sh` itself.
- The upload queue shape, retry counts and thread shutdown.
- How the manifest handles a checkpoint directory that changes under it.
- Which port `--local` uses, and whether it shares the saved view.
- The tests: config parsing, mirror URI, upload retry, repair idempotence.