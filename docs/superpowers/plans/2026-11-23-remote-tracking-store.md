# Remote Tracking Store Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One MLflow store on this VM that accumulates every run, with the box sending metrics live over an SSH tunnel and checkpoints archived to the bucket.

**Architecture:** The mirror stops hardcoding a local SQLite path and takes a tracking URI and artifact root from config. A watcher thread uploads checkpoints and the run folder to the bucket with a scoped key. Deploy files hold the server unit, the backup timer and the tunnel launcher.

**Tech Stack:** Python 3.12, MLflow 3.18.0, boto3 1.44.9, SQLite, systemd, SSH port forwarding, rclone only for manual restores.

**Spec:** `docs/superpowers/specs/2026-11-23-remote-tracking-store-design.md`

## Global Constraints

- `mlflow==3.18.0` and `boto3==1.44.9` are pinned exactly in `pyproject.toml` under `[project.optional-dependencies] tracking`.
- The server binds `127.0.0.1:8081` and runs with `--workers 1`.
- Nothing listens on a public interface. No ngrok, no tailscale.
- The run folder stays the record. A failed mirror or upload never stops a step.
- Bucket prefixes: `s3://opencodeagent1/runs/v1/<arm>/<started>/` and `s3://opencodeagent1/mlflow/v1/`.
- A run's identity is the `rlvr/run_name` tag, the value `f"{arm}-{timestamp}"`.
- `dumps/` is never touched by this work.
- Plain prose, no semicolons or em dashes in docs and messages.

## Review Focus

- A missing `[mlflow]` table must keep the old local SQLite behaviour. A config written before this change must not start a server connection.
- An upload failure, including a client that raises on every call, must not raise into training and must not stop the uploader thread from handling later files.
- `sync-mlflow` run twice on the same folder must leave one parent run and one child run, not four.
- The tunnel script must exit nonzero when the health check never answers, instead of leaving a detached run with no store.
- A run with no checkpoints must still upload `meta.json`, `metrics.jsonl` and a manifest.

---

### Task 1: Config keys for the remote store and the bucket

**Files:**
- Modify: `src/rlvr/config.py:186-205`, `src/rlvr/config.py:568-580`, the module docstring near line 57
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `MlflowSettings(enabled: bool = True, gpu_sample_seconds: float = 1.0, tracking_uri: str | None = None, artifact_root: str | None = None, bucket: str | None = None, upload_concurrency: int = 2)`
- Consumes: `_optional_string(raw, dotted, default)`, `_optional_positive_int(raw, dotted, default)`

- [ ] **Step 1: Write the failing tests**

```python
def test_mlflow_remote_keys_parse(tmp_path):
    config = load_config(write_toml(tmp_path, """
[mlflow]
tracking_uri = "http://172.17.0.1:8081"
artifact_root = "s3://opencodeagent1/mlflow/v1/artifacts"
bucket = "s3://opencodeagent1/runs/v1"
upload_concurrency = 3
"""))
    assert config.mlflow.tracking_uri == "http://172.17.0.1:8081"
    assert config.mlflow.artifact_root == "s3://opencodeagent1/mlflow/v1/artifacts"
    assert config.mlflow.bucket == "s3://opencodeagent1/runs/v1"
    assert config.mlflow.upload_concurrency == 3


def test_mlflow_remote_keys_default_to_none(tmp_path):
    config = load_config(write_toml(tmp_path, ""))
    assert config.mlflow.tracking_uri is None
    assert config.mlflow.artifact_root is None
    assert config.mlflow.bucket is None
    assert config.mlflow.upload_concurrency == 2


def test_mlflow_upload_concurrency_must_be_positive(tmp_path):
    with pytest.raises(ConfigError):
        load_config(write_toml(tmp_path, "[mlflow]\nupload_concurrency = 0\n"))
```

Use the file's existing helpers for writing a config. Keep the helper names in this file, do not invent new ones.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_config.py -k "mlflow_remote or upload_concurrency" -v`
Expected: FAIL, `MlflowSettings` has no attribute `tracking_uri`.

- [ ] **Step 3: Add the fields and the parsing**

Add the four fields to `MlflowSettings` and parse them in `load_config` with `_optional_string(raw, "mlflow.tracking_uri", None)`, `_optional_string(raw, "mlflow.artifact_root", None)`, `_optional_string(raw, "mlflow.bucket", None)` and `_optional_positive_int(raw, "mlflow.upload_concurrency", DEFAULT_UPLOAD_CONCURRENCY)`.

Add `DEFAULT_UPLOAD_CONCURRENCY = 2` beside the other `[mlflow]` defaults. Add the four keys to the module docstring block at line 57. Add `boto3==1.44.9` to the `tracking` extra in `pyproject.toml`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_config.py -v`
Expected: PASS, all config tests.

- [ ] **Step 5: Commit**

```bash
git add src/rlvr/config.py tests/test_config.py pyproject.toml
git commit -m "Read the remote tracking URI, artifact root and bucket from config"
```

---

### Task 2: The mirror uses the configured URI and artifact root

**Files:**
- Modify: `src/rlvr/mlflow_log.py:399-450`, `src/rlvr/mlflow_log.py:953-970`, `src/rlvr/cli.py:735`
- Test: `tests/test_mlflow_log.py`

**Interfaces:**
- Consumes: `MlflowSettings.tracking_uri` and `artifact_root` from Task 1.
- Produces:
  - `MlflowMirror(runs_root: Path, run_dir: Path, *, tracking_uri: str | None = None, artifact_root: str | None = None)`
  - `open_mirror(runs_root: Path, run_dir: Path, enabled: bool, *, tracking_uri: str | None = None, artifact_root: str | None = None) -> Mirror`

- [ ] **Step 1: Write the failing tests**

```python
def test_mirror_uses_the_configured_tracking_uri(tmp_path: Path) -> None:
    store = tmp_path / "store"
    store.mkdir()
    run_dir = make_run(tmp_path)
    uri = f"sqlite:///{store / 'other.db'}"
    mirror = MlflowMirror(tmp_path / "runs", run_dir, tracking_uri=uri)
    mirror.start({"arm": "code-score"})
    assert (store / "other.db").exists()
    assert not (tmp_path / "runs" / "mlflow.db").exists()


def test_mirror_creates_the_experiment_at_the_artifact_root(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path)
    root = "s3://opencodeagent1/mlflow/v1/artifacts"
    mirror = MlflowMirror(tmp_path / "runs", run_dir, artifact_root=root)
    mirror.start({"arm": "code-score"})
    client = MlflowClient()
    experiment = client.get_experiment_by_name("rlvr")
    assert experiment.artifact_location == root


def test_mirror_without_a_uri_keeps_the_local_store(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path)
    mirror = MlflowMirror(tmp_path / "runs", run_dir)
    mirror.start({"arm": "code-score"})
    assert (tmp_path / "runs" / "mlflow.db").exists()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_mlflow_log.py -k "configured_tracking_uri or artifact_root" -v`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument`.

- [ ] **Step 3: Implement the constructor and connect step**

Store both values on the mirror. In `_connect`, call
`mlflow.set_tracking_uri(self.tracking_uri or f"sqlite:///{self.runs_root / DATABASE_NAME}")`.
When `artifact_root` is set, `create_experiment(EXPERIMENT_NAME, artifact_location=self.artifact_root)`. Otherwise keep `str(self.runs_root / ARTIFACTS_NAME)`.

Pass both through `open_mirror`. In `run_command`, pass `config.mlflow.tracking_uri` and `config.mlflow.artifact_root`.

- [ ] **Step 4: Run the mirror tests**

Run: `.venv/bin/pytest tests/test_mlflow_log.py tests/test_mlflow_resume.py tests/test_cli.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/rlvr/mlflow_log.py src/rlvr/cli.py tests/test_mlflow_log.py
git commit -m "Point the mirror at the configured store and artifact root"
```

---

### Task 3: Repair matches on the run name, not the folder path

**Files:**
- Modify: `src/rlvr/mlflow_log.py:807-835`
- Test: `tests/test_mlflow_resume.py`

**Interfaces:**
- Consumes: `rlvr/run_name`, already written at `start` as `f"{arm}-{timestamp}"`.
- Produces: `MlflowMirror.run_key: str`, computed from `meta.json` when present, else from the folder names.

- [ ] **Step 1: Write the failing test**

```python
def test_rebuild_finds_a_run_copied_to_another_path(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    first = make_run(tmp_path)
    mirror = MlflowMirror(runs_root, first)
    mirror.start({"arm": "code-score", "timestamp": RUN_NAME})
    mirror.close("FINISHED")

    copy = tmp_path / "elsewhere" / "code-score" / RUN_NAME
    shutil.copytree(first, copy)
    rebuilt = MlflowMirror(runs_root, copy)
    rebuilt.rebuild()
    rebuilt.close("FINISHED")

    clients = MlflowClient()
    parents = [
        run for run in clients.search_runs(
            experiment_ids=[rebuilt.experiment_id],
            filter_string=(
                "attributes.status = 'FINISHED' and "
                f"attributes.run_name = 'code-score-{RUN_NAME}'"
            ),
        )
        if run.data.tags.get("mlflow.parentRunId") is None
    ]
    assert len(parents) == 1
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/pytest tests/test_mlflow_resume.py -k copied_to_another_path -v`
Expected: FAIL, two runs, because `rlvr/run_dir` differs.

- [ ] **Step 3: Implement the identity**

In `__init__`, read `meta.json` from the run directory when it exists and compute
`self.run_key = f"{arm}-{timestamp}"` with `_run_name`. Fall back to the folder names. In `_runs_for_folder`, filter on
`f"tags.\`rlvr/run_name\` = '{escaped_key}'"`. Keep the child-run lookup.

- [ ] **Step 4: Run the resume and import tests**

Run: `.venv/bin/pytest tests/test_mlflow_resume.py tests/test_mlflow_import.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/rlvr/mlflow_log.py tests/test_mlflow_resume.py
git commit -m "Match a repaired run by name so a copied folder replaces it"
```

---

### Task 4: The bucket uploader

**Files:**
- Create: `src/rlvr/upload.py`
- Test: `tests/test_upload.py`

**Interfaces:**
- Produces:
  - `upload_prefix(destination: str, arm: str, started: str) -> str` returning `<destination>/<arm>/<started>` with the trailing slash stripped.
  - `RunUploader(run_dir: Path, arm: str, started: str, destination: str, *, client: Any = None, concurrency: int = 2, interval: float = 10.0, retries: int = 3)`
  - `.start() -> None`, `.close(timeout: float = 60.0) -> None`, `.wait_for_idle(timeout: float = 10.0) -> None`, `.errors: list[str]`
  - `MANIFEST_NAME = "manifest.json"`, schema value `1`.
- Consumes: a boto3 client with `put_object(Bucket=..., Key=..., Body=...)`, or an injected stub with the same call.

- [ ] **Step 1: Write the failing tests**

```python
class StubS3:
    def __init__(self, fail_times: int = 0):
        self.calls: list[tuple[str, bytes]] = []
        self.fail_times = fail_times

    def put_object(self, Bucket, Key, Body):
        if self.fail_times > 0:
            self.fail_times -= 1
            raise OSError("boom")
        self.calls.append((Key, Body.read()))


def test_checkpoint_is_uploaded_under_the_run_prefix(tmp_path):
    run_dir = make_run(tmp_path)
    (run_dir / "checkpoints" / "000010").mkdir(parents=True)
    (run_dir / "checkpoints" / "000010" / "adapter.bin").write_bytes(b"x")
    client = StubS3()
    uploader = RunUploader(run_dir, "code-score", RUN_NAME, "s3://b/runs/v1",
                           client=client, interval=0.01)
    uploader.start()
    uploader.wait_for_idle()
    uploader.close()
    keys = [key for key, _ in client.calls]
    assert f"runs/v1/code-score/{RUN_NAME}/checkpoints/000010/adapter.bin" in keys


def test_manifest_records_a_hash_per_file(tmp_path):
    ...  # start, wait, close, then read manifest.json from the stub calls
    assert entry["sha256"] == hashlib.sha256(b"x").hexdigest()
    assert manifest["schema"] == 1


def test_upload_retries_then_gives_up_without_raising(tmp_path):
    client = StubS3(fail_times=100)
    uploader = RunUploader(..., client=client, retries=3, interval=0.01)
    uploader.start()
    uploader.wait_for_idle()
    uploader.close()
    assert uploader.errors


def test_close_uploads_the_folder_with_no_checkpoints(tmp_path):
    ...  # meta.json and metrics.jsonl, then close, assert both keys and the manifest
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_upload.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'rlvr.upload'`.

- [ ] **Step 3: Implement the watcher**

A background thread scans `checkpoints/` every `interval` seconds and queues files whose `(size, mtime_ns)` changed since the last scan. A `ThreadPoolExecutor(max_workers=concurrency)` uploads, retrying `retries` times with 1, 2, 4 second backoff. `close` stops the scan, waits for the queue, uploads `meta.json`, `metrics.jsonl`, `eval.json`, `labels.jsonl` and `control-events.jsonl` when present, then uploads `manifest.json` last. Compute sha256 while reading the file once. Every exception goes to `.errors` and never escapes. Import `boto3` inside the factory that builds the default client, so an injected stub needs no boto3.

- [ ] **Step 4: Run the uploader tests**

Run: `.venv/bin/pytest tests/test_upload.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/rlvr/upload.py tests/test_upload.py
git commit -m "Upload checkpoints and the run folder to the bucket"
```

---

### Task 5: Start the uploader in a run, and add `sync-mlflow`

**Files:**
- Modify: `src/rlvr/cli.py` (run command around line 735, parser around line 323, import command around line 954)
- Modify: `src/rlvr/mlflow_import.py:55`
- Test: `tests/test_cli.py`, `tests/test_mlflow_import.py`

**Interfaces:**
- Consumes: `RunUploader` from Task 4, `import_run` from `mlflow_import`.
- Produces:
  - `import_run(runs_root: Path, run_dir: Path, *, rebuild: bool = False) -> str | None`
  - `rlvr sync-mlflow <run folder> [--config PATH]` which calls `import_run(..., rebuild=True)`.

- [ ] **Step 1: Write the failing tests**

```python
def test_sync_mlflow_twice_leaves_one_run(tmp_path, capsys):
    folder = make_run(tmp_path)
    args = Namespace(run_dir=folder, config=None)
    assert sync_mlflow_command(args) == 0
    first = count_runs()
    assert sync_mlflow_command(args) == 0
    assert count_runs() == first


def test_run_starts_the_uploader_when_a_bucket_is_set(tmp_path, monkeypatch):
    ...  # patch RunUploader, run_command with [mlflow] bucket set, assert start and close called
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_cli.py -k sync_mlflow -v`
Expected: FAIL, no such command.

- [ ] **Step 3: Implement**

`sync_mlflow_command` resolves the tracking URI from `--config`, else the `config` snapshot in `meta.json`, else the environment, opens a mirror over the folder with `rebuild=True`, and prints the run id. `import_run` gains `rebuild`, which calls `mirror.rebuild()` instead of skipping an existing `mlflow.json`. In `run_command`, when `config.mlflow.bucket` is set, build `RunUploader(store.run_dir, arm, started, config.mlflow.bucket, concurrency=config.mlflow.upload_concurrency)`, call `start()` before training, set the tag `rlvr/checkpoints_uri` to `upload_prefix(...)` with `MlflowClient().set_tag(mirror.run_id, ...)` once the mirror has a run id, and call `close()` in the existing finally block.

- [ ] **Step 4: Run the CLI tests**

Run: `.venv/bin/pytest tests/test_cli.py tests/test_mlflow_import.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/rlvr/cli.py src/rlvr/mlflow_import.py tests/test_cli.py tests/test_mlflow_import.py
git commit -m "Start the uploader with a run and repair a folder on demand"
```

---

### Task 6: `rlvr dashboard` points at the remote store

**Files:**
- Modify: `src/rlvr/cli.py:989-1027` and the dashboard parser
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `config.mlflow.tracking_uri`.
- Produces: `dashboard_command` returns 0 after printing the URI when `--local` is absent. `--local` keeps the old server path on `--port`, default 8099.

- [ ] **Step 1: Write the failing tests**

```python
def test_dashboard_prints_the_remote_store(capsys):
    args = Namespace(local=False, port=8081, runs_root=Path("runs"))
    assert dashboard_command(args) == 0
    assert "http://127.0.0.1:8081" in capsys.readouterr().out


def test_dashboard_local_still_starts_a_server(monkeypatch):
    ...  # patch os.execv, assert argv holds "--backend-store-uri" and the local db
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_cli.py -k dashboard -v`
Expected: FAIL, `Namespace` has no `local`.

- [ ] **Step 3: Implement**

Add `--local` to the dashboard parser. Without it, print the configured tracking URI, or `http://127.0.0.1:8081` when unset, then return 0. With it, run the existing code and default the port to 8099.

- [ ] **Step 4: Run the CLI tests**

Run: `.venv/bin/pytest tests/test_cli.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/rlvr/cli.py tests/test_cli.py
git commit -m "Point the dashboard command at the remote store"
```

---

### Task 7: Deploy files for the server and the backup

**Files:**
- Create: `deploy/mlflow.service`, `deploy/mlflow.env.example`, `deploy/backup.py`, `deploy/mlflow-backup.service`, `deploy/mlflow-backup.timer`, `deploy/install.sh`, `deploy/README.md`
- Test: `tests/test_packaging.py`

**Interfaces:**
- Produces: `deploy/install.sh` which copies the unit files, creates `/home/opencode/Work/rl-grpo-data/mlflow`, and reloads systemd. `deploy/backup.py` which copies the SQLite file to `s3://opencodeagent1/mlflow/v1/backups/mlflow-<date>.db` using `sqlite3.Connection.backup` and boto3.

- [ ] **Step 1: Write the failing tests**

```python
def test_service_runs_the_repo_venv():
    text = (ROOT / "deploy" / "mlflow.service").read_text()
    assert "/home/opencode/Work/rl-grpo/.venv/bin/mlflow" in text
    assert "--workers 1" in text
    assert "127.0.0.1" in text


def test_env_example_lists_the_bucket_variables():
    text = (ROOT / "deploy" / "mlflow.env.example").read_text()
    for name in ("MLFLOW_S3_ENDPOINT_URL", "AWS_ACCESS_KEY_ID",
                 "AWS_SECRET_ACCESS_KEY", "AWS_DEFAULT_REGION"):
        assert name in text


def test_backup_script_is_importable_and_has_a_main():
    ...  # import deploy/backup.py by path, assert main exists
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_packaging.py -k "service or env_example or backup" -v`
Expected: FAIL, file not found.

- [ ] **Step 3: Write the files**

The unit runs `mlflow server --host 127.0.0.1 --port 8081 --backend-store-uri sqlite:////home/opencode/Work/rl-grpo-data/mlflow/mlflow.db --artifacts-destination s3://opencodeagent1/mlflow/v1 --serve-artifacts --workers 1` with `EnvironmentFile=/home/opencode/.config/mlflow/env`, `Restart=always`, `User=opencode`. The timer runs daily at 03:00 UTC. `install.sh` is idempotent and prints the two commands it cannot run.

- [ ] **Step 4: Run the packaging tests**

Run: `.venv/bin/pytest tests/test_packaging.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add deploy tests/test_packaging.py
git commit -m "Add the server unit, the backup timer and the install script"
```

---

### Task 8: The tunnel and the launch script

**Files:**
- Create: `scripts/start-remote-run.sh`, `scripts/open-tunnel.sh`
- Modify: `scripts/docker-setup.sh` (set `GatewayPorts clientspecified`)
- Test: `tests/test_remote_run.py`

**Interfaces:**
- Produces:
  - `scripts/open-tunnel.sh <host> [port]` opens `ssh -N -R <gateway>:<port>:127.0.0.1:<port>`, writes a pid file, retries with `ServerAliveInterval=15`.
  - `scripts/start-remote-run.sh <arm> <host> [extra args]` calls `open-tunnel.sh`, waits for `http://127.0.0.1:<port>/health` through the box, then launches `docker-run.sh` detached and nonzero-exits when the health check fails 30 times.

- [ ] **Step 1: Write the failing test**

```python
def test_start_remote_run_uses_a_reverse_forward(tmp_path):
    log = tmp_path / "ssh.log"
    run_script_with_fake_binaries(tmp_path, log)
    assert "-R" in log.read_text()
    assert "GatewayPorts" not in log.read_text()


def test_start_remote_run_fails_when_health_never_answers(tmp_path):
    result = run_script_with_dead_health(tmp_path)
    assert result.returncode != 0
```

Use a temp `PATH` holding a stub `ssh` that logs argv and exits 0, a stub `curl` that exits 1, and a stub `docker` that logs.

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/pytest tests/test_remote_run.py -v`
Expected: FAIL, script not found.

- [ ] **Step 3: Write the scripts**

`open-tunnel.sh` discovers the gateway on the box with
`ssh "$host" "docker network inspect bridge --format '{{(index .IPAM.Config 0).Gateway}}'"`,
defaults to 172.17.0.1, writes `~/.cache/rlvr/tunnel-<host>.pid`, and passes
`-o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=3`.
`start-remote-run.sh` runs `open-tunnel.sh`, polls the health endpoint 30 times at 2 seconds, exits 1 with a message when it never answers, then runs `docker-run.sh` under `setsid nohup`.

`docker-setup.sh` adds `GatewayPorts clientspecified` to `/etc/ssh/sshd_config` when absent and reloads sshd, printing the command instead when it lacks root.

- [ ] **Step 4: Run the script tests**

Run: `.venv/bin/pytest tests/test_remote_run.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/start-remote-run.sh scripts/open-tunnel.sh scripts/docker-setup.sh tests/test_remote_run.py
git commit -m "Open the tunnel and launch a run against the remote store"
```

---

### Task 10: Wire the bucket credentials and the default config

**Why this exists:** the plan left two design requirements unassigned. Without
them the live path cannot write a checkpoint or an eval table to the bucket.

**Files:**
- Modify: `scripts/docker-run.sh`
- Modify: `configs/real.toml`, `configs/lfm-spread.toml`
- Test: `tests/test_packaging.py`

**Interfaces:**
- Consumes: `MlflowSettings.bucket` and `artifact_root` from Task 1, `RunUploader` from Task 4.
- Produces: the container receives `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `MLFLOW_S3_ENDPOINT_URL` and `AWS_DEFAULT_REGION` from the host shell, and the two box configs set `bucket = "s3://opencodeagent1/runs/v1"` and `artifact_root = "s3://opencodeagent1/mlflow/v1/artifacts"`.

- [ ] **Step 1: Write the failing tests**

```python
def test_docker_run_passes_the_bucket_credentials():
    text = (REPO_ROOT / "scripts" / "docker-run.sh").read_text()
    for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
                 "MLFLOW_S3_ENDPOINT_URL", "AWS_DEFAULT_REGION"):
        assert f"-e {name}" in text


def test_the_box_configs_name_the_bucket():
    for name in ("real.toml", "lfm-spread.toml"):
        config = load_config(REPO_ROOT / "configs" / name)
        assert config.mlflow.bucket == "s3://opencodeagent1/runs/v1"
        assert config.mlflow.artifact_root == "s3://opencodeagent1/mlflow/v1/artifacts"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_packaging.py -k "credentials or name_the_bucket" -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

Add the four `-e` flags to the training `docker run` in `scripts/docker-run.sh`, beside the existing ones. Add the `[mlflow]` table with the two keys to both configs. Do not add a `tracking_uri` to either file, because the tunnel supplies it per box.

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/pytest tests/test_packaging.py tests/test_config.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/docker-run.sh configs/real.toml configs/lfm-spread.toml tests/test_packaging.py
git commit -m "Pass the bucket credentials to the container and name the bucket in config"
```

---

### Task 9: Document the new path

**Files:**
- Modify: `docs/DEPLOY.md`, `README.md`, `docs/DECISIONS.md`, `docs/INDEX.md`
- Test: none. Reviewed by reading.

**Interfaces:**
- Consumes: everything above.

- [ ] **Step 1: Rewrite the Dashboard section of `docs/DEPLOY.md`**

State the server on the VM, the tunnel, `./scripts/start-remote-run.sh`, the UI through the Mac tunnel on 8083, and the backup timer. Remove the claim that the store starts empty on the box.

- [ ] **Step 2: Add the decisions**

Number after the last entry in `docs/DECISIONS.md`: the store lives on the VM, the tunnel replaces a public port, checkpoints go straight to the bucket, repair matches on the run name, the store starts empty.

- [ ] **Step 3: Add the plan and spec to `docs/INDEX.md`**

- [ ] **Step 4: Run the whole suite**

Run: `.venv/bin/pytest -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add docs README.md
git commit -m "Document the remote store, the tunnel and the bucket layout"
```
