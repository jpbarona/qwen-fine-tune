"""Tests for the packaging: the one runtime extra and the shipped configs."""

from __future__ import annotations

import importlib.util
import os
import sqlite3
import subprocess
import tomllib
from pathlib import Path
from types import ModuleType

import pytest

from rlvr.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_backup_module() -> ModuleType:
    path = REPO_ROOT / "deploy" / "backup.py"
    spec = importlib.util.spec_from_file_location("deploy_backup_under_test", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_vllm_extra_pins_the_whole_gpu_stack() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = pyproject["project"]["optional-dependencies"]
    expected = {
        "vllm": "0.34.0",
        "torch": "2.14.0",
        "transformers": "5.22.2",
        "peft": "0.18.1",
        "accelerate": "1.8.1",
    }
    pins = {}
    for requirement in extras["vllm"]:
        name, _, version = requirement.partition("==")
        assert version, f"{requirement!r} must pin an exact version"
        pins[name] = version
    assert pins == expected


def test_the_old_environment_extras_are_gone() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = pyproject["project"]["optional-dependencies"]

    assert set(extras) == {"dev", "tracking", "vllm"}


def test_the_tracking_extra_pins_mlflow() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = pyproject["project"]["optional-dependencies"]
    expected = {"mlflow": "3.18.0", "boto3": "1.44.9"}
    pins = {}
    for requirement in extras["tracking"]:
        name, _, version = requirement.partition("==")
        assert version, f"{requirement!r} must pin an exact version"
        pins[name] = version
    assert pins == expected


def test_no_extra_pins_torch_via_a_range() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    vllm = pyproject["project"]["optional-dependencies"]["vllm"]
    assert all(">=" not in requirement and "<" not in requirement for requirement in vllm)


def test_requires_python_is_3_12_or_newer() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["requires-python"] == ">=3.12"


def test_service_runs_the_repo_venv() -> None:
    text = (REPO_ROOT / "deploy" / "mlflow.service").read_text(encoding="utf-8")
    assert "/home/opencode/Work/rl-grpo/.venv/bin/mlflow" in text
    assert "--workers 1" in text
    assert "127.0.0.1" in text


def test_env_example_lists_the_bucket_variables() -> None:
    text = (REPO_ROOT / "deploy" / "mlflow.env.example").read_text(encoding="utf-8")
    for name in (
        "MLFLOW_S3_ENDPOINT_URL",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_DEFAULT_REGION",
    ):
        assert name in text


def docker_run_records(tmp_path: Path) -> list[list[str]]:
    """Run the launcher with a stub docker, and return each docker argv.

    The stub records every call with a `---` record separator, so the
    `docker image inspect` call and the `docker run` call stay separate.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "docker.argv"
    stub = bin_dir / "docker"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        'printf "%s\\n" "---" "$@" >> "$FAKE_DOCKER_ARGV"\n'
        "exit 0\n",
        encoding="utf-8",
    )
    stub.chmod(0o755)
    environment = dict(os.environ)
    environment["PATH"] = f"{bin_dir}:{environment['PATH']}"
    environment["FAKE_DOCKER_ARGV"] = str(log)
    result = subprocess.run(
        ["bash", str(REPO_ROOT / "scripts" / "docker-run.sh"), "code-score"],
        capture_output=True,
        text=True,
        env=environment,
        cwd=REPO_ROOT,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    records = log.read_text(encoding="utf-8").split("---\n")
    return [
        [argument for argument in record.splitlines() if argument]
        for record in records
        if record.strip()
    ]


def test_docker_run_passes_the_bucket_credentials(tmp_path: Path) -> None:
    records = docker_run_records(tmp_path)
    run = next(record for record in records if record[:1] == ["run"])
    forwarded = {
        run[index + 1]
        for index, argument in enumerate(run)
        if argument == "-e" and index + 1 < len(run)
    }
    assert {
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "MLFLOW_S3_ENDPOINT_URL",
        "AWS_DEFAULT_REGION",
        "MLFLOW_TRACKING_URI",
    } <= forwarded


def test_the_box_configs_name_the_bucket() -> None:
    for name in ("real.toml", "lfm-spread.toml"):
        config = load_config(REPO_ROOT / "configs" / name)
        assert config.mlflow.bucket == "s3://opencodeagent1/runs/v1"
        assert config.mlflow.artifact_root == "s3://opencodeagent1/mlflow/v1/artifacts"


def test_backup_script_is_importable_and_has_a_main() -> None:
    module = _load_backup_module()
    assert callable(module.main)


def test_the_copy_holds_committed_rows_only(tmp_path: Path) -> None:
    module = _load_backup_module()
    source = tmp_path / "mlflow.db"
    writer = sqlite3.connect(source)
    try:
        writer.execute("pragma journal_mode=wal")
        writer.execute("create table runs (id integer)")
        writer.execute("insert into runs values (1)")
        writer.commit()
        writer.execute("insert into runs values (2)")

        target = tmp_path / "copy.db"
        module.copy_database(source, target)

        copied = sqlite3.connect(target)
        try:
            rows = copied.execute("select id from runs order by id").fetchall()
        finally:
            copied.close()
    finally:
        writer.rollback()
        writer.close()

    assert rows == [(1,)]


def test_the_backup_returns_nonzero_when_the_upload_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_backup_module()
    database = tmp_path / "mlflow.db"
    sqlite3.connect(database).close()

    class FailingClient:
        def upload_file(self, *args: object, **kwargs: object) -> None:
            raise OSError("bucket is down")

    monkeypatch.setattr(module, "_client", lambda: FailingClient())

    assert module.main(["--db", str(database)]) == 1


def test_prune_leaves_objects_that_are_not_backups() -> None:
    module = _load_backup_module()
    prefix = "mlflow/v1/backups"
    backups = [f"{prefix}/mlflow-2026-11-{day:02d}.db" for day in range(1, 17)]
    foreign = f"{prefix}/aaa-notes.txt"

    class FakeClient:
        def __init__(self) -> None:
            self.deleted: list[str] = []

        def list_objects_v2(self, **kwargs: object) -> dict[str, object]:
            return {"Contents": [{"Key": key} for key in [*backups, foreign]]}

        def delete_object(self, **kwargs: object) -> None:
            self.deleted.append(str(kwargs["Key"]))

    client = FakeClient()
    deleted = module.prune(client, "bucket", prefix, 14)

    assert deleted == backups[:2]
    assert foreign not in client.deleted


def test_install_script_checks_its_prerequisites() -> None:
    text = (REPO_ROOT / "deploy" / "install.sh").read_text(encoding="utf-8")
    assert "import boto3" in text
    assert "replace-with-the-access-key" in text
    assert 'chmod 0600 "$ENV_FILE"' in text
    assert 'chown -R opencode:opencode "$DATA_DIR"' in text
    assert "systemctl restart mlflow.service" in text
