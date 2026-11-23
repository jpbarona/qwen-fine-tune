"""Tests for the bucket uploader, against a stub S3 client.

Nothing here touches a network or real AWS. The stub records the key and the
body of every `put_object` call, and can fail a fixed number of times first.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from rlvr.upload import MANIFEST_NAME, RunUploader, upload_prefix

RUN_NAME = "20260101T000000Z"
DESTINATION = "s3://b/runs/v1"


class StubS3:
    def __init__(self, fail_times: int = 0):
        self.calls: list[tuple[str, bytes]] = []
        self.fail_times = fail_times

    def put_object(self, Bucket, Key, Body):
        if self.fail_times > 0:
            self.fail_times -= 1
            raise OSError("boom")
        self.calls.append((Key, Body.read()))


def make_run(tmp_path: Path) -> Path:
    """A run folder with meta.json, metrics.jsonl and no checkpoints."""
    run_dir = tmp_path / "runs" / "code-score" / RUN_NAME
    run_dir.mkdir(parents=True)
    meta = {
        "arm": "code-score",
        "timestamp": RUN_NAME,
        "mode": "fake",
        "config": {"steps": 2, "learning_rate": 0.0001},
    }
    (run_dir / "meta.json").write_text(json.dumps(meta) + "\n", encoding="utf-8")
    (run_dir / "metrics.jsonl").write_text(
        json.dumps({"step": 1, "pass_rate": 0.5}) + "\n", encoding="utf-8"
    )
    return run_dir


def make_checkpoint(run_dir: Path, step: str = "000010", payload: bytes = b"x") -> Path:
    """One checkpoint file under the run folder, and return its path."""
    folder = run_dir / "checkpoints" / step
    folder.mkdir(parents=True)
    path = folder / "adapter.bin"
    path.write_bytes(payload)
    return path


def uploaded(stub: StubS3) -> dict[str, bytes]:
    """The keys the stub accepted, mapped to the bytes of the body."""
    return dict(stub.calls)


def test_upload_prefix_strips_the_trailing_slash() -> None:
    assert upload_prefix("s3://b/runs/v1/", "code-score", RUN_NAME) == (
        f"s3://b/runs/v1/code-score/{RUN_NAME}"
    )
    assert upload_prefix("s3://b/runs/v1", "code-score", RUN_NAME) == (
        f"s3://b/runs/v1/code-score/{RUN_NAME}"
    )


def test_checkpoint_is_uploaded_under_the_run_prefix(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path)
    make_checkpoint(run_dir)
    client = StubS3()
    uploader = RunUploader(
        run_dir, "code-score", RUN_NAME, DESTINATION, client=client, interval=0.01
    )
    uploader.start()
    uploader.wait_for_idle()
    uploader.close()

    keys = [key for key, _ in client.calls]
    assert f"runs/v1/code-score/{RUN_NAME}/checkpoints/000010/adapter.bin" in keys
    assert keys[-1] == f"runs/v1/code-score/{RUN_NAME}/{MANIFEST_NAME}"
    assert not uploader.errors


def test_manifest_records_a_hash_per_file(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path)
    make_checkpoint(run_dir, payload=b"x")
    client = StubS3()
    uploader = RunUploader(
        run_dir, "code-score", RUN_NAME, DESTINATION, client=client, interval=0.01
    )
    uploader.start()
    uploader.wait_for_idle()
    uploader.close()

    bodies = uploaded(client)
    manifest = json.loads(
        bodies[f"runs/v1/code-score/{RUN_NAME}/{MANIFEST_NAME}"].decode("utf-8")
    )
    assert manifest["schema"] == 1
    assert len(manifest["config_hash"]) == 64
    entry = next(
        row
        for row in manifest["files"]
        if row["path"] == "checkpoints/000010/adapter.bin"
    )
    assert entry["sha256"] == hashlib.sha256(b"x").hexdigest()
    assert entry["size"] == 1
    assert entry["key"] == (
        f"runs/v1/code-score/{RUN_NAME}/checkpoints/000010/adapter.bin"
    )


def test_a_checkpoint_written_after_start_is_uploaded(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path)
    client = StubS3()
    uploader = RunUploader(
        run_dir, "code-score", RUN_NAME, DESTINATION, client=client, interval=0.01
    )
    uploader.start()
    key = f"runs/v1/code-score/{RUN_NAME}/checkpoints/000011/adapter.bin"
    try:
        make_checkpoint(run_dir, step="000011", payload=b"later")
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and key not in uploaded(client):
            uploader.wait_for_idle(0.1)
            time.sleep(0.01)
    finally:
        uploader.close()

    assert key in uploaded(client)
    assert not uploader.errors


def test_upload_retries_then_gives_up_without_raising(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path)
    make_checkpoint(run_dir)
    client = StubS3(fail_times=100)
    uploader = RunUploader(
        run_dir,
        "code-score",
        RUN_NAME,
        DESTINATION,
        client=client,
        retries=3,
        interval=0.01,
        backoff=0.01,
    )
    uploader.start()
    uploader.wait_for_idle()
    uploader.close()

    assert uploader.errors
    assert client.calls == []


def test_close_uploads_the_folder_with_no_checkpoints(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path)
    client = StubS3()
    uploader = RunUploader(
        run_dir, "code-score", RUN_NAME, DESTINATION, client=client, interval=0.01
    )
    # No start call: close still uploads the run folder.
    uploader.close()

    keys = [key for key, _ in client.calls]
    assert f"runs/v1/code-score/{RUN_NAME}/meta.json" in keys
    assert f"runs/v1/code-score/{RUN_NAME}/metrics.jsonl" in keys
    assert keys[-1] == f"runs/v1/code-score/{RUN_NAME}/{MANIFEST_NAME}"
    assert not uploader.errors


def test_close_uploads_only_the_run_files_that_exist(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path)
    (run_dir / "eval.json").write_text("{}\n", encoding="utf-8")
    client = StubS3()
    uploader = RunUploader(
        run_dir, "code-score", RUN_NAME, DESTINATION, client=client, interval=0.01
    )
    uploader.close()

    keys = [key for key, _ in client.calls]
    prefix = f"runs/v1/code-score/{RUN_NAME}/"
    assert keys[:3] == [
        prefix + "meta.json",
        prefix + "metrics.jsonl",
        prefix + "eval.json",
    ]
    assert prefix + "labels.jsonl" not in keys
    assert prefix + "control-events.jsonl" not in keys
