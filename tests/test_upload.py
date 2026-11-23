"""Tests for the bucket uploader, against a stub S3 client.

Nothing here touches a network or real AWS. The stub records the key and the
body of every `put_object` call, can fail a fixed number of times, and can
block one key so a close is caught with an upload in flight.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path

from rlvr.upload import MANIFEST_NAME, RunUploader, upload_prefix

RUN_NAME = "20260101T000000Z"
DESTINATION = "s3://b/runs/v1"
PREFIX = f"runs/v1/code-score/{RUN_NAME}"


class StubS3:
    def __init__(self, fail_times: int = 0, fail_suffix: str | None = None):
        self.calls: list[tuple[str, bytes]] = []
        self.attempts: list[tuple[str, float]] = []
        self.fail_times = fail_times
        self.fail_suffix = fail_suffix
        self.lock = threading.Lock()

    def put_object(self, Bucket, Key, Body):
        with self.lock:
            self.attempts.append((Key, time.monotonic()))
            should_fail = self.fail_times > 0 and (
                self.fail_suffix is None or Key.endswith(self.fail_suffix)
            )
            if should_fail:
                self.fail_times -= 1
        if should_fail:
            raise OSError("boom")
        with self.lock:
            self.calls.append((Key, Body.read()))

    def attempt_times(self, suffix: str) -> list[float]:
        with self.lock:
            return [when for key, when in self.attempts if key.endswith(suffix)]


class CountingStub(StubS3):
    """A stub that records how many transfers run at the same moment."""

    def __init__(self, delay: float = 0.05):
        super().__init__()
        self.delay = delay
        self.active = 0
        self.max_active = 0

    def put_object(self, Bucket, Key, Body):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(self.delay)
            return super().put_object(Bucket, Key, Body)
        finally:
            with self.lock:
                self.active -= 1


class BlockingStub(StubS3):
    """A stub that holds one checkpoint upload until the test releases it."""

    def __init__(self, blocked_suffix: str = "adapter.bin"):
        super().__init__()
        self.blocked_suffix = blocked_suffix
        self.started = threading.Event()
        self.release = threading.Event()

    def put_object(self, Bucket, Key, Body):
        if Key.endswith(self.blocked_suffix):
            self.started.set()
            self.release.wait(5.0)
        super().put_object(Bucket, Key, Body)


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


def make_checkpoint(
    run_dir: Path, step: str = "000010", payload: bytes = b"x"
) -> Path:
    """A complete checkpoint: the adapter, then the manifest the trainer writes last."""
    folder = run_dir / "checkpoints" / step
    folder.mkdir(parents=True)
    path = folder / "adapter.bin"
    path.write_bytes(payload)
    (folder / "checkpoint.json").write_text(
        json.dumps({"step": int(step)}) + "\n", encoding="utf-8"
    )
    return path


def uploaded(stub: StubS3) -> dict[str, bytes]:
    """The keys the stub accepted, mapped to the bytes of the body."""
    with stub.lock:
        return dict(stub.calls)


def keys_of(stub: StubS3) -> list[str]:
    with stub.lock:
        return [key for key, _ in stub.calls]


def manifest_of(stub: StubS3) -> dict:
    body = uploaded(stub)[f"{PREFIX}/{MANIFEST_NAME}"]
    return json.loads(body.decode("utf-8"))


def wait_for_key(
    stub: StubS3, key: str, uploader: RunUploader, timeout: float = 2.0
) -> bool:
    """Poll the stub while nudging the uploader, until the key appears."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if key in uploaded(stub):
            return True
        uploader.wait_for_idle(0.1)
        time.sleep(0.01)
    return key in uploaded(stub)


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

    keys = keys_of(client)
    assert f"{PREFIX}/checkpoints/000010/adapter.bin" in keys
    assert keys[-1] == f"{PREFIX}/{MANIFEST_NAME}"
    # The close walk skips the checkpoint folder, so nothing is sent twice.
    assert len(keys) == len(set(keys))
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

    manifest = manifest_of(client)
    assert manifest["schema"] == 1
    assert len(manifest["config_hash"]) == 64
    entry = next(
        row
        for row in manifest["files"]
        if row["path"] == "checkpoints/000010/adapter.bin"
    )
    assert entry["sha256"] == hashlib.sha256(b"x").hexdigest()
    assert entry["size"] == 1
    assert entry["key"] == f"{PREFIX}/checkpoints/000010/adapter.bin"


def test_a_checkpoint_written_after_start_is_uploaded(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path)
    client = StubS3()
    uploader = RunUploader(
        run_dir, "code-score", RUN_NAME, DESTINATION, client=client, interval=0.01
    )
    key = f"{PREFIX}/checkpoints/000011/adapter.bin"
    uploader.start()
    try:
        make_checkpoint(run_dir, step="000011", payload=b"later")
        assert wait_for_key(client, key, uploader)
    finally:
        uploader.close()

    assert key in uploaded(client)
    assert not uploader.errors


def test_a_checkpoint_without_its_manifest_is_not_uploaded(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path)
    folder = run_dir / "checkpoints" / "000010"
    folder.mkdir(parents=True)
    (folder / "adapter.bin").write_bytes(b"half")
    client = StubS3()
    uploader = RunUploader(
        run_dir, "code-score", RUN_NAME, DESTINATION, client=client, interval=0.01
    )
    uploader.start()
    uploader.wait_for_idle()
    uploader.close()

    assert f"{PREFIX}/checkpoints/000010/adapter.bin" not in keys_of(client)
    assert not uploader.errors


def test_a_checkpoint_is_uploaded_after_its_manifest_appears(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path)
    folder = run_dir / "checkpoints" / "000010"
    folder.mkdir(parents=True)
    (folder / "adapter.bin").write_bytes(b"x")
    client = StubS3()
    uploader = RunUploader(
        run_dir, "code-score", RUN_NAME, DESTINATION, client=client, interval=0.01
    )
    key = f"{PREFIX}/checkpoints/000010/adapter.bin"
    uploader.start()
    try:
        uploader.wait_for_idle()
        assert key not in keys_of(client)
        (folder / "checkpoint.json").write_text("{}\n", encoding="utf-8")
        assert wait_for_key(client, key, uploader)
    finally:
        uploader.close()

    assert key in uploaded(client)
    assert not uploader.errors


def test_a_file_that_fails_its_retries_is_uploaded_on_a_later_scan(
    tmp_path: Path,
) -> None:
    run_dir = make_run(tmp_path)
    make_checkpoint(run_dir, payload=b"x")
    client = StubS3(fail_times=4, fail_suffix="adapter.bin")
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
    key = f"{PREFIX}/checkpoints/000010/adapter.bin"
    uploader.start()
    try:
        assert wait_for_key(client, key, uploader)
    finally:
        uploader.close()

    assert key in uploaded(client)
    relative = key[len(PREFIX) + 1 :]
    entry = next(
        row for row in manifest_of(client)["files"] if row["path"] == relative
    )
    assert entry["sha256"] == hashlib.sha256(b"x").hexdigest()
    assert uploader.errors


def test_a_failing_file_does_not_stop_a_later_file(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path)
    make_checkpoint(run_dir, step="000010", payload=b"bad")
    make_checkpoint(run_dir, step="000011", payload=b"good")
    client = StubS3(fail_times=1000, fail_suffix="000010/adapter.bin")
    uploader = RunUploader(
        run_dir,
        "code-score",
        RUN_NAME,
        DESTINATION,
        client=client,
        retries=1,
        interval=0.01,
        backoff=0.01,
    )
    good = f"{PREFIX}/checkpoints/000011/adapter.bin"
    uploader.start()
    try:
        assert wait_for_key(client, good, uploader)
    finally:
        uploader.close()

    assert good in uploaded(client)
    assert f"{PREFIX}/checkpoints/000010/adapter.bin" not in uploaded(client)
    assert any("000010/adapter.bin" in error for error in uploader.errors)


def test_close_uploads_a_checkpoint_written_after_the_last_scan(
    tmp_path: Path,
) -> None:
    run_dir = make_run(tmp_path)
    client = StubS3()
    uploader = RunUploader(
        run_dir, "code-score", RUN_NAME, DESTINATION, client=client, interval=10.0
    )
    uploader.start()
    make_checkpoint(run_dir, step="000012", payload=b"last")
    uploader.close()

    key = f"{PREFIX}/checkpoints/000012/adapter.bin"
    assert key in uploaded(client)
    relative = key[len(PREFIX) + 1 :]
    assert any(row["path"] == relative for row in manifest_of(client)["files"])
    assert not uploader.errors


def test_close_waits_for_an_upload_in_flight_and_writes_manifest_last(
    tmp_path: Path,
) -> None:
    run_dir = make_run(tmp_path)
    make_checkpoint(run_dir)
    client = BlockingStub()
    uploader = RunUploader(
        run_dir, "code-score", RUN_NAME, DESTINATION, client=client, interval=0.01
    )
    uploader.start()
    assert client.started.wait(2.0)
    closer = threading.Thread(target=uploader.close)
    closer.start()
    try:
        time.sleep(0.05)
        assert closer.is_alive()
    finally:
        client.release.set()
        closer.join(5.0)

    assert not closer.is_alive()
    keys = keys_of(client)
    assert f"{PREFIX}/checkpoints/000010/adapter.bin" in keys
    assert keys[-1] == f"{PREFIX}/{MANIFEST_NAME}"
    assert not uploader.errors


def test_close_records_an_error_when_uploads_do_not_finish(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path)
    make_checkpoint(run_dir)
    client = BlockingStub()
    uploader = RunUploader(
        run_dir, "code-score", RUN_NAME, DESTINATION, client=client, interval=0.01
    )
    uploader.start()
    try:
        assert client.started.wait(2.0)
        uploader.close(timeout=0.05)
    finally:
        client.release.set()

    assert any("did not finish in time" in error for error in uploader.errors)


def test_close_writes_the_upload_errors_and_prints_the_count(
    tmp_path: Path, capsys
) -> None:
    run_dir = make_run(tmp_path)
    make_checkpoint(run_dir)
    client = StubS3(fail_times=100)
    uploader = RunUploader(
        run_dir,
        "code-score",
        RUN_NAME,
        DESTINATION,
        client=client,
        retries=1,
        interval=0.01,
        backoff=0.01,
    )
    uploader.start()
    uploader.wait_for_idle()
    uploader.close()

    log = run_dir / "upload-errors.log"
    assert log.is_file()
    text = log.read_text(encoding="utf-8")
    assert "OSError: boom" in text
    err = capsys.readouterr().err
    assert f"uploader: {len(uploader.errors)} upload error(s)" in err
    assert "upload-errors.log" in err


def test_close_writes_no_error_log_without_failures(tmp_path: Path, capsys) -> None:
    run_dir = make_run(tmp_path)
    client = StubS3()
    uploader = RunUploader(
        run_dir, "code-score", RUN_NAME, DESTINATION, client=client, interval=0.01
    )
    uploader.close()

    assert not (run_dir / "upload-errors.log").exists()
    assert "upload error" not in capsys.readouterr().err


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


def test_retries_keep_their_backoff_after_close_starts(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path)
    make_checkpoint(run_dir)
    client = StubS3(fail_times=2, fail_suffix="adapter.bin")
    uploader = RunUploader(
        run_dir,
        "code-score",
        RUN_NAME,
        DESTINATION,
        client=client,
        retries=2,
        interval=10.0,
        backoff=0.1,
    )
    uploader.start()
    deadline = time.monotonic() + 2.0
    while not client.attempt_times("adapter.bin") and time.monotonic() < deadline:
        time.sleep(0.01)
    uploader.close()

    times = client.attempt_times("adapter.bin")
    assert len(times) == 3
    assert times[1] - times[0] >= 0.08
    assert times[2] - times[1] >= 0.16


def test_close_uploads_the_folder_with_no_checkpoints(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path)
    client = StubS3()
    uploader = RunUploader(
        run_dir, "code-score", RUN_NAME, DESTINATION, client=client, interval=0.01
    )
    # No start call: close still uploads the run folder.
    uploader.close()

    keys = keys_of(client)
    assert f"{PREFIX}/meta.json" in keys
    assert f"{PREFIX}/metrics.jsonl" in keys
    assert keys[-1] == f"{PREFIX}/{MANIFEST_NAME}"
    assert not uploader.errors


def test_close_uploads_only_the_run_files_that_exist(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path)
    (run_dir / "eval.json").write_text("{}\n", encoding="utf-8")
    client = StubS3()
    uploader = RunUploader(
        run_dir, "code-score", RUN_NAME, DESTINATION, client=client, interval=0.01
    )
    uploader.close()

    keys = keys_of(client)
    assert set(keys[:-1]) == {
        f"{PREFIX}/meta.json",
        f"{PREFIX}/metrics.jsonl",
        f"{PREFIX}/eval.json",
    }
    assert keys[-1] == f"{PREFIX}/{MANIFEST_NAME}"


def test_close_uploads_every_remaining_run_file(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path)
    for name in ("gpu.jsonl", "rollouts.jsonl", "mlflow.json"):
        (run_dir / name).write_text("{}\n", encoding="utf-8")
    client = StubS3()
    uploader = RunUploader(
        run_dir, "code-score", RUN_NAME, DESTINATION, client=client, interval=0.01
    )
    uploader.close()

    keys = keys_of(client)
    for name in ("gpu.jsonl", "rollouts.jsonl", "mlflow.json"):
        assert f"{PREFIX}/{name}" in keys
    assert keys[-1] == f"{PREFIX}/{MANIFEST_NAME}"
    assert not uploader.errors


def test_close_is_idempotent(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path)
    client = StubS3()
    uploader = RunUploader(
        run_dir, "code-score", RUN_NAME, DESTINATION, client=client, interval=0.01
    )
    uploader.close()
    calls = len(keys_of(client))
    uploader.close()

    assert len(keys_of(client)) == calls


def test_close_marks_itself_closed_after_a_failure(tmp_path: Path, monkeypatch) -> None:
    run_dir = make_run(tmp_path)
    client = StubS3()
    uploader = RunUploader(
        run_dir, "code-score", RUN_NAME, DESTINATION, client=client, interval=0.01
    )

    def boom() -> None:
        raise RuntimeError("manifest boom")

    monkeypatch.setattr(uploader, "_upload_manifest", boom)
    uploader.close()
    calls = len(keys_of(client))
    uploader.close()

    assert any("manifest boom" in error for error in uploader.errors)
    assert len(keys_of(client)) == calls


def test_two_uploaders_share_two_cross_process_slots(tmp_path: Path) -> None:
    first = make_run(tmp_path)
    second = tmp_path / "runs" / "hybrid" / RUN_NAME
    second.mkdir(parents=True)
    for name in ("meta.json", "metrics.jsonl"):
        (second / name).write_bytes((first / name).read_bytes())
    for step in ("000010", "000011", "000012"):
        make_checkpoint(first, step=step)
        make_checkpoint(second, step=step)

    client = CountingStub()
    uploaders = [
        RunUploader(
            first,
            "code-score",
            RUN_NAME,
            DESTINATION,
            client=client,
            concurrency=2,
            interval=0.01,
            retries=0,
        ),
        RunUploader(
            second,
            "hybrid",
            RUN_NAME,
            DESTINATION,
            client=client,
            concurrency=2,
            interval=0.01,
            retries=0,
        ),
    ]
    for uploader in uploaders:
        uploader.start()
    for uploader in uploaders:
        uploader.wait_for_idle(20.0)
    for uploader in uploaders:
        uploader.close()

    assert client.max_active <= 2
    assert client.max_active >= 2
    assert (tmp_path / "runs" / ".upload-slots").is_dir()
    assert all(not uploader.errors for uploader in uploaders)


def test_worker_threads_are_daemons(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path)
    client = StubS3()
    uploader = RunUploader(
        run_dir, "code-score", RUN_NAME, DESTINATION, client=client, interval=0.01
    )
    uploader.start()
    try:
        workers = [
            thread
            for thread in threading.enumerate()
            if thread.name.startswith("rlvr-upload-")
            and thread.name != "rlvr-upload-scan"
        ]
        assert workers
        assert all(thread.daemon for thread in workers)
    finally:
        uploader.close()
