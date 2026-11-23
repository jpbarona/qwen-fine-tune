"""Copy a run folder and its checkpoints to the bucket.

A training step must never wait on the network. `RunUploader` starts one
background thread that scans `checkpoints/` every few seconds and hands changed
files to a small worker pool. `close` stops the scan, drains the queue, uploads
the small run files, then writes `manifest.json` last.

Every failure is retried with backoff and then recorded in `errors`. No upload
raises into the caller, so a bad bucket or a dead network leaves the run
folder intact and the run keeps going.

The bucket destination looks like `s3://bucket/runs/v1`. The run lands under
`s3://bucket/runs/v1/<arm>/<started>/`. The default client is built with boto3,
which is imported only when no client is injected. Tests inject a stub, so
boto3 is not needed to run them.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

__all__ = ["MANIFEST_NAME", "MANIFEST_SCHEMA", "RunUploader", "upload_prefix"]

MANIFEST_NAME = "manifest.json"

#: Version of the manifest layout. Bump it when a reader would need new code.
MANIFEST_SCHEMA = 1

#: Files uploaded from the run folder root when the uploader closes.
RUN_FILES = (
    "meta.json",
    "metrics.jsonl",
    "eval.json",
    "labels.jsonl",
    "control-events.jsonl",
)

CHECKPOINTS_NAME = "checkpoints"


def upload_prefix(destination: str, arm: str, started: str) -> str:
    """The bucket prefix for one run, with no trailing slash."""
    return f"{destination.rstrip('/')}/{arm}/{started}".rstrip("/")


def _split_destination(destination: str) -> tuple[str, str]:
    """Split `s3://bucket/prefix` into the bucket and the key prefix."""
    scheme = "s3://"
    if not destination.startswith(scheme):
        raise ValueError(f"destination must start with {scheme}: {destination}")
    rest = destination[len(scheme) :]
    bucket, _, prefix = rest.partition("/")
    if not bucket:
        raise ValueError(f"destination has no bucket: {destination}")
    return bucket, prefix.strip("/")


def _default_client() -> Any:
    """Build the boto3 S3 client from the environment.

    Imported here so a caller that injects a stub never needs boto3 installed.
    """
    import boto3

    options: dict[str, str] = {}
    endpoint = os.environ.get("MLFLOW_S3_ENDPOINT_URL") or os.environ.get(
        "AWS_ENDPOINT_URL"
    )
    if endpoint:
        options["endpoint_url"] = endpoint
    region = os.environ.get("AWS_DEFAULT_REGION")
    if region:
        options["region_name"] = region
    return boto3.client("s3", **options)


class RunUploader:
    """Best effort copy of one run folder to the bucket.

    Construct, call `start`, and call `close` when the run ends. If `start` is
    never called, `close` still copies the run files and the manifest.
    """

    def __init__(
        self,
        run_dir: Path,
        arm: str,
        started: str,
        destination: str,
        *,
        client: Any = None,
        concurrency: int = 2,
        interval: float = 10.0,
        retries: int = 3,
        backoff: float = 1.0,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.arm = arm
        self.started = started
        self.destination = destination
        self.concurrency = max(1, int(concurrency))
        self.interval = max(0.0, float(interval))
        self.retries = max(0, int(retries))
        self.backoff = max(0.0, float(backoff))

        self.prefix = upload_prefix(destination, arm, started)
        self.bucket, key_prefix = _split_destination(self.prefix)
        self.key_prefix = key_prefix

        self.errors: list[str] = []

        self._client = client if client is not None else _default_client()
        self._lock = threading.Lock()
        self._idle = threading.Condition(self._lock)
        self._active = 0
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._seen: dict[str, tuple[int, int]] = {}
        self._files: dict[str, dict[str, Any]] = {}
        self._thread: threading.Thread | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._closed = False

    def start(self) -> None:
        """Start the background scan. Safe to call more than once."""
        if self._closed or self._thread is not None:
            return
        try:
            self._executor = ThreadPoolExecutor(
                max_workers=self.concurrency, thread_name_prefix="rlvr-upload"
            )
            self._thread = threading.Thread(
                target=self._scan_loop, name="rlvr-upload-scan", daemon=True
            )
            self._thread.start()
        except Exception as error:
            self._record(f"start: {error}")

    def wait_for_idle(self, timeout: float = 10.0) -> None:
        """Wait until the first scan and every queued file has finished."""
        if self._thread is None:
            return
        deadline = time.monotonic() + max(0.0, timeout)
        remaining = deadline - time.monotonic()
        if not self._ready.wait(remaining):
            return
        remaining = max(0.0, deadline - time.monotonic())
        self._wait_idle(remaining)

    def close(self, timeout: float = 60.0) -> None:
        """Stop the scan, drain the queue, upload the run files and manifest."""
        if self._closed:
            return
        self._closed = True
        try:
            self._close(max(0.0, timeout))
        except Exception as error:
            self._record(f"close: {error}")

    def _close(self, timeout: float) -> None:
        self._stop.set()
        deadline = time.monotonic() + timeout
        thread = self._thread
        if thread is not None:
            thread.join(max(0.0, deadline - time.monotonic()))
            if thread.is_alive():
                self._record("close: scan thread did not stop in time")
        if self._thread is not None:
            self._wait_idle(max(0.0, deadline - time.monotonic()))

        for name in RUN_FILES:
            path = self.run_dir / name
            if path.is_file():
                self._upload_file(path, name)

        self._upload_manifest()

        executor = self._executor
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)

    def _scan_loop(self) -> None:
        try:
            self._scan_once()
        except Exception as error:
            self._record(f"scan: {error}")
        finally:
            self._ready.set()
        while not self._stop.wait(max(self.interval, 0.01)):
            try:
                self._scan_once()
            except Exception as error:
                self._record(f"scan: {error}")

    def _scan_once(self) -> None:
        root = self.run_dir / CHECKPOINTS_NAME
        if not root.is_dir():
            return
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            try:
                stat = path.stat()
            except OSError as error:
                self._record(f"{path}: {error}")
                continue
            stamp = (stat.st_size, stat.st_mtime_ns)
            relative = path.relative_to(self.run_dir).as_posix()
            if self._seen.get(relative) == stamp:
                continue
            self._seen[relative] = stamp
            self._submit(path, relative)

    def _submit(self, path: Path, relative: str) -> None:
        executor = self._executor
        if executor is None:
            return
        with self._idle:
            self._active += 1
        try:
            executor.submit(self._job, path, relative)
        except Exception as error:
            with self._idle:
                self._active -= 1
                self._idle.notify_all()
            self._record(f"{relative}: {error}")

    def _job(self, path: Path, relative: str) -> None:
        try:
            self._upload_file(path, relative)
        finally:
            with self._idle:
                self._active -= 1
                self._idle.notify_all()

    def _upload_file(self, path: Path, relative: str) -> bool:
        try:
            data = path.read_bytes()
        except OSError as error:
            self._record(f"{relative}: {error}")
            return False
        key = f"{self.key_prefix}/{relative}"
        if not self._put_bytes(key, data):
            return False
        entry = {
            "path": relative,
            "key": key,
            "size": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
        with self._lock:
            self._files[relative] = entry
        return True

    def _put_bytes(self, key: str, data: bytes) -> bool:
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            if attempt:
                self._stop.wait(self.backoff * (2 ** (attempt - 1)))
            try:
                self._client.put_object(
                    Bucket=self.bucket, Key=key, Body=io.BytesIO(data)
                )
            except Exception as error:
                last = error
                continue
            return True
        detail = f"{type(last).__name__}: {last}" if last is not None else "no attempt"
        self._record(f"{key}: {detail}")
        return False

    def _upload_manifest(self) -> None:
        with self._lock:
            files = [dict(row) for _, row in sorted(self._files.items())]
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "arm": self.arm,
            "started": self.started,
            "prefix": self.prefix,
            "config_hash": self._config_hash(),
            "files": files,
        }
        text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        key = f"{self.key_prefix}/{MANIFEST_NAME}"
        self._put_bytes(key, text.encode("utf-8"))

    def _config_hash(self) -> str | None:
        path = self.run_dir / "meta.json"
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(meta, dict):
            return None
        config = meta.get("config")
        if config is None:
            return None
        material = json.dumps(
            config, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def _wait_idle(self, timeout: float) -> bool:
        with self._idle:
            return self._idle.wait_for(lambda: self._active == 0, timeout)

    def _record(self, message: str) -> None:
        with self._lock:
            self.errors.append(message)
