"""Copy a run folder and its checkpoints to the bucket.

A training step must never wait on the network. `RunUploader` starts one
background thread that scans `checkpoints/` every few seconds and hands changed
files to a small pool of daemon worker threads. `close` stops the scan, runs
one last scan, drains the queue, uploads every remaining file in the run folder
except `manifest.json` and `checkpoints/`, then writes `manifest.json` last.

Every transfer, checkpoint or run file, holds one of two slots shared through
`flock` files under `<runs root>/.upload-slots/`. Four arms on one box therefore
run at most two uploads at once, whatever their per process concurrency.

A checkpoint file is queued only when its step directory holds
`checkpoint.json`. The trainer writes that manifest last, so a half written
adapter or optimizer file is never uploaded. A file that fails its retries is
left unseen, so the next scan tries it again. A file pruned while it sits in
the queue is skipped without an error.

Every failure is retried with backoff and then recorded in `errors`. On close,
a non-empty `errors` list is written to `upload-errors.log` in the run folder
and the count is printed once to stderr. No upload raises into the caller, so a
bad bucket or a dead network leaves the run folder intact and the run keeps
going.

The bucket destination looks like `s3://bucket/runs/v1`. The run lands under
`s3://bucket/runs/v1/<arm>/<started>/`. The default client is built with boto3,
which is imported only when no client is injected. Tests inject a stub, so
boto3 is not needed to run them.
"""

from __future__ import annotations

import fcntl
import hashlib
import io
import json
import os
import queue
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

__all__ = [
    "MANIFEST_NAME",
    "MANIFEST_SCHEMA",
    "UPLOAD_ERRORS_NAME",
    "RunUploader",
    "upload_prefix",
]

MANIFEST_NAME = "manifest.json"

#: Failure log written in the run folder when an upload gave up.
UPLOAD_ERRORS_NAME = "upload-errors.log"

#: Upload slots shared by every process that writes under one runs root.
SLOT_DIR_NAME = ".upload-slots"
SLOT_COUNT = 2

#: Version of the manifest layout. Bump it when a reader would need new code.
MANIFEST_SCHEMA = 1

CHECKPOINTS_NAME = "checkpoints"

#: Marker written last in a checkpoint directory. A directory without it may
#: still be receiving files, so its contents are not uploaded yet.
CHECKPOINT_MANIFEST = "checkpoint.json"


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
        # The run folder is <runs root>/<arm>/<started>. Four arms share one
        # runs root, so the slots under it cap the whole box, not one process.
        self.slot_dir = self.run_dir.parent.parent / SLOT_DIR_NAME

        self.errors: list[str] = []

        self._client = client if client is not None else _default_client()
        self._lock = threading.Lock()
        self._idle = threading.Condition(self._lock)
        self._active = 0
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._seen: dict[str, tuple[int, int]] = {}
        self._inflight: set[str] = set()
        self._files: dict[str, dict[str, Any]] = {}
        self._work: queue.Queue[tuple[Path, str, tuple[int, int]] | None] = (
            queue.Queue()
        )
        self._workers: list[threading.Thread] = []
        self._thread: threading.Thread | None = None
        self._closed = False

    def start(self) -> None:
        """Start the background scan. Safe to call more than once."""
        if self._closed or self._thread is not None:
            return
        try:
            for index in range(self.concurrency):
                worker = threading.Thread(
                    target=self._worker_loop,
                    name=f"rlvr-upload-{index}",
                    daemon=True,
                )
                worker.start()
                self._workers.append(worker)
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
        """Stop the scan, run one last scan, then upload the folder and manifest."""
        if self._closed:
            return
        try:
            self._close(max(0.0, timeout))
        except Exception as error:
            self._record(f"close: {error}")
        finally:
            self._closed = True
            self._write_errors()

    def _write_errors(self) -> None:
        """Write the failure log and print its count once, when there is one."""
        with self._lock:
            errors = list(self.errors)
        if not errors:
            return
        path = self.run_dir / UPLOAD_ERRORS_NAME
        try:
            path.write_text("\n".join(errors) + "\n", encoding="utf-8")
        except OSError as error:
            print(f"uploader: cannot write {path}: {error}", file=sys.stderr)
        print(
            f"uploader: {len(errors)} upload error(s), see {path}",
            file=sys.stderr,
        )

    def _close(self, timeout: float) -> None:
        self._stop.set()
        deadline = time.monotonic() + timeout
        thread = self._thread
        if thread is not None:
            thread.join(max(0.0, deadline - time.monotonic()))
            if thread.is_alive():
                self._record("close: scan thread did not stop in time")
        if self._workers:
            try:
                self._scan_once()
            except Exception as error:
                self._record(f"scan: {error}")
            if not self._wait_idle(max(0.0, deadline - time.monotonic())):
                self._record("close: uploads did not finish in time")

        # Copy every file the run left behind, not a fixed list. The checkpoint
        # folder is already handled by the scan, and the manifest is written to
        # the bucket last.
        for path in sorted(self.run_dir.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(self.run_dir).as_posix()
            if relative == MANIFEST_NAME:
                continue
            if relative == CHECKPOINTS_NAME or relative.startswith(
                f"{CHECKPOINTS_NAME}/"
            ):
                continue
            self._upload_file(path, relative)

        self._upload_manifest()

        for _ in self._workers:
            self._work.put_nowait(None)

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
            relative = path.relative_to(self.run_dir).as_posix()
            checkpoint = self._checkpoint_dir(relative)
            if checkpoint is None:
                continue
            if not (checkpoint / CHECKPOINT_MANIFEST).is_file():
                continue
            try:
                stat = path.stat()
            except FileNotFoundError:
                continue
            except OSError as error:
                self._record(f"{relative}: {error}")
                continue
            stamp = (stat.st_size, stat.st_mtime_ns)
            with self._lock:
                if self._seen.get(relative) == stamp:
                    continue
                if relative in self._inflight:
                    continue
                self._inflight.add(relative)
            self._submit(path, relative, stamp)

    def _checkpoint_dir(self, relative: str) -> Path | None:
        """The step directory that owns `relative`, or None outside one."""
        parts = Path(relative).parts
        if len(parts) < 3 or parts[0] != CHECKPOINTS_NAME:
            return None
        return self.run_dir / parts[0] / parts[1]

    def _submit(self, path: Path, relative: str, stamp: tuple[int, int]) -> None:
        with self._idle:
            self._active += 1
        try:
            self._work.put_nowait((path, relative, stamp))
        except Exception as error:
            with self._idle:
                self._active -= 1
                self._idle.notify_all()
            with self._lock:
                self._inflight.discard(relative)
            self._record(f"{relative}: {error}")

    def _worker_loop(self) -> None:
        while True:
            item = self._work.get()
            if item is None:
                return
            path, relative, stamp = item
            try:
                self._upload_checkpoint(path, relative, stamp)
            except Exception as error:
                self._record(f"{relative}: {error}")
            finally:
                with self._idle:
                    self._active -= 1
                    self._idle.notify_all()

    def _upload_checkpoint(
        self, path: Path, relative: str, stamp: tuple[int, int]
    ) -> None:
        ok = False
        try:
            ok = self._upload_file(path, relative)
            if ok:
                try:
                    stat = path.stat()
                except OSError:
                    stat = None
                if stat is None or (stat.st_size, stat.st_mtime_ns) != stamp:
                    # The file changed while it was sent. Queue it again.
                    ok = False
        finally:
            with self._lock:
                self._inflight.discard(relative)
                if ok:
                    self._seen[relative] = stamp

    def _upload_file(self, path: Path, relative: str) -> bool:
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            return False
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
                time.sleep(self.backoff * (2 ** (attempt - 1)))
            try:
                with self._upload_slot():
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

    @contextmanager
    def _upload_slot(self) -> Iterator[None]:
        """Hold one of the shared upload slots for the duration of the block.

        The lock is a `flock` on a file under `<runs root>/.upload-slots/`, so
        it holds across processes. A slot frees when the file handle closes,
        and the kernel drops it if the process dies.
        """
        self.slot_dir.mkdir(parents=True, exist_ok=True)
        while True:
            for index in range(SLOT_COUNT):
                handle = open(self.slot_dir / f"slot-{index}", "a")
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError:
                    handle.close()
                    continue
                try:
                    yield
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                    handle.close()
                return
            time.sleep(0.01)

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
