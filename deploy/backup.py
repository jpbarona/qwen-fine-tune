"""Copy the MLflow SQLite store to object storage.

The store is one SQLite file. A plain file copy can catch a write in
progress, so this module uses the sqlite3 backup API to make a consistent
copy first. The copy goes to a temporary file and from there to the bucket.

boto3 is imported inside the functions that need it. That keeps the module
importable for the packaging tests, which run without the tracking extra.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_DB = Path("/home/opencode/Work/rl-grpo-data/mlflow/mlflow.db")
DEFAULT_BUCKET = "opencodeagent1"
DEFAULT_PREFIX = "mlflow/v1/backups"
DEFAULT_KEEP = 14


def backup_key(now: datetime | None = None, prefix: str = DEFAULT_PREFIX) -> str:
    """Return the object key for a backup taken at the given time."""
    moment = now or datetime.now(UTC)
    return f"{prefix}/mlflow-{moment:%Y-%m-%d}.db"


def copy_database(source: Path, target: Path) -> None:
    """Write a consistent copy of the SQLite database at source to target."""
    source_conn = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    try:
        target_conn = sqlite3.connect(target)
        try:
            source_conn.backup(target_conn)
        finally:
            target_conn.close()
    finally:
        source_conn.close()


def _client():
    """Return an S3 client that points at the configured endpoint."""
    import boto3

    return boto3.client(
        "s3",
        endpoint_url=os.environ.get("MLFLOW_S3_ENDPOINT_URL") or None,
    )


def upload(client, local: Path, bucket: str, key: str) -> None:
    """Upload one local file to the bucket under key."""
    client.upload_file(str(local), bucket, key)


def prune(client, bucket: str, prefix: str, keep: int) -> list[str]:
    """Delete old backups so that at most keep objects remain.

    Returns the keys that were deleted.
    """
    keys: list[str] = []
    token = None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": f"{prefix}/"}
        if token:
            kwargs["ContinuationToken"] = token
        response = client.list_objects_v2(**kwargs)
        keys.extend(obj["Key"] for obj in response.get("Contents", []))
        token = response.get("NextContinuationToken")
        if not token:
            break

    keys.sort()
    stale = keys[:-keep] if keep > 0 else []
    for key in stale:
        client.delete_object(Bucket=bucket, Key=key)
    return stale


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Back up the MLflow SQLite store.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="SQLite file to copy")
    parser.add_argument("--bucket", default=DEFAULT_BUCKET, help="destination bucket")
    parser.add_argument("--prefix", default=DEFAULT_PREFIX, help="destination prefix")
    parser.add_argument("--keep", type=int, default=DEFAULT_KEEP, help="backups to keep")
    args = parser.parse_args(argv)

    if not args.db.is_file():
        print(f"no SQLite file at {args.db}", file=sys.stderr)
        return 1

    key = backup_key(prefix=args.prefix)
    try:
        client = _client()
    except ImportError:
        print("boto3 is missing, install the tracking extra", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory(prefix="mlflow-backup-") as tmp:
        local = Path(tmp) / "mlflow.db"
        copy_database(args.db, local)
        upload(client, local, args.bucket, key)

    deleted = prune(client, args.bucket, args.prefix, args.keep)
    print(f"uploaded s3://{args.bucket}/{key}")
    for old in deleted:
        print(f"deleted s3://{args.bucket}/{old}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
