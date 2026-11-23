# Deploy the MLflow tracking store

This directory holds the systemd units, the backup script and the install
script for the MLflow server on this VM. The server owns the SQLite store at
`/home/opencode/Work/rl-grpo-data/mlflow/mlflow.db` and writes artifacts to
`s3://opencodeagent1/mlflow/v1`. It listens on `127.0.0.1:8081` only. A GPU box
reaches it through an SSH reverse tunnel.

## Files

- `mlflow.service` runs the server from the repo venv as user `opencode`.
- `mlflow.env.example` lists the object storage variables the server reads.
- `mlflow-backup.service` runs `backup.py` once.
- `mlflow-backup.timer` runs that service every day at 03:00 UTC.
- `backup.py` copies the store with the sqlite3 backup API and uploads it.
- `install.sh` installs the units and the store directory.

## Install

First make the environment file. The install script copies the example when
the file is absent, so this step is only needed to set the values early.

```
install -m 0600 deploy/mlflow.env.example /home/opencode/.config/mlflow/env
$EDITOR /home/opencode/.config/mlflow/env
```

Fill `MLFLOW_S3_ENDPOINT_URL`, `AWS_ACCESS_KEY_ID`,
`AWS_SECRET_ACCESS_KEY` and `AWS_DEFAULT_REGION` with the key for the
`opencodeagent1` bucket. The key must allow `mlflow/` reads and writes.

Then run the install script as root.

```
sudo deploy/install.sh
```

Without root the script creates the store directory and the environment file,
then prints the two commands to run as root. It does not print them when it
runs as root.

The script is safe to repeat. It overwrites the three units, keeps the store
and the environment file, and reloads systemd. It does not touch the database.

## Check the server

```
systemctl status mlflow.service
curl -fsS http://127.0.0.1:8081/health
journalctl -u mlflow.service -n 50 --no-pager
```

The Mac reaches the UI through the existing Mac to VM tunnel on port 8083.
Do not change `--host` to a public address.

## Backup

`backup.py` uses `sqlite3.Connection.backup`, so the copy is consistent even
while the server writes. It writes a temporary file, uploads it to
`s3://opencodeagent1/mlflow/v1/backups/mlflow-<date>.db`, then deletes older
objects in that prefix so that 14 remain. The date is UTC.

Run it by hand with:

```
systemctl start mlflow-backup.service
journalctl -u mlflow-backup.service -n 20 --no-pager
```

Or from the repo venv with `boto3` installed:

```
.venv/bin/python deploy/backup.py
```

Options are `--db`, `--bucket`, `--prefix` and `--keep`.

To restore, stop the server, copy the chosen backup over the live database,
and start the server again. The store can also be rebuilt from the run folders
with `rlvr sync-mlflow`, so the backup only saves that work.

## Tests

The packaging tests check that the unit calls the repo venv, that the example
lists the four variables, and that the backup script imports.

```
.venv/bin/pytest tests/test_packaging.py -v
```
