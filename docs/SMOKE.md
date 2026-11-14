# Smoke test result

Status: passing.

## What was run

Synced the local `main` tree to the Massed Compute instance and ran the smoke arm:

```
./run.sh code-score --config configs/smoke.toml
```

The command exited 0. The log has the summary line and the run folder has all five required artifacts.

## Failures found

None on this run. The smoke passed on the first attempt after syncing the latest local commit.

The prior attempts on the instance failed on two bugs. Both were already fixed in the local tree by the two most recent commits:

- The unsupported `generator` keyword passed to `model.generate`. Fixed by `c96c2f4`.
- `mean_entropy` underflowing to NaN. Fixed by `ab6922e`.

I did not change any code or tests. The synced tree already contained both fixes.

## Verification

The mean entropy stayed finite on every step. Values ranged from 0.42 to 0.55 nats.

The remote suite passes:

```
464 passed in 60.38s
```

Run with `.venv/bin/pytest -q -W error`.

## Final metrics

```
arm=code-score steps=8 rollouts=64
mean_reward=0.0000 pass_at_1=0.0000 stop_reason=max_steps run folder: runs/code-score/20261113T021629Z
```

## Run folder

`/home/Ubuntu/rl-grpo/runs/code-score/20261113T021629Z`

Contents:

- `meta.json`
- `metrics.jsonl`
- `rollouts.jsonl` (64 rollouts)
- `eval.json`
- `checkpoints/` (step-00000004 and step-00000008)

`git_commit` is null in `meta.json` because the instance copy has no `.git` directory. This is the documented fallback in `repro.py`, not a bug.

## Committed and pushed

`ab6922e Guard mean entropy against underflow to NaN` pushed to `origin/main`.

No new commits were needed for this run.
