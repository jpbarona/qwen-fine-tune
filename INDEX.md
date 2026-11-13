# rl-grpo index

Qwen3-0.6B GRPO reward study. Pointers only.

- `journal.md` — running problem and fix chunks, 10 to 12 words each.
- `DECISIONS.md` — one arm, reward shape, curriculum stages.
- `issues/09-spec-single-arm-curriculum.md` — current spec, single arm.
- `SPEC.md` — study design and the four reward arms.
- `PLAN.md` — ticket breakdown.
- `SMOKE.md` — smoke run result.
- `DEPLOY.md` — instance and deploy notes.
- `README.md` — install, run, and test commands.
- `configs/` — `real.toml` and `smoke.toml`.
- `data/mbpp_full.jsonl` — 964 MBPP rows, fixed 714 and 250 split.
- `src/rlvr/` — package source.
- `tests/` — 513 tests, no GPU tests.
- `runs/` — run folders, kept as study data.

The dead GPU instance home is backed up at
`/home/opencode/Work/rl-grpo-instance/home-Ubuntu/`.
