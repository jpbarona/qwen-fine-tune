# rl-grpo docs index

Qwen3-0.6B GRPO reward study. Documentation lives here. Code and data stay in the repo root.

Docs, all in this folder:

- `DECISIONS.md` — decisions as they are made, newest last.
- `journal.md` — running problem and fix chunks, 10 to 12 words each.
- `SPEC.md` — the deferred four-arm study design.
- `PLAN.md` — the original ticket breakdown.
- `SMOKE.md` — smoke run result.
- `DEPLOY.md` — instance and deploy notes.
- `JUDGE_MODEL.md` — judge model research, local only.

Repo items, relative to this folder:

- `../issues/09-spec-single-arm-curriculum.md` — current spec, single arm.
- `../issues/10` to `../issues/16` — the implemented tickets.
- `../README.md` — install, run, and test commands.
- `../configs/` — `real.toml` and `smoke.toml`.
- `../data/mbpp_full.jsonl` — 964 MBPP rows, fixed 714 and 250 split.
- `../data/difficulty_labels.jsonl` — one band per training row.
- `../data/synthetic_easy.jsonl` — 50 very easy rows for the first stage.
- `../src/rlvr/` — package source.
- `../tests/` — 725 tests, no GPU tests.
- `../runs/` — run folders, kept as study data.

The dead GPU instance data is backed up at
`/home/opencode/Work/rl-grpo-instance/`.

To update the docs, say "go and update the docs". That means: bring
`DECISIONS.md` and `journal.md` up to date, keep this index accurate, and leave
the code untouched.
