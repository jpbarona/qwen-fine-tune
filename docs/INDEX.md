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
- `../issues/10` to `../issues/17` — the implemented tickets.
- `../issues/18-reward-and-checkpoint-fixes.md` — the reward and checkpoint fix plan.
- `../README.md` — install, run, and test commands.
- `../configs/` — `real.toml` and `smoke.toml`.
- `../dashboard/` — React and Tailwind source for the live dashboard.
- `../src/rlvr/web/index.html` — the committed single file build.
- `../data/mbpp_full.jsonl` — 964 MBPP rows, fixed 714 and 250 split.
- `../data/difficulty_labels.jsonl` — one band per training row.
- `../data/synthetic_easy.jsonl` — 50 very easy rows for the first stage.
- `../src/rlvr/` — package source.
- `../tests/` — 840 tests, no GPU tests.
- `../runs/` — run folders, kept as study data.

Pulled run data lives outside this repo, in
`/home/opencode/Work/rl-grpo-data/`, one folder per pull.

To update the docs, say "go and update the docs". That means: bring
`DECISIONS.md` and `journal.md` up to date, keep this index accurate, and leave
the code untouched.
