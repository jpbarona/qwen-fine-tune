# rl-grpo docs index

Qwen3-0.6B and LFM2.5-2.6B GRPO reward study. Documentation lives here. Code and data stay in the repo root.

Docs, all in this folder:

- [postmortem-run17.md](postmortem-run17.md) — why the 313 step curve looked flat, and what the model actually did.
- [postmortem-run17.pdf](postmortem-run17.pdf) — the same report with three charts, three pages.
- [6.1-sol-xhigh-pass.md](6.1-sol-xhigh-pass.md) — repository organization, LFM support, throughput changes, assumptions, and validation.
- `DECISIONS.md` — decisions as they are made, newest last.
- [superpowers/specs/2026-11-19-lfm-all-linear-lora-design.md](superpowers/specs/2026-11-19-lfm-all-linear-lora-design.md) — the approved design for all-linear LoRA on LFM.
- [superpowers/plans/2026-11-19-lfm-all-linear-lora.md](superpowers/plans/2026-11-19-lfm-all-linear-lora.md) — the task breakdown for that change.
- [../issues/19-parked-pools-spec.md](../issues/19-parked-pools-spec.md) — spec for measured pools and the two parked re-screen policies.
- [superpowers/plans/2026-11-21-parked-pools.md](superpowers/plans/2026-11-21-parked-pools.md) — task breakdown with the binding values.
- `../issues/20` to `../issues/26` — the seven tickets for that change.
- [superpowers/plans/2026-11-21-parked-pools.md](superpowers/plans/2026-11-21-parked-pools.md): the task breakdown for measured pools and the two parked re-screen arms.
- [IDEAS.md](IDEAS.md): one or two chunk proposals, newest first, with dates. The long version is in `/home/opencode/Work/Garbage/ideas-old.md`.
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
- `../issues/19-parked-pools-spec.md`: the measured pool spec and its two re-screen policies.
- `../issues/20` to `../issues/26`: the tickets for the vLLM-only engine, pool build, arm configs and runbook.
- `../README.md`: install, run, and test commands, including the two parked-pool arms and `rlvr eval-band`.
- `../configs/`: `real.toml`, `smoke.toml`, `lfm.toml`, `lfm-smoke.toml`, the two parked-pool arms `lfm-spread.toml` and `lfm-sweep.toml`, the two twenty step Qwen probes `probe-1e4.toml` and `probe-1e3.toml`, the two LFM probes `lfm-probe-1e4.toml` and `lfm-probe-1e3.toml`, and the sixty step Qwen baseline `qwen-alllinear-60.toml`.
- `../dashboard/` — React and Tailwind source for the live dashboard.
- `../src/rlvr/web/index.html` — the committed single file build.
- `../data/mbpp_full.jsonl` — 961 MBPP rows, pinned split of 711 train and 250 test, with `../data/split_seed1234.json` holding the ids.
- `../data/difficulty_labels.jsonl` — one band per training row.
- `../data/synthetic_easy.jsonl` — 50 very easy rows for the first stage.
- `../src/rlvr/` — package source.
- `../tests/` — CPU regression tests and optional model-runtime and Docker integration checks.
- `../runs/` — run folders, kept as study data.

Pulled run data lives outside this repo, in
`/home/opencode/Work/rl-grpo-data/`, one folder per pull.

To update the docs, say "go and update the docs". That means: bring
`DECISIONS.md` and `journal.md` up to date, keep this index accurate, and leave
the code untouched.
