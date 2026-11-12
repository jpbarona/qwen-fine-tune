# 01: Fake end-to-end pipeline

**What to build:** A runnable command that executes one arm from start to finish
with fake model, judge, and test-runner backends, and writes the full run folder.
This proves the orchestration seam, config, and record schema before any GPU or
API is involved.

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent

- [ ] `run --arm <code-score|hybrid|gated-judge|gated-process> --fake` completes and exits 0.
- [ ] Run folder contains `meta.json`, `metrics.jsonl`, `rollouts.jsonl`, and `eval.json`.
- [ ] Each rollout record has the agreed fields with correct types.
- [ ] Unknown arm and missing config fail with a clear message.
- [ ] `pytest` passes on CPU with no network and no GPU.
