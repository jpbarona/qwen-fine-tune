# 07: Instrumentation, evaluation, summary, report

**What to build:** Entropy and length on every rollout, checkpoint evaluation every
100 steps, final pass@1 on 250 test questions, a summary command, and a static
report published to GitHub Pages.

**Blocked by:** 03.

**Status:** ready-for-agent

- [ ] Every rollout records token entropy and completion length.
- [ ] Evaluation runs every 100 steps on 100 fixed test questions with greedy decoding.
- [ ] Final evaluation reports pass@1 on all 250 test questions.
- [ ] Records are append-only JSONL with a stable schema.
- [ ] `summary <run-dir>` prints reward, pass@1, entropy, and length per step.
- [ ] `summary` writes a static HTML report with one section per arm.
- [ ] A publish command pushes the report to a GitHub Pages branch.
- [ ] Metrics functions are unit tested on synthetic records.
