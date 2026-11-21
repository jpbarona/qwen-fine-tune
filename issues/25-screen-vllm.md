# 25: Screen on vLLM at the training group size

**What to build:** The screen samples with vLLM, at sixteen rollouts per prompt
by default, and writes rows the pools can read. A full screen over the training
split is a minutes-long job.

**Blocked by:** 20.

**Status:** ready-for-agent

- [ ] The screen uses the same sampling backend as training.
- [ ] The sample count defaults to sixteen and is overridable.
- [ ] A row records the prompt id, sample count, passes and pass rate.
- [ ] A restart skips rows already written and tolerates a torn last line.
- [ ] A screen over 711 prompts finishes without manual intervention, and the
      summary reports the zero, partial and full counts.
- [ ] Tests cover the row schema, the restart path and the summary counts.
