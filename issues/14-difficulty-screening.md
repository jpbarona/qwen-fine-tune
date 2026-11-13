# 14: Difficulty screening command

**What to build:** A command that samples the base model on a set of questions and
writes a difficulty table, so the curriculum bands come from measurement rather
than labels. It never trains and never writes a checkpoint.

**Blocked by:** 11.

**Status:** ready-for-agent

- [ ] A command takes a question set, a sample count and an output path, and runs the base model with thinking off at the current cap.
- [ ] It writes one row per question with the question id, samples run, samples fully passed and pass rate.
- [ ] It writes a summary with the pass rate histogram and the count of questions at zero, partial and full pass.
- [ ] It holds no optimizer state, writes no checkpoint and cannot resume a training run.
- [ ] An interrupted screen resumes from the rows already written.
- [ ] A fake mode completes without a GPU and produces a deterministic table.
- [ ] The whole suite passes.
