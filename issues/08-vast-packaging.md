# 08: Vast packaging and full runs

**What to build:** A pinned environment, one-command launch, checkpoint resume, and
budget guards, so all four arms run on a fresh 3090 without babysitting.

**Blocked by:** 06, 07.

**Status:** ready-for-agent

- [ ] Pinned requirements install cleanly on a CUDA 12 image.
- [ ] `run.sh <arm>` starts a run with no manual edits.
- [ ] Resume continues from the last checkpoint after a restart.
- [ ] Budget guard stops at the configured step or judge-call limit.
- [ ] `meta.json` records package versions, model revision, judge model, split seed, and commit.
- [ ] The smoke profile finishes in under 15 minutes on a 3090.
- [ ] One full arm finishes without manual intervention.
- [ ] The report for a finished arm is reachable at a GitHub Pages URL.
