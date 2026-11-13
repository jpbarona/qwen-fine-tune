# 15: Difficulty labels and the synthetic easy set

**What to build:** Every one of the 714 MBPP training rows carries a difficulty
label, and a set of 50 very easy synthetic questions exists with tests, so the
curriculum has a real first stage and a prior for banding.

**Blocked by:** 13.

**Status:** ready-for-agent

- [ ] One label per training row, in a committed data file, with the row id and the band.
- [ ] Labels are produced in batches of 20 by an agent and stored in the same order as the split.
- [ ] A loader validates the label file against the split and fails on a missing or unknown id.
- [ ] 50 synthetic questions exist in the same schema as MBPP rows, each with a reference solution and at least two tests.
- [ ] Every synthetic reference solution passes its own tests in the Docker runner.
- [ ] Every synthetic question is solvable in under 20 lines of Python and needs no imports beyond the standard library.
- [ ] No model or network call happens at test time.
- [ ] The whole suite passes.
