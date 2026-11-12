# 02: Merged split and Docker test runner

**What to build:** Load MBPP, merge the three real splits into 964 questions, cut
a fixed 714/250 split, and run candidate code against task tests in Docker with
per-test results.

**Blocked by:** 01.

**Status:** ready-for-agent

- [ ] Loader merges train, validation, and test into 964 questions.
- [ ] Split with a fixed seed gives 714 train and 250 test.
- [ ] Every arm reads the same split and prompt format.
- [ ] Candidate code runs in Docker with no network, a non-root user, a read-only mount, and a 5 second timeout.
- [ ] A hung candidate is killed and classified as a timeout.
- [ ] No secret from the parent environment reaches the container.
- [ ] Fixture code produces each expected error class.
