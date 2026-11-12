# 04: Code score arm

**What to build:** The code-score arm. Reward is the fraction of tests passed, with
no judge and no gate. This is the dense baseline.

**Blocked by:** 02.

**Status:** ready-for-agent

- [ ] Reward equals passed tests divided by total tests.
- [ ] A rollout that passes all tests gets 1.0.
- [ ] A rollout that passes none gets 0.
- [ ] No judge call is made in this arm.
- [ ] A short GPU run logs a spread of rewards between 0 and 1.
- [ ] Unit tests cover 0, partial, and full pass.
