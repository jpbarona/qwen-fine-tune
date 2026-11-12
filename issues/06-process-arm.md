# 06: Gated process arm

**What to build:** The process arm. Any test failure gives 0. On success the reward
is `0.5 * process + 0.5 * judge`, where the process judge reads only the reasoning
trace.

**Blocked by:** 05.

**Status:** ready-for-agent

- [ ] Thinking traces are captured separately from the final code.
- [ ] The process judge receives the trace and problem only.
- [ ] Failing code gets reward 0 and no judge calls.
- [ ] Passing code gets `0.5 * process + 0.5 * judge`.
- [ ] Rollout records carry both scores and the combined reward.
- [ ] A short GPU run completes with all fields populated.
