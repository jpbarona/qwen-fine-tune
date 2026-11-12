# 05: Hybrid and gated judge arms

**What to build:** Two arms that gate on all tests passing. Hybrid gives
`0.5 + 0.5 * judge`. Gated judge gives the judge score.

**Blocked by:** 03, 04.

**Status:** ready-for-agent

- [ ] Any test failure gives reward 0 and no judge call.
- [ ] Hybrid on a pass gives `0.5 + 0.5 * judge`.
- [ ] Gated judge on a pass gives the judge score.
- [ ] The judge sees no test code.
- [ ] Unit tests cover pass, fail, and timeout for both arms.
- [ ] A short GPU run completes for both arms with populated scores.
