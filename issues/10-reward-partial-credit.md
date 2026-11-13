# 10: Reward with partial credit and penalties

**What to build:** The code arm scores rollouts so that a group of failures still
carries a gradient. Partial credit for tests passed, penalties for truncation and
syntax errors, a length penalty, and a clip. The fake run shows the new numbers
and the ordering invariant holds for every length.

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent

- [ ] Truncated scores -0.6, no code or syntax error scores -0.5, compiles but fails every test scores 0.0.
- [ ] Partial tests score passed over total, and all tests passed scores 1.0.
- [ ] Length term is 0.001 per completion token above 900, floored at 0.2, applied to the whole output.
- [ ] The sum is clipped to -1 to 1.
- [ ] The ordering invariant is covered by a test across completion lengths from 0 to 4096, worst correct answer 0.8 beats best wrong answer 0.0.
- [ ] The soft limit and per token rate are config values with defaults, and bad values are rejected.
- [ ] The other three arms keep their current rewards.
- [ ] The fake run writes the new rewards and the whole suite passes.
