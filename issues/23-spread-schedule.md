# 23: Spread parked prompts across steps

**What to build:** A schedule seam at the step draw, and the spread policy behind
it. Each step asks the schedule for a plan of train prompts and parked prompts.
Parked rollouts join the same update, and their counts update the labels.

**Blocked by:** 21, 22.

**Status:** ready-for-agent

- [ ] The step draw asks a schedule for its prompts and no longer reads a pool
      directly.
- [ ] The schedule takes a parked slice of `min(cap, ceil(P / refresh))` prompts
      per step, where the cap is a quarter of the rollouts per step divided by
      the group size.
- [ ] Parked rollouts are part of the same gradient update as train rollouts.
- [ ] A step row records the policy name, the train count, the parked count and
      the share of groups with a non-zero advantage.
- [ ] Parked labels update after the step, and a prompt reaching two of sixteen
      is promoted.
- [ ] Ten consecutive full-failure groups demote a train prompt.
- [ ] Tests cover the slice arithmetic, an empty parked pool, a full parked pool
      and the promotion and demotion boundaries.
