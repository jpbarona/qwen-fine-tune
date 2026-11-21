# 24: Sweep parked prompts in dedicated batches

**What to build:** The second schedule adapter. Every gap of training steps, the
parked pool is covered in consecutive steps of parked prompts only, at the same
rollout budget per step, with gradients kept.

**Blocked by:** 23.

**Status:** ready-for-agent

- [ ] A config key selects the sweep policy, and the spread policy stays the
      default until one arm is chosen.
- [ ] The sweep length is the parked count divided by the prompts per step,
      rounded up.
- [ ] The gap between sweeps is a ratio of the sweep length, configurable from
      three to ten, defaulting to three.
- [ ] A sweep step uses the same rollout budget as a training step.
- [ ] Labels update at the end of the sweep, with the same promotion bar.
- [ ] Tests cover the sweep length, the gap, a parked pool smaller than one step
      and a resumed run landing mid-sweep.
