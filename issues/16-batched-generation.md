# 16: Batched generation to saturate the GPU

**What to build:** A step generates several prompts at once, with 8 samples each,
so the card is busy instead of waiting on one Python thread. Batch size grows
until memory is nearly full, and the backward pass runs per generation so the
larger batch fits.

**Blocked by:** 11, 12.

**Status:** ready-for-agent

- [ ] One step generates at least two prompts, each with its own group of 8 samples.
- [ ] Advantage is normalised inside each prompt group, never across the whole batch.
- [ ] The backward pass runs per generation and gradients accumulate across the group.
- [ ] Batch size starts at 2 prompts and grows while free memory stays above the floor, then stays at the largest size that fit.
- [ ] The chosen prompt count and sequence count are recorded per step, and the dashboard shows them.
- [ ] Resume restores correctly when it lands on a batch boundary.
- [ ] At the 2,048 cap, a measured step shows lower wall clock per sequence than the unbatched path.
- [ ] The whole suite passes.
