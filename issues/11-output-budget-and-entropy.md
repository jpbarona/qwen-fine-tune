# 11: Output budget and sampled entropy

**What to build:** Thinking and code both fit in one completion, and the per
token logits stop being stored. The cap defaults to 2,048. Entropy is measured
during the update pass every N updates, default 500, and zero disables it.

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent

- [ ] The completion cap is configurable and defaults to 2,048.
- [ ] Generation no longer requests per token scores or keeps logits for every step.
- [ ] Entropy is computed on the update pass, every `entropy_every` updates, default 500.
- [ ] A metric record written on a non measurement step carries no entropy value, and readers treat it as absent.
- [ ] Truncation is still detected and still reported per step.
- [ ] The dashboard renders sparse entropy points without a crash or a gap bug.
- [ ] The fake run at the new cap completes and the whole suite passes.
