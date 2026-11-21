# 21: Pools built from measured pass counts

**What to build:** A module that turns screen rows into three pools. Train holds
prompts with a partial pass count, rehearsal holds prompts that passed every
sample, parked holds prompts that passed none. The module owns the promotion bar
and the counts, and nothing else in the codebase decides pool membership.

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent

- [ ] Given rows and a sample count, the module returns train, rehearsal and
      parked lists, each sorted deterministically.
- [ ] A row with zero passes is parked, a row with every pass is rehearsal, and
      anything between is train.
- [ ] A row measured with fewer samples than the configured count is rejected
      with an error naming the row.
- [ ] Promotion needs two passes out of sixteen or more.
- [ ] Empty pools are valid and the counts are reported.
- [ ] Tests cover the boundaries, an empty input and a short-sample row.
