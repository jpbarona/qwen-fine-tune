# 26: Two arms, one hundred steps, from the base model

**What to build:** Two configs that differ only in the pool policy, and a runbook
that takes a fresh box from screen to two finished arms.

**Blocked by:** 24, 25.

**Status:** ready-for-agent

- [ ] A spread arm config and a sweep arm config share every value except the
      policy key.
- [ ] Both run one hundred steps from base weights, group size sixteen, and an
      equal rollout budget.
- [ ] The run metadata records the policy, the screen path and the budget.
- [ ] The runbook covers install, screen, both arms, and the files to pull.
- [ ] The dashboard shows both arms side by side.
- [ ] `docs/DECISIONS.md`, `docs/journal.md` and the docs index record the
      change, including the engine decision.
- [ ] The suite passes.
