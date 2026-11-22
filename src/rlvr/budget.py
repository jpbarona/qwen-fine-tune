"""One fixed thinking budget for the whole run.

The budget is a config value only. vLLM 0.34 has no per-request logits
processors, so the sampler does not enforce the budget and a rollout is not
ended when it reaches it. There is no per-stage ladder after the curriculum was
removed from the training path.

Pure Python, no torch.
"""

from __future__ import annotations

__all__ = ["DEFAULT_THINKING_BUDGET"]

#: Tokens allowed for the reasoning block, the same on every step.
DEFAULT_THINKING_BUDGET = 1024
