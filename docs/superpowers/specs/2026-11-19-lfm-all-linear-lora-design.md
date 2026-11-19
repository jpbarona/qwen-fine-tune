# LFM all-linear LoRA design

2026-11-19. Approved by the operator before any code changed.

## Intent

Train LFM2.5-2.6B with GRPO on MBPP, then compare how far the gain transfers
to other languages against Qwen3-0.6B. The adapter placement and rank decide
whether that comparison is fair and whether the model can learn at all.

## What was decided

- Adapter targets: all linear layers. Attention projections, both MLP matrices
  and the two convolution projections in every LFM block.
- Rank 16, alpha 32.
- Learning-rate probes at 1e-4 and 1e-3, twenty steps each, one per card.
- After the probes, the training run, plus a 60-step Qwen3-0.6B all-linear
  baseline so the comparison holds placement constant.

## Why

- Current code gives LFM adapters only to the four attention projections
  (`model_profiles.py:23-33`, applied at `qwen.py:610`). LFM2 has 8 attention
  blocks and 22 convolution blocks, so most of the network stays frozen.
- Every framework the research subagent checked defaults to all-linear for RL:
  verl, OpenRLHF, TRL. Placement mattered more than rank in the Thinking
  Machines experiments.
- RL needs little adapter capacity. Thinking Machines matched full fine-tuning
  at rank 1 on 8B models, and a 1B-3B GRPO study found rank 16 against 32
  inside noise. Liquid ships rank 16 for LFM2.5 fine-tuning.
- Sizes at all-linear on this model: rank 8 is 12.2M, rank 16 is 24.5M, rank
  32 is 48.9M. Rank 16 is 0.95 percent of the 2.57B model.

## Non-goals

- No rehearsal set. The operator wants the stage pool to empty.
- No full fine-tune.
- No MultiPL-E evaluation in this change.
- No change to Qwen's committed behaviour. Its configs keep attention-only
  unless a config asks otherwise.

## Approaches considered

1. Attention only, both models. Rejected. Below current practice, and freezes
   22 of 30 blocks.
2. All-linear as the only option. Rejected. It would silently change every
   existing Qwen config and invalidate past runs.
3. All-linear behind a config key, default attention. Chosen.

## Constraints

- The held-out split stays pinned in `data/split_seed1234.json`.
- LFM runs use the Hugging Face engine, which needs transformers 5.
- Checkpoints carry a target-set fingerprint. A checkpoint whose placement
  differs from the running config must still be rejected.
- No GPU work in this change. The code must be testable on CPU.

## Left to the builder

- Exact key name and allowed values.
- Test names and fixtures.
- Config file names for the probe runs and the Qwen baseline.
