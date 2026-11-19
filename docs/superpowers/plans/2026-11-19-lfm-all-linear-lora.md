# LFM all-linear LoRA plan

Design: `docs/superpowers/specs/2026-11-19-lfm-all-linear-lora-design.md`

## Global Constraints

- Existing behaviour stays default. A config without `lora_targets` keeps the
  profile's attention projections, so every committed Qwen config is unchanged.
- Allowed values are exactly `attention` and `all-linear`. Any other value is a
  config error.
- Rank 16, alpha 32 for the new LFM and Qwen baseline configs. Adapter maths:
  all-linear is 24.5M parameters at rank 16.
- LFM runs use `engine = "hf"`. The Qwen baseline uses `engine = "vllm"`.
- Do not touch `data/`, the split file, or the reward code.
- No new dependencies. No GPU work.
- Tests: the suite must pass, currently 958 passed and 4 skipped.
- Doc style: plain English, no semicolons, no em dashes.

## Task 1: Add the lora_targets setting

Goal: a config can choose all-linear adapters without changing any existing
config's behaviour.

Files:

- `src/rlvr/config.py`. Add `lora_targets: str = "attention"` to the training
  config dataclass, next to `lora_rank` and `lora_alpha`. Parse
  `training.lora_targets` with the same style as the other optional keys.
  Reject any value outside `attention` and `all-linear` with a `ConfigError`
  naming the key and the allowed values.
- `src/rlvr/qwen.py`. The `LoraConfig` call currently passes
  `target_modules=list(self.profile.attention_targets)`. Accept the new value
  through the generator constructor and pass `"all-linear"` straight to PEFT
  when the setting is `all-linear`. Follow the value from the config through
  the existing construction path in `src/rlvr/experiment.py` or `src/rlvr/cli.py`.
- Checkpoint validation. Confirm a checkpoint written with attention-only
  adapters is rejected when the running config asks for all-linear, and the
  reverse. If the current check compares target names, it already does this and
  needs only a test.

Tests, in `tests/test_config.py` and `tests/test_qwen.py`:

- default is `attention`
- `all-linear` parses
- an unknown value raises
- the generator builds a PEFT model whose adapter keys include convolution and
  MLP names when the setting is `all-linear`, using the existing fake model
  fixture, not a download
- the checkpoint guard rejects a mismatched target set

Report file: `task-1-report.md`.

## Task 2: Add the run configs and update the docs

Goal: configs the operator can start tonight, and the written record.

Files:

- `configs/lfm.toml`: `lora_rank = 16`, `lora_alpha = 32`,
  `lora_targets = "all-linear"`, `learning_rate = 1e-4`. Update the header
  comment that says attention-only.
- `configs/lfm-probe-1e4.toml` and `configs/lfm-probe-1e3.toml`: copies of
  `configs/lfm.toml` with `[budget] max_steps = 20` and the matching rate.
- `configs/qwen-alllinear-60.toml`: Qwen3-0.6B at revision
  `c1899de289a04d12100db370d81485cdf75e47ca`, `engine = "vllm"`, rank 16,
  alpha 32, `lora_targets = "all-linear"`, `learning_rate = 1e-4`,
  `[budget] max_steps = 60`.
- `docs/DECISIONS.md`: D26 records all-linear placement, rank 16, alpha 32 and
  the evidence, plus the 60-step Qwen baseline.
- `docs/journal.md`: item 98, one line.
- `docs/INDEX.md`: list the design doc, the plan, and the three new configs.
- `README.md`: replace the attention-only sentence in the LoRA section with the
  config key and its two values.

Report file: `task-2-report.md`.

## Task 3: Verify the change

Goal: evidence that the default is unchanged and the new path works on CPU.

Steps:

- `.venv/bin/python -m pytest -q` and report the last line.
- Load each new config through the config loader and print `lora_targets`,
  `lora_rank`, `lora_alpha`, `engine`, `max_steps`.
- Load an existing config (`configs/real.toml`) and confirm `lora_targets` is
  `attention`.
- `git diff --stat` over the change.
- Confirm no file under `data/` changed.

Report file: `task-3-report.md`.
