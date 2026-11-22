# rlvr

A study of whether an LLM judge can train a small code model as well as unit
tests. See `docs/SPEC.md` for the study design and `docs/PLAN.md` for the task breakdown.
`docs/journal.md` holds the running problem and fix notes. `docs/DECISIONS.md` records
the current choices.

## Install

One environment installs the trainer and the vLLM sampler:

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[vllm]" "pytest>=9,<10"
```

The `vllm` extra pins vLLM 0.34.0, torch 2.14.0, Transformers 5.22.2, PEFT
0.18.1 and Accelerate 1.8.1. It needs a CUDA 12 host whose driver supports the
torch wheel. Every rollout in training and in the screen samples with vLLM.

## LFM2.5-2.6B

LFM uses the same environment, GRPO trainer, reward arms, measured pool,
evaluation, controls, and checkpoint workflow. Its model profile selects the
tokenizer behavior and attention projection names. vLLM must be 0.17 or newer
for LoRA over the fused MLP and short convolution, and the extra pins 0.34.0.
`scripts/smoke_lfm_vllm.py` serves the model, loads an adapter, samples, and
swaps in a second adapter without training.

Use separate run roots for the smoke and full runs if both need to be resumed.
Add `--resume` with the original config and run root to continue a checkpoint.
Model identity, revision, split seed, and fake/real mode are checked against
saved metadata before run records are truncated.

The LFM profile pins `LiquidAI/LFM2.5-2.6B` at revision
`7141b56cd4650af95c7c7b86983443aed050c96a`.
The native LFM template always opens a reasoning block, including for screening
and evaluation. The comparison starts from Qwen's hyperparameters. The
`[training] lora_targets` key chooses the adapter placement: `attention` adapts
the four attention projections, and `all-linear` adapts every linear layer.
`configs/lfm.toml` and `configs/qwen-alllinear-60.toml` use `all-linear`.
`configs/lfm-smoke.toml` stays `attention`. Full-weight GPU performance still
needs measurement.
The assumptions, implementation details, and validation are recorded in
[the pass notes](docs/6.1-sol-xhigh-pass.md).

## Development and throughput

```bash
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest -q
cd dashboard
npm ci
npm run build
```

The dashboard build embeds JavaScript and CSS in `src/rlvr/web/index.html`.
Commit this generated file with frontend changes. Pull request checks run the
CPU suite and rebuild the dashboard to detect stale embedded assets.

Training can run up to `training.test_workers` isolated test containers at once
(default 4). `training.prefetch_scoring` defaults to true and scores the next
already-generated group while the current group runs its backward pass.
`training.eval_batch_size` defaults to 8 and shrinks on CUDA memory exhaustion.
The training thread owns model calls and optimizer updates, with one optimizer
step per generated prompt batch.

Metrics record generation time, scoring time, time waiting for scores, update
time, step time, and completion tokens per second. Scoring and update times can
overlap. Step time covers generation through the optimizer update. Evaluation
records its own duration. Use these together with the dashboard's GPU samples
when comparing runs on the same GPU and profile.

The source map and GPU validation procedure are in
[the pass notes](docs/6.1-sol-xhigh-pass.md).

## Run the fake pipeline

The fake path needs no GPU, no network, no Docker, and no model download. It is
deterministic: the same command produces the same rollout records. The CLI
still reads the committed MBPP split and the synthetic easy file, so the run
trains on the 711 MBPP train questions plus the 50 synthetic easy questions, and
evaluates the 250 held-out test questions. Only the model, judge, and Docker
runner are replaced by fakes.

```bash
.venv/bin/rlvr run --arm code-score --fake
```

Arms: `code-score`, `hybrid`, `gated-judge`, `gated-process`. All four are
implemented on the fake path. The process arm judges the reasoning trace and
the code separately, and only on a full test pass.
Flags: `--config PATH` (default: the packaged config at
`src/rlvr/resources/default_config.toml`) and `--runs-root PATH` (default
`./runs`). Unknown arms, a missing config, and an invalid config print one
message on stderr and exit 2.

## Difficulty screening

The screen samples the base model on a question set and writes a difficulty
table. It never trains, builds no optimizer, writes no checkpoint, and writes
into its own output folder, not a run folder.

```bash
.venv/bin/rlvr screen \
    --questions data/train_split.jsonl \
    --output screens/mbpp-base
```

`--questions` is a JSONL file in the MBPP row schema, so the synthetic easy set
and the MBPP rows both work. `--samples` is the number of model samples per
question. It defaults to sixteen, which is the group size of the two parked-pool
arms. The screen asks for thinking off, at the completion cap from
`[training] max_completion_tokens`, because training uses that cap. Qwen honors
the toggle. The LFM template always opens a reasoning block, so the toggle is
ignored for LFM and the screen samples thinking at the same completion cap.

A screen for a parked-pool arm must cover exactly the 711 training questions.
The arm runbook below writes that question file. A screen over the full 961 row
file also measures the held-out questions, which the pool builder rejects.

The output folder holds two files:

```
rows.jsonl      one row per question: prompt_id, samples, passed, pass_rate
summary.json    the pass rate histogram from 0.0 to 1.0, and the
                zero_pass, partial_pass, and full_pass question counts
```

A sample counts as passed only when every task test passed. A rerun reads the
rows already written and skips those question ids, so an interrupted screen
continues where it stopped. A final row without a newline is rewritten with one,
and a torn final line is dropped in that rewrite, before the restart appends.
Rows must keep the same sample count and belong to the same question set, so two
screens cannot be mixed by accident.

`--fake` replaces the model and Docker with the deterministic fakes and needs no
GPU, network, or Docker. It exercises the same writer, summary, and resume path.

## The two parked-pool arms

The study compares two ways to re-screen parked prompts at equal compute.
`configs/lfm-spread.toml` joins a parked slice to every step.
`configs/lfm-sweep.toml` covers the parked pool in consecutive steps, then
trains for a gap. The two configs are identical except the policy keys. Both
run 100 steps from base weights at group size 16, a 304 sequence rollout budget,
rate 3e-5, all-linear LoRA rank 16 and alpha 32, a 1,024 token completion cap,
and a recorded but unenforced thinking budget of 1,024.

Install with the one vLLM environment in the Install section. The screen uses
the same environment.

First the screen input. `data/train_split.jsonl` already holds the 711 training
questions of the pinned split, in the MBPP row schema, so no generation step is
needed. The arm config reads `screens/mbpp-base/rows.jsonl`, so the screen output
folder must be `screens/mbpp-base`.

Evaluations run every 5 steps and checkpoints every 10. A resumed run evaluates
the checkpoint it loads before the first new step.

Then screen the base model. Leave out `--samples` so the default sixteen is
used. The command resumes if it is interrupted.

```bash
.venv/bin/rlvr screen \
    --questions screens/mbpp-train.jsonl \
    --output screens/mbpp-base \
    --config configs/lfm-spread.toml
```

The summary must report 711 questions, and `zero_pass`, `partial_pass` and
`full_pass` must add up to 711. The zero and full counts are the parked and
rehearsal pools. The partial count is the train pool. `samples_per_question`
must be 16, the same as `[training] generations_per_prompt`. A run stops at load
if they differ, so an old eight sample folder cannot be reused by accident.

Run the arms one at a time. Use a separate run root for each so `--resume`
cannot pick the other arm's folder.

```bash
OPENROUTER_API_KEY=... ./run.sh code-score \
    --config configs/lfm-spread.toml --runs-root runs-spread
OPENROUTER_API_KEY=... ./run.sh code-score \
    --config configs/lfm-sweep.toml --runs-root runs-sweep
```

After a crash, resume with the matching config and run root:

```bash
OPENROUTER_API_KEY=... ./run.sh code-score \
    --config configs/lfm-sweep.toml --runs-root runs-sweep --resume
```

Pull these files back. Keep the screen folder, because the run metadata records
its path.

```
screens/mbpp-base/rows.jsonl
screens/mbpp-base/summary.json
runs-spread/code-score/<timestamp>/meta.json
runs-spread/code-score/<timestamp>/metrics.jsonl
runs-spread/code-score/<timestamp>/rollouts.jsonl
runs-spread/code-score/<timestamp>/labels.jsonl
runs-spread/code-score/<timestamp>/eval.json
runs-spread/code-score/<timestamp>/checkpoints/
runs-sweep/code-score/<timestamp>/  the same files
```

Pull them back before you destroy the box. The run folders exist only on that
machine until they are copied.

```bash
rsync -av <host>:<repo>/screens/mbpp-base screens/mbpp-base
rsync -av <host>:<repo>/runs-spread/ runs-spread/
rsync -av <host>:<repo>/runs-sweep/ runs-sweep/
```

To see both runs side by side, pass the two run folders to `summary`, or point
the dashboard at the two arm folders:

```bash
.venv/bin/rlvr summary runs-spread/code-score/<timestamp> \
    runs-sweep/code-score/<timestamp> --html report.html
.venv/bin/rlvr dashboard runs-spread/code-score runs-sweep/code-score
```

The four measurements are read from those files:

1. Non-zero advantage share. Average `advantage_share` over the `metrics.jsonl`
   rows. A row counts groups whose rewards hold more than one distinct value.
2. Mean absolute advantage. Group `rollouts.jsonl` by step and prompt id, apply
   the training normalisation `(reward - group mean) / group std`, and average
   the absolute values. A group with zero std contributes zero.
3. Promotions per thousand parked rollouts. Count `labels.jsonl` rows with
   `reason = "promoted"`. Divide by the parked rollouts, which is the sum of
   `parked_prompts * 16` over `metrics.jsonl`, and multiply by 1,000.
4. Pass@1 on the pinned 100. Read the checkpoint `pass_at_1` values from
   `eval.json`. The base score on the same cohort is 0.51.

A sweep run spends part of its 100 step budget on parked steps. Its train
passes are therefore fewer than a spread run at the same step count, even though
both spend the same share of rollouts on parked work.

## Real training

One command per arm, no manual edits:

```bash
OPENROUTER_API_KEY=... ./run.sh code-score
```

`run.sh` reads the judge key from the environment, passes every extra argument
to `rlvr run`, and defaults to `configs/real.toml`. The four arms are
`code-score`, `hybrid`, `gated-judge`, `gated-process`. The real path trains in
`src/rlvr/qwen.py` and samples with vLLM through `src/rlvr/vllm_backend.py`,
behind the same `Experiment` seam. The Docker test runner and the judge client
sit behind their own seams.

A smoke run validates a fresh GPU in minutes:

```bash
OPENROUTER_API_KEY=... ./run.sh code-score --config configs/smoke.toml
```

The smoke profile runs 8 training steps. A step starts with two prompt groups
and grows while free GPU memory stays above the floor, so it runs at least 128
generations. It also runs two checkpoint evaluations on 4 questions and a final
evaluation on 8 questions. Expected runtime is roughly 5 to 10 minutes,
dominated by the Docker test runs. This estimate is unverified: there is no GPU
in this environment.

Without a pool policy, each step takes a batch of prompts, one group of the
configured group size per prompt, all sampled in one padded call. The batch
starts at two prompts. After each step the free memory is read. While it stays
at or above `[training] memory_floor_mib` the
next step takes one more prompt. The first step below the floor settles the size
at the largest batch that fit, and the run keeps that size.
`[training] max_prompts` caps the prompt count per step when memory would allow
more. An absent key means no cap. The cap limits sequences per step. A run with
a pool policy does not use this growth path. Its step size is the rollout budget
divided by the group size, which is 19 prompts in the two arm configs. Step time
is unverified, because test containers and judge calls run outside the batch.
Gradients accumulate
across the prompt groups and the optimizer steps once at the end of the step, so
the step count is the number of weight updates. A group is normalised on its own
rewards, never across the batch.

## Deploy on Massed Compute

Rent one RTX A6000: 48GB VRAM, 96GB system RAM, 300GB disk, $0.57 per hour. Use the
Ubuntu Server 24.04 image and set `OPENROUTER_API_KEY` at deploy time.

After the instance boots, install Docker, then run the install, smoke, and full
arm commands from the sections above:

```bash
sudo apt install -y docker.io
OPENROUTER_API_KEY=... ./run.sh code-score --config configs/smoke.toml
OPENROUTER_API_KEY=... ./run.sh code-score
```

## Checkpoints and resume

A checkpoint is saved every `[run] checkpoint_every_steps` steps (default 20). `configs/real.toml` sets 10. Fifteen checkpoints are kept per run.
Each checkpoint directory holds the LoRA adapter, the optimizer state, the step
number, and a `checkpoint.json` manifest. After a crash, resume from the last
checkpoint:

```bash
OPENROUTER_API_KEY=... ./run.sh code-score --resume
```

Resume opens the latest run folder for the arm, restores the model, optimizer,
and step, truncates any records past the checkpoint, and continues appending to
the same folder. Records are never duplicated.

## Budget guards

`[budget] max_steps` and `[budget] max_judge_calls` cap the run. The loop stops
cleanly at the cap, writes the final evaluation, and records the reason
(`completed`, `max_steps`, `max_judge_calls`, or `out_of_memory`) in
`eval.json`.

## Tests

```bash
.venv/bin/pytest
```

CPU only. No network, no GPU. Docker is used by two files and those tests skip
when the image is missing. The suite runs in about 35 seconds.

`tests/test_environment_solvable.py` is the cold check on the environment: it
runs every synthetic reference solution through the real container harness, so a
prompt that names the wrong function fails here instead of during a run.

## Run folder layout

```
runs/<arm>/<timestamp>/
    meta.json        arm, timestamp, git commit, versions, model revision,
                     judge model, split seed, the full config
    rollouts.jsonl   one record per rollout, schema below
    metrics.jsonl    one row per step: mean reward, pass rate, completion
                     length, cache hits, the GPU memory split, and entropy
                     on a measurement step
    eval.json        final pass@1 plus every checkpoint evaluation
    checkpoints/     one directory per saved step: LoRA adapter, optimizer
                     state, and a checkpoint.json manifest
```

`<timestamp>` is UTC as `YYYYMMDDTHHMMSSZ`.

## Evaluation

Every `[run] eval_every_steps` steps (default 100) the run evaluates a fixed
subset of the test pool with greedy decoding. The subset is
`[run] eval_questions` questions (default 100), chosen once per run by
`[run] eval_subset_seed` (default 2026). That seed is separate from the data
split seed, and the subset is a seeded sample, not the first questions in file
order. The subset keeps the pool order.

### Scoring one difficulty band

`eval-band` scores a fixed number of prompts from one labelled band, so a
checkpoint can be compared per band without a full run.

```bash
.venv/bin/rlvr eval-band --config configs/lfm.toml --band medium \
    --count 100 --seed 2026 --checkpoint runs/lfm/20261119T101500Z/checkpoints/step-000020
```

The bands are `very easy`, `medium`, `hard`, and `very hard`. The prompts come
from the training split, which carries the labels. Once training uses those
prompts they are no longer held out, so a band score after training is a
training score for that band, not a generalization score. The command prints
one JSON line with `model`, `band`, `pass_at_1`, `count`, `passed`, and
`seconds`. `--out PATH` writes the same line to a file. Without `--checkpoint`
it scores the base model. A band that holds fewer prompts than `--count` stops
with an error rather than scoring a smaller set.

## Live dashboard

The dashboard reads a run folder and shows reward, pass rate, pass@1, completion
length, truncation, and memory. Two things worth knowing:

- the system tiles report GPU memory in GiB,
- the stage completion panel plots how much of the current band has been solved.
  A prompt counts as solved only when all 8 of its rollouts pass, and the
  denominator is the band's starting pool from `meta.json`, so the line shows
  progress that does not move backwards.

A pooled run names its re-screen policy in the run header and adds a pool
composition panel. The upper chart plots the train prompts and the parked
prompts per step. The lower chart plots the advantage share from 0 to 1. All
three series come from the step rows, so an older run without them shows the
panel empty rather than an error.

The questions tab lists every prompt the run sampled, one row each: steps,
rollouts, full-pass steps, per-rollout pass rate, best result, last error, last
step, and solved. It streams `rollouts.jsonl` once and keeps counters per
prompt. Prompts with no full pass come first and stay red. The sort and stage
and solved filters run in the page. Prompt text is read from
`data/mbpp_full.jsonl` and `data/synthetic_easy.jsonl` when those files are
present, and the tab still works from prompt ids when they are missing.

Dropped prompts are hidden by default. The tab says how many were dropped and
offers a control to show them, with the step they were dropped.

GPU utilisation is `nvidia-smi`'s figure: the share of the sample window in
which any kernel was running. It reads 70 to 85 percent on a step of 16
sequences and on a step of 152, because one kernel runs almost continuously
either way. It says busy, not saturated. Throughput, not utilisation, is the
number that moves with batch size, and it is not logged yet.

Status comes from the process table, not only from file age. A live
`rlvr run --arm <arm>` keeps the run `running` through a long evaluation. A run
whose process is gone turns `stopped` after a two minute grace. A GPU reading
that cannot be taken renders as `-`, never as zero, and the strip carries
`last sample Ns ago` so a stalled sampler is visible. Progress counts prompt
samples against the plan rather than steps against a starting batch size.

The page is a React and Tailwind app in `dashboard/`. The build is committed as
`src/rlvr/web/index.html`, a single file with the bundle inlined, so serving it
needs no network. Rebuild after changing the UI:

```bash
cd dashboard && npm install && npm run build
```

## Measuring a short run

Two optional keys make a short experiment cheap. `[budget] max_steps` caps a
real run, and `training.eval_every_steps` sets the evaluation cadence, so a
twenty step probe is a config copy with a changed learning rate and
`max_steps = 20`. `configs/probe-1e4.toml` and `configs/probe-1e3.toml` are
that copy.

Every step row carries two norm fields. `grad_norm` is the L2 norm of the
trainable gradients, taken after the accumulated gradient is divided by the
group count and before the optimiser step. `update_norm` is the L2 norm of the
parameter delta that the step produced. Both are null when the value is not
finite.

Read `update_norm` when comparing learning rates. AdamW normalises the step by
the gradient history, so `grad_norm` stays close across rates and only shows a
collapse. `update_norm` tracks how far the weights actually moved.

## Editing a run in place

Each run folder holds `control.json`. Edit it while training runs and the change
applies at the next step, no restart:

```json
{"drop": ["mbpp-679", "mbpp-536"],
 "tests": {"mbpp-535": ["assert topbottom_surfacearea(10) == 314.15000000000003"]}}
```

- `drop` removes a prompt from the train pool or the parked pool. Rollouts
  already recorded stay in the history.
- `tests` replaces the hidden tests for one prompt from the next step.
- A bad key or a test that does not compile is rejected. The previous state
  stays in force, and the error lands in `control-events.jsonl`.

`--control PATH` uses a different file. The event log records the step of every
change, so a prompt edited mid-run can be split out of an analysis.

A resume must leave the adapter trainable. `PeftModel.from_pretrained` defaults
to a frozen adapter, which then builds an optimizer over no parameters and
fails with `optimizer got an empty parameter list`. The loader passes
`is_trainable=True`.

## One measured pool

Training runs from one pool built from a screen. Each training prompt is
measured at sixteen rollouts. A prompt that passes some but not all of them goes
to train. A prompt that passes all sixteen goes to rehearsal and is not trained.
A prompt that passes none goes to parked. There are no curriculum stages.

The labels come from screens. Training rollouts may demote a train prompt after
ten consecutive full-failure groups. They never promote a prompt, because a
prompt selected for a low pass count regresses upward on the next measurement.
A parked prompt is promoted at two passes of sixteen or more.

Two schedules put parked prompts back into training. The spread policy joins a
slice to every step. The sweep policy covers the whole parked pool in
consecutive steps, then trains for a gap. Both use the same rollout budget per
step. The sweep spends about a quarter of its steps on parked work. The spread
slice is capped at a quarter of the step, and it reaches that cap only when the
parked pool holds 321 prompts or more. Below that the two arms are not equal
compute, and the run prints a warning with the parked count. The arm runbook
above runs one arm per policy.

There is one pool and no stage ladder. Every record keeps the `stage` field at
`unstaged` so older readers still find it. Older runs also wrote a per-stage
evaluation entry with `kind` `"stage"` and a band name. New runs have no stage,
so `eval.json` holds only checkpoint and final entries.

The final evaluation always covers the whole test pool, which is the 250
held-out questions for a real run. pass@1 is the only metric: the fraction of
questions whose greedy generation passed every task test.

`eval.json` has the final result at the top level and every evaluation in a
list. It is replaced atomically after each evaluation, so a crash keeps the
last checkpoint result.

```json
{
  "arm": "hybrid",
  "timestamp": "20261112T000000Z",
  "step": 1000,
  "pass_at_1": 0.42,
  "count": 250,
  "subset": {"method": "seeded_sample", "seed": 2026,
             "requested": 100, "selected": 100, "pool_size": 250},
  "evaluations": [
    {"step": 100, "kind": "checkpoint", "pass_at_1": 0.31, "count": 100,
     "prompt_ids": ["..."], "subset": {"...": "the same selector"}},
    {"step": 400, "kind": "stage", "stage": "very easy", "pass_at_1": 0.55,
     "count": 12, "prompt_ids": ["..."], "subset": null},
    {"step": 1000, "kind": "final", "pass_at_1": 0.42, "count": 250,
     "prompt_ids": ["..."], "subset": null}
  ]
}
```

The CLI loads the pinned MBPP split with `data.load_split`, uses the 711 train
questions plus the 50 synthetic easy questions as training prompts, and the 250
test questions as eval prompts. Fake mode uses the same set, so a fake run also
evaluates held-out questions and never evaluates a training question. The
`Experiment` itself has no fallback. It refuses to run when `eval_prompts` is
empty. A wrong or empty split stops the command before a run folder is made.

The prompt names the function the hidden tests call. MBPP problem text often
leaves the name out, so the model used to invent one and every test failed with
`NameError`. `data.expected_function_name` takes every call in every test,
drops keywords and module attributes, and keeps the one name all tests share.
`format_prompt` puts it in the first line, with the number of arguments the
first test passes:
``Write a Python function named `topbottom_surfacearea` that takes 1 argument ...``.
All 1014 rows get a name and an argument count. Test bodies and expected values
never enter the prompt.

## Completion length and sampled entropy

`completion_tokens` is the completion length. It feeds the length penalty.

Entropy is not a rollout field. It is measured on the update pass, on the same
forward logits the policy loss uses, every `[training] entropy_every_steps`
training steps (default 500). Every prompt group updated in that step is
measured. Zero disables it. The value is the mean Shannon entropy in nats
over the generated tokens, estimated from at most 64 evenly spaced positions
when a completion is longer than that. It is logged, not rewarded, and it is
not bounded by 1.

`metrics.jsonl` carries `mean_entropy` only on a step that measured it. A step
without a measurement omits the key. The summary and the dashboard treat the
key as absent, never as zero. The fake generator returns one deterministic
value per prompt on the same cadence.

## GPU memory

Each metric row carries the allocator split in MiB: `gpu_live_mib`,
`gpu_cached_mib`, `gpu_fragmented_mib`, `gpu_total_mib`, and
`gpu_unaccounted_mib`. The first three come from `torch.cuda.memory_stats()`
inside the training process, read after the update. Live plus cached plus
fragmented is the allocator's reserved total. Unaccounted is the device total
minus that, which covers other processes, the CUDA context, and the driver.

A run sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` at the entry
point, before torch is imported, so segments grow instead of fragmenting on a
new size. A value already set by the caller is left alone. The fake path writes
zeros, so the dashboard memory panel renders on a machine with no GPU.

## Rollout record schema (frozen)

One JSON object per line, fields in this order. Defined in
`src/rlvr/schema.py`. Later tasks read these records, so field names, order,
and types are a contract.

| Field | Type | Notes |
|---|---|---|
| `arm` | str | one of the four arm names |
| `step` | int | 1-based step within the run |
| `prompt_id` | str | MBPP-style id, stable across arms |
| `reasoning` | str | thinking trace, kept apart from code |
| `code` | str | generated code only |
| `prompt_tokens` | int | |
| `completion_tokens` | int | completion length for the length study |
| `tests_passed` | int | |
| `tests_total` | int | |
| `error_class` | str | `pass` when every test passed, otherwise the failure class set by task 2 |
| `code_score` | float | passed / total, in [0, 1] |
| `judge_score` | float or null | null when the arm does not judge or tests failed |
| `process_score` | float or null | null except in `gated-process` on a pass |
| `reward` | float | the arm reward, clipped to [-1, 1] |
| `stage` | str | `unstaged`, kept so older readers find the field |
| `cache_hit` | bool | true when any judge call was served from cache |

Timestamps and truncation flags are deliberately out of this record. Truncation
rate is a step-level metric, added in a later task.

## Summary and static report

`summary` takes one or more run folders. A `runs/` root, or a single arm folder,
also works and expands to every `<timestamp>` folder below it.

```bash
.venv/bin/rlvr summary runs
.venv/bin/rlvr summary runs/code-score/20261112T000000Z runs/hybrid/20261113T000000Z
.venv/bin/rlvr summary runs --html report.html
.venv/bin/rlvr summary runs --no-html
```

It prints four plain tables, one per metric: reward, pass@1, entropy, and
completion length. Steps are rows and arms are columns, so arms sit side by
side. pass@1 shows only at steps where an evaluation ran. The entropy column
shows a value only on a step that measured it, and a dash elsewhere.

The HTML report is written to `report.html` in the current folder unless
`--html` says otherwise. It is one static file: inline CSS, no script, no
external font, stylesheet, or image. There is one section per arm with a
per-step table, and a small comparison table at the top.

## Publish to GitHub Pages

`publish` copies the report into a throwaway git repository as `index.html`,
adds `.nojekyll`, commits it, and force-pushes that commit to the Pages branch.
The branch is replaced each time, which is intended for generated output.

```bash
.venv/bin/rlvr publish --report report.html \
    --remote git@github.com:<user>/<repo>.git --branch gh-pages
```

Credentials come from the caller's git setup. This command does not read a
token. Use a deploy key with write access to the repository. It is not run in
the test suite, which pushes to a local bare repository instead.

## The seam

`Experiment` (`src/rlvr/experiment.py`) is the only orchestration point. It
takes four injected services and nothing else touches a model, API, container,
or the filesystem during a run.

```python
Generator.generate(prompt, *, n, temperature) -> list[Generation]
Generator.generate_batch(prompts, *, n, temperature) -> list[list[Generation]]
Generator.accumulate(prompt, generations, rewards, *, measure_entropy) -> float | None
Generator.apply_update() -> None
Generator.save_checkpoint(directory) -> None
Generator.load_checkpoint(directory) -> None
TestRunner.run(prompt, code) -> Outcome
Judge.score_output(prompt, code) -> JudgeResult
Judge.score_process(prompt, reasoning) -> JudgeResult
Store.write_meta(meta) / append_rollout(record) / append_metric(row) / write_eval(payload)
Store.arm, Store.timestamp, Store.run_dir, Store.checkpoints_dir
```

`temperature` is `0.0` for greedy evaluation. `n` is the number of generations
per prompt, which is the configured group size in training and one in
evaluation. One training step samples
every prompt group in one `generate_batch` call, runs and judges each group,
logs the records, accumulates the group rewards one group at a time, then calls
`apply_update` once. That is the only weight update in the step. The judge sees
the problem and the code, or the problem and the trace. It never sees the tests.

Fakes in `src/rlvr/fake.py`: output depends on prompt id and decoding mode, test
outcome on prompt id and code, judge score on judge kind, prompt id, and text.
Judge scores and test outcomes come from SHA-256, so they repeat across
machines. They carry no meaning.

## Judge client

`JudgeClient` in `src/rlvr/judge.py` implements the `Judge` seam. It uses
OpenRouter by default: endpoint `https://openrouter.ai/api/v1/chat/completions`,
model `deepseek/deepseek-v4.1-flash-1017`, temperature 0, JSON output, and
`reasoning_effort` `medium`. The model id is the one listed by
`https://openrouter.ai/api/v1/models`, and `reasoning_effort` is one of that
model's supported parameters. OpenRouter requests send `Authorization`,
`Content-Type`, `User-Agent: rlvr/0.1 (research judge)`, and `X-Title: rlvr`.
`HTTP-Referer` is not sent: the docs mark it optional and describe it as app
attribution for the public rankings page.

The `[judge]` table picks the backend:

| Key | Default | Meaning |
|---|---|---|
| `provider` | `openrouter` | `openrouter` or `opencode` |
| `base_url` | the provider endpoint | full chat completions URL |
| `api_key_env` | the provider variable | name of the env var that holds the key |
| `model` | (required) | model id, must match the provider |

With `provider = "opencode"` the client uses
`https://opencode.ai/inference/go/openai/v1/chat/completions`, model
`deepseek-v4.1-flash`, and the same four headers plus `Accept` and a stable
`x-opencode-session` header. That gateway rejects the default urllib User-Agent.
The OpenCode-only session header is never sent to OpenRouter.

So the two run configurations are: `openrouter` with
`model = "deepseek/deepseek-v4.1-flash-1017"` and `OPENROUTER_API_KEY`, or
`opencode` with `model = "deepseek-v4.1-flash"` and `OPENCODE_GO_API_KEY`.
Provider, base URL, and model must be changed together.

The key comes from the variable named by `[judge] api_key_env`, which defaults
to `OPENROUTER_API_KEY`. `JudgeClient.from_config` reads it and fails with
`MissingAPIKeyError` before any request. The key is only used in the
Authorization header. It is never written to the cache, an error message, or
`meta.json`.

The output judge sees the problem and code. The process judge sees the problem
and reasoning trace. Neither prompt has a way to receive task tests.

Calls retry on 408, 429, 5xx, timeouts, and connection errors with exponential
backoff plus jitter. `[judge] max_attempts` caps the attempts. Other 4xx
responses raise `UnrecoverableJudgeError` at once.

Scores are cached in `[judge] cache_dir` (default `judge-cache/`). The key is
SHA-256 over provider, base URL, model, judge kind, full prompt text, and rubric
version. Writes use `os.replace`. Identical requests return `cache_hit=true`.
Changing the provider or the base URL starts a separate cache generation, so a
switched provider cannot reuse the other one's scores.

`score_batch` runs requests on a bounded pool of `[judge] concurrency` threads
(default 32) and preserves input order.

## Reward table (frozen, from SPEC.md)

| Arm | Tests fail | Tests pass |
|---|---|---|
| `code-score` | 0.45 at one test, up to 0.75 | 1.0 |
| `hybrid` | 0 | `0.5 + 0.5 * judge` |
| `gated-judge` | 0 | `judge` |
| `gated-process` | 0 | `0.5 * process + 0.5 * judge` |

Fixed outcomes in `code-score`:

| Outcome | Reward |
|---|---|
| truncated rollout | `-1.0` |
| code that does not compile | `-0.3` |
| code that raises while it runs | `0.0` |
| closing marker, zero tests passed | flat `0.05` |
| clean failure, not closed | `0.0` |
| length above 900 tokens | `-0.001` per token, capped at `-0.2` |

The sum is clipped to `-1` and `1`. The thinking budget is one fixed value per
run, `[training] thinking_budget`, which is 1,024 in the arm configs. vLLM 0.34
records it but does not enforce it.

Any failed test closes the gate, so a gated arm makes no judge call. For the
gated arms zero tests counts as a failure. For `code-score` zero tests is an
error: a rollout with nothing to run must fail loudly, not score 0.

`gated-process` calls the process judge with the problem and the reasoning
trace, then the output judge with the problem and the code. Both scores are
stored in the rollout record. An empty trace is sent to the judge like any
other trace.

For a real run, the judge client is built at startup only for arms that use the
judge, so a missing `OPENROUTER_API_KEY` fails before any rollout. The fake
path uses the fake judge and needs no key.

## Status

Tasks 1 to 8 are done: the fake end-to-end path, config, schema, run folders,
merged MBPP split, Docker test runner, the judge client, all four reward arms,
entropy and length logging, the checkpoint and final evaluation schedule, the
summary command, the static HTML report, and the Pages publish command. Task 8
adds the real Qwen generator and GRPO trainer, the pinned GPU environment, one
command per arm, checkpoint resume, budget guards, and the smoke profile.

The real generator, GRPO update, and the 15-minute smoke runtime are not
verified here because this environment has no GPU. The Pages publish command
exists but is not run live here: it needs a writable remote.

## Remote instance

See `docs/DEPLOY.md` for the live host, user, SSH key, and run commands.
