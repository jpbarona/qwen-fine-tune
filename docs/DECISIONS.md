# Decisions

Study decisions, newest first. Each one says what was chosen and why.

## D9. Several prompts per step, sized by free memory

A step takes several prompt groups, one group of 8 samples each. The batch starts
at two prompts. Free memory is read after each step, and the batch grows by one
while free memory stays at or above the larger of `[training] memory_floor_mib`
and one prompt group's measured footprint from the previous step. The first
step below that floor settles the batch at the largest size that fit, and it
stays there. A CUDA out-of-memory error does the same instead of crashing. It
never goes below one prompt. There is no percentage rule.

Advantage is normalised inside one prompt group. The backward pass runs once per
generation and gradients accumulate across the group, so only one autograd graph
is alive at a time. This is what makes the 2,048 token cap fit at a larger batch.

A checkpoint records the prompt cursor, the batch policy state, and the
curriculum position. A resume continues on the next batch boundary with the same
prompt position, batch size, and stage pool.

## D1. One arm only for now

The four-arm comparison is paused. The judge and process arms, hybrid and
gated-process, all gate on unit tests. If the model almost never passes tests,
all four arms score zero and the study cannot answer its own question.

Code arm only until the model passes tests at a usable rate. Revisit the arms
after that.

## D2. Thinking on, 2,048 token cap

Thinking stays on, since the target behaviour is concise thinking plus correct
code. The completion cap rises from 768 to 2,048, counted over the whole
output, reasoning and code together.

Prerequisite: drop `output_scores` before raising the cap. Stored logits cost
about 10 GB at 2,048 and 34 GB at 7,000, which does not fit beside the rest.
This is now decided, see D8.

## D3. Reward

| Case | Reward |
|---|---|
| Truncated at the cap | -1.0 |
| No code parsed, or syntax error | -0.3 |
| Runtime error | 0.0 |
| Compiles, tests fail, closed | 0.05 |
| Compiles, tests fail, not closed | 0.0 |
| Partial tests | 0.45 at one test, rising to 0.75 just below a full pass |
| All tests pass | 1.0 |
| Length above 900 tokens | -0.001 per token above 900, floor -0.2 |

Clip the sum to the range -1 to 1.

The length penalty applies to answers that passed a test, over the whole
output, thinking included. It does not touch the failure tiers, so a long clean
failure cannot fall below a crash. It is also clamped just above the tier
beneath it, so a long answer that passed more tests never scores below a short
answer that passed fewer. Two of four at 1,100 tokens floors at 0.451, above one
of four at 0.45, instead of dropping to 0.40.

Invariant, a correct answer that finishes scores above an unclosed wrong
answer, at any length:

- Best unclosed wrong answer is 0.0, compiles and fails every test.
- Worst correct answer is 1.0 less the 0.2 floor, which is 0.8.
- 0.8 is greater than 0.0, so this holds.

The failure tiers are flat and ordered: truncated -1.0, syntax error -0.3,
runtime error 0.0, clean failure that closed 0.05. Closing pays a flat 0.05
when no test passed. It buys termination, not proximity, so one passing test at
0.45 beats it by a wide margin and a crash can never collect it. Closing does
not touch a partial score, so a 3-of-4 answer stays 0.75 and a full pass keeps
at least 0.8.

Why the close payment is small: run 16 paid 0.3 for a closed answer that passed
nothing. By step 67 the model had 203 of 208 rollouts crashing at runtime and
zero passes, with reward stuck at 0.297. The payment was farmed, so it is now
0.05 and crashes score 0.0.

Why partial credit carries the early training: a group of 8 all-zero rollouts
gives zero advantage, so no penalty helps. Partial credit creates the
variance that GRPO needs.

## D4. Curriculum stages

Four bands, in order: very easy, medium, hard, very hard.

- Very easy: 50 synthetic questions, each with its own tests. Written by an
  agent. Start here.
- Medium and hard: cut from the 711 MBPP training rows by the committed
  difficulty labels.
- Very hard: last stage only.

Each stage keeps about 20 percent of the previous band, so the model does not
forget it.

Within each stage, 80 percent of questions train and 20 percent are held for
evaluation.

## D5. Difficulty labels

Labels come from subagents, in batches of 20 questions. Labels are accepted as
imperfect. The screening command measures pass rates, but it does not re-band
anything. Band assignment stays on the committed label file.

## D6. Skip mastered questions

A run mode where a question leaves the pool once a full pass is observed for
it. Mastered questions give zero advantage, so keeping them burns steps.

## D7. Dashboard

Show the current stage band, the step count, and pass rate per band.

## D8. Entropy is sampled, not recorded every step

`output_scores` comes out of `generate`. It is not used by the loss, only for
the per-step entropy chart.

Entropy is measured every N training steps instead, default 500, set by
`[training] entropy_every_steps` in the config. At that point the update pass
already has logits for the tokens, so the measurement is free. Set to 0 to
disable.

The dashboard entropy chart shows one point per measurement, not per step.

## Open

- Eval frequency and size for the new setup.
- How the synthetic easy set is checked, beyond its own tests.
- Whether the 250 held out MBPP rows stay as the final evaluation set.
- Smallest workable group size. Try GRPO with as few samples per prompt as
  still trains, 4 or 6, as a later experiment. Default stays 8 for now.

## Finding: run two collapsed at step 22

Run `20261114T042144Z` reached reward 0.31 by step 2, then collapsed. From
step 22 every completion ran to the 2,048 token cap without closing the
reasoning block, so no code was parsed, every reward was -0.8, and every group
had zero advantage. No checkpoint existed before step 50, so there was nothing
to resume. The run was stopped at step 35.

Contributing causes, in order:

1. One optimizer update per prompt. A 26 prompt step meant 26 updates, about
   300 updates by step 22, so the policy moved far faster than the step counter
   suggests.
2. No KL anchor, so nothing held the policy near the base model.
3. Checkpoint and eval cadence of 50 and 100 steps, too coarse for a collapse
   at step 22.
4. Batching ran prompt groups one at a time, so the batch per generate call
   stayed at 8 sequences.

## D9. One update per step

Status: implemented. `Experiment._training_step` accumulates every prompt group
and calls `Generator.apply_update` once, and `QwenGenerator.step` counts calls
to that method, so the step number is the weight update count.

Gradients accumulate across every prompt group in a step, and the optimizer
steps once at the end. Reason: per prompt updates multiply the learning rate by
the batch size and caused the collapse. The step counter now matches the number
of weight updates.

Advantage normalisation stays inside one prompt group. The optimizer is left
alone.

## D10. Several prompts in one generate call

Status: implemented. `Generator.generate_batch` takes the prompt list and returns
one group per prompt. `Experiment._training_step` pairs the returned groups with
the prompts by position and checks the shape before any scoring.

Prompts are padded to a common length and sampled together, eight answers each,
in one call. Reason: decode is CPU bound at 152 tokens per second, about 5
percent of the roofline. One call per prompt leaves the card idle between calls.
Gradients, rewards and parsing do not change.

Padding is on the left, so every completion starts at the same column and one
prefix length slices every row. The attention mask marks the pads, is checked
before the call, and is stored with each group so the update pass feeds the same
mask back.

## D11. vLLM for rollouts

The seam is built. `VLLMGenerator` implements the `Generator` protocol and
samples token ids with a vLLM engine while the Hugging Face model keeps the LoRA
adapter and does the update. The engine is optional: `[training] engine`
defaults to `hf`, so the existing path is unchanged, and `vllm` is a startup
error when the package is missing.

Weight handoff uses LoRA, not a full model copy. After each optimizer step the
trainer writes the PEFT adapter to disk and the engine gets a new `LoRARequest`
with a fresh integer id, so it cannot keep serving the previous adapter. Only
the two newest adapter folders stay on disk. One handoff per optimizer step,
never per prompt. Training still recomputes old logprobs on the training model,
so engine logprobs are not used.

Status: the shipped real config uses `vllm`, with `vllm==0.10.1` pinned in the
`vllm` extra. The path is unit tested with a fake engine. It has not yet been
executed against a real vLLM engine on a GPU.

## D12. Thinking budget schedule

Status: implemented. `[training] thinking_budget_start` defaults to 768 and
doubles at each curriculum stage, capped at `max_completion_tokens`, so the
four bands get 768, 1536, 2048, 2048. A rollout that reaches the budget without
closing is ended there and scores as truncated.

Closing the block pays up to 0.4 when no test passed, capped below one test so
one pass always beats closing alone. With three tests it pays 0.3. It does not
touch a partial score.

Reason: while every rollout fails to close, every reward is identical and no
shaping term can help. A short budget makes closing reachable and creates the
variance the bonus needs.

## D13. Checkpoint every 10 steps, eval every 20

Status: implemented. `configs/real.toml` sets `checkpoint_every_steps = 10` and
`eval_every_steps = 20`. Fifteen checkpoints are kept per run, the five newest
consecutive plus older ones at doubling gaps, so an early collapse still has a
state to resume from and a long plateau has more points to fall back to.

Reason: the collapse arrived at step 22 and the first checkpoint was at 50. The
medium plateau on run `20261115T212843Z` showed that 20 steps between saves
loses up to 40 minutes of work on a restart.

## D14. Docs live in docs/

Every markdown document moves to `docs/`. `README.md` stays at the root for
GitHub. "Go and update the docs" means update `DECISIONS.md` and `journal.md`,
keep the docs index accurate, and leave the code alone.

## Note on the length penalty

The length term is implemented and on by default: soft limit 900 tokens, 0.001
per token above it, floor 0.2. It is covered by tests. The three keys now sit in
`configs/real.toml` under `[training]`, so the term is visible and tunable.


## D15. The prompt names the function the tests call

Status: implemented. `data.expected_function_name` reads the name from the first
`assert name(` in the test list. `format_prompt` puts it in the first line of
the prompt. When no test calls an identifier, the old template is used, which is
2 rows out of 964.

Reason: run 17 collapsed with a 92 percent `runtime_error` rate. 540 of 544
sampled failing rollouts defined a function with a different name than the test
called, so every test raised `NameError`. The model was never shown the name. A
local docker replay of 16 distinct failing tasks passed 10 of 16 after renaming
the candidate function to the expected name, so the code was mostly right and
only the identifier was wrong.

Not in the prompt: test bodies, expected values, or the number of tests. The
name alone is enough to fix the failure and does not let a model hardcode an
answer.

Arity wording: the prompt says the tests call the function with exactly N
arguments, not that the function takes N. The wording names the tests as the
source, so the model cannot read it as a claim about its own signature. One
output rule follows: output only the function definition, no prose and no code
fences.

Reason: in 300 medium traces, 245 restated the arity phrase, 27 guessed a
signature, 23 contradicted their own code, 61 debated the output format, and 29
produced code that would not parse. The fix removes the ambiguity without
changing the task.

## D16. A prompt is solved only when every rollout passes

Status: implemented in `experiment._prompt_group`. A group counts as solved when
all of its generations pass. One pass in eight no longer removes the prompt from
the stage pool. The docstring on the method says so.

Reason: on 2026-11-15 the very easy pool went from 69 prompts to 5 by step 12,
because a single lucky pass retired each prompt. What stayed behind was exactly
the set the model could not solve, so every later step trained on five
impossible prompts, the reward went negative, and the run could never leave the
band. The plotted collapse was that selection effect, not the model getting
worse.

Consequence, not yet decided: a stage ends when its pool is empty, and a strict
rule makes the pool empty much later. The stage needs a second exit. Options are
an attempt cap per prompt, or a step cap per band. Decide before the next full
run.

## D17. The RL environment needs cold tests, not live-run discovery

Status: implemented. 899 tests, 9 of them cold checks over the committed data.
The prompt contract reads every one of the 961 MBPP rows and the 50 synthetic
rows:

- every row resolves to a function name, and the name is not a keyword,
- the name appears in the prompt and in every test of the row,
- no test calls a helper the row does not provide (`KNOWN_MISSING_HELPERS` in
  `tests/test_data.py` records the row that currently fails this),
- every test string compiles,
- no test is blank, and only `KNOWN_DUPLICATE_TESTS` repeats one,
- the prompt never contains a test body or an expected value,
- prompt ids are unique across both sets.

The synthetic set gets the stricter check, because it is the first stage and it
was written by an agent:

- the prompt name is the name the reference solution defines,
- every reference solution passes its own tests in-process,
- every reference solution passes in the real container harness
  (`tests/test_environment_solvable.py`),
- a candidate that uses the wrong name is reported as a runtime error.

Stage pools are checked for duplicates and for train/held-out overlap.

Reason: three environment bugs were found only by burning GPU hours. The prompt
never named the function, `abs` was mistaken for the function under test in a
wrap-around assert, and `assert (` was parsed as a call named `assert`. A cold
test would have caught each one before a run started.

The rule for new environment code: if a bug can be found by reading the data and
the prompt, a test must exist for it. A live run is not the place to discover a
data defect.

## D18. The prompt states the argument count, and one row is renamed

Status: implemented. `data.expected_call_arity` reads the first test call with
`ast` and counts positional and keyword arguments. `format_prompt` writes
"takes 1 argument" or "takes 2 arguments" into the first line. Rows whose call
cannot be parsed keep the older prompt.

Row 644 is renamed. The MBPP tests called `reverse_Array_Upto_K`, the model
kept writing `reverse_array_up_to_k`, and the run saw 23 passes in 120 rollouts.
The tests now use the lowercase name the model writes. 202 other MBPP rows still
carry mixed case names and were left alone.

Reason: with the name but not the arity, the model wrote `union_elements(one)`
where the tests pass two tuples, and `topbottom_surfacearea(radius, height)`
where the tests pass one number. Four prompts failed every rollout that way, and
the stage pool ended up holding only them.

## D19. A run can be edited in place through a control file

Status: implemented. Each run folder gets `control.json`, created empty at
startup:

```json
{"drop": ["mbpp-679"], "tests": {"mbpp-535": ["assert f(10) == 314.15"]}}
```

The loop reads it at the top of every step, re-reading only when the text
changes. `drop` removes a prompt from the current stage pool. If the pool
empties, the stage advances or the curriculum completes. `tests` replaces one
prompt's test list from the next step. Unknown prompt ids are ignored, unknown
keys and test lists that do not compile are rejected, and a broken file is
logged as an `error` event while the previous state stays in force. Every
change writes a line to `control-events.jsonl` with the step, so an edit is
never silent.

`--control PATH` points a run at a different file.

Reason: every defect found so far was found mid-run, and the only way to act on
it was to stop the run and throw away GPU hours. Three prompts that can never
pass pinned the very easy stage with no way out.

Caveat: a run with edits is still a run, but a prompt id can mean two different
tasks before and after an edit. The event log records the step, so an analysis
can split there. Drops are cleaner than test edits for that reason.

## D20. The dashboard reports what it can prove

Status: implemented. Three rules, after the resume that died and left the page
claiming RUNNING with a zeroed GPU strip:

- Status comes from the process table. A live `rlvr run --arm <arm>` keeps the
  run `running` through a long evaluation. A missing process turns it `stopped`
  after a two minute grace. If the process table cannot be read, the old
  metrics-age rule is the fallback.
- A missing GPU reading is `-`, never `0%`. The strip shows `last sample Ns ago`
  and a sample where every field is unreadable is not recorded at all. Zero
  stays zero, so an idle GPU is still visible as idle.
- Dropped prompts leave the questions tab, keeping their history behind a
  "show dropped" control, and stage completion subtracts them.

Reason: an idle GPU and a broken sampler look the same if zero is the fallback,
and a dead trainer looked alive for fifteen minutes.

## D21. Proposed: raise generations per prompt when the pool is small

Status: proposed, not implemented. With three prompts left, a step is 24
sequences and the card is idle. The proposal is a floor of 8 generations per
prompt that rises as the pool shrinks, up to a cap (32 is the suggestion), so a
step keeps roughly constant work.

Why it is worth doing: group size is what the GRPO baseline is computed from, so
a larger group lowers gradient variance, not just idle time.

The catch is the mastery rule in D16. If solved means every rollout passed,
raising `n` to 32 makes retirement far less likely and the stage stalls again, a
failure already seen twice. The workable version keeps the rule by judging the
first 8 rollouts and letting the extra generations feed only the advantage.

Not decided, not built.

## D22. Every run measures the base model at step 0

Status: done by hand for run 17, not built into the runner. The run's own
evaluations began at step 20, which made a 300 step curve look flat: the
untrained model solved 12 of the 100 held-out questions, step 20 solved 32 and
step 300 solved 37. Two thirds of the gain happened before the first data point.

Reason: a curve with no starting point cannot show improvement, and the
conclusion drawn from run 17 was wrong because of it. A base evaluation on the
same cohort costs about five minutes.

The cheap version of this, for a run that has not started, is one greedy
evaluation of the base model on the run's own held-out ids, saved beside the
run folder.

## D23. Log the weight update norm, not only the gradient norm

Status: implemented. Every step records `grad_norm`, the L2 norm of the
trainable gradients after the group divide and before the optimiser step, and
`update_norm`, the L2 norm of the parameter delta the step produced. Both write
null when the value is not finite, so a diverging run records a row instead of
crashing the metrics write.

Reason: AdamW normalises each parameter's step by its gradient history, so the
gradient norm barely moves between learning rates. In the probe it read 0.11 to
0.33 at 1e-4 and 0.05 to 0.59 at 1e-3, overlapping. The update norm read 0.08
against 0.7 to 2.2, a gap of 25 times. Only the second number answers "is this
rate moving the weights too far".

No gradient clipping was added. Clipping would not stop a rate that is simply
too large under AdamW, and it would hide the divergence the probe is looking
for.

## D24. Learning rate: 1e-4 is safe, 1e-3 diverges

Status: measured with two 20 step probes on separate cards, same config, same
cohort, each capped with `[budget] max_steps = 20`.

- 1e-4: reward stayed between 0.44 and 0.91, pass@1 at step 20 was 0.31,
  update norm 0.08.
- 1e-3: collapsed at step 2. By step 14 the reward was -0.99, the pass rate
  zero, and every answer ran to the 2,048 token cap. The probe was stopped at
  step 16.
- 1e-5, from run 17: pass@1 0.32 at step 20, 0.33 at step 60.

So a 10x rate is safe but has bought nothing after 20 steps, and a 100x rate
destroys the policy. The next question is whether 1e-4 pulls ahead over 60
steps, which is one resume of the probe checkpoint.

## D25. Three unpassable very easy questions are out of the dataset

`mbpp-535`, `mbpp-536` and `mbpp-679` never passed in run 17 and were dropped by
the control file at step 121. They are removed from `data/mbpp_full.jsonl` and
from the label file, so no run spends steps on them again.

Deleting rows would reshuffle the seed 1234 cut and replace 187 of the 250
held-out questions, which breaks every past evaluation. So the split is pinned
in `data/split_seed1234.json`: 711 train and 250 test, with the same test ids as
before. `load_split` reads the pinned file when the config seed is 1234 and cuts
fresh for any other seed.

Training goes from 714 rows to 711.

## D26. All-linear LoRA at rank 16, alpha 32

LFM2.5-2.6B and the Qwen baseline both adapt every linear layer. On LFM this
covers the attention projections, both MLP matrices, and the two linear
projections in each convolution block, `in_proj` and `out_proj`. PEFT
`all-linear` covers those two `nn.Linear` projections and not the depthwise
`nn.Conv1d` kernel. Rank 16, alpha 32.

Why. The code before this change gave LFM only the four attention projections,
which left 22 of 30 blocks frozen. The default is still `attention`, set by
`lora_targets`, and `all-linear` is opt-in. Every framework checked defaults to
all-linear for RL: verl, OpenRLHF, TRL. Placement mattered more than rank in the Thinking
Machines experiments, where attention-only at a matched parameter count lost to
all-linear.

Why rank 16. Thinking Machines matched full fine-tuning at rank 1 on 8B models.
A 1B to 3B GRPO study found rank 16 against 32 inside noise, with higher ranks
drifting further from the base policy. Liquid ships rank 16 for LFM2.5. The one
result favouring a higher rank is code supervised fine-tuning at 7B, where rank
64 beat 16, and that run carried far more signal than a short RL run.

Sizes on LFM2.5-2.6B, all-linear: rank 8 is 12.2M, rank 16 is 24.5M, rank 32 is
48.9M. Rank 16 is 0.95 percent of the model.

The setting is a config key, `lora_targets`, with `attention` as the default, so
every existing Qwen config keeps its committed behaviour. A 60 step Qwen
all-linear run at rank 16 supplies the matched baseline for the model
comparison.

