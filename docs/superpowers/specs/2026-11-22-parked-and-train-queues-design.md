# Parked and train queues, 22 November 2026

Status: approved design, not implemented.
Branch: `feat/parked-train-queues`.
Plan: `docs/superpowers/plans/2026-11-22-parked-and-train-queues.md`.

## Why

The spread schedule picks each step's prompts by multiplying the step number by
the block size. Two events break that arithmetic. A change in block size moves
the start backwards. A promotion or demotion reshuffles the list. In the 22
November run the parked block missed 52 of 267 prompts and repeated 101 of 316
slots. The train block has the same fault and nobody measured it.

The screen also puts 174 solved prompts into a rehearsal pool. Training never
draws them. The run record still counts them as usable. A group where all 16
samples pass has zero reward spread, so it cannot produce a gradient.

The demotion rule needs 10 consecutive zero-pass groups. The train block shows
each prompt about once every 27 steps, so the rule needs more steps than the run
has. No prompt demoted in either pull.

## Goals

- Show every parked prompt once before any parked prompt repeats.
- Show every train prompt once before any train prompt repeats.
- Retire screen prompts where all samples pass, and stop counting them as usable.
- Make demotion reachable inside a 100-step run.
- Keep the state across a restart.

## Non-goals

- No eight-sample probe. The 16-sample group and the 2-of-16 promotion bar stay.
- No second passing group. The promotion test stays one group.
- No change to the reward, the optimizer, the learning rate, the eval cohort or
  the split.
- No GPU run in this change.

## Decisions

### Parked share

The parked block is `max(4, floor(prompts_per_step / 2))`. At 19 prompts per
step it is 9. The minimum of 4 keeps a small step from losing the parked block.
The train block is the rest, 10 at 19 prompts. If the step is so small that the
minimum would leave no train prompt, the parked block is `prompts_per_step - 1`.

`refresh_steps` no longer controls spread. Remove the key and its validation.
The two arm configs drop it.

### One queue per pool

Each pool has a queue of prompt ids. A step takes a block from the front. A
prompt that stays in its pool goes to the back. A prompt that changes pool leaves
the queue.

The first pass covers every prompt in the pool before any prompt repeats. After
that the queue cycles in the same order, oldest first.

A promoted parked prompt joins the back of the train queue. A demoted train
prompt joins the back of the parked queue. The back placement means a new member
waits one tour.

The parked and the train pools use the same queue code.

### State

After each step the run writes the full parked queue and the full train queue to
`labels.jsonl`, in a `queues` row. A restart restores both. Ids that left the
pool are dropped. Ids that joined the pool are appended in sorted order. The
queue is the source of rotation, not the step number.

### Demotion

A train prompt demotes after 3 zero-pass groups across the run. A pass does not
clear the count. The count resets after a demotion. The threshold is 3, not 4,
because at 10 train prompts per step each train prompt appears about once every
27 steps. Three failures can fit inside 100 steps. Four cannot fit for most
prompts.

### Retirement

A screen prompt with 16 passes out of 16 is retired. It leaves the train and
parked pools and the usable count. The run record lists `retired` separately.
Training skips retired prompts, as it skips rehearsal today.

Replace the `rehearsal` field with `retired` in `Pools` and in `meta.json`.
Remove the rehearsal idea from `docs/IDEAS.md`.

### Step size

A resume that changes `rollouts_per_step`, `generations_per_prompt` or the group
size is rejected. The queue arithmetic assumes one block size for the run.

## What the run records

- `meta.json`: `pools.train`, `pools.parked`, `pools.retired`, `pools.samples`.
- `metrics.jsonl`: unchanged. `train_prompts` and `parked_prompts` still count
  the groups in the step.
- `labels.jsonl`: label changes, failure counts, and one `queues` row per step.

## Tests

- Queue coverage: a pool of P prompts with a block of k covers every prompt
  before any repeat.
- Queue order: after a full pass, the next block is the oldest block again.
- Promotion: a promoted id leaves the parked queue and joins the back of the
  train queue.
- Demotion: a demoted id leaves the train queue and joins the back of the parked
  queue.
- Resume: a restart restores both queues and continues the same order.
- Step size: a resume with a different `rollouts_per_step` is rejected.
- Retirement: a 16-of-16 prompt is retired and absent from train and parked.
- Demotion threshold: 3 zero-pass groups demote, and a pass between them does
  not reset the count.
- Spread share: 19 prompts per step gives 9 parked and 10 train. A small step
  keeps at least 1 train prompt.

## Open

The remote `main` still holds the old history with `OpenCode` authors. Landing
this work on `main` needs a decision about the protected branch. That is
separate from this design.
