# Ideas

Two chunks per idea. Sorted newest first inside each topic. The date marks first appearance, from git history for the older ideas and from this session for the newer ones. The full entries are in `/home/opencode/Work/Garbage/ideas-old.md`.

## Training

- **Spread against sweep.** Compare the two parked schedules at equal compute. Pick the schedule with more signal, and record a null result either way (2026-11-21).
- **Re-screen parked prompts.** Re-screen parked prompts inside training steps so their rollouts produce gradients. The test covers two schedules, spread and sweep (2026-11-21).
- **Stopping rule.** Write a stop rule for each run and add a rate decay. A run at the wrong rate spends hours on nothing (2026-11-19).
- **Adaptive generations.** Raise group size when the pool shrinks, and judge mastery on the first eight. Extra samples feed the advantage only (2026-11-19).
- **Judge arms.** Add a judge term smaller than half a test. All-zero groups carry signal, and a full pass wins (2026-11-19).
- **Rehearsal set.** Keep a fixed slice of solved prompts in every batch. Easy prompts hold reward variance, and the model keeps seeing them (2026-11-19).

## Evaluation

- **Test bands.** Label the 250 test questions by difficulty band. You read per-band scores from the pinned hundred on unseen prompts (2026-11-19).
- **Cross-language.** Score the same adapter on MultiPL-E and mHumanEval. The first needs a runtime per language, the second reuses the Python runner (2026-11-19).
- **Base evaluation.** Score the base model on the run's own cohort before step one. Write the row into eval.json so step one has a starting point (2026-11-19).
- **Evaluating state.** Show an evaluating state on the dashboard while the pinned hundred runs. You know why the GPU looks busy (2026-11-19).

## Engine and data

- **Meta refresh.** Rewrite the config block in meta.json on every resume. The folder matches the run, and no stale value remains (2026-11-21).
- **Thinking budget.** Cap tokens at the budget plus a fixed answer allowance. The alternative is a stop string on the closing marker (2026-11-21).
- **vLLM upgrade.** Upgrade vLLM so LFM uses the fast path. The repository pins 0.10.1, and the Hugging Face fallback took an hour for twenty steps (2026-11-19).

## Sampling and pooling

The research question: which sampling and weighting rule wins at a fixed rollout budget? Four arms run against vanilla on the pinned hundred. The arms are sorted newest first (2026-11-22).

- **Temperature lift.** Raise the sampling temperature for flat groups until signal appears. Cap the lift, and log the realised temperature per group (2026-11-22).
- **Within-group top-up** (secondary). Add samples to one flat group until its spread clears the threshold. Extra samples waste compute when the pass rate sits near zero (2026-11-22).
- **Soft adaptive** (secondary). Give signal-rich groups extra rollouts inside the same total budget. Flat groups stop at the minimum, so hard prompts can starve (2026-11-22).
- **Drop low-signal.** Drop groups under a std threshold and renormalize the rest. Vanilla zeroes exact zeros, so set the bar higher (2026-11-22).
- **Replace low-signal.** Resample groups under the threshold inside the same rollout budget. You keep equal compute and drop the dead groups (2026-11-22).
- **Std-weighted.** Weight each group's update by its reward std. Flat groups contribute nothing, and wide groups drive the step (2026-11-22).

## Model generated

A model review of the screen and two spread pulls proposed these five. None is chosen.

- **Parked coverage.** Draw each parked prompt once before repeating any. The current rotation missed 52 prompts and repeated 101 slots in one run (2026-11-22).
- **Eight-sample probe.** Probe parked prompts with eight samples and confirm promotions with sixteen. The two-stage screen uses 37 to 44 percent less parked compute (2026-11-22).
- **Demotion on draws.** Count demotion on draws, not calendar steps. Ten failed groups need about 130 steps at the current revisit gap (2026-11-22).
- **Second passing group.** Require a second passing group before a promotion enters train. Five of sixteen promotions showed no passing group in train (2026-11-22).
- **Retire rehearsal.** Retire rehearsal from training and stop counting it as usable. All 174 prompts sat unused across 43,168 rollouts from two pulls (2026-11-22).
