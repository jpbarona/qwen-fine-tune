# Run 17 analysis and a small cross-language probe

This folder has two parts. Part 1 recomputes run 17 from its saved files. Part 2
runs a 20 problem HumanEval style probe on the base model and the step 300
adapter on the GPU box. All numbers below come from the files listed at the end
unless the text names another source.

## Part 1: run 17

### Configuration

Run 17 trained Qwen/Qwen3-0.6B at revision
c1899de289a04d12100db370d81485cdf75e47ca with GRPO on MBPP.

| Item | Value |
| --- | --- |
| Arm | code-score |
| Steps | 313 |
| LoRA | rank 32, alpha 64, attention only |
| LoRA target modules | q_proj, k_proj, v_proj, o_proj in every layer |
| Learning rate | 1e-5 |
| Rollouts per prompt | 8 |
| Sampling temperature | 0.9 |
| Max completion tokens | 2048 |
| KL beta | 0.0 |
| Clip epsilon | 0.2 |
| Train split | 714 questions |
| Test split | 250 questions |
| Eval subset | 100 questions, seed 2026, pool 250 |
| Curriculum | very easy, medium, hard, very hard |
| Start stage | very easy |
| Software | Python 3.12.16, torch 2.7.1 cu128, transformers 4.57.1, peft 0.17.1 |

The run wrote 313 metric rows and 132,408 rollout rows. Every rollout group has
8 rows, so the file holds 16,551 groups. The stage counts are 160 steps in very
easy and 153 steps in medium. Hard and very hard were never reached.

### Base against step 300

The saved files contain no base measurement. The run's own evaluations start at
step 20. The base numbers below come from the run 17 postmortem, which
re-evaluated the base model and several checkpoints on the run's own 100
question cohort.

| Measurement | Untrained | Step 300 |
| --- | --- | --- |
| Greedy pass@1, reasoning on | 0.19 | 0.37 |
| Greedy pass@1, reasoning off | 0.120 | 0.240 |
| Solved questions, reasoning off | 12 of 100 | 24 of 100 |
| Mean completion tokens, reasoning off | 34 | 46 |

Paired on the same questions, 13 were gained and 1 was lost. The postmortem
reports an exact McNemar p of 0.002.

The step 300 checkpoint evaluation inside eval.json reports 0.35, not 0.37.
The difference is a second measurement of the same checkpoint, not a second
checkpoint.

### The flat curve artefact

The run evaluated 15 checkpoints, at steps 20, 40, 60, 80, 100, 120, 140, 160,
180, 200, 220, 240, 260, 280 and 300. The values were:

0.32, 0.33, 0.33, 0.37, 0.34, 0.40, 0.34, 0.33, 0.37, 0.40, 0.36, 0.34,
0.35, 0.35, 0.35.

The first value is 0.32 and the last is 0.35. The range is 0.32 to 0.40. A
reader who starts at step 20 sees 300 steps of noise around a line with almost
no slope.

The postmortem gives the missing part. The untrained model scores 0.19 and
step 300 scores 0.37. The first logged point is already 0.32. Most of the gain
therefore happened before the first evaluation. The flat part is real for the
steps it covers, but it is not the whole run.

The curve is also noisy. With 100 questions, one question is one percentage
point, and the postmortem puts noise at about five points. Differences smaller
than that are not readable from this curve.

### Zero variance and retirement

| Statistic | Count | Share of groups |
| --- | --- | --- |
| Groups | 16,551 | 1.000 |
| Zero variance groups | 1,713 | 0.1035 |
| Groups with reward 0 for all 8 | 548 | 0.0331 |
| Groups with reward 1 for all 8 | 127 | 0.0077 |
| Groups where all 8 failed every test | 7,311 | 0.4417 |
| Groups with a repeated code body | 3,262 | 0.1971 |

The mean number of distinct code bodies in a group is 7.65 out of 8. A
zero variance group carries no advantage, so about one step in ten taught
nothing on that prompt.

Reward is not binary. It carries partial credit, so "all tests failed" is not
the same as "reward zero". This file has 7,311 groups where all eight rollouts
passed zero tests, but only 548 groups where all eight rewards were exactly
zero. The postmortem reports 1,508 all zero groups. That number does not match
the reward column in rollouts.jsonl. I report the file value and note the
difference rather than reconcile it.

Retirement uses the rule in the training code. A prompt is removed from the
stage pool when every rollout in one group passes.

| Stage | Entered | Removed | Mean pass rate of removed | Mean pass rate of remaining |
| --- | --- | --- | --- | --- |
| very easy | 69 | 35 | 0.860 | 0.741 |
| medium | 434 | 92 | 0.863 | 0.205 |

The removed prompts are the ones the model could already solve. The remaining
medium pool averages 0.205 over all its history. The postmortem reports 446
entered and 109 removed for medium, with 0.856 for removed and 0.121 for the
ones left behind. My counts differ because I attribute a prompt to the stage
where it first appears and average over every group it ever had. The direction
is the same in both readings. The pool keeps the harder questions.

Other training statistics:

- 503 unique prompts appear in the rollout file.
- 69 prompts received exactly one group, which is one gradient update.
- The median step had 20 prompts and 160 sequences. The largest had 175
  prompts and 1,400 sequences.
- 42 steps ran three prompts or fewer.
- Error classes in medium: wrong answer 49.2 percent, runtime error 28.2
  percent, pass 15.0 percent, syntax error 7.1 percent, timeout 0.6 percent.
- Mean completion length was 809 tokens in very easy and 572 in medium.

### What the run does not establish

- It does not contain a base evaluation. The 0.19 figure comes from work done
  after the run, not from the run's own log.
- It does not establish that the gain is from GRPO rather than from the LoRA
  adapter, the prompt format or the change in answer length.
- It does not establish transfer. Training and evaluation are MBPP only, with
  no second dataset and no second language.
- It does not establish that the late rise would continue. The last logged
  checkpoint is step 300 and the run stops 13 steps later.
- It does not measure the hard or very hard curriculum bands.
- It does not log advantage spread or gradient norm. The postmortem says these
  were never recorded. Entropy was set to every 500 steps and the run ended at
  313, so the series is at most one point.
- It does not settle the learning rate. The 1e-4 probe ran 20 steps and the
  1e-3 probe collapsed. The comparison past 20 steps was not run.
- The 100 question cohort is the run's own evaluation set, not a second
  held-out set. The same ids were used for every checkpoint, which makes the
  curve internally comparable but does not make it an unbiased estimate for
  the full 250 question test split.
- A single seed cannot separate the run's result from seed noise.

## Part 2: cross-language probe

### Question

Does the step 300 adapter, trained on Python MBPP, do better than the base
model on HumanEval problems in Python, JavaScript, Java and C++?

This is a cheap directional probe, not a benchmark result.

### Design

- 20 problems, sampled once with seed 2026 from the 158 HumanEval numbers that
  exist in all four language files.
- Greedy decoding, one attempt per problem.
- Max 512 new tokens.
- Raw completion prompt, no chat template. This is the standard HumanEval and
  MultiPL-E protocol. A chat wrapper made both models restart the function
  instead of continuing it, so I dropped it.
- Stop strings from each MultiPL-E row. Python uses the standard HumanEval
  stop strings.
- Python problems and tests come from the original HumanEval file. The other
  three languages come from nuprl/MultiPL-E, configs humaneval-js,
  humaneval-java and humaneval-cpp.
- Grading compiles and runs each answer with the shipped test text. Python
  appends the HumanEval `test` block and a `check(entry_point)` call.
- Java runs with assertions enabled and with javatuples 1.2 on the classpath,
  because the shipped Java tests import it.
- JavaScript runs on Node 22.11.0. C++ compiles with g++ and C++17.
- One normalisation: for Java and C++, if the completion ends with a line that
  is only `}` and the shipped test text begins with `}`, that one completion
  line is removed. The tests supply the closing brace. Tests were otherwise
  used without changes. Both models received the same treatment.

The adapter was built with the repository generator, attention LoRA, rank 32,
alpha 64, then loaded from step-00000300. The box has transformers 5.0.0 and
peft 0.18.1, while the run used 4.57.1 and 0.17.1. The adapter loaded without
shape or key errors.

### Results

Pass counts out of 20 per language.

| Language | Base | Adapter | Change |
| --- | --- | --- | --- |
| Python | 3 | 6 | +3 |
| JavaScript | 4 | 6 | +2 |
| Java | 4 | 5 | +1 |
| C++ | 6 | 6 | 0 |
| Total | 17 of 80 | 23 of 80 | +6 |

Paired per question, the adapter gained 7 cells and lost 1. The gains were
problems 0, 152 and 159 in Python, 136 and 152 in JavaScript, 152 in Java and
76 in C++. The loss was problem 0 in C++.

The adapter is clearly active. Its completions differ from the base model on
13 of 20 Python problems and 12 of 20 C++ problems, and the grading output
shows different failure reasons.

### What the probe says and does not say

The adapter is 6 passes ahead over 80 cells. All of that is on the easier half
of the sample and the C++ count is unchanged. With 20 problems, one question
moves a language score by 5 points. The comparison cannot support a claim about
cross-language transfer. It can support a narrower statement: the adapter does
not collapse when moved from MBPP Python to HumanEval in four languages, and it
passes a few more questions than the base model on this sample.

### Exclusions

No language was skipped. All four toolchains installed and ran. Docker was not
used. The run made no repository changes and no pushes. The LFM job on the GPU
box was left running.

Python is not a MultiPL-E translation. MultiPL-E does not ship a Python config,
so Python uses the original HumanEval prompts and tests. The other three use
translated prompts and translated tests exactly as shipped.

### Biggest limitation

The sample is 20 problems with one greedy attempt. Each language count is
therefore a coarse count out of 20, and the whole experiment is 80 paired
cells. It is enough to see that the adapter is not broken on other languages.
It is not enough to rank the two models.

## Reproduce

The scripts are in `scripts/`. `analyse_run17.py` reads the run folder and
writes `results/run17-stats.json`. `select_problems.py` builds
`data/manifest.json`. `generate.py` and `grade.py` run on the GPU box.
`summarize_benchmark.py` combines the eight grading files into
`results/benchmark-summary.json`. `download_data.py` fetches the benchmark
files and converts the parquet copies to jsonl.

## Sources

- `/home/opencode/Work/rl-grpo/docs/postmortem-run17.md`
- `/home/opencode/Work/rl-grpo-data/2026-11-16-run17-step313/metrics.jsonl`
- `/home/opencode/Work/rl-grpo-data/2026-11-16-run17-step313/eval.json`
- `/home/opencode/Work/rl-grpo-data/2026-11-16-run17-step313/rollouts.jsonl`
- `/home/opencode/Work/rl-grpo-data/2026-11-16-run17-step313/meta.json`
- `/home/opencode/Work/rl-grpo-data/2026-11-16-run17-step313/checkpoints/step-00000300`
- `nuprl/MultiPL-E` on Hugging Face
- `openai/openai_humaneval` test file from the openai/human-eval repository
