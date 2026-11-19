# Journal

Running notes on this study. One chunk per line, 10 to 12 words.

Format: each line names a problem and its fix in one short chunk, joined by an arrow. Newest at the bottom. No paragraph prose, no restating the same point twice. If an idea is not tried yet, it still belongs here.

1. Random task order wastes steps → order tasks by measured difficulty
2. Thinking eats the token cap → raise cap or disable thinking
3. Truncated traces become syntax errors → split on closing think marker
4. Eval costs a third of wall clock → evaluate less often
5. One prompt per step is slow → batch several prompts
6. GPU sits at 21 percent → add batch, not memory
7. All-zero reward groups give no gradient → filter for solvable tasks
8. Tests run serially in Docker → run them in a pool
9. Pass rate is 2 percent → fix output budget before curriculum
10. Easy tasks must fade out or the model overfits them
11. 7k cap keeps every logit → drop scores before raising cap
12. Copying 10 GB off the box beats beta persistent storage
13. Four-arm study paused until the model passes some tests
14. All-zero reward groups give zero advantage → add partial test credit
15. Partial test credit creates variance where binary pass gives none
16. Any correct answer must beat any wrong one at any length
17. Each curriculum stage trains on 80, holds out the other 20
18. Entropy measured every 500 updates → drop per-token logit storage
19. Batch several prompts per step → raise memory use and throughput
20. Always-failed questions are handled by the cohort, not a drop rule
21. Try GRPO with the fewest samples per prompt, later on
22. Eight graphs alive at once wastes memory → backward per generation
23. One prompt per step leaves the card idle → grow the batch to the memory floor
24. Second run hit reward 0.31 by step two, then collapsed
25. By step 22 the model stopped closing its reasoning block
26. Every rollout ran to 2048 tokens, so reward froze at -0.8
27. No checkpoint before step 50, so collapse could not be resumed
28. Batching added groups per step but still ran them serially
29. Decode is CPU bound, one core pinned at 97 percent
30. Measured 152 tokens per second against a 2900 roofline
31. Batch stopped at 26 because mastered questions shrank the pool
32. No KL anchor, so nothing held the policy near the base
33. Checkpoint and eval cadence was too coarse for this collapse
34. Run two data saved locally before the VM was killed
35. One optimizer update per step → accumulate gradients across groups
36. Several prompts padded and sampled in one generate call
37. Left padding keeps every completion at one fixed column
38. Attention mask stored with each group and reused on update
39. Step counter now equals the number of weight updates
40. Truncation now scores minus one, a full point below syntax error
41. Closing pays 0.3 only when no test passes, capped below one
42. Partial credit runs from 0.45 at one test to 0.75
43. Thinking budget starts at 768 and doubles per stage to 2048
44. An unclosed rollout is ended at the budget and scored truncated
45. Checkpoints and evals run every 20 steps, not 50 and 100
46. Ten checkpoints kept: five recent, older ones at doubling gaps
47. vLLM samples from the LoRA adapter written after each update
48. Each adapter handoff uses a new id and drops the previous
49. vLLM 0.10.1 needs CUDA 12.8, so torch uses cu128
50. Removed the false claim that bands come from measured pass rate
51. Removed the false claim that gradient division fixes update size
52. Run 16 reached step 67 with zero passes and reward 0.297
53. Close payment paid 0.3 for crashing code → make it flat 0.05
54. Runtime errors score 0.0 → below a clean failure, above syntax
55. Length penalty could invert test order → clamp above the lower tier
56. Run 17 died with 92 percent runtime errors, all name mismatches
57. Prompt now states the function name the hidden tests call
58. Twenty local replays: renaming the function passed ten of sixteen
59. A prompt counts as solved only when all eight rollouts pass
60. One lucky pass used to retire a prompt, leaving only unsolvable leftovers
61. Cold tests now read every row before a run can start
62. Prompt name extractor skips builtins, keywords and module attributes
63. Cold tests cover all 1014 rows before a GPU run starts
64. Synthetic reference solutions run through the real container harness
65. Prompt may never carry a test body or an expected value
66. Stage pools checked for repeats and train overlap
67. README now states the mastery rule and the stage exit cost
68. Dashboard GPU tiles now report GiB instead of thousands of MiB
69. Dashboard now plots the share of the current stage solved
70. A one minute stale window called live runs stopped
71. Dashboard rebuilt in React and Tailwind, one file, no CDN
72. Percent axes were 100x low, negative rewards binned wrongly
73. Stage completion reads every rollout, not the last four megabytes
74. Prompt now states how many arguments the first test passes
75. Four prompts failed every rollout for a missing argument count
76. Row 644 renamed to the lowercase function name the model writes
77. A control file now edits a run between steps, no restart
78. Drop removes a prompt from the pool and can advance the stage
79. Test edits are logged by step so analysis can split the run
80. A crashed resume left a fresh metrics file reading as running
81. Run status now requires a live rlvr run process for that arm
82. An nvidia-smi field of [N/A] is a missing reading, not zero
83. The system strip shows a dash when nvidia-smi cannot be read
84. A dropped unsolved prompt now leaves the stage pool total
85. Progress now counts prompt samples, not starting-batch steps
86. Resume froze the LoRA adapter, so AdamW got an empty list
80. Resume froze the LoRA adapter, so the optimizer had no parameters
81. Resumed run drops three never-passing prompts at step 121
82. Dashboard now checks for a live process, not just file age
83. Missing GPU readings render as a dash, never as zero
84. Dropped prompts leave the questions tab and keep their history
85. Run status now comes from the process, not file age
86. Proposed raising group size as the pool shrinks, not built
87. nvidia-smi utilisation means busy, not saturated, at any batch size
88. Each evaluation pauses training about ten minutes every twenty steps
89. Very easy stage emptied at step 161, medium started with 448
56. Prompt states the test arity directly → fewer invented signatures
57. One output rule ends the code-fence debate in the prompt
58. Checkpoints every 10 steps, fifteen kept per run
90. AdamW hides the gradient norm, so the weight update norm is logged too
91. A non-finite norm is recorded as null, not written as a crash
92. Run 17 doubled pass@1 from 12 to 24 of 100
93. The flat curve was an artefact of starting at step 20
94. 1e-3 diverges within two steps, answers pinned at the cap
95. 1e-4 is safe but no faster after twenty steps
96. Gradient norm cannot separate rates under AdamW, update norm can
97. Three dead very easy questions removed, held-out set pinned
98. All-linear adapters at rank 16, Qwen baseline at the same placement
99. Band scoring reads training prompts, so it is not held out
