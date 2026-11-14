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
