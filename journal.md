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
