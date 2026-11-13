# 13: Curriculum stages and mastered-question removal

**What to build:** Questions arrive in four ordered bands, very easy, medium,
hard, very hard. Each stage trains on 80 percent of its band and holds out 20
percent for evaluation, and carries over a share of the previous band. A question
leaves the pool once the model fully passes it. The run folder and dashboard show
which stage is active.

**Blocked by:** 10, 11.

**Status:** ready-for-agent

- [ ] A pure stage builder takes labelled rows, band definitions and a carry-over ratio, and returns ordered pools plus the held out slice per stage.
- [ ] Bands are ordered very easy, medium, hard, very hard, and a run starts at a chosen stage.
- [ ] Each stage splits 80 percent train and 20 percent held out, with a fixed seed.
- [ ] The carry-over ratio defaults to 0.2 and takes the previous band, not the next.
- [ ] Rollout records carry the stage name and metric records carry the stage.
- [ ] A question leaves the pool after a full pass, and an empty pool advances the stage or ends the run with a clear message.
- [ ] The fake run moves from the first stage to the second when the first is exhausted.
- [ ] The dashboard shows the active stage, the step count and pass rate per band.
- [ ] The whole suite passes.
