# 22: A single pool, no curriculum stages

**What to build:** Training draws from one pool. The stage build, the stage
ladder and the thinking budget ladder are deleted from the training path. Records
carry the existing unstaged value. The difficulty band stays in the data for
reporting, and no training code reads it.

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent

- [ ] A run trains from the pool file without building stages.
- [ ] The thinking budget comes from one config value.
- [ ] Step and eval records carry the unstaged stage value.
- [ ] Checkpoints written before this change still resume, since the stage fields
      become optional.
- [ ] The dashboard renders an unstaged run without an error.
- [ ] `eval-band` and the band column of the label file still work.
- [ ] Tests cover a run with no stages and a resume with no stage position.
