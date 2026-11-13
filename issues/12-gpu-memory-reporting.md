# 12: GPU memory reporting

**What to build:** Each step reports where GPU memory is going, split into live,
allocator cache and fragmentation, instead of one total. The run configures the
allocator so segments grow rather than fragment.

**Blocked by:** 11.

**Status:** ready-for-agent

- [ ] Each metric record carries live, cached and fragmented GPU memory in MiB, read from the allocator inside the process.
- [ ] The allocator is configured with expandable segments when the run starts.
- [ ] The three figures sum consistently with the device total, or the mismatch is visible.
- [ ] The fake backend supplies zeros and the dashboard shows the panel without a GPU.
- [ ] The dashboard separates the three lines, and the total device figure stays visible.
- [ ] The whole suite passes.
