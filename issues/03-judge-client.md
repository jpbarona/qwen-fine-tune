# 03: Judge client

**What to build:** A frozen judge client for OpenCode Go that scores output code
and, separately, the reasoning trace. This feeds arms 2, 3, and 4.

**Blocked by:** 02.

**Status:** ready-for-agent

- [ ] Requests go to the OpenCode Go OpenAI-compatible endpoint with `deepseek-v4.1-flash` at normal reasoning effort.
- [ ] Output judge prompt contains the problem and code only, never the tests.
- [ ] Process judge prompt contains the problem and reasoning trace only, never the tests.
- [ ] Responses parse into a bounded score and a short reason.
- [ ] Transient failures retry with backoff.
- [ ] Identical requests hit the cache.
- [ ] The key is read from the environment and never written to disk or logs.
- [ ] A missing key fails before training starts.
