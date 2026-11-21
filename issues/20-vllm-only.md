# 20: vLLM is the only sampling engine

**What to build:** Every rollout, in training and in the screen, goes through
vLLM. The Hugging Face sampling path, the engine config key and the engine gate
stop existing.

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent

- [ ] A config with no engine key trains and screens on vLLM.
- [ ] The engine config key is rejected as unknown when present.
- [ ] The Hugging Face generate path is deleted, including its tests.
- [ ] The base model still loads in process for the optimizer, and sampling goes
      through the vLLM backend.
- [ ] One optional dependency extra installs vLLM, torch, transformers and peft
      at versions that work together.
- [ ] The suite passes with no import of a Hugging Face generation helper in the
      sampling path.
- [ ] README install steps describe one environment.
