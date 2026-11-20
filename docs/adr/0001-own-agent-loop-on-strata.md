# ADR: Run Jarvis on its own single-agent loop over Strata's API

**Status:** accepted
**Date:** 2026-11-20

## Context and problem statement

Jarvis needs an agent that reasons over spoken requests and calls tools. The owner requires that it use
no global agent instructions, skills or plugins except the Omarchy OS skills, and no subagents. It runs a
local quantized model (Qwen3.8-Flash-Next IQ2_XS on Strata), and a voice assistant has to answer within a
few seconds.

## Considered options

1. Point an existing agent harness (Claude Code, OpenCode, Hermes, Codex CLI) at Strata's
   OpenAI/Anthropic-compatible API and restrict it by configuration.
2. Write Jarvis's own small agent loop against Strata's OpenAI Chat Completions API.

## Decision outcome

Chosen option: "own agent loop", because the owner's restrictions then hold by construction: there is no
instruction, skill, plugin or MCP loader to switch off and no subagent tool to remove, whereas a harness
needs every one of those disabled and stays disabled only as long as its configuration and its next
release cooperate. The loop's prompt is a few thousand tokens instead of a harness's tens of thousands,
which matters for time-to-first-token on a local model. Boundaries are enforced in code before any tool
runs, which a harness's permission system cannot express for typed operations, provenance and
voice-bound confirmation.

### Consequences

- **Good:** full control of the prompt, tools, policy and latency; no dependency on a harness's release cycle.
- **Bad:** Jarvis owns tool-call validation, retries, step and time caps and the conversation state that
  a harness would provide; the phase-1 model evaluation has to prove the loop and model together.

## Related

- `docs/plan.md` — Architecture, Agent loop, Boundaries
- `docs/adr/0002-typed-operation-policy.md` — how the loop's tools are bounded
