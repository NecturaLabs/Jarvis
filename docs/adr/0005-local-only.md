# ADR: Run entirely on this PC, with no hosted backend

**Status:** accepted
**Date:** 2026-11-21

## Context and problem statement

A hosted model would be larger and quicker to start, but it would send what the owner says to a third party,
cost money per request and depend on an account. The owner wants Jarvis to run locally and be free to use and
maintain, and to be publishable for others on the same terms. The planning session also looked at a hosted
service (Foundry or Azure speech) for voice activation, and chose the local LiveKit wake word instead.

## Considered options

1. A hosted model backend, either primary or as a fallback when the local model is slow or unavailable.
2. Local only: the model on Strata, speech-to-text on Voxtype, the wake word on this machine.

## Decision outcome

Chosen option: "local only", because nothing the owner says then leaves the PC and there is no bill or account
to maintain. The model (Strata), speech-to-text (Voxtype), the wake word and the voice all run locally. Config
keys for a hosted model (`model.provider`, `fallback`, `remote_base_url`, `remote_model`) were added in commit
`b4c6d69` and removed in `a62872b` before any code read them. Spotify's own API is the one outside service
Jarvis calls, because controlling Spotify needs it.

### Consequences

- **Good:** no data leaves the machine except the Spotify requests; no running cost.
- **Bad:** the model has to fit this PC. On the reference machine a loaded model holds about 38 GB of RAM and
  7.3 GiB of VRAM, a cold load takes about 107 s, and an answer from the model takes seconds
  (`docs/models.md`). Plain commands avoid most of that (`docs/adr/0004-model-free-path-for-plain-commands.md`).
- A product rule in `AGENTS.md` keeps a hosted or paid backend from coming back unasked.

## Related

- `docs/adr/0001-own-agent-loop-on-strata.md`
- `docs/configuration.md`: `[model]` and `[strata]`
