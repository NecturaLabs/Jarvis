# ADR: Publish Jarvis under GPL-3.0-or-later

**Status:** accepted
**Date:** 2026-11-21

## Context and problem statement

Jarvis goes public, so it needs a licence. Both of its voices run through `piper-tts` 1.8.0
(OHF-Voice/piper1-gpl), which is GPL-3.0-or-later and is imported into the Jarvis process: Kokoro uses
Piper's espeak-ng phonemizer, and Piper voices use Piper itself (`docs/models.md`). A program that imports a
GPL library in-process and is distributed with it is distributed under the GPL's terms as a whole.

## Considered options

1. GPL-3.0-or-later for the whole project.
2. AGPL-3.0-or-later.
3. A permissive licence (MIT or Apache-2.0) for Jarvis's own files, with the combined program still bound by
   Piper's GPL.

## Decision outcome

Chosen option: "GPL-3.0-or-later", decided by the owner on 2026-11-21, because it is the licence the
distributed program carries anyway, so one licence describes every file and no reader has to work out how a
permissive licence and Piper's GPL combine. AGPL's network clause adds nothing for an agent that runs on its
owner's own PC.

### Consequences

- **Good:** one licence for the code, the docs and the distributed whole; `LICENSE`, `REUSE.toml`, the
  `license` field in `pyproject.toml` and `uv run reuse lint` all state it.
- **Bad:** every new dependency, model and voice must be compatible with GPL-3.0-or-later before it is added
  (`AGENTS.md`, "Open source"). Each Piper voice keeps its own licence, which Jarvis does not check.
- Contributions come in under the same licence (inbound = outbound, `CONTRIBUTING.md`).

## Related

- `docs/adr/0005-local-only.md`
- `docs/models.md`: the voices and their licences
