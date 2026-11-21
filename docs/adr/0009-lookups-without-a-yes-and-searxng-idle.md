# ADR: Keep read-only lookups free of a yes after web text, and keep SearXNG running while idle

**Status:** accepted
**Date:** 2026-11-21

Settles the two points [ADR 0007](0007-outside-services-for-answers.md) left open to the owner. The owner
asked for the open decisions to be researched and settled; the research is below.

## Context and problem statement

ADR 0007 made every search and lookup a read (`side_effect=False`) that runs without a yes, even while text
from a web page is in the conversation. A page can then steer the model into putting up to 200 characters of
the conversation in a search (80 in a place name). It left open whether to ask for a spoken yes for every
lookup while outside text is in the conversation.

It also measured the SearXNG container at 0.10 % CPU and 134 MiB while idle, against Jarvis's rule of no CPU
while idle, and left open whether to accept that.

## Considered options

For lookups after web text:

- Keep them free of a yes (as built).
- Ask for a spoken yes for every search or lookup while outside text is in the conversation.

For SearXNG while idle:

- Keep it running (`restart: unless-stopped`, as built).
- Start it on demand with systemd socket activation and stop it when idle.

## Decision outcome

Accepted on 2026-11-21: lookups stay free of a yes, and SearXNG keeps running.

- **Lookups.** Guidance read that day puts the risk in an agent that, at the same time, holds private data,
  reads untrusted text and can send data out
  ([the lethal trifecta](https://simonwillison.net/2025/Jun/16/the-lethal-trifecta/),
  [Meta's Rule of Two](https://ai.meta.com/blog/practical-ai-agent-security/)). Here the channel out reaches
  search engines and Open-Meteo, not the page's author: a page can open no address of its own choosing, and
  only an address a search in the session returned can be read. A yes for each lookup would ask the owner to
  judge, by ear, a query they cannot see whole.
- **SearXNG.** Its idle use is less than the 1.4 % of one core `jarvis.service` uses while listening for the
  wake word ([models.md](../models.md)). Starting it on demand would save 134 MiB beside a model that holds
  about 38 GB, and would add about 1.4 s to the first search after it stopped (a fresh container answered
  `/healthz` after 1.36 s). No SearXNG guide read that day recommends starting it on demand.

### Consequences

- Good: research does not ask for a yes at every step, and the first search is as fast as the others.
- Bad: the low-bandwidth channel ADR 0007 describes stays open, and SearXNG uses a little CPU while Jarvis is
  idle.
- Revisit the lookups if Jarvis gains a fetch of an address of its own choosing, a lookup whose logs the
  page's author can read, or private data such as mail or a calendar in the conversation; the fix then is a
  spoken yes for lookups while outside text is in the conversation.

## Related

- [ADR 0007](0007-outside-services-for-answers.md), [ADR 0002](0002-typed-operation-policy.md)
- `deploy/searxng/`, [privacy.md](../privacy.md)
