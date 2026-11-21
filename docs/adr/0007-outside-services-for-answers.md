# ADR: Reach outside services to answer questions: SearXNG's engines, Open-Meteo and Arctic Shift

**Status:** accepted
**Date:** 2026-11-21

## Context and problem statement

The owner wants Jarvis to tell the time and the weather, research online and search Reddit. Only the time
can be answered on this PC. [ADR 0005](0005-local-only.md) keeps the model, speech and the wake word
local and names Spotify as the one outside service; answering these questions needs more, and each one
sends something to someone. Reddit refuses unauthenticated and automated clients, and new apps need its
approval to use the official API.

## Considered options

- Search: a local SearXNG; the Brave Search API (an account and a key); scraping DuckDuckGo.
- Weather: Open-Meteo (free for non-commercial use, no account or key, CC BY 4.0 data); wttr.in (one
  community server, often rate-limited); asking SearXNG and reading the result.
- Reddit: SearXNG limited to reddit.com, then the Arctic Shift archive to read a thread; SearXNG alone
  (no comments); the official Reddit API with an approved app and OAuth tokens.
- Reading pages: trafilatura (the best extraction, but its dependency `regex` carries the CNRI-Python
  licence, which SPDX treats as GPL-incompatible); jusText (BSD-2-Clause); an extractor of our own.

## Decision outcome

Chosen by the owner on 2026-11-20 and 2026-11-21: a local SearXNG for web search (issues #10 and #20),
Open-Meteo for weather and for finding a place's time zone, SearXNG plus Arctic Shift for Reddit, and
trafilatura for pages, with `regex` accepted as a named exception in `tools/check_licences.py`.

What leaves the PC, and to whom:

- **SearXNG** runs in a container on `127.0.0.1` (`deploy/searxng/`). It forwards the search, with no
  cookies and no account, to the search engines it is configured with, which see the search and this
  PC's public address.
- **Open-Meteo** gets the home coordinates (or the home place name, looked up once per run) and any
  place the owner names.
- **Arctic Shift** gets the ids of the Reddit threads a search found.
- **Web pages**: Jarvis opens only a page a search in the same session returned, through a client that
  reaches only public addresses on ports 80 and 443, checked at every connection and every redirect
  (`web/guard.py`).

No request carries anything that names the owner, the PC or the project: the user agent is generic.

### Consequences

- **Good:** none of these needs an account, a key or money, and nothing is sent until the owner asks a
  question that needs it. Each operation can be refused with a `deny` class in `[boundaries.classes]`.
- **Bad:** searches, places and thread ids leave the PC, and ADR 0005's "Spotify is the one outside service"
  no longer holds. Search results depend on outside engines that rate-limit or block (DuckDuckGo answers
  with a CAPTCHA at times). Arctic Shift is a third-party copy of Reddit and can lag behind it. SearXNG is
  one more container to run: idle, it measured 0.10 % CPU and 134 MiB (`docker stats`, 2026-11-21, on
  the reference PC), so it is not quite zero CPU while Jarvis waits.
- Text from all of these is marked as web data: it never raises an operation's class, and while it is in
  the conversation any action needs a spoken yes. Weather and time answers are built only from numbers,
  codes and checked identifiers, so they do not count as outside text.

## Related

- `docs/adr/0002-typed-operation-policy.md`, `docs/adr/0005-local-only.md`
- `docs/privacy.md`, `docs/configuration.md` (`[weather]`, `[research]`)
