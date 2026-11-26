# ADR: Give Jarvis a headless browser, fenced by a local proxy and a yes for every click and typing

**Status:** accepted
**Date:** 2026-11-26

## Context and problem statement

The owner asked for Jarvis to have the `agent-browser` program, headless only. `web_read` reads only what a
server sends. A page built by scripts comes back empty, and a page that answers only after a search box or a
button cannot be used at all. A real browser can do both, and it can show the chat a picture of the page.

A real browser is also the most dangerous tool Jarvis has had on the web. The page decides what it loads:
its scripts can fetch, open WebSockets, frame other pages, redirect and use WebRTC. `web/guard.py` checks
every connection Jarvis's own HTTP client makes, but it cannot sit inside Chrome. Clicking can send a form,
post, buy or log in. Typing sends the owner's words to whoever runs the page, which ADR 0007 never
accepted: there, a page could only steer at most 200 characters into a search that goes to the search
engines.

## Considered options

- Where the browser may go:
  - agent-browser's own domain allowlist, which blocks by name, not by address;
  - a local proxy that applies `guard.py`'s public-address check to every connection, with Chrome forced
    through it;
  - a network namespace with no route out but the proxy (needs `unshare`, which Jarvis does not run for
    this).
- Acting on pages:
  - read-only (open, follow links, read, screenshots);
  - clicks and typing as ordinary `confirm` actions, which `bypass_consent` runs unasked;
  - clicks and typing that always ask, consent bypassed or not.
- Where it is offered: both channels; the chat only; neither until switched on.

## Decision outcome

Chosen on 2026-11-26 at the owner's request, which left the design to the builder ("do what you think is
best"), and with the owner's config split: the chat is its own setup, and the voice a short-term setup for
quick commands.

- **One way out.** `web/proxy.py` (`EgressProxy`) listens on 127.0.0.1 on a free port for each browser
  session. For every connection it resolves the host, refuses it unless every address is public and the
  port is 80 or 443, and connects to the address it checked. HTTPS and secure WebSockets are tunnelled
  unread. Plain HTTP goes on with `Connection: close`. Anything else is refused. One session moves at most
  200 MB, and its connections are capped at 48 at once. Chrome gets `--proxy-server` with
  `--proxy-bypass-list=<-loopback>`, so loopback goes through the proxy too, and a host-resolver rule that
  resolves no name itself. WebRTC may not send UDP past the proxy, and QUIC is off. A page's scripts,
  frames, images, WebSockets and redirects therefore reach neither Strata (127.0.0.1:8080), SearXNG
  (127.0.0.1:8888), the home network, a cloud metadata address nor agent-browser's own debugging port. A
  test with the real browser (`tests/test_web_browser_local.py`, `local`) checks this: a page whose
  scripts try every one of these reaches nothing.
- **Nothing of the owner's.** Each session gets a throwaway profile and download folder in the runtime
  folder (`$XDG_RUNTIME_DIR/jarvis/browser/session-<pid>-…`), deleted when the browser closes. Folders a
  stopped Jarvis left behind are deleted at the next start. agent-browser reads no config of its own
  (`--config` points at an empty file), so no extensions, plugins or saved logins. It runs headless,
  muted, with no display or sound-server variables and no version check, which would fetch past the
  proxy. Its action policy denies everything but the commands Jarvis sends: no scripts (`eval`), uploads,
  downloads, cookies, saved state, tabs, back or clipboard. Permission prompts are denied.
- **Idle means closed.** The browser closes after `[browser] idle_close_s` (60 s) without use, and when
  the conversation is forgotten. agent-browser's own idle timeout, a minute later, ends it if Jarvis dies.
  Closing stops the proxy, which ends every connection still open, so a browser whose own close failed has
  no way out while it waits for that timeout.
  Between requests it uses no CPU and holds no memory.
- **Operations.**
  - `browser_open` opens only an address `web_read` would open: one a search in the conversation returned,
    or one the owner wrote (`WebOps.admit`). It also follows a link the open page shows, by its ref, and
    reads further parts of the page. It needs no yes.
  - `browser_screenshot` (the chat only) shows one screen of the page. It needs no yes.
  - `browser_act` clicks an element or types into a text box. It is `confirm`, its lowest class is
    `confirm`, and it is `always_ask`: the owner is asked even with `boundaries.bypass_consent` on. The
    question names the element, its site and the exact words ('Should I type "red shoes" into the
    searchbox "Search" on shop.example and press Enter?'), and for a link where it goes, since its name can
    say anything ('Should I click the link "More" on shop.example, which goes to pay.example/checkout?').
    The yes is bound to that element on that page, and a link to that address: if the page changed
    meanwhile, or the browser closed while the owner was asked (a question open longer than
    `idle_close_s`), nothing is done and no new browser is started for it. Typing is `outbound`, at most
    200 characters, never into a password, hidden or file field.
  - Everything the browser returns is web text (`Origin.WEB`), cut to the channel's room for one result.
  - Each call to agent-browser gets at most 40 s and ends 25 s before the request's own time limit
    (`agent.max_seconds`), so a click or typing with its checks, or a browser that does not answer and is
    closed, never runs past the request.
- **Where.** `chat.browser` is on and `agent.browser` (voice and `jarvis ask`) is off by default. The tools
  are offered only when agent-browser is on PATH; `jarvis doctor` notes when it is missing. Their
  definitions take about 470 tokens of the chat's context. The chat's default `memory_tokens` went from 7000
  to 6500 so that the default context still holds the whole remembered conversation.

### Previews of the owner's own builds (agreed with coding mode)

Coding mode (planned) will want to open what it built. That never goes through `file://`, which would let
page scripts read other files, and never through this web session. A preview is a separate session whose
proxy allows exactly one loopback port and refuses everything else, the public internet included. Behind
that port is a read-only file server Jarvis starts over one project folder, which coding mode hands over as
a resolved path inside the file write roots (`app.files_boundary`). It serves regular files under that
folder only, follows no symbolic link out, and stops with the session. Its text, errors and screenshots are
web text too. Built: `web/preview.py` (`FolderServer`, `preview_browser`), `EgressProxy(local_port=…)` and
`Browser(origin=…)`; the operation that asks for a preview comes with coding mode.

### Consequences

- **Good:** pages built by scripts can be read, search boxes used, and the chat can see a page. The browser
  cannot reach this PC or the home network, carries nothing of the owner's, and asks before anything on a
  page is clicked or typed.
- **Bad:** sites see a real Chrome (agent-browser's Chrome for Testing) and this PC's public address, as a
  visit would show them. Following links needs no yes, so a page can steer which link is opened next. Like
  ADR 0007's searches, that is a low-bandwidth channel to the sites linked, not to the page's author unless
  the author runs the linked site. The chat's remembered conversation is 500 tokens smaller by default. The
  proxy does not pass on a second request on a plain-HTTP connection to another host; Chrome does not
  reuse such a connection, and the request would still reach only the first, checked public address.
- Not built in this version: scrolling a page that loads more as it scrolls, more than one tab, and pages
  that need a login.

## Related

- `docs/adr/0002-typed-operation-policy.md`, `docs/adr/0007-outside-services-for-answers.md`,
  `docs/adr/0009-lookups-without-a-yes-and-searxng-idle.md`
- `docs/configuration.md` (`[browser]`, `[chat]`, `[agent]`), `docs/privacy.md`
