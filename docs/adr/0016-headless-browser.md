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
  session. For every connection it resolves the host, refuses it unless every address is public, none is
  on one of this PC's own networks (its own addresses, and any network it reaches without a router, such
  as the home network's global IPv6 prefix; read from the routing tables at each connection) and the port
  is 80 or 443, and connects to the address it checked. Site-local IPv6 (`fec0::/10`), which Python counts
  as global, is refused too. HTTPS and secure WebSockets are tunnelled
  unread (a `CONNECT` must name a host and a port and nothing else). Plain HTTP goes on with
  `Connection: close`. Anything else is refused. Only the session's own browser may use the proxy: each
  proxy has a random credential, which agent-browser is given in its environment (a program's arguments
  are readable by every user of the PC) and sends after the proxy's `407`, compared in constant time; any
  other program on the PC is answered `407`. One session moves at most 200 MB. Its connections are capped
  at 48 at once, a slot taken before a thread is made for one; a request head must arrive whole within
  10 s, and at most 16 name lookups run at once, counting those the system never answered. A lookup that
  ends after the proxy stopped dials nothing. Chrome gets `--proxy-server` with
  `--proxy-bypass-list=<-loopback>`, so loopback goes through the proxy too, and a host-resolver rule that
  resolves no name itself. WebRTC may not send UDP past the proxy, and QUIC is off. A page's scripts,
  frames, images, WebSockets and redirects therefore reach neither Strata (127.0.0.1:8080), SearXNG
  (127.0.0.1:8888), the home network, a cloud metadata address nor agent-browser's own debugging port. A
  test with the real browser (`tests/test_web_browser_local.py`, `local`) checks this: a page whose
  scripts try every one of these reaches nothing.
- **Nothing of the owner's.** Each session gets a throwaway profile and download folder in the runtime
  folder (`$XDG_RUNTIME_DIR/jarvis/browser/session-<pid>-…`), deleted once the browser has ended. Its home
  and XDG folders are the session's too, with their Downloads read-only and named as the download folder,
  so nothing Chrome or agent-browser keeps or falls back to lands in the owner's home; Jarvis names the
  Chrome agent-browser installed (`--executable-path`), which agent-browser would otherwise look for under
  that home. Each launch has its own agent-browser session name and socket folder, so a web and a preview
  session in one Jarvis, or a launch after a close that did not end the last browser, never share one. A
  browser whose close did not end it keeps its folder until a later start finds it gone (agent-browser's
  record of it, in the socket folder, says so); the folders of a stopped Jarvis are swept the same way. A
  folder that could not be made whole is removed. agent-browser reads no config of its own
  (`--config` points at an empty file), so no extensions, plugins or saved logins. It runs headless,
  muted, with no display or sound-server variables and no version check, which would fetch past the
  proxy. Its action policy denies everything but the commands Jarvis sends: no scripts (`eval`), uploads,
  downloads, cookies, saved state, tabs, back or clipboard. Permission prompts are denied. A page can still
  start a download itself (a link, a redirect, a script); agent-browser has Chrome save every download in
  the session's download folder, which Jarvis makes read-only, so the download fails and nothing is saved.
- **Idle means closed.** The browser closes after `[browser] idle_close_s` (60 s) without use, and when
  the conversation is forgotten. agent-browser's own idle timeout, a minute later, ends it if Jarvis dies.
  Closing stops the proxy, which ends every connection still open, so a browser whose own close failed has
  no way out while it waits for that timeout. At exit Jarvis waits at most 15 s for a command still running
  before it stops the proxy alone.
  Between requests it uses no CPU and holds no memory.
- **Operations.**
  - `browser_open` opens only an address `web_read` would open: one a search in the conversation returned,
    or one the owner wrote (`WebOps.admit`). It also follows a link the open page shows, by its ref, and
    reads further parts of the page. It needs no yes.
  - `browser_screenshot` (the chat only) shows one screen of the page. It needs no yes.
  - `browser_act` clicks an element or types into a text box. It is `confirm`, its lowest class is
    `confirm`, and it is `always_ask`: the owner is asked even with `boundaries.bypass_consent` on, and
    `jarvis ask --yes`, which answers other questions by itself, asks this one on the terminal and refuses
    it when there is no terminal to answer on (`Hooks.confirm_always`). The
    question names the element, its site and the exact words ('Should I type "red shoes" into the
    searchbox "Search" on shop.example and press Enter?'), and for a link where it goes, since its name can
    say anything ('Should I click the link "More" on shop.example, which goes to pay.example/checkout?').
    Where a link goes is the browser's own address for it (`this.href`, from `snapshot -u`), resolved
    against the page's `<base href>` or its frame as a click would be, never joined by Jarvis.
    The yes is bound to that element on that page, and a link to that address: if the page changed
    meanwhile, or the browser closed while the owner was asked (a question open longer than
    `idle_close_s`), nothing is done and no new browser is started for it. agent-browser keeps one ref for
    one element of one document, so an element the page replaced has a new ref and is not clicked; the
    check runs a moment before the click, in the call before it, and a page that swaps the element in that
    moment for one of the same role and name gets the click (agent-browser then finds the element by role
    and name); binding the click to the node itself would take a script in the page, which the action
    policy keeps off. What a click or a key does on a page is the page's own to decide either way. Typing is
    `outbound`, at most 200 characters, never into a password, hidden or file field, which is refused
    before the question (in the accessibility tree a password field is a plain text box, so its input type
    is read). Enter goes to whatever has focus, so the field is focused again just before it, in the same
    call as the typing. Messages Jarvis writes for the model about an element never carry its name, which
    is the page's words. Once a click or typing is sent, a browser
    that does not answer, an answer too big or garbled to read, or a later step that fails (typing went in,
    Enter did not) is an uncertain outcome for the model, never a failure that invites a second yes; a check
    before acting that times out has done nothing and says so.
  - Everything the browser returns is web text (`Origin.WEB`), cut to the channel's room for one result.
    A page's text is the browser's own (`innerText`, which leaves out what CSS hides), less what
    `web_read` also leaves out: text under `aria-hidden` or the `hidden` attribute, or made transparent or
    of no size by an inline style, found in the page's rendered HTML (`web/page.py`, `without_hidden`).
    This is best effort, and the text stays web text either way: each hidden element's words are taken
    out after the last one taken out, in page order, whatever whitespace `innerText` put between them; a
    hidden text shorter than three characters stays (it could be part of a visible number), where the same
    words also stand visible before it those are taken instead, and a page whose hidden texts cost more than
    50 searched characters for each of its own is cut where the search stopped. Elements whose name is the
    words of an element inside a hidden one are not offered (the accessibility tree keeps what is only
    transparent or of no size); a name is at most 80 characters, and the list at most a third of a result.
  - Each call to agent-browser gets at most 40 s and ends 25 s before the request's own time limit
    (`agent.max_seconds`), so a click or typing with its checks, or a browser that does not answer and is
    closed, never runs past the request. Its answer is read as it comes, at most 8 MB, and its error
    stream is dropped; a call past either limit is stopped and the session closed.
- **Where.** `chat.browser` is on and `agent.browser` (voice and `jarvis ask`) is off by default. The tools
  are offered only when agent-browser is on PATH; `jarvis doctor` notes when it is missing, or has no
  Chrome to drive (`agent-browser install`). Their definitions take about 470 tokens of the chat's context.
  The chat's default `memory_tokens` went from 7000 to 6500 so that the default context still holds the
  whole remembered conversation.

### Previews of the owner's own builds (agreed with coding mode)

Coding mode (planned) will want to open what it built. That never goes through `file://`, which would let
page scripts read other files, and never through this web session. A preview is a separate session whose
proxy allows exactly one loopback port and refuses everything else, the public internet included. Behind
that port is a read-only file server Jarvis starts over one project folder, which coding mode hands over as
a resolved path inside the file write roots (`app.files_boundary`). It serves regular files under that
folder only, each with one name on disk (a hard link could be a file from outside), opened one folder at a
time from the folder it holds open since it started and following no link, so a file or folder a build
swaps for a link meanwhile is not followed. A type it does not know is sent as `application/octet-stream`,
never as a page, with `X-Content-Type-Options: nosniff`. It answers only its preview proxy: the proxy adds
a random token of the server's to every request it sends on (replacing any the page sent), and the server
also wants its own `127.0.0.1:<port>` as the request's host, so neither another program on this PC nor a
page in the owner's own browser renamed to 127.0.0.1 (DNS rebinding) reads the project. A program running
as the owner could still read the token from Jarvis's memory; only OS isolation would stop that. It stops
when the session is closed. Its text, errors and screenshots are
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
