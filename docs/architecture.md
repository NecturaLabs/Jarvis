# Architecture

This is a map of how Jarvis is put together. If it differs from the code, the code wins; please fix
this file. The decisions behind the shape are in [plan.md](plan.md) and the [ADRs](adr/); this file
does not repeat them.

Jarvis is one Python process, `jarvis run`, started as a systemd user unit. One thread owns the
microphone and runs the whole request path: wake word, speech capture, transcription, the agent, the
spoken reply. There is one agent and no subagents. Strata runs as a separate service on loopback
by default. It can also serve the model from another machine you own. Every tool call goes
through a typed operation and a policy decision before it runs, and
every step is written to an audit log. The tools are Spotify's, and the time, the weather, web search,
Reddit and a headless browser (`web/`).

```text
 jarvis.service (jarvis run)                          jarvis-strata.service
 ┌──────────────────────────────────────────┐        ┌──────────────────────┐
 │ audio.py  pw-record ─▶ wake ─▶ VAD       │        │ strata serve         │
 │ voice.py  VoiceLoop (main thread)        │  HTTP  │ 127.0.0.1, API key,  │
 │    │  voxtype transcribe (child process) │ ─────▶ │ lazy load, idle      │
 │    ▼                                     │ 127.0.0.1  unload            │
 │ agent.py  Agent ─▶ policy.py ─▶ Registry │        └──────────────────────┘
 │    │          shortcuts.py (no model)    │
 │    ▼                                     │        Spotify Web API (HTTPS)
 │ spotify/ops.py ─▶ spotify/api.py ────────┼──────▶ and the Spotify app
 │ web/ops.py ─▶ web/searxng.py ────────────┼──────▶ SearXNG container, 127.0.0.1:8888
 │    ├─▶ web/openmeteo.py, web/arctic.py ──┼──────▶ Open-Meteo, Arctic Shift (HTTPS)
 │    ├─▶ web/page.py ─▶ web/guard.py ──────┼──────▶ public web pages only
 │    └─▶ web/browserops.py, web/browser.py│
 │         agent-browser, headless Chrome    │
 │           └─▶ web/proxy.py on loopback ───┼──────▶ public web pages only
 │    audit.py ─▶ audit.jsonl               │
 │ audio.py  Speaker ─▶ pw-play             │
 └──────────────────────────────────────────┘
```

In remote mode the HTTP arrow leads to the owner's Strata backend, normally through an SSH
tunnel. The client still owns wake, speech-to-text, the agent loop, policy, tools, audit and
speech. It sends a bearer key and the model's conversation input, which can include tool results
and pictures. Backend-only setup installs the Strata unit under the user's `default.target`.
The remote voice unit has no dependency on Strata. See
[ADR 0020](adr/0020-model-on-another-machine.md) and [privacy.md](privacy.md).

## Modules

All paths are under `src/jarvis/`. The tables say what a module owns and what it must not do.

### Voice pipeline

| Module | Owns | Must not |
|---|---|---|
| `voice.py` | `VoiceLoop` and its `State`: wake, capture, transcribe, think, confirm, speak, follow-up; the PC-audio listener; `speakable` and `spoken_question`. While paused, a request in flight ends without an apology and says nothing more, no follow-up window opens and nothing heard in an open one is taken, and PC audio frames are read but not listened to; after `jarvis hush` the request in flight carries on in silence, and a sound still being made for it (a song) is stopped (`Hooks.hushed`). While an operation makes a sound, the loop listens for the wake word as it does while speaking (`Hooks.making_sound`), so the owner can speak over a song still being made. A microphone whose capture ended is reopened (state `no-mic` meanwhile); a jump of the boot clock ahead of the monotonic one is a suspend, after which the loop listens afresh and calls `after_sleep`; with `audio.listen_when_locked` off, no request is taken at the lock screen (wake word, follow-up, spoken over a reply, PC audio). A wake word said while your Voxtype dictation runs (or spoken over a reply then) is ignored with one `wake_ignored` audit event and no chime, and dictation that starts while a request is being captured drops that request without a word (the chime has already played, as it starts with the wake word; Voxtype's recording is never touched), recorded as a `wake_ignored` event with the reason `dictation started mid-request`; a running dictation or meeting, or a status that cannot be read, keeps follow-ups off. Speech ends below `audio.vad_neg_threshold`, and a request after the wake word needs 0.15 s of speech; 320 ms from before speech is detected is kept, so a soft first sound is not cut off. A request cut short by a lost microphone or speaker ends without an apology. Between requests it takes the due reminders and scheduled requests (`due`, `JobStore.take_due`), holding them while you dictate or a meeting runs: a reminder is said after the chime, a scheduled request goes to the agent as your own request, one that came too late is only audited (`schedule_missed`), and none opens a follow-up window. | Decide what an operation may do (that is `policy.py`). Touch a device directly; it uses `audio.py` objects. |
| `audio.py` | The `pw-record` microphone and per-app capture (`Microphone`, `AppAudio`: each stream the apps play by its own pinned recorder, mixed), `PausableMic` (lets go of the microphone for `jarvis pause`: an open reader ends with `MicPausedError`, a new one waits for `resume`; `recover` starts a capture again after it ended on its own, waiting 1 s and doubling up to 30 s; a `Microphone` with `stall_s` counts a capture that sends nothing that long as ended), whether the Omarchy lock screen is up (`lock_state`, asked of `omarchy-shell`), the wake word (LiveKit's `WakeWord`, or `AzureWake` talking to the helper), Silero VAD, `voxtype transcribe`, whether your dictation or a meeting is running (`VoxtypeWatch`, one long-lived `voxtype status --follow`, read off the voice thread), the `pw-play` speaker (sentences, and sounds such as a song queued among them with `queue_sound`, played whole with their silences and checked for the wake word in chunks of 2.5 s, each with the words timed in it), voice selection, and `run_singer`, which runs SoulX-Singer's program for one song in a process group of its own, killed with everything it started once the program ends, runs too long or its song is stopped, and reads its stderr without blocking, so no child left holding the pipe can hold Jarvis. | Edit Voxtype's config or drive its daemon. It runs only `voxtype transcribe`, with Jarvis's own settings file, and reads `voxtype status` (followed, never polled). Start the Azure helper any way but `azure_wake_command` (no network). |
| `azure_wake_helper.py` | MIT. A separate program, run with its own Python, that feeds audio from standard input to an Azure custom keyword model and prints `wake`, with 1.5 s of silence put into the engine's stream first and after each keyword ([ADR 0010](adr/0010-azure-keyword-wake-word.md)). | Be imported by Jarvis, import Jarvis, or run with a network. |
| `espeak.py` | Espeak-ng from the `piper-tts` wheel, for everything that phonemizes: `clauses` for Kokoro, and `sounds`, how a name sounds, for Spotify's name matching (kept per name). Every call holds Piper's own espeak lock. | Call espeak without that lock: it is not safe from two threads. |
| `kokoro.py` | Kokoro text to speech on ONNX Runtime, phonemized by espeak-ng (`espeak.py`); with `audio.kokoro_phoneme_map`, espeak's phonemes are rewritten first (`phonemes.py`). `timed` also gives how long each phoneme symbol lasts (the model's second output), for singing. A model file loads once per process, whichever voices use it. | Play audio. It returns samples. |
| `phonemes.py` | The table that rewrites espeak's English phonemes into the symbols Kokoro was trained on (from misaki). | Import anything but the standard library: it is a table and two small functions. |
| `speechtest.py` | `jarvis speech-test`: the cards, recording a private set (WAVs and `set.jsonl`, 0700/0600) through a capture the CLI passes in, scoring it with a transcriber (word errors, names heard, time), and comparing two scores pair by pair (exact sign and McNemar tests). | Open the microphone or run Voxtype itself: `cli.py` passes both in. Send anything anywhere. |
| `levels.py` | The sound levels of one capture (`levels`): peak, speech level, room floor, signal-to-noise ratio, how loud its first 20 ms already were, and the share of clipped samples. `voice.py` logs them as one `levels:` line before each transcription, numbers only. | Keep or log the sound itself. |
| `sentences.py` | Where a sentence of a reply ends (`sentence_end`, `split_sentences`), so speech starts before the reply is whole; every end is also one for `agent.claims_action`. | Decide what may be spoken; the agent's reply stream does. |
| `models.py` | The pinned sha256 of each model file Jarvis loads, for `jarvis doctor`. | Download anything or fail a request over a digest. |

### Agent and policy

| Module | Owns | Must not |
|---|---|---|
| `agent.py` | `Agent`: one request into at most `max_steps` model calls and `max_tool_calls` tool calls within `max_seconds` (`[agent]` for speech, `[chat]` for the chat window); each answer capped at `reply_tokens` (a reply the cap ends is kept and says so); the request's thinking effort, starting at `thinking` and raised only by a `think_harder` call, up to `thinking_max`, which is not counted as an action and whose repeat is not a stuck loop; with `tools = "routed"`, the tools a call is offered (the core, then the groups `use_tools` asked for in this request, in the registry's order), and a call to a tool of a group not offered when it was written is not run, and is no action tried (the model is told which group to ask for; a plain command matched in code is never refused); the reply call (`no_action`), refused by neither the cap on tool calls nor the time, so a request that used them up still gets its answer; the oldest tool results of a long request cleared to fit the context window; the remembered conversation (`memory.py`), with each request's tool calls and results where `memory_detail` is `full` (`kept_work`), condensed after a reply that made it outgrow its room by leaving out the oldest work, then by the model's summary (while the reply is spoken, never inside a request's time), and said in one line through `Hooks.condensed`, after which `context_use` shows what the next request starts from; provenance on message parts (pastes longer than `MAX_TRUSTED_PASTE` together are `paste` data while Jarvis asks first); every question's words made printable; running each call through validation, policy, the cap on actions in a window (`ratelimit.py`), confirmation and audit, an operation's `hashed` arguments recorded only as a SHA-256 and a length; neighbouring calls of one model message to `concurrent` (read-only) operations decided and audited one by one, then run side by side (at most `MAX_SIDE_BY_SIDE` at once, each with the request's context), their results added and audited in the calls' order; every other call, and every action, run on its own, in order; the check that a reply does not claim an action no tool did; an answer without words is "Done." after an action that succeeded, and otherwise asked for once more in words, then said to have no answer. A request stopped by the owner (Esc in the chat, speech over a reply) is still remembered, with what it looked up and a note that it was stopped, and its audited `reply` says `stopped`. A tool call the reply cap cut off is never run: the model is told how much one answer holds and to send long text in parts (with `append` where the tool has it), the cut text is not kept, and a second cut call ends the request with that said. Thinking reaches the stream as wordless `Delta(thinking=True)` pieces, so a stop takes effect while the model still thinks. `forget` and the idle expiry, checked before every request (one from PC audio too), also reset the registry, so the web operations forget what searches returned. | Run an operation that policy refused, or more side effects than `boundaries.max_actions` allows. Talk to the owner except through `Hooks`. |
| `confirmkey.py` | `ConfirmKey`: at most one `strong` action waiting for the key, bound to its action hash; `arm` once the question was heard, `press` from the socket, `disarm` says whether it was pressed in time. A press with nothing waiting, or under 0.3 s after `arm`, allows nothing and holds the next `arm` off for 30 s. Each press and each refused wait is audited (`confirm_key`). | Let a press carry over to another action, or allow anything after a timeout. |
| `policy.py` | `Origin` (provenance; Omarchy's guides are `os_skill` text, outside like a web page), `OpClass`, `decide` (the decision table: outside text raises side effects to `confirm`, and file text or a shell command's output raises `outbound` operations to it; an owner's class below an operation's `lowest_class` is raised to it), `action_hash`, `is_yes`. Plain data and pure functions. | Do I/O, or let text from outside the owner's speech raise an operation's class. |
| `operations.py` | `Operation` (with its `preview` of what a write puts in, and the `hashed` arguments the audit log keeps only as digests), `Registry` (with the `resets` that clear what operations remember of a conversation), `Result`, argument validation against a flat JSON schema, the error types; `inert` and `cut_inert`, which spell out outside text's angle brackets and size a result as the agent will send it, so a part's line saying where the next starts is never cut; `request_scope` and `current_request`, which tell an operation which request it runs in, whether it was heard in PC audio, the owner's own words of it (`Request.words`: what they typed, and what they typed into it while it ran) how much of its time is left (`Request.time_left`: the request's own limit, so time spent waiting for a yes is not counted) and whether it was cut short (`Request.cancelled`), nothing more of it can be heard (`Request.hushed`) or it makes a sound (`Request.making_sound`); `OperationStoppedError`, for an operation stopped part way by those signals. | Accept an unknown argument or an unknown operation. |
| `ratelimit.py` | `ActionRate`: at most `boundaries.max_actions` side-effecting actions in any `actions_window_s`, counted in `actions.json` in the state folder (0600, no link followed, locked while read and written, the lock waited for 2 s at most; written over, then cut to length) so the voice service, the chat and `jarvis ask` share one count; a process that cannot use the file keeps its own; `full` looks without counting. | Lift the cap: a damaged or unusable file restarts or narrows the count, never removes it. Block a request forever on the lock. |
| `shortcuts.py` | Plain commands ("pause", "volume 40") that skip the model: normalising and matching a request to one call. | Guess. A request that fits two calls, or none exactly, matches nothing. Take a new phrasing ([ADR 0013](adr/0013-free-speech-no-new-shortcuts.md)). |
| `audit.py` | `AuditLog`: append-only JSONL, mode 0600, rotated at 5 MiB with five older generations kept (`.1` to `.5`). Every Jarvis process writes it under `flock` on the open log, so the size check, the rotation and the write are one step across processes: a full log is rotated once and no event is lost to another process's write. | Fail the caller; a write error goes to the service log. |
| `app.py` | `build`: assembles the model client, Spotify, operations, prompt, audit log and agent from the config; the `no_action` operation; `stay_quiet` (voice only), refused in a request that read outside text; `think_harder`, offered when `thinking_max` is above `thinking`; the speech-to-text vocabulary hint. | Hold state of its own beyond the objects it returns. |
| `prompts/core.md`, `prompts/voice.md`, `prompts/chat.md` | The shipped system prompt: what Jarvis does and how it acts (`core.md`), then how it replies, spoken (`voice.md`) or written in Markdown within the chat's reply budget (`chat.md`). `app.py` fills in the name, tone, skills and word budget. | Be the only guard: boundaries are enforced in code, not in this prompt. |
| `prompts/skills/*.md`, `skills.py` | Jarvis's skills: instructions for a kind of task, the shipped ones (web research, game look-ups) and the owner's own in `~/.config/jarvis/skills/` (the folder and each file theirs alone, no link followed, size-capped, read once per process; never readable or writable through the file operations; [ADR 0024](adr/0024-owner-skills.md)), listed by name and description in the prompt and read whole through the safe `read_skill` operation. | Read anything but the shipped skill files and the owner's skill folder, or widen what an operation may do. |
| `toolgroups.py`, `router.py` | The tool router ([ADR 0022](adr/0022-tool-router.md)): every operation belongs to one group (`toolgroups.py`, one list of names; a test fails for an operation none names) with a one-line summary, and the core tools every request is offered. `Registry.groups()` and `tool_schemas(loaded)` give the groups present and the tools of the core plus the loaded groups in one fixed order. `router.py` has `use_tools`, which names the groups a request needs and changes nothing itself, and the toolbox of the system prompt (each group, its summary and tool names) that a routed agent has. | Match the owner's words against a group, or run a tool. |
| `osskills.py` | Omarchy's guides for agents (`/usr/share/omarchy/default/agents/skills/`): listed at start (real folders and Markdown files only, root's and writable by nobody else, each opened from `/` with no link followed; at most `MAX_CATALOG_CHARS` in all), shown in the prompt by each description's first sentence, and read one file at a time, in parts, through the safe `os_skill` operation as `os_skill` origin text. | Read a path the model wrote, follow a link, or treat the guides' commands as something Jarvis may run. |
| `memory.py` | `Conversation`: the remembered turns and a summary of older ones, measured in estimated tokens against a budget (a message's tool calls counted with their arguments); with `memory_detail = "full"` a turn keeps its work, the tool calls and results between its request and reply as the model last read them, and the oldest work goes first when room runs short (`shed_work`); a turn that drew on outside text is kept as `<data>`; the chat's conversation is saved 0600 in the state folder between sessions. | Decide what is trusted: the agent marks outside turns and keeps side effects behind a yes while one is remembered. |

### Operations and Spotify

| Module | Owns | Must not |
|---|---|---|
| `spotify/ops.py` | `SpotifyOps`: the fourteen `spotify_*` operations (`spotify_seek` bound to its position, `spotify_resume` to a named device, and `spotify_library`, `confirm`, to the song playing and one of the owner's own or collaborative playlists, all before the question), name matching (by spelling and by sound, `likeness`; a close hit gives way only to an item of the owner's that is more alike still), device handling, the library cache and its refresh thread (every `library_refresh_s`; a minute after a refresh that failed or in which no part of the library loaded). | Call Spotify except through `spotify/api.py`. Tell the model a Spotify-written string as trusted. |
| `spotify/api.py` | The Web API client: playback, search, library, devices; retries, rate limits. | Replay a request whose outcome is unknown (`UncertainOutcomeError`). |
| `spotify/auth.py` | The PKCE login, the loopback-only callback, and `TokenStore` (file-locked refresh). | Use a client secret, or listen on anything but `127.0.0.1`. |
| `spotify/errors.py` | `SpotifyError` and friends, with the origin of their words. | Mark Spotify's own text as trusted. |

### Singing

| Module | Owns | Must not |
|---|---|---|
| `sing/score.py` | The score the model writes for `sing` (`TEXT PITCH BEATS`, comma-separated; `-` joins syllables, `~` holds, `_` rests), read and checked before any sound: at most 64 notes and 30 s, pitches C2 to C6, lengths 0.125 to 8 beats, tempo 40 to 220, letters only; `Score.timed_words` gives the seconds each word is sung in. | Hold a song or a phrase of its own: every song is the model's score. |
| `sing/psola.py` | `PsolaSinger`, the speaking-voice singer: Kokoro says each phrase (with one more word after it, dropped, so the last word does not fall and creak), each word is split into syllables at its vowels, and Praat's overlap-add stretches each vowel's clearly voiced part to its notes and sets their pitch, with glides and a vibrato on long notes; the song is moved by whole octaves towards the voice; the source is padded so Praat's output, at most three times its input, holds the whole song. Its `Budget` is checked before each phrase is said and before each is sung. | Play sound. Run anything but Praat in this process. |
| `sing/arpabet.py` | A word's ARPAbet phones, mapped from espeak's American English IPA, for SoulX-Singer. | Guess from the spelling. |
| `sing/soulx.py` | `SoulxSinger`: before every song the runner bundle is checked against its pins (`verify`, `runner_problem`: every file of `models.SOULX_RUNNER` a regular file, not a link, with its size and sha256, hashed again only once it changed) and the model by its size, and a bundle that fails sings the speaking voice's song; the same song sung first by `PsolaSinger` is the reference clip; both scores are written as SoulX-Singer's metadata in a private folder under the runtime folder, `audiocpp_cli` runs once on the GPU (through `audio.run_singer`) and exits, and its song is read back, checked against the length its metadata asked for, and cut to the score with the edges of silence it was asked for taken off. A failure (an exit with an error, a sound file it cannot read, one with fewer samples than its header says, or a song more than 0.25 s off its length) or a run past 20 s or the request's time left sings the reference instead, as does a request with less than 5 s left, when the program is not started; a song stopped (the wake word over it, `jarvis hush`, a pause) is not started, or is ended at once, and its folder removed. | Download anything, keep a process between songs, or run on the CPU. |
| `sing/budget.py` | `Budget`, what making one song may spend: `stopped` (cut short, hushed, paused or stopping) and the request's `time_left`, checked by each singer between its pieces of work; `SongStoppedError` and `OutOfTimeError`. | Hold any state of its own. |
| `sing/engines.py` | Which singer `[sing] engine` makes (on the first song), the singing voice's Kokoro files, and `jarvis doctor`'s checks and notes for them, the runner bundle checked whole (`soulx.runner_problem`). | Load anything before the first song. |
| `sing/ops.py` | `SingOps`: the `sing` operation (`safe`, an action), which checks the score before any question (`prepare`), asks with the song's length and all its words when a yes is needed, renders it inside `Request.making_sound` with a `Budget` that stops once the request is cut short or hushed, and returns it as the result's `sound`; a song stopped before it was finished raises `OperationStoppedError`. | Play sound itself: the agent's `play` hook does, and only the voice channel has one. |

### The time, weather, web and Reddit

| Module | Owns | Must not |
|---|---|---|
| `web/zones.py` | A country's clocks from the time zone database on the PC (`zone.tab`, `iso3166.tab`, `zonenow.tab`, the system's copy or the `tzdata` package's): `find_country` by its names (accents dropped), `clocks` one zone for each time it shows now, east to west, named by its largest city (`_PRINCIPAL` where the database's ranking names a smaller one). | Send anything off the PC. |
| `web/regions.py` | US states, Canadian provinces and territories and Australian states and territories (`find_region`), each with its main city's point and time zone, so the time and weather in one are that city's: the geocoder finds places, not regions. Plain "Georgia" is left to the country, plain "Washington" and "Quebec" to the geocoder, with the region as the other reading (`also_meant`) that `time_now` adds when its clock differs; the configured home never comes here. | Send anything off the PC. |
| `web/ops.py` | `WebOps`: `time_now`, `weather`, `web_search`, `web_read`, `reddit_search`, `reddit_read`, their shortcuts and spoken sentences, and the addresses and thread ids recent searches returned (behind one lock, since the look-ups run side by side), forgotten with the conversation (`reset`); in the chat also `image_search` and `show_image`, with the picture ids (`i1`…) an image search or a read page's own picture gave. | Open a page, thread or picture no search returned, except an address the owner wrote, exactly, in the request (`Request.words`). Pass search, page or Reddit text on as trusted, or put outside text into a weather or time answer. |
| `web/guard.py` | `public_client`: an httpx client over an httpcore pool whose every connection is resolved (within 5 s), checked to be a public address on port 80 or 443 and on none of this PC's own networks (`own_networks`, from Linux route and local-address data, including embedded IPv4). Unavailable data blocks the connection. Successful connections are pinned to that address; `read_capped` for the 1 MB cap on service answers. | Follow a redirect itself, use a proxy from the environment, or send anything that names the owner. |
| `web/page.py` | `PageReader`: one redirect hop at a time with each address checked, 15 s for the whole read (each read its own deadline, so reads side by side never stretch each other's), a 2 MB cap counted after inflating gzip (every member of it) or deflate itself, text types only, read in the header's charset, else a byte order mark's or the page's own `<meta>` one, else UTF-8, trafilatura's main-text extraction (its cross-page duplicate check off, so a page read again is read whole) and the page's own picture (`og:image`) at a full, checked address; `picture` fetches PNG, JPEG, WebP, GIF or BMP the same way, up to 8 MB; the text a page hides (the `hidden` attribute, `aria-hidden`, inline styles that hide text or shrink it to nothing) is dropped before extraction, and `hidden_parts` with `without_hidden` filters headless browser text and elements within element and search budgets. | Read past the cap or the deadline, or open another scheme, port or a URL with a password. |
| `web/proxy.py` | `EgressProxy`: an HTTP proxy on 127.0.0.1, one for each browser session, the browser's only way out: `CONNECT` tunnels and plain-HTTP requests (sent on with `Connection: close`) only to hosts whose every address is public, on port 80 or 443, connected at the address checked (`guard.public_addresses`); a random credential of its own each (`407` without it), 200 MB for a session, 48 connections at once with up to eight awaiting authentication, request heads up to 16 KiB within two seconds, 16 name lookups at once, and the hosts it refused (`was_refused`). | Serve another program on this PC, reach this PC, the home network or any other port, or read or change a tunnel's bytes. |
| `web/browser.py` | `Browser`: one headless agent-browser session, started on first use: its throwaway folder in `$XDG_RUNTIME_DIR/jarvis/browser/` (an empty agent-browser config, the default-deny action policy, the profile, a read-only download folder, so a download a page starts fails, and the browser's own home, XDG and socket folders), a session name of its own for each launch, the proxy with Chrome forced through it (loopback too, no name resolved by Chrome, no WebRTC UDP, no QUIC), a minimal environment (no display, no sound server, no version check), commands sent as one `batch` each, what the page shows (`View`, excluding ambiguous snapshot refs), and closing after `idle_close_s` or at exit, sweeping folders whose browser has ended at launch and exit. | Open anything but a plain http or https address, run a page script, upload, download, set cookies, open a tab, type into a password field, or act on an element the page no longer shows as it did. |
| `web/browserops.py` | `browser_open` (an address `WebOps.admit` allows, a link the page shows, or another part of the page), `browser_act` (`confirm`, never lower, `always_ask`, `outbound` when it types; `prepare` binds the element, its page and link destination before the question) and, in the chat, `browser_screenshot`; closing the browser when the conversation is forgotten. | Pass page text on as trusted, or open an address `web_read` would not. |
| `web/preview.py` | `FolderServer`: one project folder verified against `Boundary.writable` after opening, served on 127.0.0.1 (GET and HEAD, up to 20 MB a file, regular files only, nothing hidden, private or reached through a link out of the folder); `preview_browser`: a browser session that opens only that origin, with a proxy reaching only that port. | Serve anything outside the folder, or let a preview reach the internet or another local port. |
| `web/searxng.py` | The client of the local SearXNG (`deploy/searxng/`): JSON results, broken and repeated links dropped; `images` for its image category (each picture once, on a standard port, with the size the engine reported). | Reach anything but the loopback URL in `research.searxng_url`. |
| `web/openmeteo.py` | Open-Meteo's place lookup and forecast (hourly values only when an hour was asked about), taking only numbers, WMO codes, times of a fixed shape, a checked time zone and a two-letter country code. | Return a place name or any other free text from the answer. |
| `web/arctic.py` | The Arctic Shift archive: posts by id, top-level comments by score. | Ask for an id or subreddit that is not plain letters and digits. |

### Services and CLI

| Module | Owns | Must not |
|---|---|---|
| `toolswitch.py` | Each tool's setting for `jarvis tools` and the chat's Tools list: its class now and as shipped, the classes it allows, and the config with one changed (its shipped class written as no entry). | Switch off how Jarvis answers (`no_action`, `think_harder`, `stay_quiet`, `use_tools`). |
| `cli.py` | The `jarvis` command: setup, download, update, run, ask, chat, doctor, status, pause, resume, hush, confirm, start, stop, restart, logs, unload, pc-audio, consent, tools, spotify-login. `cmd_run` wires the threads and the signal handlers. | Hold logic that tests need elsewhere; it only wires parts together. |
| `cliout.py` | How commands write their answers: a check, a note and the closing count in one look, colour only on a terminal (never with `NO_COLOR`, `TERM=dumb` or in the agent format), the agent format's one JSON line, and catching all a command and its child programs write so that line is the only output. | Decide what a command checks or says; `cli.py` passes it the lines. |
| `services.py` | The Strata profile and the two systemd user units, rendered and written. | Write a value with a control character into a unit, or a relative program path. |
| `install.sh` and `tools/get_models.py` | The installer checks native prerequisites, installs the Git checkout and locked environment, then runs setup and doctor. The downloader resumes partial model files and verifies their upstream release pins before publishing them. | Run a system package manager or replace a user's existing model. |
| `update.py` | Fetch one tracked branch and plan or apply a fast-forward. A separate process in the updated environment regenerates the units and runs health checks. The parent holds the mode lock and both request locks through update or rollback. A user timer can run the same command, and each attempt is audited ([ADR 0019](adr/0019-install-and-update.md)). | Cut a request, overwrite checkout edits or start stopped services. |
| `modes.py` | Coding mode ([ADR 0017](adr/0017-coding-mode.md)): the mode in a state file every Jarvis process reads, `effective` (the config with `[coding]` laid over it: `[coding.chat]` over `[chat]`, `[coding.voice]` over `[agent]`, the coding run config and context), and `Answering` (the shared lock each process holds while it answers a request), and `Switcher`, which under a lock, and only while no request is in flight in any process, writes the model server's profile and unit for the mode, restarts the server when they changed, loads the model, goes back when the server does not return, audits `mode_switched` and goes back to normal after `coding.back_after_s` unused. | Run `systemctl` itself (the CLI hands it the unit), run two model servers, or decide what counts as programming work: the model does, with `switch_model`. |
| `control.py` | The control socket `jarvis run` serves in `$XDG_RUNTIME_DIR/jarvis/` (folder 0700, socket 0600, same-user peers only by `SO_PEERCRED`, one JSON line each way, 1 KB and 2 s per request, each connection on its own thread, a `control.lock` held while serving so a second run cannot take the socket over) and `Controls`, which carries out `status`, `pause`, `resume`, `hush` and `confirm` and audits each change (`control`, and `control_refused` for another user's connection or a `confirm` from a program Jarvis started: `started_by_jarvis` follows the peer's parents and, under systemd, its control group); unloading and loading take turns, the start-up load and a wake word's load (`warm_at_start`, `prewarm`) included; after the load at start or on resume, `prime` has the model read its prompt once (`Agent.prime`, up to 60 s), and a pause or a wake word's load meanwhile waits for it; `public_state` maps the loop and the model server to the bar's states (idle, listening, thinking, speaking, paused, model loading, unavailable; a lost microphone is unavailable); `after_sleep` checks the model server after a suspend and loads the model again only if its load failed, and not while paused. | Take any command outside `COMMANDS`, answer another user, or block the loop: unloading and loading run in the background. |
| `model.py` | `StrataModel`: the OpenAI-style chat client, plus status, load, warm and unload; each call asks for the request's current `Thinking` (Strata's `reasoning_effort` and `reasoning_budget_tokens`, with `max_tokens` raised by the cap so the reply keeps its room) or switches thinking off, and the streamed thinking (`reasoning_content`) is neither passed on nor kept; the status is `loading` while a load Jarvis asked for is under way, since Strata reports only loaded or unloaded. A server that cannot be reached while its model file (the run config's `--native`) is missing raises `ModelMissingError` at once (systemd skips Strata then), which the agent speaks as such. A streamed answer is read on a thread of its own, so a stream that stalls or trickles ends at the call's time limit. | Generate tokens to check health. Use the network beyond Strata's URL. |
| `chat/app.py` | The Textual window for `jarvis chat`, laid out like Claude Code's: typed requests to the window's own agent, its replies rendered as Markdown while they are written (`Hooks.draft`), the voice service's requests and replies from the audit log, steps shown by verbosity, a yes-or-no choice for questions (with the first lines of what a write puts in under it, `Hooks.preview`), the text pasted into the input passed on with the request so a long paste can be marked as outside data, Esc to stop a request, slash commands, the `/config` panel, the toolbar, and pictures attached as `[Image #N]` tags (Ctrl+V, a dropped file, `/image`). | Trust log text; it shows it as plain characters (a reply's Markdown draws nothing on the terminal). |
| `chat/commands.py` | The slash commands and what the menu suggests while one is typed: prefix first, then letters in order, then a command's arguments. | Run anything; the window does. |
| `chat/archive.py` | `ChatArchive`: every chat conversation in `chat-archive/` under the state folder (0700; files 0600 through `write_atomic`, read back through `read_private`): the conversation as `Conversation.save` writes it, its log of the window's own audit events (`.log.jsonl`, only added to), and a small file with its name, times and request count, which is all the list and the cap read (a first-format file, with the log inside, is moved to this one when read). Written only once a request ended in it or it changed after it was kept, never while the agent works on it (the window defers to the request's worker, which archives a request the closing window stopped); listed newest first, resumed through the agent (`Agent.resume`, which keeps the provenance flags and resets what operations remember), capped at `chat.archive_keep` (0 deletes what is kept when the chat opens), deleted by `/forget all`, which also forgets the conversation in the window. | Load a conversation into the model itself, or keep more than `archive_keep`. |
| `chat/clipboard.py` | `read`: what Ctrl+V pastes, from `wl-paste` with a fixed argument list (a picture type before text, then a copied file's address), within 20 MB and 3 s; a reader over either is stopped with its process group. | Pass a media type the clipboard offered on to `wl-paste`, or read the clipboard without a key press. |
| `chat/settings.py` | The `/config` screen, in the transcript's place: `SettingsList`, the settings under headings (Chat, Voice, Listening, Safety, Tools), each with every value it takes (the one in force lit), when a change applies (now, next chat, on close, or now and on close for `bypass_consent`, which the open chat's agent takes through `set_bypass_consent`) and a line on what it does; Enter, → and space move a value on, ← back. A `risky` setting asks before it is turned on (`Pending`: the same step again to the same value; a step to another value asks about that one, Esc or moving away leaves it off) and is marked red with ⚠ while on. `ToolsList`: each tool's name, setting in colour, how it ships when changed, and what it does. | Save anything itself; the window and `cli` do, and audit it after the save. |
| `chat/history.py` | `InputHistory`: the typed lines ↑ brings back, with what was pasted into each, kept 0600 in `chat-history` under the state folder. | Keep more than `chat.history_size` lines. |
| `chat/widgets.py` | The window's pieces: requests, replies (Textual's Markdown, set as Claude Code sets them: dashes for lists, numbers at the text's edge), steps, the multi-line input, the toolbar line with its meters (shortened to fit the width), `PointerList` (the menus and choices, the highlighted row pointed at with `❯`), `Chain` (calls of one tool in a row as one step that counts them and names their targets, with the newest result; every result listed with every detail; a call that asks first or a result that fails ends it and keeps its own row), and `PictureView`, a shown picture drawn by textual-image (kitty's graphics protocol, Sixel, or half cells), at most 64 cells wide and 24 rows tall. | Choose colours other than the terminal's own, or draw on its black, which Omarchy's terminals make the background. |
| `pc/workspace.py` | Jarvis's own folder (`files.workspace`, [ADR 0025](adr/0025-workspace-folder.md)): `workspace_problem` makes it mode 0700 or says why it cannot be used (a link, another user's, writable by others; an existing folder is never chmodded), and `Ledger` keeps the device, inode, times and size of each file Jarvis made there in the state folder, so Jarvis tells its own file from one the owner put there. | Change a folder the owner made, or take a file for its own that is not. |
| `pc/files.py` | `Boundary` (the read and write roots, plus the workspace, resolved through links, and the secrets never touched: `SECRET_NAMES`, `SECRET_PATHS`, `files.deny`, Jarvis's state; hidden things directly under home shut unless `files.hidden_allow` opens them, except inside the workspace; a relative path is in the workspace) and `FileOps`: `list_dir`, `read_file` (text only, in parts by line; a line too long for one part in pieces, read on with `from_char`) and `write_file` (`confirm`; the path, the folders it makes and a replaced file's size and identity (inode, modification and change times) bound before the question and checked again before the write; a replace keeps the old file's read and write permissions; a read-only file is refused before the question; `unasked` lets a new file in the workspace, or a later write to one Jarvis made there, run without it: `decide(..., unasked=...)` lowers that call to `safe`, the audit's `tool_call` reason says so, and outside text raises it back). Every open walks the resolved path one folder at a time from `/` with `O_NOFOLLOW`, and a file under home with a second hard link is not read or added to. | Read or write outside the roots, follow a link anywhere on the way, open a pipe or device, write a hidden file, make a file executable or a folder the question did not name, or pass file text on as trusted. |
| `pc/hyprland.py` | Hyprland's own socket (`$XDG_RUNTIME_DIR/hypr/<signature>/.socket.sock`): `clients` (the windows, `j/clients`), `monitors` (`j/monitors`), `run_lua` (`hyprctl eval`) and `lua_string`, each answer read whole up to `MAX_HYPR_BYTES` within `HYPR_TIMEOUT_S`. | Start a program itself, or wait on Hyprland without a timeout. |
| `pc/apps.py` | `AppOps`: `open_app` (`safe`): one program from `[apps]`, on the workspace the owner named (1 to 10) or in the background (the special workspace `jarvis`), started by Hyprland with a `[workspace N silent]` rule, then read back through the socket until its window shows, moved if it landed elsewhere and checked again a second later. Refused in a request heard in PC audio, on a workspace that has a fullscreen window, and while a workspace on screen has one (a program may open where the owner looks); a window list Hyprland does not give refuses too. Its stray windows are never moved off a workspace with a fullscreen window, and a fullscreen window is never taken for the new one (a game there can crash when the windows around it change). | Start a program not in `[apps]`, choose a workspace itself, or open a program again that is already open. |
| `pc/windows.py` | `WindowOps`: `pc_windows` (`safe`, read-only; the mapped windows by number with program, title, workspace and whether each shows, read from Hyprland's socket, as `process` text), `pc_window` (`safe`; focus, move to a workspace without following it, fullscreen on, float on or off) and `pc_window_close` (`confirm`; Hyprland's polite close): only a number from the request's latest `pc_windows`, once the model has read it (`Request.message`: a call written beside the listing in the same model message is refused), bound to the window's address and title before the question and checked again before it runs, then read back until the window shows the change (Hyprland answers `ok` to a dispatch on a missing window). A window list Hyprland does not give is an error. Fullscreen is Hyprland's live `fullscreen` or `fullscreenClient`: such a window is never moved, resized, toggled or closed, nothing is moved onto its workspace, no other window there is touched, and no special workspace is shown over a monitor showing one. `pc_window` and `pc_window_close` refuse a request heard in PC audio; `pc_window_close` also one holding outside text other than the listing. A program's class is cleaned and cut before it is said. | Start a program, act on a window not in the latest listing the model read, or touch a fullscreen window or its workspace beyond focusing the fullscreen window. |
| `pc/screenshot.py` | `ScreenshotOps`: `pc_screenshot` (`safe`): the screen, a workspace, a listed window or the focused one, taken by `grim` (a window by its Hyprland id, so one off screen is taken without showing it; a workspace off screen, or under a shown special workspace, is made from its windows at the monitor's full resolution). Only on the owner's own request, never one heard in PC audio. Saved as `screenshot-<date>_<time>.png` (0600) where Omarchy saves screenshots, copied with `wl-copy`, shown in the chat; where the model can see pictures it is also given the picture in the next message (`Result.look`, shrunk to `MAX_SIDE`) as outside data, so the request is tainted like a shared picture and the conversation keeps its name only (`Agent._show_looks`, `kept_work`). Read-only toward the desktop. | Move, focus or resize anything. |
| `pc/watch.py` | `ScreenWatch`: `pc_watch` (`safe`; a listed window, a workspace or the screen, the model's own `condition` and optional `say` line (`MAX_SAY`), `every_s` 10 to 300, `for_min` up to 720, `repeat`), `pc_watches` (`safe`, read-only, with each `say` line) and `pc_watch_stop` (`safe`), at most two at once, refused for PC audio and when the model cannot see pictures; a window binds like the window tools' (a number from a listing the model has read). The voice service makes the checks between requests (`VoiceLoop._served_watch`, a thread of its own, never while Jarvis speaks, is paused, locked or the owner dictates): the `pc_screenshot` capture, shrunk to `SHOWN_SIDE`, then one model call without memory and with nothing run (`Agent.glance`: Jarvis's own prompt and tools for the server's pinned prefix, `no_action` forced, its `reply` read as YES or NO, the picture as outside data), dropped when a request begins; times are the wall clock and, on a YES that follows a NO (`Watch.showing`), a fixed line built from the condition or the `say` line, never words read off the screen. Audited at start, stop and each line. Each check is a model call with a picture. Watches live in the service's memory ([ADR 0021](adr/0021-screen-watch.md)). A process watch (`of=process`, a PID `find_processes` listed in an earlier model message of the request, bound by its start time through the shared `ProcessOps.watched`, refused with outside text) reads only `/proc/<pid>/stat` and `status` at each check: gone, another process on the PID, or a zombie with one thread is said as exited and ends the watch; `T`/`t` as stopped; `D` at `STUCK_CHECKS` checks in a row spanning `STUCK_S` as stuck (any other state breaks the wait); each change said once, and the return to running too. No picture, no model call, up to `MAX_PROCESS_WATCHES` beside the screen watches. | Act on the desktop, run a tool, keep a picture, or say what the screen shows. |
| `pc/processes.py` | `ProcessOps`: `find_processes` (`safe`; the owner's own processes matching a spoken name by command, executable, argv[0], or Hyprland window class and title read from Hyprland's socket, at most 8, children folded into their program) and `stop_process` (`strong`, never lower): only a PID `find_processes` listed in the same request, bound to its program and start time before the question, checked again once held by a pidfd, then SIGTERM (SIGKILL with `force`); refused in a request heard in PC audio or holding outside text other than its own listing (`Request.outside`, noted by `Agent`). The protected list (`PROTECTED_PROGRAMS`, `PROTECTED_UNITS`, `PROTECTED_RULES`), each entry with its reason. | Stop anything by name or pattern, a PID it did not list in this request, another user's process, a protected one, or anything for PC audio; start a program. |
| `pc/shell.py` | `ShellOps`: `pc_shell` (`strong`, never lower), offered only when `[shell] enabled` is set, starting in the workspace (the start folder bound into the arguments in `prepare`, so the question and the run name the same one): one command the owner said, read out before the key character by character (`spelled`) and refused in `prepare` when it cannot be (runs of whitespace, too long), when the request was heard in PC audio, or when it holds outside text. It runs once by `systemd-run --user --scope --expand-environment=no` with a fixed description and a minimal launcher environment (a unit of its own, with `MemoryMax`, no swap, `CPUQuota`, `TasksMax` and `RuntimeMaxSec`), behind `setpriv --no-new-privs` and `env -i` with only `PATH`, `HOME`, `LANG` and `TERM`; when it returns or times out, the scope is killed with SIGKILL and checked gone. Output is read unbuffered, capped as it is read and returned as `shell` text: outside text, remembered as such, and private like a file's, so a later command or stopped process is refused and a search that sends text off the PC waits for a yes; the audit keeps only its digest (`Operation.audit_text`). | Run anything without the confirm key, with raised rights, with Jarvis's environment, or for PC audio. |
| `pc/stats.py` | `StatsOps`: `pc_stats` (specs, CPU, memory, graphics card, VRAM, temperatures, disks, uptime) with its shortcuts and spoken sentence. | Start a program. |
| `pc/volume.py` | `VolumeOps`: `pc_volume` (`safe`): read, set, turn up or down, or mute PipeWire's default output, every program's sound, through `audio.wpctl`. A change is bound to the volume it starts from and ends at before any question; never above 100 %. | Change one program's volume, or pick an output device. |
| `pc/hearing.py` | `PcHearing` and `HearingOps`: `pc_audio_listen` (`safe`, read-only), offered on the voice channel only with `audio.listen_to_pc`: the next 3 to 60 seconds of the frames the PC watcher reads, or with `audio.pc_audio_recall_s` the last ones, kept in memory (dropped on a pause, the recall also at the end of the stream; a listen ends on a pause or a suspend), transcribed by the unhinted Voxtype transcriber and returned as `pc_audio` text: outside text, private like a file's, so a later stopped process or command is refused and, unless `boundaries.bypass_consent` is on, a search that sends text off the PC waits for a yes; the audit keeps only its digest. Refused in a request heard in PC audio, in a scheduled one and outside the voice service; a listen ends when the request is stopped (`Request.cancelled`), and is shortened to end `RESERVE_S` before the request's time runs out, and refused when too little is left. | Open a capture of its own, write audio to disk, or hear while Jarvis is paused or speaks. |
| `pc/media.py` | `MediaOps`: `pc_media` (`safe`): play, pause, skip or say what plays in other MPRIS players (a browser video, VLC) through `audio.playerctl`. Spotify is left to its own tools unless `everything` is set. Only a player's plain name is said, never a title. | Read a title, open a player, or seek. |
| `schedule.py` | `ScheduleOps`: `schedule_add` (a reminder or a request; in seconds or minutes, or at a time of day on a day, once or repeating), `schedule_list` (a read) and `schedule_cancel` (a number or all); `JobStore`, the jobs file under a lock file, and `take_due`, which looks at the file at most once a second and only when it changed or a job's time came, and moves each due job on to its next time or removes it; each job keeps where its request came from (`foreign`, `outside`, from `current_request`), which the voice loop runs it with (`Agent.handle(steered_by=…)`); `Due.missed` and the late rules; `reminder_words`. | Say a job, or run one; the voice loop does both. Write a systemd unit or a cron line. |
| `about.py` | `AboutOps`: `about_jarvis` (what Jarvis is and how it is set up, worked out from the config and Strata's run config) and `read_jarvis_file` (one of `jarvis_files`, a fixed list of its own configuration files, or one `[section]` of it, cut to the channel's room): resolved through links, refused when the `[files]` boundary keeps it private (`Boundary.secret`), opened without following a last link, regular files only, values under secret-looking names hidden (`hide_secrets`), returned as `file` text. Both are reads. | Take a path from the model, read the state folder, or change a configuration file. |
| `calc.py` | `calculate`: works out an arithmetic expression exactly (fractions, a few functions and constants) from Python's syntax tree, with every step bounded. | Run the text, read a name it does not list, or work out a sum without a bound. |
| `sysinfo.py` | `Sampler`: the CPU's busy share and temperature, memory, and the biggest graphics card's load, VRAM and temperature, read from `/proc` and `/sys`. | Start a program or fail on a missing file; a figure it cannot read is left out. |
| `activity.py` | `AuditTail` (follows the audit log) and how the chat window shows each event: its kind, words (a request and a reply whole, a step on one line), a quiet note, whether it belongs to the step above, the verbosity that shows it, and what the working line says. | Write the log. |
| `config.py` | `config.toml`: a validated, versioned pydantic schema. | Accept an unknown key or a value with a control character. |

### Support

| Module | Owns |
|---|---|
| `paths.py` | XDG config, state and data folders, and the fixed file names inside them. |
| `files.py` | `write_atomic`: write through a temp file and a rename. |
| `pictures.py` | `Picture` and `prepare`: Pillow's PNG, JPEG, WebP, GIF and BMP decoders only, the pixel count checked before decoding (40 million at most), the first frame turned upright and shrunk to 1,568 pixels on its longest side, then encoded again as PNG (JPEG for a large photo); `read_file` (a regular file of at most 20 MB), `pasted_path` (a dropped file's path or `file://` address) and `newest` (the latest screenshot); `PictureCache`, the pictures the chat shows, 0600 in `chat-pictures` under the state folder, named by their content, the newest 50 kept. |
| `secretfile.py` | `write_private` and `read_private`: mode 0600 files, symlinks and foreign owners refused; the Strata API key reader. |
| `jsonutil.py` | `obj` and `objs`: typed access to JSON from outside. |

## The voice loop

`State` in `voice.py` has six values. `VoiceLoop.run` reads frames on the main thread; the state
changes as one request moves through.

| From | To | When |
|---|---|---|
| `IDLE` | `CAPTURE` | The wake word is heard on the microphone (`handle_wake`). |
| `IDLE` | `TRANSCRIBE` | A request captured from PC audio is waiting (`_answer_pc`). |
| `IDLE` | `SPEAK` or `THINK` | A reminder came due (`SPEAK`, after the chime) or a scheduled request did (`THINK`, as your own request), and you are not dictating or in a meeting (`_run_due`). |
| `CAPTURE` | `IDLE` | No speech within `no_speech_timeout_s`, or your Voxtype dictation started meanwhile (the request is dropped without a word, one `wake_ignored` audit event). |
| `CAPTURE` | `TRANSCRIBE` | Trailing silence, or the length cap, ends the utterance. |
| `TRANSCRIBE` | `THINK` | Voxtype returned text. |
| `TRANSCRIBE` | `SPEAK` | Your own speech could not be transcribed, or came back empty: Jarvis says it did not catch that. |
| `TRANSCRIBE` | `IDLE` | The text is a phantom (a say-nothing hallucination), or PC audio failed or came back empty. |
| `THINK` | `SPEAK` | A waiting line while the model loads (then back to `THINK`), or the final reply. |
| `THINK` | `CONFIRM` | The agent asks for a spoken yes: the question is spoken (`SPEAK`), then the loop listens in `CONFIRM`. |
| `CONFIRM` | `THINK` or `SPEAK` | The answer is in; the call runs or is dropped, and the reply is spoken. |
| `SPEAK` | `IDLE` | The reply was spoken, or the request failed after an apology (`_serve`). |
| `SPEAK` | `FOLLOW_UP` | The reply to your own voice was spoken and `audio.follow_up_s` is above 0, and Voxtype is not dictating or in a meeting (or its status cannot be read). |
| `FOLLOW_UP` | `TRANSCRIBE` | Speech in the window: a request without the wake word. Unclear speech is dropped silently (`TRANSCRIBE` to `IDLE`). |
| `FOLLOW_UP` | `CAPTURE` | The wake word in the window: a request as usual. |
| `FOLLOW_UP` | `IDLE` | No speech in the window, Jarvis is stopping or paused (nothing heard in the window is taken once paused), or Voxtype started dictating meanwhile. Speech shorter than 0.3 s is dropped untranscribed and the window listens on for the rest of its time. |

Notes. There is no separate wake state: wake detection runs on the mic frames in `IDLE`. While
Jarvis speaks, the `jarvis-barge-in` thread reads the mic and listens for the wake word
(`audio.barge_in`) through `WakeWord.heard_over_speech`, which needs at most
`audio.barge_in_threshold`; hearing it hushes the speaker, cancels the request and makes `_serve` take a new
one (`SPEAK` to `CAPTURE`). A stretch of speech that ends without it logs the closest score heard from
`NEAR_MISS` (0.2) up. A request the agent answers with no words, or one where the owner asked for quiet
(`stay_quiet`, which the agent passes on through `Hooks.quiet` even when words were heard first; it also hushes what was queued, unless a song was queued in the request, which then plays in full), opens no
follow-up window. It reads frames only while speech plays or an operation makes a sound (`Hooks.making_sound`, at the usual wake score then, since nothing plays), so it never takes one the loop
needs, and a sentence the speaker found wake-like (`Speaker.interruptible`) is not spoken over while it plays and for `GUARD_TAIL_S` (1 s) after. After
speech that was not interrupted, `_end_speaking` drains the capture so Jarvis cannot hear itself. A
line spoken while the agent works (`_cue`: a waiting line, a confirmation question) moves the loop to
`SPEAK` and back to `THINK`. Every capture has a timeout, and a failed request ends in `IDLE`.

## One request, from wake word to spoken reply

1. `audio.py`: `Microphone` streams 80 ms frames from `pw-record`. `WakeWord` scores each frame (or
   `AzureWake` passes it to the Azure helper, with `audio.wake_engine = "azure"`).
2. `voice.py`: `handle_wake` calls `capture_utterance` (Silero `Vad` finds the end of speech) and
   `_reply` runs `audio.transcribe` (`voxtype transcribe`). Speech from the mic is tagged
   `Origin.OWNER_SPEECH`; speech captured from PC audio is tagged `Origin.PC_AUDIO`, and the same
   request heard on both is counted as PC audio. **Provenance is first attached here.**
3. `agent.py`, `Agent.handle`: records a `request` audit event. A request that is not the
   owner's is wrapped as someone else's words and kept out of the history. While Jarvis asks
   first, a chat paste over 200 characters is wrapped as `<data origin="paste">` and marks the
   request as untrusted.
4. `shortcuts.match`: a plain command skips the model and becomes a tool call. Anything else goes to
   `StrataModel.chat` with the system prompt, the remembered conversation (its newest turns that fit,
   when it is over its budget; it is summarized after the reply, not inside the request) and the tool
   schemas.
5. Each call goes through `Agent._run_call`:
   - `operations.validate_args` checks it against the operation's schema; an unknown name or bad
     argument is refused and audited.
   - **The policy decides** here: `policy.decide` returns run, confirm, strong or deny from the
     operation's class (the owner's override, else the default), with the rule that any side effect
     while outside text is in the conversation needs at least a confirmation.
   - Only when the call is not refused, `Operation.prepare` may bind "whatever is current" to a
     concrete target (`spotify_play`, `spotify_change_volume` and `spotify_like`; `write_file`
     binds the resolved path, the folders it would make and a replaced file's size and identity),
     or refuse it (`stop_process` refuses a PID `find_processes` did not list in this request, and
     any request heard in PC audio: `Agent.handle` runs each request in its own `request_scope`).
   - **The cap on actions**, in both consent modes: a side effect the policy lets through is refused
     before any question when `ratelimit.ActionRate` has no room (`full`, which counts nothing); it
     takes its place (`take`) only once it may run, right before `op.run` (`Agent._run_counted`), so
     a declined action takes none. Past `boundaries.max_actions` in the window it is refused,
     audited as `action_limit`, and the owner hears why.
   - **The confirmation is bound to one exact action**: `action_hash` of the operation and its
     final arguments is computed after `prepare`, before the question, and is the digest the call
     runs and is audited under. `VoiceLoop._ask` speaks the question and listens for an answer
     (the chat shows it, with `Operation.preview` under it);
     `policy.is_yes` accepts only a clear yes. Anything else, a timeout or a failure is no. A
     `strong` call instead goes to `Hooks.confirm_key`: `VoiceLoop._await_key` speaks the question,
     arms `ConfirmKey` with the digest and waits for `jarvis confirm` through the control socket,
     while still reading the mic so the wake word can say no. A channel without the hook (the chat,
     `jarvis ask`) refuses the call, and `boundaries.bypass_consent` never skips it.
   - **The audit entry is written**: `tool_call` (with the verdict and digest, and an operation's
     `hashed` arguments, such as a write's content, as a SHA-256 and a length) before the call runs,
     then `confirmation` and `tool_result`, `tool_failed`, or `tool_stopped` for an operation stopped part way (a song still being made when the owner spoke over Jarvis or hushed it). A `confirmation` that is not a yes
     carries a `reason` from `NoReason` (`said_no`, `no_answer`, `slept`, `mic_lost` and the rest):
     the hook returns `Declined` with it, only `True` approves, and anything else is a no. A
     question not heard out (hushed, paused or not played) is `not_heard` and nothing is listened
     for. A confirm-key no that ended a wait has its reason in the `confirm_key` event before it;
     one the key never armed for (`not_heard`, `key_held_off`, `spoken_over`) says it here. The first and last events of a
     request are `request` and `reply`.
6. A tool result keeps its origin. Text from outside (Spotify names) is wrapped as
   `<data origin="...">` for the model and marks the rest of the request as untrusted. Jarvis's own
   status text stays trusted.
7. The reply is streamed (`StrataModel.chat` with `on_delta`): `agent._ReplyStream` passes each
   finished sentence (`sentences.sentence_end`) of the message's text, or of a first `no_action`
   call's `reply`, to `Hooks.speak` while the model writes on. It holds a sentence claiming an
   action before any succeeded, and stops once a tool call follows: sentences of the text already
   passed on play out, unless one claims an action; then, as for a `no_action` reply that is not the
   answer, the speaker is hushed (what is playing ends, the rest is dropped).
   `Agent._final` checks the whole reply that claims an action when no action succeeded (a tool
   that only looks something up, `side_effect=False`, does not count; one that acted but did not
   end cleanly, `Result.acted`, does); it challenges the model once, then speaks a plain "I didn't
   manage that". `Agent._deliver` speaks
   what was not streamed yet. `voice.speakable` turns each part into plain words and
   `audio.Speaker` synthesizes each sentence while the one before plays, through one `pw-play`.

## Threads and processes

Threads in the `jarvis run` process:

| Thread | Does |
|---|---|
| Main | `VoiceLoop.run`: mic frames, wake, capture, transcribe, agent, speech, confirmation. One request at a time. |
| Microphone reader | `Microphone._read`: moves `pw-record` output into a bounded queue. Each PC-audio capture has its own. |
| PC watcher | `VoiceLoop._watch_pc`: scans PC audio for the wake word and hands one request at a time to the main thread, and hands each frame it reads to `PcHearing` while a `pc_audio_listen` runs or a recall is set (`VoiceLoop._heard`). Each wake word it drops is logged with why. Only with `audio.listen_to_pc`. |
| `jarvis-speech` | `Speaker._work`: synthesizes queued sentences in order and hands them to one `pw-play` while they keep coming. |
| `jarvis-barge-in` | `VoiceLoop._watch_speech`: while Jarvis speaks, reads the mic for the wake word. Only with `audio.barge_in`. |
| `jarvis-azure-wake` | `AzureWake._read`: reads the Azure helper's `ready` and `wake` lines. Only with `audio.wake_engine = "azure"`. |
| `jarvis-stopper` | Waits for SIGTERM or SIGINT, then stops the speaker, mic and PC capture. |
| `jarvis-control-work` | `Controls`' slow work, off the request path: the model load at start and on resume (then one read of the prompt, `Agent.prime`), the unload after a pause, the check after a suspend. |
| `jarvis-lookup-N` | The read-only look-ups of one model message, side by side (`ThreadPoolExecutor`, at most `MAX_SIDE_BY_SIDE`); they end with the request's step. |
| `jarvis-speech-warm` | `FileWarmer`: reads the speech model's file into the file cache at start and at each wake word, never within 5 minutes of a whole read; ends once read. |
| `jarvis-spotify-refresh` | Keeps the Spotify login and library fresh. |
| Short-lived | A model prewarm on each wake, the chime, the "waking up" lines (`jarvis-waking` and a timer). |

Every `threading.Thread` Jarvis starts itself, and the waiting-line timer, is a daemon thread, so
Python ends the process without waiting for any. That is deliberate (#32): Python's docs warn that a
daemon thread may stop in the middle of its work. Most state files are replaced whole through
`files.write_atomic` (the private ones too), so a thread cut off mid-write leaves the old file. Two
files are not: `actions.json` (`ratelimit.py`) is written over in place and then cut to length, so a
cut-off write leaves the old count, or old bytes behind new ones, which the next reader takes as
damaged and starts again; and the audit log (`audit.py`) is appended to with one `os.write`, so a
cut-off append can leave one partial last line. `cmd_run`'s `finally` closes the control socket, the
Voxtype watcher and the Azure helper explicitly. No lost write has been seen; make a thread
non-daemon, stopped by an event, only if one is.

The look-up threads are not daemon threads either: the agent's thread waits for them within the step, each bounded by its own deadline, and Python joins any left at exit.

The other exception is `jarvis chat`: Textual's `run_worker(..., thread=True)` workers (the agent request,
status and stats reads, picture and paste handling) are not daemon threads, so the process waits for
them when the window closes.

`jarvis chat` runs the agent on a Textual worker thread. `jarvis spotify-login` runs a loopback HTTP
server on its own thread until the callback arrives.

Child processes Jarvis starts: `pw-record` (microphone and PC audio), `pw-dump` (find app streams),
`pw-play` (speech, songs and chime), `voxtype transcribe` (one per utterance), `audiocpp_cli` (one per
song, when SoulX-Singer is installed; `sing/soulx.py` through `audio.run_singer`), `systemctl --user` and
`journalctl --user` (the lifecycle commands in `cli.py`), `xdg-open` (the login link, unless
`--no-browser`), `wl-paste` (the chat's Ctrl+V, `chat/clipboard.py`), `grim` and `wl-copy` (`pc_screenshot`, `pc/screenshot.py`; `grim` also for `pc_watch`) and, only when `[shell]` is on,
`systemd-run --user --scope` and `systemctl --user stop` for `pc_shell` (`pc/shell.py`). A program the owner asks
to open (`open_app`, `pc/apps.py`) is started by Hyprland, not by Jarvis: Jarvis sends the command
through Hyprland's socket.

Strata is a different service, `jarvis-strata.service`. By default it listens on `127.0.0.1` with an API key
that Jarvis reads from the state folder, and it serves only Jarvis. In local mode `jarvis.service`
starts after it and loads the model. The remote client uses `GET /v1/models` for health and
capabilities and leaves model lifecycle control on the backend. Health checks never generate.
The service units and profile come from `services.py`.

## Where state lives

Config, state and model files are described in
[configuration.md](configuration.md#where-things-live). In short: `config.toml` and the generated
Strata profile and Voxtype settings under the config folder; the Strata API key, Spotify tokens,
audit log and Strata log under the state folder (the first two mode 0600); model files in
`data_dir`. `paths.py` is the only module that knows the folder names. The voice service keeps its
conversation in memory only and drops it after `agent.memory_idle_s`; the chat saves its own, 0600,
in `chat-conversation.json` under the state folder. The model-file list is in
[models.md](models.md).

## Security-sensitive code

AGENTS.md ("Boundaries") names the security-sensitive areas, and
[`.github/CODEOWNERS`](../.github/CODEOWNERS) lists most of their files by path (every other file
falls under its catch-all entry, so a change anywhere still asks for review). The code:

- the policy engine: `policy.py`, `operations.py`, `shortcuts.py`, `ratelimit.py` and the call path
  in `agent.py`;
- confirmation: `agent.py` and `voice.py`, with `is_yes` in `policy.py`;
- the audit log: `audit.py`;
- the Spotify OAuth flow and token files: `spotify/`, `secretfile.py`;
- the units and the config that become systemd lines: `services.py`, `config.py`;
- the system prompt: `prompts/core.md`;
- the web tools: `web/`, above all `web/guard.py`, `web/page.py`, `web/proxy.py`, `web/browser.py` and
  `web/browserops.py`;
- the PC-operation tools: `pc/`, above all `pc/files.py`, `pc/processes.py` and `pc/shell.py`;
- pictures: `pictures.py`, which decodes picture bytes anyone may have made, and `chat/clipboard.py`,
  which starts `wl-paste`.
- the remembered conversation and its provenance marks: `memory.py`, and the chat's archive of it with
  your private data, `chat/archive.py`;
- the chat window, which can turn `bypass_consent` on: `chat/app.py` and `chat/settings.py`.

Report a vulnerability as [SECURITY.md](../SECURITY.md) says.
