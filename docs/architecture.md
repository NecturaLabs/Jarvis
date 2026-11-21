# Architecture

This is a map of how Jarvis is put together. If it differs from the code, the code wins; please fix
this file. The decisions behind the shape are in [plan.md](plan.md) and the [ADRs](adr/); this file
does not repeat them.

Jarvis is one Python process, `jarvis run`, started as a systemd user unit. One thread owns the
microphone and runs the whole request path: wake word, speech capture, transcription, the agent, the
spoken reply. There is one agent and no subagents. The model is a separate service, Strata, on
loopback. Every tool call goes through a typed operation and a policy decision before it runs, and
every step is written to an audit log. The tools are Spotify's, and the time, the weather, web search and
Reddit (`web/`).

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
 │    └─▶ web/page.py ─▶ web/guard.py ──────┼──────▶ public web pages only
 │    audit.py ─▶ audit.jsonl               │
 │ audio.py  Speaker ─▶ pw-play             │
 └──────────────────────────────────────────┘
```

## Modules

All paths are under `src/jarvis/`. The tables say what a module owns and what it must not do.

### Voice pipeline

| Module | Owns | Must not |
|---|---|---|
| `voice.py` | `VoiceLoop` and its `State`: wake, capture, transcribe, think, confirm, speak, follow-up; the PC-audio listener; `speakable` and `spoken_question`. | Decide what an operation may do (that is `policy.py`). Touch a device directly; it uses `audio.py` objects. |
| `audio.py` | The `pw-record` microphone and per-app capture (`Microphone`, `AppAudio`), the wake word (LiveKit's `WakeWord`, or `AzureWake` talking to the helper), Silero VAD, `voxtype transcribe`, whether your dictation is running (`voxtype status`), the `pw-play` speaker, voice selection. | Edit Voxtype's config or drive its daemon. It runs only `voxtype transcribe`, with Jarvis's own settings file, and reads `voxtype status`. Start the Azure helper any way but `azure_wake_command` (no network). |
| `azure_wake_helper.py` | MIT. A separate program, run with its own Python, that feeds audio from standard input to an Azure custom keyword model and prints `wake` ([ADR 0010](adr/0010-azure-keyword-wake-word.md)). | Be imported by Jarvis, import Jarvis, or run with a network. |
| `kokoro.py` | Kokoro text to speech on ONNX Runtime, phonemized by espeak-ng. | Play audio. It returns samples. |
| `sentences.py` | Where a sentence of a reply ends (`sentence_end`, `split_sentences`), so speech starts before the reply is whole; every end is also one for `agent.claims_action`. | Decide what may be spoken; the agent's reply stream does. |
| `models.py` | The pinned sha256 of each model file Jarvis loads, for `jarvis doctor`. | Download anything or fail a request over a digest. |

### Agent and policy

| Module | Owns | Must not |
|---|---|---|
| `agent.py` | `Agent`: one request into at most `max_steps` model calls and `max_tool_calls` tool calls within `max_seconds` (`[agent]` for speech, `[chat]` for the chat window); each answer capped at `reply_tokens` (a reply the cap ends is kept and says so); the oldest tool results of a long request cleared to fit the context window; the remembered conversation (`memory.py`), summarized by the model when it outgrows its room; provenance on message parts; running each call through validation, policy, confirmation and audit; the check that a reply does not claim an action no tool did. | Run an operation that policy refused. Talk to the owner except through `Hooks`. |
| `policy.py` | `Origin` (provenance), `OpClass`, `decide` (the decision table: outside text raises side effects to `confirm`, and file text raises `outbound` operations to it), `action_hash`, `is_yes`. Plain data and pure functions. | Do I/O, or let text from outside the owner's speech raise an operation's class. |
| `operations.py` | `Operation`, `Registry`, `Result`, argument validation against a flat JSON schema, the error types. | Accept an unknown argument or an unknown operation. |
| `shortcuts.py` | Plain commands ("pause", "volume 40") that skip the model: normalising and matching a request to one call. | Guess. A request that fits two calls, or none exactly, matches nothing. |
| `audit.py` | `AuditLog`: append-only JSONL, mode 0600, rotated once at 5 MiB. | Fail the caller; a write error goes to the service log. |
| `app.py` | `build`: assembles the model client, Spotify, operations, prompt, audit log and agent from the config; the `no_action` operation; the speech-to-text vocabulary hint. | Hold state of its own beyond the objects it returns. |
| `prompts/core.md`, `prompts/voice.md`, `prompts/chat.md` | The shipped system prompt: what Jarvis does and how it acts (`core.md`), then how it replies, spoken (`voice.md`) or written in Markdown within the chat's reply budget (`chat.md`). `app.py` fills in the name, tone, skills and word budget. | Be the only guard: boundaries are enforced in code, not in this prompt. |
| `prompts/skills/*.md`, `skills.py` | Jarvis's own skills: instructions for a kind of task (web research, game look-ups), listed by name and description in the prompt and read whole through the safe `read_skill` operation. | Read anything but the shipped skill files. |
| `memory.py` | `Conversation`: the remembered turns and a summary of older ones, measured in estimated tokens against a budget; a turn that drew on outside text is kept as `<data>`; the chat's conversation is saved 0600 in the state folder between sessions. | Decide what is trusted: the agent marks outside turns and keeps side effects behind a yes while one is remembered. |

### Operations and Spotify

| Module | Owns | Must not |
|---|---|---|
| `spotify/ops.py` | `SpotifyOps`: the twelve `spotify_*` operations, name matching, device handling, the library cache and its refresh thread. | Call Spotify except through `spotify/api.py`. Tell the model a Spotify-written string as trusted. |
| `spotify/api.py` | The Web API client: playback, search, library, devices; retries, rate limits. | Replay a request whose outcome is unknown (`UncertainOutcomeError`). |
| `spotify/auth.py` | The PKCE login, the loopback-only callback, and `TokenStore` (file-locked refresh). | Use a client secret, or listen on anything but `127.0.0.1`. |
| `spotify/errors.py` | `SpotifyError` and friends, with the origin of their words. | Mark Spotify's own text as trusted. |

### The time, weather, web and Reddit

| Module | Owns | Must not |
|---|---|---|
| `web/ops.py` | `WebOps`: `time_now`, `weather`, `web_search`, `web_read`, `reddit_search`, `reddit_read`, their shortcuts and spoken sentences, and the addresses and thread ids recent searches returned; in the chat also `image_search` and `show_image`, with the picture ids (`i1`…) an image search or a read page's own picture gave. | Open a page, thread or picture no search returned. Pass search, page or Reddit text on as trusted, or put outside text into a weather or time answer. |
| `web/guard.py` | `public_client`: an httpx client over an httpcore pool whose every connection is resolved (within 5 s), checked to be a public address on port 80 or 443, and pinned to that address; `read_capped` for the 1 MB cap on service answers. | Follow a redirect itself, use a proxy from the environment, or send anything that names the owner. |
| `web/page.py` | `PageReader`: one redirect hop at a time with each address checked, 15 s for the whole read, a 2 MB cap counted after inflating gzip or deflate itself, text types only, trafilatura's main-text extraction (its cross-page duplicate check off, so a page read again is read whole) and the page's own picture (`og:image`) at a full, checked address; `picture` fetches PNG, JPEG, WebP, GIF or BMP the same way, up to 8 MB. | Read past the cap or the deadline, or open another scheme, port or a URL with a password. |
| `web/searxng.py` | The client of the local SearXNG (`deploy/searxng/`): JSON results, broken and repeated links dropped; `images` for its image category (each picture once, on a standard port, with the size the engine reported). | Reach anything but the loopback URL in `research.searxng_url`. |
| `web/openmeteo.py` | Open-Meteo's place lookup and forecast, taking only numbers, WMO codes, a checked time zone and a two-letter country code. | Return a place name or any other free text from the answer. |
| `web/arctic.py` | The Arctic Shift archive: posts by id, top-level comments by score. | Ask for an id or subreddit that is not plain letters and digits. |

### Services and CLI

| Module | Owns | Must not |
|---|---|---|
| `cli.py` | The `jarvis` command: setup, run, ask, chat, doctor, status, start, stop, restart, logs, unload, pc-audio, spotify-login. `cmd_run` wires the threads and the signal handlers. | Hold logic that tests need elsewhere; it only wires parts together. |
| `services.py` | The Strata profile and the two systemd user units, rendered and written. | Write a value with a control character into a unit, or a relative program path. |
| `model.py` | `StrataModel`: the OpenAI-style chat client, plus status, load, warm and unload. | Generate tokens to check health. Use the network beyond Strata's URL. |
| `chat/app.py` | The Textual window for `jarvis chat`, laid out like Claude Code's: typed requests to the window's own agent, its replies rendered as Markdown while they are written (`Hooks.draft`), the voice service's requests and replies from the audit log, steps shown by verbosity, a yes-or-no choice for questions, Esc to stop a request, slash commands, the `/config` panel, the toolbar, and pictures attached as `[Image #N]` tags (Ctrl+V, a dropped file, `/image`). | Trust log text; it shows it as plain characters (a reply's Markdown draws nothing on the terminal). |
| `chat/commands.py` | The slash commands and what the menu suggests while one is typed: prefix first, then letters in order, then a command's arguments. | Run anything; the window does. |
| `chat/clipboard.py` | `read`: what Ctrl+V pastes, from `wl-paste` with a fixed argument list (a picture type before text, then a copied file's address), within 20 MB and 3 s; a reader over either is stopped with its process group. | Pass a media type the clipboard offered on to `wl-paste`, or read the clipboard without a key press. |
| `chat/history.py` | `InputHistory`: the typed lines ↑ brings back, kept 0600 in `chat-history` under the state folder. | Keep more than `chat.history_size` lines. |
| `chat/widgets.py` | The window's pieces: requests, replies (Textual's Markdown), steps, the multi-line input, the toolbar line with its meters (shortened to fit the width), `PointerList` (the menus and choices, the highlighted row pointed at with `❯`), and `PictureView`, a shown picture drawn by textual-image (kitty's graphics protocol, Sixel, or half cells), at most 64 cells wide and 24 rows tall. | Choose colours other than the terminal's own, or draw on its black, which Omarchy's terminals make the background. |
| `pc/files.py` | `Boundary` (the read and write roots, resolved through links, and the secrets never touched: `SECRET_NAMES`, `SECRET_PATHS`, `files.deny`, Jarvis's state) and `FileOps`: `list_dir`, `read_file` (text only, in parts by line) and `write_file` (`confirm`; the path bound before the question and checked again before the write, through the folder's descriptor with no link followed at the last name). | Read or write outside the roots, follow a link out of them, open a pipe or device, write a hidden file, make a file executable, or pass file text on as trusted. |
| `pc/stats.py` | `StatsOps`: `pc_stats` (specs, CPU, memory, graphics card, VRAM, temperatures, disks, uptime) with its shortcuts and spoken sentence. | Start a program. |
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
| `CAPTURE` | `IDLE` | No speech within `no_speech_timeout_s`. |
| `CAPTURE` | `TRANSCRIBE` | Trailing silence, or the length cap, ends the utterance. |
| `TRANSCRIBE` | `THINK` | Voxtype returned text. |
| `TRANSCRIBE` | `SPEAK` | Your own speech could not be transcribed, or came back empty: Jarvis says it did not catch that. |
| `TRANSCRIBE` | `IDLE` | The text is a phantom (a say-nothing hallucination), or PC audio failed or came back empty. |
| `THINK` | `SPEAK` | A waiting line while the model loads (then back to `THINK`), or the final reply. |
| `THINK` | `CONFIRM` | The agent asks for a spoken yes: the question is spoken (`SPEAK`), then the loop listens in `CONFIRM`. |
| `CONFIRM` | `THINK` or `SPEAK` | The answer is in; the call runs or is dropped, and the reply is spoken. |
| `SPEAK` | `IDLE` | The reply was spoken, or the request failed after an apology (`_serve`). |
| `SPEAK` | `FOLLOW_UP` | The reply to your own voice was spoken and `audio.follow_up_s` is above 0, and Voxtype is not dictating. |
| `FOLLOW_UP` | `TRANSCRIBE` | Speech in the window: a request without the wake word. Unclear speech is dropped silently (`TRANSCRIBE` to `IDLE`). |
| `FOLLOW_UP` | `CAPTURE` | The wake word in the window: a request as usual. |
| `FOLLOW_UP` | `IDLE` | No speech in the window, Jarvis is stopping, or Voxtype started dictating meanwhile. Speech shorter than 0.3 s is dropped untranscribed and the window listens on for the rest of its time. |

Notes. There is no separate wake state: wake detection runs on the mic frames in `IDLE`. While
Jarvis speaks, the `jarvis-barge-in` thread reads the mic and listens for the wake word
(`audio.barge_in`); hearing it hushes the speaker, cancels the request and makes `_serve` take a new
one (`SPEAK` to `CAPTURE`). It reads frames only while speech plays, so it never takes one the loop
needs, and a sentence the speaker found wake-like (`Speaker.interruptible`) is not spoken over. After
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
   owner's is wrapped as someone else's words and kept out of the history.
4. `shortcuts.match`: a plain command skips the model and becomes a tool call. Anything else goes to
   `StrataModel.chat` with the system prompt, the remembered conversation (summarized first when it
   outgrew its budget) and the tool schemas.
5. Each call goes through `Agent._run_call`:
   - `operations.validate_args` checks it against the operation's schema; an unknown name or bad
     argument is refused and audited.
   - **The policy decides** here: `policy.decide` returns run, confirm, strong or deny from the
     operation's class (the owner's override, else the default), with the rule that any side effect
     while outside text is in the conversation needs at least a confirmation.
   - Only when the call is not refused, `Operation.prepare` may bind "whatever is current" to a
     concrete target (`spotify_play`, `spotify_change_volume` and `spotify_like`).
   - **The confirmation is bound to one exact action**: `action_hash` of the operation and its
     final arguments is computed after `prepare`, before the question, and is the digest the call
     runs and is audited under. `VoiceLoop._ask` speaks the question and listens for an answer;
     `policy.is_yes` accepts only a clear yes. Anything else, a timeout or a failure is no.
   - **The audit entry is written**: `tool_call` (with the verdict and digest) before the call runs,
     then `confirmation` and `tool_result` or `tool_failed`. The first and last events of a request
     are `request` and `reply`.
6. A tool result keeps its origin. Text from outside (Spotify names) is wrapped as
   `<data origin="...">` for the model and marks the rest of the request as untrusted. Jarvis's own
   status text stays trusted.
7. The reply is streamed (`StrataModel.chat` with `on_delta`): `agent._ReplyStream` passes each
   finished sentence (`sentences.sentence_end`) of the message's text, or of a first `no_action`
   call's `reply`, to `Hooks.speak` while the model writes on. It holds a sentence claiming an
   action before any succeeded, and stops once a tool call follows: sentences of the text already
   passed on play out, unless one claims an action; then, as for a `no_action` reply that is not the
   answer, the speaker is hushed (what is playing ends, the rest is dropped).
   `Agent._final` checks the whole reply that claims an action when no tool succeeded; it
   challenges the model once, then speaks a plain "I didn't manage that". `Agent._deliver` speaks
   what was not streamed yet. `voice.speakable` turns each part into plain words and
   `audio.Speaker` synthesizes each sentence while the one before plays, through one `pw-play`.

## Threads and processes

Threads in the `jarvis run` process:

| Thread | Does |
|---|---|
| Main | `VoiceLoop.run`: mic frames, wake, capture, transcribe, agent, speech, confirmation. One request at a time. |
| Microphone reader | `Microphone._read`: moves `pw-record` output into a bounded queue. Each PC-audio capture has its own. |
| PC watcher | `VoiceLoop._watch_pc`: scans PC audio for the wake word and hands one request at a time to the main thread. Only with `audio.listen_to_pc`. |
| `jarvis-speech` | `Speaker._work`: synthesizes queued sentences in order and hands them to one `pw-play` while they keep coming. |
| `jarvis-barge-in` | `VoiceLoop._watch_speech`: while Jarvis speaks, reads the mic for the wake word. Only with `audio.barge_in`. |
| `jarvis-azure-wake` | `AzureWake._read`: reads the Azure helper's `ready` and `wake` lines. Only with `audio.wake_engine = "azure"`. |
| `jarvis-stopper` | Waits for SIGTERM or SIGINT, then stops the speaker, mic and PC capture. |
| `jarvis-model-warm` | Loads the model at start, off the request path. |
| `jarvis-spotify-refresh` | Keeps the Spotify login and library fresh. |
| Short-lived | A model prewarm on each wake, the chime, the "waking up" lines (`jarvis-waking` and a timer). |

`jarvis chat` runs the agent on a Textual worker thread. `jarvis spotify-login` runs a loopback HTTP
server on its own thread until the callback arrives.

Child processes Jarvis starts: `pw-record` (microphone and PC audio), `pw-dump` (find app streams),
`pw-play` (speech and chime), `voxtype transcribe` (one per utterance), `systemctl --user` and
`journalctl --user` (the lifecycle commands in `cli.py`), `xdg-open` (the login link, unless
`--no-browser`), the Spotify app itself (`omarchy-launch-spotify`, when no player is up) and
`wl-paste` (the chat's Ctrl+V, `chat/clipboard.py`).

Strata is a different service, `jarvis-strata.service`. It listens on `127.0.0.1` with an API key
that Jarvis reads from the state folder, and it serves only Jarvis. `jarvis.service` starts after it
and loads the model at start. Health checks (`StrataModel.status`) never generate. The service
units and profile come from `services.py`.

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

- the policy engine: `policy.py`, `operations.py`, `shortcuts.py` and the call path in `agent.py`;
- confirmation: `agent.py` and `voice.py`, with `is_yes` in `policy.py`;
- the audit log: `audit.py`;
- the Spotify OAuth flow and token files: `spotify/`, `secretfile.py`;
- the units and the config that become systemd lines: `services.py`, `config.py`;
- the system prompt: `prompts/core.md`;
- the web tools: `web/`, above all `web/guard.py` and `web/page.py`;
- the PC-operation tools: `pc/`, above all `pc/files.py`;
- pictures: `pictures.py`, which decodes picture bytes anyone may have made, and `chat/clipboard.py`,
  which starts `wl-paste`.

Report a vulnerability as [SECURITY.md](../SECURITY.md) says.
