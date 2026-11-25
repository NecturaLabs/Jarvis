# Jarvis — build plan

Living design document: update it in place when the design changes. Drafted 2026-11-20 and reviewed by
Codex (gpt-6.1-sol) as co-author; decisions with lasting consequences are also recorded in `docs/adr/`.

An always-running, voice-activated local agent for this Omarchy PC. You say "Hey Jarvis"; Omarchy's local
dictation (Voxtype) transcribes; a local model on Strata reasons; typed tools act within boundaries you set;
Piper answers by voice (Kokoro is the default voice now). First core feature: Spotify.

## Build status

Checked against the code on `feature/spotify-voice-control`. The decisions below stand; this section says
which parts exist. Details of what is built are in [configuration.md](configuration.md) and
[models.md](models.md).

| Phase | Status | Built, and where it differs from the text below |
|---|---|---|
| 0. Project setup | Built | Git, `AGENTS.md`, docs, uv project, CI. |
| 1. Strata | Mostly built | `jarvis setup` writes the Strata profile, from the model's own Strata run config (`strata.run_config`, since 2026-11-23; nothing model-specific in Jarvis), and `jarvis-strata.service` (lazy load, idle unload, `MemoryHigh`/`MemoryMax`, `CPUWeight`, a start limit, sandboxing options, 127.0.0.1, API key). Defaults differ from the sketch: `expert_vram_gib` 1.0, no `--mmap-experts` flag, `idle_unload_s` 86400 (24 hours, so the model stays loaded; 300 until 2026-11-22), `memory_max` 40G, context 24576 (raised from 8192 for research, 2026-11-21). Not recorded in the repository: the 200-request model eval and its gate. Load-refusal below a memory reserve is not built. `jarvis run` loads the model at start (owner decision 2026-11-21: keep it loaded). |
| 2. Ears | Mostly built | `pw-record` capture, the LiveKit wake word, Silero VAD and the state machine (IDLE, CAPTURE, TRANSCRIBE, THINK, CONFIRM, SPEAK). Transcription is **path B** (`voxtype transcribe`, with Jarvis's own `-c` settings); path A (the daemon contract) is not built, and neither is volume lowering over MPRIS. The wake test runs (8 h, 100 wakes) are not recorded. |
| 3. Boundaries, setup and voice | Partly built | Built: config (one schema version, no migrations), the policy engine and its decision table, provenance, spoken confirmation bound to one action, the audit log, the agent loop, Kokoro and Piper voices. Not built: the Textual wizard (`jarvis setup` takes flags instead), quiet hours. The `strong` class waits for the confirm key (`jarvis confirm`), from the voice service only; `stop_process` is the first shipped `strong` operation, and an owner override cannot lower it (`Operation.lowest_class`). |
| 4. Spotify | Built | PKCE login, Web API client, and the operations play (track, album, artist, playlist, liked songs), pause, resume, next, previous, set and change volume, shuffle, repeat, queue, like, now playing. Players are found through the Web API, and Spotify is never opened unasked: with no player Jarvis says so and asks (`open_app`, #36); MPRIS is not used. No seek, no separate transfer-to-device or my-playlists operation (playing "my playlist" is `play` with `mine`). |
| 5. PC operations | Partly built | Built: `pc_stats`, `calculate` (exact arithmetic on an expression the model writes, parsed and never run), `list_dir`, `read_file` within `[files]` read roots and `write_file` (`confirm`) within write roots, secrets never touched, file text as data that puts outbound operations behind a yes; `find_processes` and `stop_process` (`strong`, #9): a PID `find_processes` listed in the same request, bound to its program and start time and held by a pidfd before the signal, a protected list of the session's and Jarvis's own processes, nothing for PC audio. `open_app` (#36): a program from `[apps]` on the workspace the owner names or in the background, placed by Hyprland and checked there, never for PC audio. `os_skill`: one file of Omarchy's guides, listed at start, read in parts as outside text, no path from the model and no link followed. `pc_volume` (the `set_volume` of the sketch: PipeWire's default output through `wpctl`) and `pc_media` (other MPRIS players through `playerctl`, Spotify left to its own tools), 2026-11-25. `pc_audio_listen` (ADR 0015): what is said in PC audio, heard for the owner from the wake-word watcher's frames, in memory only, 2026-11-26. Not built: `workspace`, `run_omarchy`, and the owner's spoken check of stopping a program. |
| 6. Research | Mostly built | `deploy/searxng/` (pinned image, JSON on, limiter off, `127.0.0.1:8888`), `web_search`, `web_read` (trafilatura, size caps, public-address checks at every connection and redirect, only pages a search returned or an address the owner wrote), plus `reddit_search` and `reddit_read` (SearXNG and the Arctic Shift archive), `time_now` and `weather` (Open-Meteo), [ADR 0007](adr/0007-outside-services-for-answers.md). Not built: SearXNG in an installer, a spoken check with the owner. |
| 7. Polish | Partly built | Built: sentence streaming and voice barge-in ([ADR 0008](adr/0008-speak-while-writing-and-barge-in.md)); echo cancellation is a documented PipeWire setup for the owner, which broke the PC's sound output when applied on 2026-11-21 and was reverted (#13); follow-up listening (`audio.follow_up_s`, 6 s by default; the owner's live run is still to come). The Omarchy bar module is built (`tools/bar_status.py`, #14) and waits for the owner to add it to the bar. |
| 8. Hardening | Planned | No `install.sh`, `jarvis update`, rollback or uninstall. `jarvis doctor`, `status` (with `--json`), `pause`, `resume`, `hush`, `logs`, `unload`, `start`, `stop` and `restart` exist. |

Also beyond the plan: Jarvis answers about itself (`about_jarvis`) and reads its own configuration files
(`read_jarvis_file`). Built beyond the plan, each with an ADR: reminders and scheduled requests, said or carried out by the
voice service ([0011](adr/0011-reminders-and-scheduled-requests.md)); speech heard in PC audio is treated as someone else's
([0003](adr/0003-pc-audio-is-not-the-owner.md)); plain commands skip the model
([0004](adr/0004-model-free-path-for-plain-commands.md); no new ones, [0013](adr/0013-free-speech-no-new-shortcuts.md)); no hosted backend
([0005](adr/0005-local-only.md)). Also built: a vocabulary hint for speech-to-text, a check that a spoken
reply does not claim an action no tool did, and `jarvis doctor` hashing model files against pinned releases.

Where the code settled a number the plan left open: a request stops after 90 s, or 270 s when the model must load first (plan: 60 s), the conversation
is token-budgeted and summarized (plan: about 10 turns), speech allows 8 model calls and 12 tool calls per request and the chat 30 and 40 (`[agent]`, `[chat]`), 3,000 characters of each result for speech and 8,000 in the chat, and 0.7 s of
trailing silence ends an utterance (plan: 1.0 s). Kokoro is the default voice; Piper is still supported
behind the same speech setting, as the Piper-licence risk below asked.

## Owner requirements (fixed)

- Always running; wakes on "Hey Jarvis" (bare "Jarvis" later, optional).
- Transcription through Omarchy's local dictation (Voxtype, the configured default), not the API backend.
- Reasoning by the local model on Strata. Strata serves only Jarvis.
- First-time configuration sets boundaries, tone/persona and the core instructions (what Jarvis is).
- Researches online and operates this PC.
- No global instructions, skills or plugins except the Omarchy OS skills; no subagents: one main agent.
- First core feature: Spotify control (needs Spotify Premium). v1 is Spotify management by voice only; PC operations
  and research come after it works.
- Light on the PC so gaming stays decent at any time: no GPU or CPU use while idle and no heavy CPU use between
  requests (decided 2026-11-21: a loaded idle model may hold RAM and VRAM; Jarvis must run beside games). The tasks are simple, so a small model is enough.
- Replies by voice (Piper); no desktop notifications.
- The owner speaks freely and never repeats a set phrase: the model decides what a request means. No new fixed
  phrasings or keyword lists for spoken requests; a new ability is a typed tool the model picks.
- Model kept loaded (owner decision, 2026-11-21, replacing "idle by default, including after reboot"): `jarvis run`
  loads it at start, and `strata.idle_unload_s` defaults to 86400 (24 hours) so it stays loaded. While idle it
  holds memory but uses no GPU or CPU. A lower value frees it sooner and loads it again on the next wake.
- Very easy to set up and maintain. Public repo `NecturaLabs/Jarvis`, GPL-3.0-or-later (`docs/adr/0006-gpl-licence.md`).
- Open source (GPL-3.0-or-later since 2026-11-21); everything committed stays publishable.
- Agents working on it open nothing on the owner's screen (hidden; else workspace 1) and never stop processes they
  did not start.

## Verified facts (the reference machine, 2026-11-20)

- Omarchy 4.5.0-3, Hyprland, RX 7900 XT 20 GB, Ryzen 7 9800X3D, 64.9 GB RAM,
  system Python 3.14 (Jarvis uses it through uv; every key dependency ships 3.14 wheels), uv via mise.
- Dictation default `voxtype` (`~/.config/omarchy/defaults/dictation`). Voxtype 1.1.1 daemon as user unit, Whisper
  `base.en` only model downloaded, CPU-only, compositor-driven, `pause_media = true`.
- Voxtype integration contract (docs/INTEGRATIONS.md): `record start --file=PATH --no-auto-submit
  --no-smart-auto-submit [--model M] [--no-osd]`, then `record stop --wait --json` -> status ok/empty/timeout/error,
  exit 0/3/4/1, default timeout 120 s. An undownloaded `--model` fails at once. Voxtype asks integrations not to
  edit its config. `voxtype transcribe FILE` builds its own transcriber and prints diagnostics beside the text.
- Spotify desktop installed, on MPRIS. Web API rules for a new app (Nov 2026): Premium for the app's owner, 5 allowlisted users,
  search limit <= 10, `/me/library` and playlist `/items` endpoints, artist top-tracks removed, shared
  per-developer quotas with `429 QUOTA_EXCEEDED` (Jul 2026). Redirect must be HTTPS or a loopback literal
  (`127.0.0.1` / `[::1]`); `localhost` is rejected. PKCE (S256) for public clients.
- Strata 0.2.16: OpenAI + Anthropic APIs on localhost; gfx1100 supported; `lazy_load`, `idle_unload_s`,
  `reasoning_budget_tokens` (0 = unlimited, so set a positive cap); foreground `strata serve --profile … --ready-file`
  for supervisors; mapped low-RAM mode (`--mmap-experts`, or `--resident-budget-gib N`) serves experts through the
  OS file cache, which the kernel can reclaim. `models import --plan` previews only. `strata --version` reports 0.2.16, engine build 0.1.62.
- Strata's engine runs two model families only (`qwen4exp`: Qwen3.8-Flash-Next; `qwen35moe`: Qwen3.6-35B-A3B);
  no small dense model. (Strata 0.2.18 and 0.2.19, read 2026-11-23, added shape engines for Tencent Hy4-Flash and
  Inkling 2 Mini; [models.md](models.md) compares the models measured here.) Qwen3.6-35B-A3B (Apache-2.0, 35B total / 3B active, 256 experts, top-8) comes as Unsloth
  UD-Q4_K_XL 22.9 GB, UD-Q6_K_XL 32.6 GB, Q8_0 37.8 GB. Its hybrid mode keeps some experts in VRAM and the rest in
  RAM: Strata measured 58-60 tok/s decode on an RTX 3060 12 GB with 37 of 256 experts per layer in VRAM (docs
  QWEN36.md). `--expert-cache auto` sizes the cache to the free VRAM; a fixed `--expert-cache N` (N x the largest
  expert blob) and `--vram-reserve-mib` cap it (docs DETAILS.md). Sources read 2026-11-21; not yet measured here.
- LiveKit WakeWord 2.1.3 (Apache-2.0 code and weights per NOTICE), Python >= 3.11, deps numpy + onnxruntime,
  pretrained `hey_jarvis.onnx`; local detection, no account.
- Piper: `piper-tts` (OHF-Voice/piper1-gpl, GPL-3.0). Latency on this PC unmeasured.
- Omarchy OS skills: `/usr/share/omarchy/default/agents/skills/{omarchy,omarchy-app,diagnose-crash}/SKILL.md`.

## Architecture

```
 jarvis.service (user, one process, one agent)
   audio: PipeWire capture 16 kHz mono, timestamped ring buffer
     └─ state machine: IDLE (wake word) → CAPTURE → TRANSCRIBE → THINK → CONFIRM? → SPEAK → IDLE
          wake: LiveKit wake words      VAD: Silero (end of speech, no-speech, max duration)
          transcribe: Voxtype (path chosen by measurement, see below)
   agent: OpenAI chat + tools against Strata, step/time/output caps, validated tool calls
   policy: typed operations, default deny, provenance tracking, action-bound confirmation, audit log
   tools: spotify · pc (typed ops) · os_skill (read Omarchy's guides, in parts) · web (later)
   voice out: Piper → PipeWire
 strata.service (user): `strata serve --profile jarvis` foreground, lazy load, idle unload, mapped experts
 searxng (docker compose, 127.0.0.1 only) — added in the research phase, isolated from voice and Spotify
```

Own agent loop rather than Claude Code/OpenCode/Hermes on Strata: the "no global instructions, skills, plugins,
subagents" rule holds by construction, the prompt stays small enough for voice latency, and boundaries are
enforced in code before any tool runs.

## Design

### Audio and transcription
- One state machine owns the mic, Voxtype calls, TTS and confirmation; every state has a timeout and a cancel path.
- Silence rules: no speech within 4 s after the wake = cancel; 0.7 s trailing silence ends an utterance (speech goes on above `audio.vad_neg_threshold`, 0.15 under the start threshold); a request after the wake word holds at least 0.15 s of speech; 20 s max.
  Wake phrase and chime audio are cut from what is transcribed.
- Device: explicit PipeWire source chosen in setup; the same source for wake, VAD and Voxtype, checked by `doctor` against the `Audio/Source` nodes in `pw-dump`. A capture that sends nothing for 5 s counts as lost and is reopened.
- **Transcription path, decided by measurement in phase 2:**
  - A. *Daemon contract* — `record start --file … --model small.en`, Jarvis VAD calls `record stop --wait --json`.
    Uses the warm daemon. Ownership rule: start only when `voxtype status --format json` reports idle, record the
    transcript path Jarvis created, and never stop or cancel a recording whose file is not Jarvis's. If the owner's
    own dictation begins mid-capture, Jarvis drops its request and says so. (Path B, built, does the nearer thing: a wake word heard while the owner's dictation runs is ignored, and no follow-up window opens during dictation or a Voxtype meeting.)
  - B. *File transcription* — Jarvis writes its buffered utterance (with pre-roll) to a WAV and runs
    `voxtype transcribe`, parsing the transcript explicitly. No shared daemon state; costs a model load per call.
  - Measure both: wake-to-text latency, first-word clipping, accuracy on 50 recorded commands. Keep one.
- `small.en` (or the measured winner) is downloaded by `install.sh` through Voxtype's own model setup command.
- Playback role (decided 2026-11-22, #29 item 4): `pw-play` keeps its default media role (Music); Jarvis's
  voice is not given another, as nothing needs it and what a role does to Spotify's volume here is unverified.
- `voxtype transcribe` output (decided 2026-11-22, #29 item 7): it has no documented format, so
  `parse_transcript` keeps filtering its header and log lines, and the pinned-version tests stay; no upstream
  issue is filed.
- Media: during capture Jarvis lowers Spotify through MPRIS volume itself and restores the volume it saved, unless
  the command changed volume or playback (then the command's result wins). Voxtype's `pause_media` is the owner's
  setting; path A inherits it, path B does not touch it.

### Speaking, barge-in, confirmation
- Built: the reply is spoken sentence by sentence while the model writes it, one player for the whole reply, and
  the wake word stays on while Jarvis speaks: saying it stops the speech and the request and starts a new one
  (`audio.barge_in`), at the lower score of `audio.barge_in_threshold` since Jarvis's own voice in the
  microphone lowers the owner's. A sentence that would trigger the wake word by itself is checked before it
  plays, against that score, and cannot be spoken over while it plays. See [ADR 0008](adr/0008-speak-while-writing-and-barge-in.md).
- Confirmation starts only after TTS playback has drained, accepts yes/no for 8 s, and is bound to one immutable
  action (hash of operation + arguments). Anything else, silence, or the wake word over the question, is no. The
  highest-risk class is confirmed by keybinding (`jarvis confirm`), never by voice, and the consent bypass
  never skips it.
- Echo cancellation (PipeWire echo-cancel module) is not built; it is the owner's system setting if Jarvis's own
  voice ever reaches the wake word through the room.

### Model (Strata)
- Profile `jarvis`: the model's own Strata run config (`strata.run_config`, written by `strata launch`) with
  Jarvis's limits on top: the engine's whole VRAM use at or under about 8 GB (`expert_vram_gib` for an engine that
  takes a size, `vram_reserve_mib` for one whose expert cache fills the card), the context, `--parallel 1`,
  `lazy_load` on, the idle unload, a positive `reasoning_budget_tokens` cap, local access with an API key, and the
  experts Jarvis uses learned and preloaded (`learn_experts`). Nothing model-specific is hardcoded (owner,
  2026-11-23). The model is Qwen3.8-Flash-Next IQ2_XS (owner's choice of default, 2026-11-23, to be tested against
  Inkling 2 Mini; measurements in [models.md](models.md)); Qwen3.6-35B-A3B UD-Q4_K_XL was the model until then,
  with a profile Jarvis built itself.
- Under systemd: `strata serve --profile jarvis --ready-file …`, absolute paths, `Restart=on-failure` with backoff,
  `MemoryHigh`/`MemoryMax`. Health checks never generate, so boot and `doctor` leave the model unloaded.
- Jarvis says "One moment, waking up" when a request finds the model unloaded; `jarvis unload` frees it at once
  (e.g. before a game). It refuses to load when available memory is below a reserve set in config (not built).
- Survives suspend/resume and engine crashes: the agent retries once after the server is ready again.

### Agent loop
- One agent; tools passed as JSON schemas; each assembled tool call is validated (schema + semantic checks) before
  the policy sees it.
- Caps per request: 6 tool steps, 60 s (built as 90 s, plus 180 s more when the model must load first, and not counting the wait for the owner's yes or confirm key), output size per tool. Repeated identical calls are stopped.
- Thinking: each request starts at `thinking` (`none` by default); the model raises its own effort for the rest of
  a request with the typed `think_harder` tool, up to `thinking_max` (`high` by default), never lowering it
  ([ADR 0014](adr/0014-model-raises-its-own-thinking.md)).
- Uncertain outcomes (timeout after sending) are never retried blindly for non-idempotent actions (queue, skip,
  like); Jarvis says "I'm not sure that went through" instead.
- Replies are short and report what happened (tool results, not intentions). The reply is streamed: a finished
  sentence is spoken while the rest is written, except a sentence claiming an action before any succeeded, and
  nothing more of a message once a tool call follows it; what was already passed on plays to its end
  ([ADR 0008](adr/0008-speak-while-writing-and-barge-in.md)).
- Session memory: the conversation is token-budgeted (`agent.memory_tokens`, `chat.memory_tokens`). The chat
  keeps earlier requests' tool calls and results too (`memory_detail = "full"`; speech keeps only each
  request, reply, actions and look-ups, `condensed`); when it outgrows its room, the oldest work goes
  first, then the model summarizes the oldest turns; speech forgets it after 30 min idle, the chat keeps it
  across sessions. Follow-up listening is built and on by default: the next request within `audio.follow_up_s`
  (6 s) of a spoken reply needs no wake word (never while Voxtype is dictating or in a meeting).
- System prompt: `core.md` (shipped: identity, tools, limits) + persona/tone + owner addendum from setup + a
  rendering of the boundaries + the Omarchy skill index (names and each description's first sentence; full text through `os_skill`).

### Boundaries (policy engine, enforced in code)
- **Typed operations, default deny.** Each tool exposes named operations with validated arguments
  (`pc.open_app(name)`, `pc.set_volume(pct)`, `pc.workspace(n)`, `pc.read_file(path)`, `pc.list_dir(path)`,
  `pc.run_omarchy(cmd)` from an allowlisted set, …). Unknown operations are refused.
- Each operation has a class: `safe` (runs), `confirm` (spoken yes), `strong` (keybinding confirm), `deny`. The
  owner sets classes in setup; default profile "ask for risky".
- Generic shell (`pc.shell`) is off by default; when the owner enables it, every call is `strong`, runs with a
  timeout under `systemd-run --user --scope` with memory/CPU limits, no sudo, in a scrubbed environment.
  Built as `pc_shell` (`src/jarvis/pc/shell.py`, `[shell]`, #19): the class cannot be set below `strong`,
  `setpriv --no-new-privs` keeps sudo out, and a request heard in PC audio or holding outside text runs
  nothing.
- Process control: stop only a PID Jarvis shows, `strong` class; never by name or pattern. Built as
  `find_processes` and `stop_process` (`src/jarvis/pc/processes.py`): the PID must come from
  `find_processes` in the same request, the class cannot be set below `strong`, and a request heard in
  PC audio stops nothing.
- File scope: allowed roots for read and write, resolved through symlinks before checking.
- Network: web access only through the web module, destination checks on every URL and redirect, no file or
  conversation data sent in URLs without confirmation.
- **Provenance:** every message part carries its origin (owner speech, tool result, web, file, Spotify metadata).
  Non-owner text is data: it cannot raise a class, and while such content is in the session any side-effecting
  operation needs at least `confirm`.
- Quiet hours (no speech output, optional mic off) and `jarvis pause` (mic released, model unloaded).
- Audit log (JSONL): request, tool call, decision, outcome.

### Spotify (first core feature)
- Setup guides: create the developer app, add the redirect `http://127.0.0.1:<port>/callback`, allowlist the
  owner's account, paste the client ID. PKCE S256 with validated `state`, loopback-only callback, scopes for
  playback state/control, library and playlists. Refresh token stored 0600 under `~/.local/state/jarvis/`,
  refreshed atomically.
- Operations: play (track / album / playlist / artist context), pause, resume, next, previous, seek, volume,
  shuffle, repeat, queue, like, now playing, my playlists, transfer to device. Preferred device chosen in setup;
  if Spotify isn't running, say so and ask whether to open it and where (`open_app`), never open it unasked.
- MPRIS only for local transport operations on the Spotify player. A Web API mutation with an uncertain result is
  never replayed through MPRIS. Quota errors (`429`) are reported, not retried in a loop.

### Research (later phase)
- SearXNG in docker compose, pinned image, JSON enabled, limiter off, bound to 127.0.0.1; `web.search` and
  `web.fetch` (trafilatura extraction, size caps, URL policy). Spoken summaries name their sources. A container
  failure never affects voice or Spotify.

### Config and first-run setup
- `~/.config/jarvis/config.toml`, pydantic-validated, schema-versioned with migrations, preserved across updates.
- `jarvis setup` (Textual wizard, run in a floating Omarchy terminal on first start): name, persona/tone, core
  addendum, boundaries profile and per-operation classes, mic choice and level test, wake test, voice choice with a
  sample, Spotify login, quiet hours. Reopenable anytime.

## Setup and maintenance

- `./install.sh` (idempotent): preflight (Omarchy, PipeWire, Voxtype contract `--wait`, docker, GPU, free disk/RAM);
  the system Python 3.14 through uv and `uv sync --frozen`; installs a pinned Strata release (its engine under
  a folder the owner picks, not `~/.local/share`) and writes the `jarvis` profile
  pointing at the model folder; downloads wake-word and Piper voice files with checksums; asks Voxtype to
  download the transcription model; writes user units; runs `jarvis setup` on first run.
- `jarvis doctor`: one line per check with its fix; never loads the model unless `--load`.
- `jarvis update`: `git pull --ff-only`, `uv sync --frozen`, Strata update only when idle, restart units, `doctor`;
  keeps the previous checkout and Strata engine for `jarvis rollback`.
- `jarvis status | logs | pause | resume | unload | stop | confirm | setup | uninstall` (uninstall keeps config and
  the model unless asked).
- Lifecycle: units bound to the graphical session; defined behavior on lock (keep listening, configurable),
  suspend/resume (re-open mic, re-check Strata), mic unplug/replug (wait and re-open), missing model disk
  (Strata unit fails cleanly, Jarvis says the model is unavailable).
- Repo `NecturaLabs/Jarvis` (public); CI on GitHub-hosted runners: ruff, pyright, pytest (unit only, no audio/GPU),
  REUSE and pip-audit.

## Open-source readiness (from day one)

- Nothing machine- or owner-specific in code: paths (the model folder), devices, voices and accounts come
  from config with documented defaults; tests use fixtures, never the owner's data.
- History stays publishable: no personal data, tokens, recordings or internal hostnames in any commit; owner
  voice fixtures stay out of the repository unless the owner approves them for publication.
- Licences checked on every dependency and model added (code, weights, voices). Piper (`piper-tts`) is
  GPL-3.0, so the release licence must be GPL-3.0-compatible or Piper must be swappable behind the speech
  interface; decided before going public (ADR then).
- Each user brings their own Spotify developer app (client ID in setup), which fits Spotify's dev-mode limit.
- At release: `necturalabs:open-source-repo` audit (history and secret scan, community files, licence, Scorecard).

## Phases (each ends with measured evidence)

0. **Project setup** (built) — git, AGENTS.md, docs (architecture, playbook, decisions), uv project, CI, private repo,
   issue backlog.
1. **Strata** (mostly built) — install the pinned CLI and engine, download Qwen3.6-35B-A3B UD-Q4_K_XL, `jarvis` profile, user
   unit. Size the expert cache to the ~8 GB VRAM limit. Measure on this PC under normal desktop load: cold and warm
   load time, TTFT, decode tok/s, peak RAM/VRAM, CPU use during a reply, release after unload, and a game running
   beside a loaded model.
   Model eval: 200+ representative and adversarial requests on Jarvis's real tool schemas, scored on correct
   operation, arguments, abstention and finished task. Gate: >= 95% correct first operation, >= 90% finished
   multi-step tasks, 0 policy-violating calls reaching execution. Fallback: UD-Q6_K_XL under the same VRAM limit
   (more RAM, same VRAM). v1 ends at phase 4 (Spotify by voice); phases 5 and 6 start after the owner confirms it.
2. **Ears** (mostly built) — capture, wake word, VAD, state machine, both transcription paths measured; pick one. Wake test:
   8 h of desk audio with music and calls for false wakes, 100 spoken wakes for misses.
3. **Boundaries, setup and voice** (partly built) — config + migrations, policy engine with a full decision table, provenance,
   confirmation, audit log, the wizard, Piper output, the agent loop with fake-model tests. MVP: "Hey Jarvis, what
   time is it?" spoken back, and a denied action refused correctly.
4. **Spotify** (built) — client, PKCE login in the wizard, MPRIS transport, device handling. Milestone: play artist / song
   / playlist, skip, volume, now playing, queue, like, all by voice, with recorded-fixture tests.
5. **PC operations** (partly built) — PC stats, files, process stop, the optional strong-class shell and os_skill built; app, workspace, volume and Omarchy commands to come.
6. **Research** (mostly built) — SearXNG, search/fetch, URL policy, sourced answers.
7. **Polish** (partly built) — sentence streaming and voice barge-in built, echo cancellation documented as a PipeWire setup;
   follow-up listening, the bar module (`tools/bar_status.py`), optional bare-"Jarvis" wake word.
8. **Hardening** (planned) — install/update/rollback/uninstall tested on a clean user, lifecycle cases, docs and playbook.

## Testing

- Unit: policy decision table, provenance rules, config migrations, tool argument validation, agent loop with a
  scripted fake model (incl. malformed and repeated calls), Spotify client on recorded HTTP fixtures, state machine
  timing with a fake clock.
- Audio fixtures: owner recordings + Piper-generated speech for wake, VAD and transcription.
- Automated runs are silent: audio output goes nowhere (`PIPEWIRE_REMOTE=/nonexistent`), and tests feed WAV
  fixtures instead of the mic.
- Local integration (marked, not in CI): Strata eval suite, Voxtype round trip, Spotify live smoke.
- Bugs get a failing regression test first; tests are never weakened to pass.
- Test runs open nothing on the owner's screen (hidden or headless; else workspace 1) and stop only processes they
  started, by recorded PID.

## Risks

- Tool-use quality and latency of Qwen3.6-35B-A3B in hybrid mode at ~8 GB VRAM on gfx1100 — phase 1 gate,
  fallback named.
- Memory and CPU under games — fixed VRAM cap, mapped experts, 5-minute idle unload, `jarvis unload`, load
  refusal below a reserve. Under RAM pressure the kernel drops cached experts and the next reply is slower.
- Voxtype contention with the owner's dictation — ownership rule or file path B.
- Spotify policy and quota changes — clear errors; MPRIS transport still works.
- Prompt injection from web pages, files or metadata — provenance and action-bound confirmation.
- Piper is GPL-3.0-or-later, so Jarvis is published under GPL-3.0-or-later (`docs/adr/0006-gpl-licence.md`).

## Co-author review record (Codex gpt-6.1-sol)

Accepted: phase reorder (boundaries before side effects), typed operations with default deny, provenance across
turns and action-bound confirmation, explicit audio state machine and timing, Voxtype ownership rule, a bigger model
eval, `strata serve` under systemd with non-generating health checks and a positive reasoning budget, a measured
resource gate, full Spotify setup and fallback rules, reproducible install with rollback, longer wake tests, v1 cuts
(no follow-up listening, no sentence streaming, one wizard library, SearXNG deferred).
Corrected facts: Omarchy 4.5.0-3; 64.9 GB RAM; only `base.en` installed; sizes in GB with units; Spotify Nov-2026
state; IPv6 loopback also allowed. Added by us: mapped-expert mode to fit beside desktop load.
