# Jarvis — build plan

Living design document: update it in place when the design changes. Drafted 2026-11-20 and reviewed by
Codex (gpt-6.1-sol) as co-author; decisions with lasting consequences are also recorded in `docs/adr/`.

An always-running, voice-activated local agent for this Omarchy PC. You say "Hey Jarvis"; Omarchy's local
dictation (Voxtype) transcribes; a local model on Strata reasons; typed tools act within boundaries you set;
Piper answers by voice. First core feature: Spotify.

## Owner requirements (fixed)

- Always running; wakes on "Hey Jarvis" (bare "Jarvis" later, optional).
- Transcription through Omarchy's local dictation (Voxtype, the configured default), not the API backend.
- Reasoning by the local model on Strata. Strata serves only Jarvis.
- First-time configuration sets boundaries, tone/persona and the core instructions (what Jarvis is).
- Researches online and operates this PC.
- No global instructions, skills or plugins except the Omarchy OS skills; no subagents: one main agent.
- First core feature: Spotify control (needs Spotify Premium).
- Replies by voice (Piper); no desktop notifications.
- Model idle by default (including after reboot), loaded on demand, unloaded when idle.
- Very easy to set up and maintain. Private repo `necturalabs/jarvis`.
- Goes open source once stable and confirmed working to our standards, so everything committed stays
  publishable from day one.
- Agents working on it open nothing on the owner's screen (hidden; else workspace 1) and never stop processes they
  did not start.

## Verified facts (this PC, 2026-11-20)

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
  OS file cache, which the kernel can reclaim. `models import --plan` previews only. CLI reinstalled since;
  model data kept.
- Model kept: Qwen3.8-Flash-Next IQ2_XS at `/mnt/models/ai/strata`: GGUF 68.0 GB + pack 1.55 GB +
  MTP 6.93 GB = 76.5 GB, file list and sizes verified after the move. Unused packs and all local image models deleted.
- LiveKit WakeWord 2.1.3 (Apache-2.0 code and weights per NOTICE), Python >= 3.11, deps numpy + onnxruntime,
  pretrained `hey_jarvis.onnx`; local detection, no account.
- Piper: `piper-tts` (OHF-Voice/piper1-gpl, GPL-3.0). Latency on this PC unmeasured.
- Omarchy OS skills: `/usr/share/omarchy/default/agents/skills/{omarchy,omarchy-app,diagnose-crash}/SKILL.md`.

## Architecture

```
 jarvis.service (user, one process, one agent)
   audio: PipeWire capture 16 kHz mono, timestamped ring buffer
     └─ state machine: IDLE → WAKE → CAPTURE → TRANSCRIBE → THINK → CONFIRM? → ACT → SPEAK → IDLE
          wake: LiveKit hey_jarvis      VAD: Silero (end of speech, no-speech, max duration)
          transcribe: Voxtype (path chosen by measurement, see below)
   agent: OpenAI chat + tools against Strata, step/time/output caps, validated tool calls
   policy: typed operations, default deny, provenance tracking, action-bound confirmation, audit log
   tools: spotify · pc (typed ops) · os_skill (read Omarchy SKILL.md) · web (later)
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
- Silence rules: no speech within 4 s after the wake = cancel; 1.0 s trailing silence ends an utterance; 20 s max.
  Wake phrase and chime audio are cut from what is transcribed.
- Device: explicit PipeWire source chosen in setup; the same source for wake, VAD and Voxtype, checked by `doctor`.
- **Transcription path, decided by measurement in phase 2:**
  - A. *Daemon contract* — `record start --file … --model small.en`, Jarvis VAD calls `record stop --wait --json`.
    Uses the warm daemon. Ownership rule: start only when `voxtype status --format json` reports idle, record the
    transcript path Jarvis created, and never stop or cancel a recording whose file is not Jarvis's. If the owner's
    own dictation begins mid-capture, Jarvis drops its request and says so.
  - B. *File transcription* — Jarvis writes its buffered utterance (with pre-roll) to a WAV and runs
    `voxtype transcribe`, parsing the transcript explicitly. No shared daemon state; costs a model load per call.
  - Measure both: wake-to-text latency, first-word clipping, accuracy on 50 recorded commands. Keep one.
- `small.en` (or the measured winner) is downloaded by `install.sh` through Voxtype's own model setup command.
- Media: during capture Jarvis lowers Spotify through MPRIS volume itself and restores the volume it saved, unless
  the command changed volume or playback (then the command's result wins). Voxtype's `pause_media` is the owner's
  setting; path A inherits it, path B does not touch it.

### Speaking, barge-in, confirmation
- v1: no voice barge-in while speaking; a keybinding (`jarvis stop`) interrupts. Wake detection is suspended during
  TTS, so Jarvis cannot trigger itself.
- Confirmation starts only after TTS playback has drained, accepts yes/no for 8 s, and is bound to one immutable
  action (hash of operation + arguments). Anything else, or silence, is no. The highest-risk class is confirmed by
  keybinding (`jarvis confirm`), never by voice.
- Voice barge-in and echo cancellation (PipeWire echo-cancel module) come in the polish phase.

### Model (Strata)
- Profile `jarvis`: Qwen3.8-Flash-Next IQ2_XS, 32K context, vision off, `lazy_load` on, `idle_unload_s` 1800,
  mapped experts (memory the kernel can reclaim), a positive `reasoning_budget_tokens` cap, local access with an
  API key.
- Under systemd: `strata serve --profile jarvis --ready-file …`, absolute paths, `Restart=on-failure` with backoff,
  `MemoryHigh`/`MemoryMax`. Health checks never generate, so boot and `doctor` leave the model unloaded.
- Jarvis says "One moment, waking up" when a request finds the model unloaded; `jarvis unload` frees it at once
  (e.g. before a game). It refuses to load when available memory is below a reserve set in config.
- Survives suspend/resume and engine crashes: the agent retries once after the server is ready again.

### Agent loop
- One agent; tools passed as JSON schemas; each assembled tool call is validated (schema + semantic checks) before
  the policy sees it.
- Caps per request: 6 tool steps, 60 s, output size per tool. Repeated identical calls are stopped.
- Uncertain outcomes (timeout after sending) are never retried blindly for non-idempotent actions (queue, skip,
  like); Jarvis says "I'm not sure that went through" instead.
- Replies are short and report what happened (tool results, not intentions). v1 speaks complete replies; sentence
  streaming later.
- Session memory: last ~10 turns, expires after 5 min idle. No follow-up listening in v1 (say "Hey Jarvis" again).
- System prompt: `core.md` (shipped: identity, tools, limits) + persona/tone + owner addendum from setup + a
  rendering of the boundaries + the Omarchy skill index (names and descriptions; full text through `os_skill`).

### Boundaries (policy engine, enforced in code)
- **Typed operations, default deny.** Each tool exposes named operations with validated arguments
  (`pc.open_app(name)`, `pc.set_volume(pct)`, `pc.workspace(n)`, `pc.read_file(path)`, `pc.list_dir(path)`,
  `pc.run_omarchy(cmd)` from an allowlisted set, …). Unknown operations are refused.
- Each operation has a class: `safe` (runs), `confirm` (spoken yes), `strong` (keybinding confirm), `deny`. The
  owner sets classes in setup; default profile "ask for risky".
- Generic shell (`pc.shell`) is off by default; when the owner enables it, every call is `strong`, runs with a
  timeout under `systemd-run --user --scope` with memory/CPU limits, no sudo, in a scrubbed environment.
- Process control: stop only a PID Jarvis shows, `strong` class; never by name or pattern.
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
  if Spotify isn't running, launch it (`omarchy-launch-spotify`) and wait for device discovery before transfer.
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
  pointing at `/mnt/models/ai/strata`; downloads wake-word and Piper voice files with checksums; asks Voxtype to
  download the transcription model; writes user units; runs `jarvis setup` on first run.
- `jarvis doctor`: one line per check with its fix; never loads the model unless `--load`.
- `jarvis update`: `git pull --ff-only`, `uv sync --frozen`, Strata update only when idle, restart units, `doctor`;
  keeps the previous checkout and Strata engine for `jarvis rollback`.
- `jarvis status | logs | pause | resume | unload | stop | confirm | setup | uninstall` (uninstall keeps config and
  the model unless asked).
- Lifecycle: units bound to the graphical session; defined behavior on lock (keep listening, configurable),
  suspend/resume (re-open mic, re-check Strata), mic unplug/replug (wait and re-open), missing `/mnt/models` mount
  (Strata unit fails cleanly, Jarvis says the model is unavailable).
- Repo `necturalabs/jarvis` (private); CI: ruff, pyright, pytest (unit only, no audio/GPU).

## Open-source readiness (from day one)

- Nothing machine- or owner-specific in code: paths (`/mnt/models/ai/strata`), devices, voices and accounts come
  from config with documented defaults; tests use fixtures, never the owner's data.
- History stays publishable: no personal data, tokens, recordings or internal hostnames in any commit; owner
  voice fixtures stay out of the repository unless the owner approves them for publication.
- Licences checked on every dependency and model added (code, weights, voices). Piper (`piper-tts`) is
  GPL-3.0, so the release licence must be GPL-3.0-compatible or Piper must be swappable behind the speech
  interface; decided before going public (ADR then).
- Each user brings their own Spotify developer app (client ID in setup), which fits Spotify's dev-mode limit.
- At release: `necturalabs:open-source-repo` audit (history and secret scan, community files, licence, Scorecard).

## Phases (each ends with measured evidence)

0. **Project setup** — git, AGENTS.md, docs (architecture, playbook, decisions), uv project, CI, private repo,
   issue backlog.
1. **Strata** — install the pinned CLI and engine, adopt IQ2_XS, `jarvis` profile, user unit. Measure on this PC
   under normal desktop load: cold and warm load time, TTFT, decode tok/s, peak RAM/VRAM, release after unload.
   Model eval: 200+ representative and adversarial requests on Jarvis's real tool schemas, scored on correct
   operation, arguments, abstention and finished task. Gate: >= 95% correct first operation, >= 90% finished
   multi-step tasks, 0 policy-violating calls reaching execution. Fallback model: Qwen3.6-35B-A3B.
2. **Ears** — capture, wake word, VAD, state machine, both transcription paths measured; pick one. Wake test:
   8 h of desk audio with music and calls for false wakes, 100 spoken wakes for misses.
3. **Boundaries, setup and voice** — config + migrations, policy engine with a full decision table, provenance,
   confirmation, audit log, the wizard, Piper output, the agent loop with fake-model tests. MVP: "Hey Jarvis, what
   time is it?" spoken back, and a denied action refused correctly.
4. **Spotify** — client, PKCE login in the wizard, MPRIS transport, device handling. Milestone: play artist / song
   / playlist, skip, volume, now playing, queue, like, all by voice, with recorded-fixture tests.
5. **PC operations** — typed operation set, os_skill, optional strong-class shell.
6. **Research** — SearXNG, search/fetch, URL policy, sourced answers.
7. **Polish** — follow-up listening, sentence streaming, voice barge-in with echo cancellation, bar indicator,
   optional bare-"Jarvis" wake word.
8. **Hardening** — install/update/rollback/uninstall tested on a clean user, lifecycle cases, docs and playbook.

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

- IQ2_XS tool-use quality and latency on gfx1100 — phase 1 gate, fallback model named.
- Memory under games — mapped experts, idle unload, `jarvis unload`, load refusal below a reserve.
- Voxtype contention with the owner's dictation — ownership rule or file path B.
- Spotify policy and quota changes — clear errors; MPRIS transport still works.
- Prompt injection from web pages, files or metadata — provenance and action-bound confirmation.
- Piper is GPL-3.0 — fine for a private repo; revisit if Jarvis is ever distributed.

## Co-author review record (Codex gpt-6.1-sol)

Accepted: phase reorder (boundaries before side effects), typed operations with default deny, provenance across
turns and action-bound confirmation, explicit audio state machine and timing, Voxtype ownership rule, a bigger model
eval, `strata serve` under systemd with non-generating health checks and a positive reasoning budget, a measured
resource gate, full Spotify setup and fallback rules, reproducible install with rollback, longer wake tests, v1 cuts
(no follow-up listening, no sentence streaming, one wizard library, SearXNG deferred).
Corrected facts: Omarchy 4.5.0-3; 64.9 GB RAM; only `base.en` installed; sizes in GB with units; Spotify Nov-2026
state; IPv6 loopback also allowed. Added by us: mapped-expert mode to fit beside desktop load.
