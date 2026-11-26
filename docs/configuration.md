# Configuration

Every setting lives in one TOML file, validated when Jarvis reads it. This page is written from
`src/jarvis/config.py`; the code wins if they differ.

## Where things live

| What | Where (default) | Set by |
|---|---|---|
| Config file | `~/.config/jarvis/config.toml` (`$XDG_CONFIG_HOME/jarvis/`) | `jarvis setup`, `jarvis pc-audio`, or by hand |
| Strata profile (generated) | `~/.config/jarvis/strata-profile.json` | `jarvis setup` |
| Jarvis's own Voxtype settings (generated) | `~/.config/jarvis/voxtype.toml`, and `voxtype-speech-test.toml` for `jarvis speech-test` | `jarvis run`, at each start; `jarvis speech-test` |
| Your own skills | `~/.config/jarvis/skills/<name>.md`, the folder and files yours alone ([ADR 0024](adr/0024-owner-skills.md)) | you, by hand; read at the next `jarvis restart` or chat window |
| User units | `~/.config/systemd/user/jarvis.service` and `jarvis-strata.service` | `jarvis setup`, `jarvis update` |
| Strata API key, Spotify tokens, audit log, action count, reminders and scheduled requests, Strata log | `~/.local/state/jarvis/` (`$XDG_STATE_HOME/jarvis/`): `strata.env`, `spotify-token.json`, `audit.jsonl` (and five older `audit.jsonl.1` to `.5`), `actions.json`, `schedule.json` (and its `schedule.json.lock`), `strata.log` | Jarvis; the key, token, log, count and jobs files are mode 0600 |
| Wake, VAD and voice model files | `data_dir`; the installer sets it to `$XDG_STATE_HOME/jarvis/models` on a fresh install | `uv run tools/get_models.py` or [manual model setup](models.md) |

[`config.example.toml`](../config.example.toml) in the repository lists every key at its default, with no
personal values: copy it to `~/.config/jarvis/config.toml` and change what you need. A missing config file
means all defaults. An unknown key, an out-of-range value, a malformed file or a
`schema_version` other than `1` is an error that names the file and key; `jarvis` exits with status 2.
Paths and names may not contain control characters, because they are written into systemd units.

## Which changes need a restart

Jarvis reads the file when a command starts, so `jarvis ask`, `status`, `unload` and `doctor` always see the
current file. The running voice service reads it once, when it starts.

- `audio`, `persona`, `boundaries`, `spotify`, `model`, `clock`, `weather`, `research`, `files` and `sing` keys:
  run `jarvis restart`.
- `strata` keys: run `jarvis restart`. It writes Strata's profile and unit from the config, restarts
  Strata when either changed since Strata started, and otherwise leaves a running Strata, and its loaded
  model, alone. Setup and updates write the voice service's unit. Until setup has run, restart writes
  nothing. A config it cannot read, or a file it cannot write, is said and the voice service restarts
  anyway. An empty `model.base_url` follows
  Strata's port; a saved URL that only repeated that port is dropped when the file is read, and
  `jarvis doctor` fails a saved URL on another port than Strata's.
- `jarvis pc-audio on|off` changes the file and restarts the voice service itself when it is running.

A command or the chat that saves a setting (`jarvis setup`, `consent`, `pc-audio`, `wake`, the chat's
`/config`) changes only that setting in the file and keeps the rest as you wrote it, comments included. It
never writes a default the file does not already hold, and takes out a key set back to its default, so a
changed default in a later version reaches you.

## Top level

| Key | Default | Allowed | What it does |
|---|---|---|---|
| `schema_version` | `1` | `1` | File format version. Any other value is refused. |
| `data_dir` | `~/.local/share/jarvis` | path | Where the wake word, VAD and voice model files live. `jarvis setup --data-dir`. |

## Update

| Key in `[update]` | Default | What it does |
|---|---|---|
| `branch` | `main` | Release branch. The installed checkout must have this branch checked out, tracking the same branch on a named Git remote. |
| `auto` | `false` | Allows the automatic update timer to apply updates. Use `jarvis update --auto on` to create and enable it, or `jarvis update --auto off` to disable and remove it. |

Run `jarvis update` to fetch and preview the release commits and file changes.
`jarvis update --apply` fast-forwards a clean checkout, syncs the checkout's `.venv` from the lockfile,
and regenerates its Strata profile and user units when needed. It restarts a running voice service
only when idle, leaves stopped services stopped and keeps a running Strata if its settings are unchanged.
If validation or service startup fails, it restores the previous commit, environment and generated
files. If another program edits the checkout during an update, rollback can stop and report the
previous commit for manual recovery.

The timer checks daily with up to two hours of randomized delay. Update attempts go to the local
audit log, with errors and applied revisions also in the timer's systemd journal. `jarvis doctor`
shows availability as of the last fetch, automatic update settings and the last audit result.
`jarvis doctor --no-service` checks the installation before the services are enabled.
Use `--essential` to report any Spotify login problems as notes.
See [ADR 0019](adr/0019-install-and-update.md) for the release trust and rollback boundaries.

## `[model]`: how Jarvis talks to Strata

| Key | Default | Allowed | What it does |
|---|---|---|---|
| `base_url` | empty | empty, or an `http://` or `https://` URL on `127.0.0.1`, `localhost` or `::1`; no user name or password | Empty means `http://127.0.0.1:<strata.port>/v1`. |
| `model` | `jarvis` | text | The model name sent in each request. |
| `request_timeout_s` | `120.0` | above 0, up to 600 | Longest one model call may wait; a request's first call waits 180 s more when the model must load first. A request also stops after `agent.max_seconds` (`chat.max_seconds` in the chat) in total. |
| `temperature` | `0.2` | 0 to 2 | Sampling temperature. |

Jarvis asks for thinking as `[agent]` and `[chat]` set it (`thinking`, `thinking_tokens`); with `none` it switches thinking off. The model may raise its own effort for the rest of one request, up to `thinking_max` (the `think_harder` tool); the next request starts at `thinking` again. A summary of older turns is always written without thinking.

Older versions saved `max_tokens` here; a file that still holds it loads, and the key is ignored: the reply cap is now `reply_tokens` in `[agent]` and
`[chat]`.

## `[agent]`: what one spoken request may do, and what Jarvis remembers

| Key | Default | Allowed | What it does |
|---|---|---|---|
| `max_steps` | `8` | 1 to 100 | Most model calls in one request; the last one may only answer, from what the others found. |
| `max_tool_calls` | `12` | 1 to 200 | Most tool calls in one request. The reply call (`no_action`) is the answer and is not counted. |
| `max_seconds` | `90.0` | 10 to 3600 | Longest one request may take (180 s more when the model must load first). Time spent asking you and waiting for your yes or the confirm key does not count. Once an action has run, the model still gets at least 15 s to say how it went, and a request that runs out of time after one says what it did. |
| `reply_tokens` | `400` | 64 to 8192 | Most tokens the model writes in one answer. The prompt asks for replies that fit; a reply the cap ends anyway is kept and says it was cut short. A tool call the cap cuts off (a long file being written) is not run: the model is told how much one answer holds and writes the text in parts, adding each to the file; a second cut call ends the request and says so. |
| `tool_text_chars` | `3000` | 500 to 50000 | Most characters of one tool result the model reads. |
| `memory_tokens` | `3000` | 0 to 200000 | Room the remembered conversation may take of the context window. When a reply makes it outgrow that, the oldest requests' tool calls and results (kept with `memory_detail = "full"`) are left out first, down to three quarters of the room, the latest request's only when it alone does not fit; when it still does not fit, the model summarizes the oldest turns right after the reply, while it is spoken and on its own time (the latest two stay word for word); if that fails, the oldest are dropped. A request that finds it over its room (a plain command's turns, a saved conversation) is given the newest turns that fit, and the summary follows its reply. Jarvis uses less when the window cannot hold it beside the prompt, a reply and two tool results. |
| `memory_detail` | `condensed` | `condensed`, `full` | What later requests keep of one. `condensed`: the request, the reply, what was done and what was looked up. `full`: also its tool calls and their results, as the model read them, so a follow-up has the pages, files and outputs a request read without fetching them again, and the model server reuses what it already read of them. Outside text in them stays data and keeps actions behind a yes while it is remembered. A request in full detail takes more of `memory_tokens`; when it runs short, the oldest work goes first, and the chat says in one line what it condensed. |
| `memory_idle_s` | `300.0` | 0 or more | Forget the conversation after this long without a request (time the PC spends asleep counts); `0` never forgets. Speech is short-term memory for quick commands: a follow-up ("louder", "the next one", "what was that song?") comes within a minute or two, and after five idle minutes an old exchange is likelier to steer the next command wrong than to help. `jarvis ask` remembers nothing between runs; a scheduled request shares speech's conversation and these limits. |
| `thinking` | `none` | `none`, `low`, `medium`, `high` | How hard the model thinks before each answer at the start of a request (Strata's reasoning effort). `none` answers at once; on everyday requests thinking made answers slower and no more accurate ([models.md](models.md#thinking)). Thinking is never spoken or shown, and costs time: at about 65 tokens a second, the full cap below is about 3 s more before the first words of each model call. |
| `thinking_tokens` | `192` | 32 to 8192 | The most the model thinks before each answer at `thinking`; Strata then closes the thinking and lets it answer. It comes on top of `reply_tokens`, so the reply keeps all its room, and the context keeps room for both. |
| `thinking_max` | `high` | `none`, `low`, `medium`, `high`, and not below `thinking` | The most the model may raise its own effort to, for the rest of one request, when it judges the request needs careful thought (several steps, a name it may have misheard, a correction, a try that failed). It does so with the `think_harder` tool, which is offered only when this is above `thinking`: set it to the same as `thinking` to keep one effort. The effort never drops within a request, and each request starts at `thinking` again. Raising to `low` or `high` can cost the model one more read of its whole prompt on Qwen3.8 ([models.md](models.md#thinking)). The tool's class is `safe`; `think_harder = "deny"` under `[boundaries.classes]` also switches it off. |
| `thinking_raised_tokens` | `768` | 32 to 8192 | The most the model thinks before each answer once it raised its effort: about 12 s at most for speech. A raised call never gets less than `thinking_tokens` when a request starts above `none`. The context keeps room for the larger of the two. |
| `browser` | `false` | `true`, `false` | Offer the headless browser tools (`browser_open`, `browser_act`; see [`[browser]`](#browser-the-headless-browser)) to spoken requests and `jarvis ask`, when agent-browser is installed. Off by default: speech is for quick commands, and clicking or typing on a page always waits for your spoken yes. |

A spoken reply that drew on outside text (a web page, a song name) is remembered as data, as in the chat: a
follow-up such as "what was the second one?" has it, and while it is remembered an action the model chooses
needs your yes; a plain command (`pause`, `volume 40`) does not. A request the model answers while such text is
remembered is marked the same way, so that holds after the turn that read it is forgotten.

## `[chat]`: the same for typed requests in `jarvis chat`

The same keys as `[agent]`, with room for longer written work: `max_steps` `30`, `max_tool_calls` `40`,
`max_seconds` `600.0`, `reply_tokens` `2048`, `tool_text_chars` `8000` (a read page fills it),
`memory_tokens` `5700` (what the default 24,576-token window holds beside the raised thinking's room, the browser tools and the window and screenshot tools; less when a thinking cap is larger; with a larger `strata.context` a larger value has room for more of the earlier requests' work), `memory_detail` `full`, `memory_idle_s` `0.0` (never forget), `thinking` `none`, `thinking_tokens` `512`, `thinking_max` `high` and `thinking_raised_tokens` `1024`, `browser` `true` (the headless browser tools, `browser_screenshot` too, when agent-browser is installed), plus:

| Key | Default | Allowed | What it does |
|---|---|---|---|
| `resume` | `true` | `true`, `false` | Open the chat on the conversation it had last time, kept 0600 in `~/.local/state/jarvis/chat-conversation.json`. `false` starts each session fresh and deletes the file. |
| `verbosity` | `replies` | `replies`, `steps`, `everything` | How much of Jarvis's work the chat shows: requests and replies only, plus each step and its result, or plus every model call and detail. Ctrl+O and `/verbosity` change it and save it here. |
| `toolbar` | `true` | `true`, `false` | The line under the input with the PC's CPU, GPU, VRAM and RAM use, temperatures, and how full the model's context is. In a narrow window it leaves out the temperatures first, so it stays on one line. |
| `while_working` | `steer` | `steer`, `queue` | What Enter does while a request runs. `steer`: the words join that request as yours, before the model's next call; words it ended before taking run next. `queue`: they wait and run as the next request. Words sent with a picture or with anything pasted into them, however short, are always queued. Esc still stops the request; queued words then run. `/config` changes it. |
| `response_format` | `concise` | `concise`, `normal`, `verbose` | How the model writes chat replies: briefly, in a few sentences and without headings; as much as the request needs; or explained in full, with headings for a long answer. `/format` and `/config` change it, from the next request on. The voice keeps its own short spoken style. |
| `archive_keep` | `50` | 0 to 1000 | Chat conversations kept in `~/.local/state/jarvis/chat-archive/` (0600) for `/resume` and reading back: each with its full log of requests, steps, results and replies. Older ones are deleted; `0` keeps none, deletes those kept before when the chat opens, and turns `/resume` off. `/forget all` deletes them all and forgets the conversation in the window too. A conversation is written once it has a request; opening and closing the chat writes no copy of the one it remembers. |
| `history_size` | `1000` | 0 to 100000 | Typed lines kept for ↑ between sessions, with what was pasted into each, 0600 in `~/.local/state/jarvis/chat-history`; `0` keeps none. |

`/config` in the chat opens the settings in sections, each setting with its values, when a change applies
and a line on what it does; ↑↓ move, Enter or → takes the next value, ← the one before, Esc closes. Under
Chat it changes `verbosity`, `toolbar`, `while_working` and `thinking` at once, and `reply_tokens`,
`max_steps`, `memory_detail`, `resume` and `browser` from the next time the chat opens, writing them here.
Under Chat it also sets `browser.idle_close_s` (30, 60, 120 or 300 seconds), under Voice
`audio.speech_speed` and `agent.browser`, under Listening and Safety `audio.listen_to_pc` (off by default)
and `boundaries.bypass_consent`, and under Tools each tool's setting; the voice service restarts once to take
these when the screen closes, and the open chat takes `bypass_consent` at once (what Jarvis says about its
own setup, `about_jarvis`, follows when the chat next opens).
Acting without asking and taking requests from PC audio are turned on only by the same step again to the same
value (a second Enter or →, or ← after ←), after the row says what that means and which value, and are marked in
red while they are on; a step to another value asks about that one instead. A restart that fails is said in the chat.

`/resume` lists the archived conversations, newest first (when, name, how many requests), and `/resume 2`
continues the second: the model gets that conversation back as it was, so what came from outside text then
still counts as data and still puts actions behind a yes. A conversation is named after its first request
until `/rename NAME` names it; the name stays when it is resumed. In Listening, `/config` also sets
`audio.pc_audio_recall_s` (0, 10, 30 or 60 seconds), asked about before it first leaves 0.

`/effort` in the chat says how hard the model thinks before it answers, and how far it may raise that itself
for a hard request. `/effort none`, `low`, `medium` or `high` sets `thinking` here, from the next request on;
when `thinking_max` is lower, it is raised to the same level. Every chat setting saved here is written to the
audit log as `setting_changed`, after the save.

The chat remembers replies that drew on outside text, as data. While one is remembered, an action with a side
effect that the model chooses needs your yes; a plain command (`pause`, `volume 40`) does not, since the
model plays no part in it.

Pictures in the chat need the model's picture reader (`strata.pictures`, below). Ctrl+V pastes a copied picture
as an `[Image #1]` tag in the input, as in Claude Code, or copied text where the cursor is; the terminal's
own paste (Ctrl+Shift+V) carries only text. A picture file dropped on the terminal, or `/image <path>`,
attaches that file, and `/image` alone attaches the newest screenshot in `OMASNAP_SCREENSHOT_DIR`,
`~/Pictures/Screenshots` or `~/Pictures`. Delete a tag to leave its picture out. Each picture is checked
and shrunk to 1,568 pixels on its longest side before it goes to the model. What a picture shows counts as
outside text: in that request, an action the model chooses needs your yes. The conversation keeps the
picture's name, not the picture.

Pasted text counts as outside text too when what you pasted into a request is longer than 200
characters in all and Jarvis asks first (`boundaries.bypass_consent = false`): the model reads it marked
as data, and in that request, and while the conversation remembers it, an action the model chooses needs
your yes. Several short pastes count together, and the same text pasted twice counts twice. Shorter
pastes and what you type stay your own words. With consent bypassed a paste is your own words whatever
its length. If you edit a paste in the box and the pastes together are long, the whole request counts.
A request brought back with ↑ keeps what was pasted into it, so a long paste sent again still counts.

When a question is about writing a file, the chat shows the first three lines of the new text under it;
by voice Jarvis says only how many characters.

Ask the chat to show you something, in any words, and it searches for pictures and draws the best
one in the window, through kitty's graphics protocol (Sixel in a Sixel terminal, coloured half cells
elsewhere). It fetches only a picture that search offered, from a public address, and does not see it itself.

## `[strata]`: how the `jarvis-strata` service runs the model

These are written into the Strata profile and unit by `jarvis setup`. Strata serves only Jarvis and listens on
`127.0.0.1`.

| Key | Default | Allowed | What it does |
|---|---|---|---|
| `home` | `~/.local/share/strata` | path | Strata's data folder (`STRATA_DATA_DIR`). `--strata-home`. |
| `components_parent` | `~/.local/share` | path | The `XDG_DATA_HOME` Strata runs with. `--strata-components-parent`. |
| `run_config` | unset | path | Strata's own run config for the model: the `strata-<model>.json` that `strata launch` writes in Strata's `install` folder. The model, its engine, its flags, its tokenizer, its name and its picture reader all come from it; Jarvis sets only the limits in this table. Once set, the Strata unit starts only if the model file it names exists, and `jarvis setup` refuses a run config naming a file that is gone (mount the drive, or run `strata launch` for the model again). A model whose drive is not mounted is still known by its path, so Jarvis says the file is missing at once and keeps pictures on for when it is back. A run config with its own `--host` or `--port` is refused: Jarvis serves on `127.0.0.1` at `port`. Unset, `jarvis setup` saves the config and stops there; `jarvis doctor` fails until it is set. `--run-config`. |
| `coding_run_config` | unset | path | Strata's run config for coding mode's own model ([`[coding]`](#coding-coding-mode)); unset: coding mode keeps `run_config`. `--coding-run-config`. |
| `port` | `8080` | 1024 to 65535 | Strata's port on `127.0.0.1`. |
| `context` | `24576` | 2048 to 262144 | Context window in tokens (`--max-context`). With `vram_reserve_mib` set, the engine's VRAM stays at its limit and a larger context takes the room from expert slots on the card instead; without it, each 1,000 tokens more costs about 0.05 GiB of VRAM ([models.md](models.md#context-size)). It holds the prompt and the tool schemas (about 5,000 tokens together), the remembered conversation (`agent.memory_tokens` or `chat.memory_tokens`), one reply and the request's tool results; when a long request outgrows it, its oldest tool results are cleared down to their start. |
| `expert_vram_gib` | `1.0` | 0 to 64 | Expert weights kept in VRAM, for an engine that takes a size: passed as `--expert-vram` or `--vram-experts-gb`, whichever the run config has; the planner keeps the other experts in RAM (Jarvis drops the run config's `--ram-experts-gb`). An engine whose expert cache fills the card (`--expert-cache auto`, Qwen3.8-Flash-Next) ignores it: limit that one with `vram_reserve_mib`. |
| `idle_unload_s` | `86400` | 30 to 86400 | Seconds without a request before Strata frees the model. The default, 24 hours, keeps it loaded through the day; a lower value gives the RAM back sooner. |
| `reasoning_budget_tokens` | `512` | 1 to 8192 | Thinking-token cap in the profile, for a request that names none; Jarvis's own requests name theirs (`thinking_tokens` in `[agent]` and `[chat]`). |
| `effort_position` | `""` | `""`, `start`, `end` | Where Strata writes a request's thinking effort, as its `effort_position` in the profile; empty keeps the run config's own (Strata's default is `start`). With `end`, a `low` or `medium` effort goes into a short turn before the answer, so a request whose model raised its effort (`think_harder`) reuses the pinned prompt instead of reading it all again; `high` stays at the start. How well the model follows an effort written there is not measured yet ([models.md](models.md#thinking)), so it is not the default. |
| `memory_max` | `40G` | digits and optional `K`, `M`, `G` or `T` | `MemoryHigh` of the Strata unit. `MemoryMax` is derived from it, 10% above (44G for 40G; a size that does not divide evenly is rounded up and written in bytes), so there is no second key. Past `MemoryHigh` the kernel throttles the unit and reclaims its memory; past `MemoryMax` it kills the unit. The size depends on the model: Qwen3.8-Flash-Next IQ2_XS needs `44G` ([models.md](models.md)). |
| `keep_in_ram` | `true` | `true` or `false` | Keep the model server's memory in RAM: its unit gets `MemorySwapMax=0`, so the kernel never swaps it out, and `LimitMEMLOCK=infinity`, so the engine can lock its model as it asks to. Off, a PC short of memory swaps part of the model out between requests and back in on the next one; while it does, the graphics driver pauses the GPU ([Keeping the model in RAM](#keeping-the-model-in-ram)). Turn it off only with swap on a disk and a game that needs that memory more than Jarvis needs a quick answer. |
| `speed_projection` | `true` | `true` or `false` | Keep a control vector the run config loads, such as Strata's experimental speed projection, which makes the model decline far fewer requests ([The speed projection](#the-speed-projection)). Off, Jarvis's profile leaves the control-vector options out, so the model declines as it was trained to. Jarvis never adds a vector itself. |
| `vram_reserve_mib` | `0` (unset) | 0, or 100 to 65536 | VRAM in MiB the engine leaves free for other programs, passed as `--vram-reserve-mib`. `0` passes nothing, so Strata's own default (700) applies. For an engine whose expert cache fills the card (`--expert-cache auto`) it is the VRAM limit and must be set: `jarvis setup` refuses such a run config without it, and `jarvis doctor` fails "VRAM left for games". On a 20 GB card, 12000 holds Qwen3.8-Flash-Next at about 8 GiB ([models.md](models.md)). A card with at most about 8 GB of VRAM cannot start with a reserve (Strata's tuner never tries one there). |
| `kv_resident` | `0` (all in VRAM) | 0, or 20480 to 262144 | Context cells kept in VRAM, passed as `--kv-resident`. Set, the whole context also lives in pinned RAM and only this many cells stay on the card, so `context` can grow past them without more VRAM; the VRAM saved goes to expert slots. Pinned RAM counts against `memory_max`. A `context` of this many cells or fewer is not streamed. Reading past the cells on the card is slower; measure before raising `context` far. |
| `learn_experts` | `true` | `true` or `false` | For an engine whose run config starts from a routing profile (`--expert-profile`): the engine saves which experts Jarvis's requests use to `~/.local/state/jarvis/strata-expert-profile-<model>.bin` (one file per model, so one model's profile never starts another) (when it stops, and every 10 minutes between requests) and starts from that file next time, so those experts are on the card from the first request. Nothing is deleted from the model. `jarvis setup` starts the file once from a copy of the run config's own profile and never overwrites what the engine learned; `jarvis doctor` fails when it is missing or empty. A routing profile the run config names by a relative path is refused. |
| `pictures` | `true` | `true` or `false` | Read pictures with the reader the run config names: Strata's reader beside the main engine runs on the processor, in RAM and no VRAM, from the first picture until `vision_idle_s` without one; Inkling's engine reads them itself. `false` drops the reader. A run config without a reader reads no pictures either way. |
| `vision_idle_s` | `120` | 10 to 86400 | Seconds without a picture before the picture reader stops. |

Fixed by Jarvis, not configurable: lazy loading on, one request at a time, `127.0.0.1`, its own log and working folder. Everything else in the profile is the run config's.

### What the two units set

Both units are written by `jarvis setup`; their text is `jarvis_unit` and `strata_unit` in `src/jarvis/services.py`.

| Setting | `jarvis.service` | `jarvis-strata.service` |
|---|---|---|
| `NoNewPrivileges=yes`, `LockPersonality=yes`, `RestrictSUIDSGID=yes` | yes | yes |
| `UMask=0077` (what the unit creates is private to you) | no | yes |
| `SystemCallArchitectures=native` | yes | no |
| `CPUWeight=50` (half the default, so a game started by Steam in the same `app.slice` wins a CPU contest) | yes | yes |
| `StartLimitIntervalSec=15min`, `StartLimitBurst=5` (in `[Unit]`) | yes | yes |
| `MemoryHigh` / `MemoryMax` | none / `4G` | `memory_max` / 10% above it |
| `LimitMEMLOCK=infinity`, `MemorySwapMax=0` | no | unless `strata.keep_in_ram` is off |

- The voice service keeps your session's umask, so a file the file tools write at your request gets the
  mode it asks for (0644, a folder 0755).
- A program you ask Jarvis to open (`open_app`) is started by Hyprland, not by the voice service, so
  it does not inherit these settings.
- Not set, each for a reason: `ProtectKernelModules=` and `ProtectKernelLogs=` on the voice service (in a user
  unit they put it in a private user namespace, which Spotify and every other program it starts would inherit),
  `RestrictNamespaces=` (the Azure wake word helper runs under `unshare --map-current-user --net`),
  `MemoryDenyWriteExecute=` (onnxruntime and Strata's HIP path may compile code at run time), `ProtectHome=` and
  `ProtectSystem=` (config, state and models live under home and on other disks), `PrivateNetwork=` and `IPAddressDeny=`
  (Spotify, Open-Meteo and web reads), `RestrictRealtime=` (`pw-record` and `pw-play` are not shown to
  do without real-time scheduling), `Nice=` and the `idle` scheduling policies (a reply would starve while a game
  runs; the model is paged in from disk, so `IOSchedulingClass=idle` would slow it too). `IOWeight=` does nothing: the
  user manager delegates only the `cpu`, `memory` and `pids` controllers.
- The start limit: a unit that fails five times in 15 minutes stays `failed` and shows in `jarvis status` instead
  of restarting for ever (each Strata restart reads the model again). `jarvis start` and `jarvis restart` run
  `systemctl --user reset-failed` first on a unit that is `failed`, so they bring it back.
- These options were checked with transient units on this PC (`systemd-run --user`), not with the model loaded
  on the GPU; [the playbook](playbook/README.md#owner-runs-for-the-service-units-33) lists what the owner still runs.

### Keeping the model loaded

`idle_unload_s` defaults to 86400 (24 hours), its largest value, which keeps the model loaded through the day. Measured on the
reference machine (RX 7900 XT, 64 GB RAM), a loaded idle model holds about 38 GB of RAM and 7.3 GiB of VRAM and
uses no GPU or CPU while idle. A cold load took about 107 s with a game running; Jarvis says "One moment,
waking up" while that happens. At 300 s, say, the model is freed after five idle minutes and loaded again on
the next wake word. The default `memory_max` of 40G leaves about 2 GB above the 38 GB measured, so
raise it before holding a larger model. `jarvis unload` frees the model at once.

### Keeping the model in RAM

Strata keeps most of the model in ordinary RAM that the GPU reads from (33 GiB for Qwen3.8-Flash-Next). The
kernel is free to swap that memory out when the PC runs short, and has to bring it back for the next
request. Each such move makes the graphics driver pause the GPU. Measured on the reference machine on
2026-11-25 (60 GiB of RAM, 9 GB of it available, 11 GB already in compressed swap, no game running): 2.7 GB of
the model was in swap, the server took 8.0 million major page faults in 4.5 hours, and Strata's log reported
the GPU paused during 12 of 119 model calls, by 1.5 s at the median and 9.8 s at most. A voice request's
first model call took a median 1.59 s and 5.28 s at the 90th percentile; the same call in runs one after
another took 0.57 s.

`strata.keep_in_ram` (on by default) keeps the model out of swap: the Strata unit gets `MemorySwapMax=0`
and `LimitMEMLOCK=infinity`. `jarvis restart` writes the unit from the config and, since it changed,
restarts the model server with it. How much of the pausing that removes has not been measured yet; it
takes days of use with the setting on, and Strata's log (`paused the GPU`) shows it.

What swapping the model out would win is small. Its weights are quantized and barely compress: a 256 MiB
slice of Qwen3.8-Flash-Next's file takes 251 MiB under zstd, so on compressed swap in RAM (zram, the
default on many desktop distributions) moving it out frees about 2 % of what it moves. Only swap on a disk
makes real room, and then the cost is the pause plus a disk read. With the setting on, the rest of the PC
swaps a little sooner instead, and `jarvis unload` still frees all of the model's memory before a heavy
game. With no swap the unit's memory limits matter more: past `strata.memory_max` the kernel slows the
server down rather than swapping it, and past `MemoryMax`, 10% higher, it stops the server, so keep that
limit above the model's own size, as the default is.
Turn the setting off when the PC swaps to a disk and a game needs the room more than Jarvis needs a quick
answer.

`MemorySwapMax=0` works in any user unit. The lock does not by default: a user unit can raise a limit only
up to its user manager's hard limit, and for locked memory that is 8 MiB on a stock systemd (the unit still
starts, with that limit). To raise the hard limit for your own user services and leave the default as it
is, as root:

```ini
# /etc/systemd/system/user@<your uid>.service.d/memlock.conf
[Service]
LimitMEMLOCK=8M:infinity
```

then `systemctl daemon-reload`, and log out and in. The swap setting alone already keeps the model from
being swapped; the lock is what Strata's own start-up text asks a systemd service for.

### The speed projection

Strata ships a control vector for Qwen3.8-Flash-Next, its "experimental speed projection" (in its
control component's `data/experimental-speed-projection/`, with a README). Strata's own README says what it
does: the engine removes one direction from the model's activations, and the model then declines far fewer
requests (1 of 50 against 50 of 50 on its authors' test). That takes away a safety behaviour of the model.
It also shifts the model's answers slightly on ordinary text, and costs 0.2 to 0.4% per token. Strata's
setup leaves it off.

To load it, add its options to the model's run config, either in `args` or in `user_args`, which Strata keeps
when its setup runs again. Take the values from the JSON file beside the vector:

```json
"user_args": ["--control-vector", "/path/to/Qwen3.8-Flash-Next-experimental-speed-projection.gguf",
              "--control-vector-layer-range", "4", "44", "--cvec-mode", "project", "--cvec-dir", "per-layer"]
```

Then run `jarvis restart`. With `strata.speed_projection` on (the default), Jarvis keeps whatever vector the run
config loads; off, it leaves the vector out without touching the run config. Jarvis's own boundaries do not
depend on the model declining anything: every action still passes the policy, its confirmation and the
audit log in code. How the vector changes Jarvis's tool calls has not been measured.

## `[coding]`: coding mode

Jarvis is in one of two modes, `normal` or `coding` ([ADR 0017](adr/0017-coding-mode.md)). Coding mode is for
programming work: more model calls, longer replies, more thinking, a larger context, a part on programming
added to the prompt, and, when you set one, a model of its own.

- **Switching.** `jarvis model` shows the mode, each mode's model and whether it is loaded; `jarvis model
  coding` and `jarvis model normal` switch, and so does `/model` in the chat. Where `auto` is on, the model
  switches by itself when a request is programming work, and back when you say that work is over
  (`switch_model`; switch it off like any tool with `jarvis tools switch_model off`). After
  `back_after_s` without a request in coding mode Jarvis goes back to normal. In the chat, `/config` has
  these under Mode: `back_after_s`, `coding.chat.auto` and `coding.voice.auto`. What the chat changed since
  it started (`/effort`, `/format`, `/config`) is kept when the mode changes; coding mode still starts at
  its own `[coding.chat]` values where it sets them.
- **The model.** `strata.coding_run_config` names Strata's run config for the coding model, as
  `strata.run_config` does for the usual one: run `strata launch` for it once, then `jarvis setup
  --coding-run-config <its strata-<model>.json>`. Unset, coding mode keeps the usual model. With two
  models, a switch restarts the model server: the first model is out of memory before the second loads, so
  only one is ever in memory, and Jarvis says before it starts how long the load took the last time. A
  switch is refused while Jarvis is answering a request anywhere (the voice service, a chat window or
  `jarvis ask`), and a request that begins during a switch waits for the model server to come back. `jarvis doctor` checks the coding
  model's files too.
- **Every switch is audited** (`mode_switched`: from, to, by the owner, the model or the idle time, the
  model's name, whether the server restarted, the seconds).

| Key | Default | What it does |
|---|---|---|
| `back_after_s` | `1800.0` | Back to normal after this long without a request in coding mode; `0` never |
| `context` | `65536` | The model server's context in coding mode (`strata.context` in normal mode) |
| `temperature` | unset | The sampling temperature in coding mode; unset: `model.temperature` |
| `vram_reserve_mib`, `expert_vram_gib`, `memory_max` | unset | The same keys of `[strata]`, for coding mode only; unset (`0`, absent, `""`): `[strata]`'s |
| `reasoning_budget_tokens` | `4096` | Strata's cap on thinking in coding mode |
| `tools_off` | `[]` | Tools not offered in coding mode |
| `write_roots` | `[]` | Folders Jarvis may also write in coding mode, besides `files.write_roots` |
| `prompt` | `""` | Your own instructions for programming work, added to the prompt in coding mode |

`[coding.chat]` is laid over `[chat]` and `[coding.voice]` over `[agent]` while Jarvis is in coding mode. Each
holds only what coding changes; a key you take out keeps the channel's own value, and everything else
(`chat.resume`, `chat.verbosity`, the voice's `memory_idle_s` and so on) is never touched.

| Key | `[coding.chat]` | `[coding.voice]` |
|---|---|---|
| `max_steps` | `60` | `20` |
| `max_tool_calls` | `120` | `40` |
| `max_seconds` | `1800.0` | `300.0` |
| `reply_tokens` | `8192` | `4096` |
| `tool_text_chars` | `20000` | unset |
| `memory_tokens` | `20000` | unset |
| `thinking` | `medium` | `low` |
| `thinking_tokens` | `1024` | `512` |
| `thinking_max` | unset | unset |
| `thinking_raised_tokens` | `4096` | `2048` |
| `auto` | `true` | `false` |

The chat's overlay is for long, written work. The voice's is small on purpose: a spoken, `jarvis ask` or
scheduled request stays a quick command with its short memory, and only gets room to write a file. `auto` is
off there because a model load in the middle of a spoken request is a long silence, and every spoken request
after it would be answered by the coding model; set `coding.voice.auto = true` to let spoken requests switch
too. `memory_tokens` is still held to what the context leaves, as in `[chat]`.

## `[spotify]`: your own Spotify developer app

| Key | Default | Allowed | What it does |
|---|---|---|---|
| `client_id` | empty | text | Your app's client ID. There is no client secret. `--spotify-client-id`. |
| `redirect_port` | `8765` | 1024 to 65535 | Port of the login callback, `http://127.0.0.1:<port>/callback`; your app must list exactly this redirect URI. |
| `device_name` | empty | text | Preferred player, matched by name ignoring case. Empty means whichever player is active or listed. |
| `close_match` | `0.8` | above 0, up to 1 | How similar a search result must be to your words to be taken as the answer. Names are compared by spelling and by sound, whichever is closer. |
| `loose_match` | `0.6` | above 0, up to 1, and not above `close_match` | The weaker similarity used to accept an artist or a library item. |
| `queue_max` | `20` | 1 to 100 | Most songs one queue request may add. |
| `search_pages` | `8` | 1 to 20 | Search result pages read while looking for a match. |
| `library_refresh_s` | `21600` | 60 or more | How often the playlist and liked-songs cache is refreshed in the background. |

The login (`jarvis spotify-login`) asks Spotify for: reading and controlling playback, reading what plays and
what played recently, reading and changing your library (Liked Songs), reading your playlists and changing
them (`playlist-modify-private`, `playlist-modify-public`, added 2026-11-25 so the song playing can go into one of
your playlists), and reading your top items and the artists you follow. A login from before 2026-11-25 lacks
the playlist ones: everything else works, and adding to a playlist says to run `jarvis spotify-login` once.

## `[audio]`: microphone, wake word, speech and voice

| Key | Default | Allowed | What it does |
|---|---|---|---|
| `source` | empty | PipeWire source node name | Microphone: a node name or serial. Empty means the default source. `jarvis doctor` checks that it names a microphone (an `Audio/Source` node) that PipeWire lists. A capture that sends no audio for 5 s counts as lost and is reopened. `--mic`. |
| `wake_words` | `["hey_jarvis", "yo_jarvis"]` | 1 to 8 distinct names, each up to 40 characters: lowercase letters, digits and `_`, starting with a letter or digit | The LiveKit wake word models that listen, side by side: each is `<data_dir>/wake/<name>.onnx`, and any of them wakes Jarvis. A model that is missing or cannot be loaded is left out with a warning in the log and the rest still listen; `jarvis doctor` fails when none of them has a file or when a file does not load (it names the file and the error), and notes each one that is missing. A model uses its score in `wake_thresholds`, else `wake_threshold`. `jarvis wake --words hey_jarvis,yo_jarvis` sets the list; `jarvis wake` lists every model in that folder as on or off, and marks one that is on but has no file. Only `hey_jarvis` is downloaded; train others as [Models](models.md#wake-word-and-end-of-speech) says. |
| `wake_threshold` | `0.5` | above 0, below 1 | LiveKit's wake word score needed. Higher means fewer false wakes and more misses. The service log shows each wake's score. |
| `wake_thresholds` | empty | up to 32 wake word names, each with a score above 0, below 1 | A wake word's own score needed, in place of `wake_threshold`, as a table: `[audio.wake_thresholds]` then `jarvis = 0.85`. Each model scores on its own scale, so one that wakes too often at the shared score can be given a higher one; [Models](models.md#wake-word-and-end-of-speech) has the measured figures. A name that is not in `wake_words` is kept but does nothing; `jarvis doctor` notes it, so a misspelling shows. `jarvis wake` shows each own score beside the model's name and in its first line. When several models pass in one frame, the log names each with its score and threshold. |
| `wake_engine` | `livekit` | `livekit` or `azure` | Which wake word model listens. `azure` uses your Azure custom keyword model; see [Azure wake word](#azure-wake-word). `jarvis wake livekit\|azure`. |
| `azure_keyword_model` | empty | path to a `.table` file | The Azure custom keyword model. `jarvis wake azure --model FILE`. |
| `azure_python` | empty | path to a Python | The Python that has the Speech SDK. Empty means `<data_dir>/azure-wake/venv/bin/python`. |
| `vad_threshold` | `0.5` | above 0, below 1 | Speech probability that starts speech. |
| `vad_neg_threshold` | `0.0` | 0 to below 1, and not above `vad_threshold` | Once speech has started it goes on while the probability stays above this, so a breath or a soft syllable inside a sentence is not taken for the end. `0` means `vad_threshold` less 0.15 (never below 0.01), as Silero's own segmenter does, 0.35 at the default. Set it equal to `vad_threshold` for no margin. It applies to the end of every request, answer and follow-up; PC audio's speech count while Jarvis waits for a yes keeps `vad_threshold`, as it only counts frames. |
| `no_speech_timeout_s` | `4.0` | above 0, up to 30 | Silence after the wake word before Jarvis gives up. |
| `end_silence_s` | `0.7` | above 0, up to 5 | Trailing silence (below the exit threshold) that ends a request. A request after the wake word, by mic or PC audio, needs at least 0.15 s of speech (two 80 ms frames) or it is dropped without a word: a click is not a request. With the Azure wake word, the keyword's last quarter second, handed over with the request, neither starts the request nor counts toward that 0.15 s. After the keyword Jarvis waits up to `no_speech_timeout_s` for speech, as with the built-in wake word. |
| `max_utterance_s` | `20.0` | above 1, up to 120 | Longest request. |
| `confirm_window_s` | `8.0` | above 1, up to 30 | How long Jarvis waits for the answer to a confirmation question. |
| `confirm_key_s` | `30.0` | 5 to 120 | How long a `strong` action waits for the confirm key (`jarvis confirm`) after Jarvis has asked about it. |
| `barge_in` | `true` | true or false | Listen for the wake word while Jarvis speaks: saying it stops Jarvis and starts a new request (a question spoken over counts as no). See [ADR 0008](adr/0008-speak-while-writing-and-barge-in.md). |
| `barge_in_threshold` | `0.5` | above 0, below 1 | While Jarvis speaks, a LiveKit wake word needs at most this score (a lower threshold of its own still holds). Jarvis's voice in the microphone lowers your score: with its voice as loud as yours, recall at 0.76 fell from 99 % to 71 %, and at 0.5 it was 94.5 % ([models.md](models.md#speaking-over-jarvis)). Each of its sentences is checked against this score before it plays, so it cannot stop itself. When the wake word is missed over its speech, the log says the closest score heard. The Azure keyword model has no score and is not affected. |
| `listen_when_locked` | `true` | true or false | `false`: while the Omarchy lock screen is up, Jarvis takes no request: not after the wake word, not in the follow-up window, not spoken over a reply (the wake word still stops its speech), and none from PC audio. It asks the shell (`omarchy-shell lock isLocked`) once such a request is heard, and a state it cannot read counts as locked; `jarvis doctor` checks it can be read. `true` lets anyone at a locked PC talk to Jarvis. |
| `follow_up_s` | `6.0` | 0 to 15 | Seconds after a spoken reply in which the next request needs no wake word; 0 turns it off. The window opens once the reply has finished playing, only after a request in your own voice, and never while Voxtype dictation is recording or transcribing or a Voxtype meeting runs, or while that cannot be read (`voxtype status`, see [Dictation and meetings](#dictation-and-meetings)); with `listen_to_pc` on, speech heard while PC audio also held speech is dropped; speech shorter than 0.3 s (a cough, a click) is ignored and the window listens on; speech in it that cannot be made out is dropped without a word. No window opens while another program plays audio (music, a video, a game, a call), or while PipeWire cannot be read (`pw-dump`), and speech heard in the window is dropped if such audio started meanwhile (see `follow_up_over_audio`). The wake word inside the window starts a request as usual. Background speech in a quiet room can still start a request: if it does, shorten the window or set 0. |
| `follow_up_over_audio` | `false` | true or false | `true` opens the follow-up window even while another program plays audio. Sung lyrics, a video or a game's voices are then taken as your next request as readily as your own voice. |
| `voice` | `kokoro:bm_george` | text | `kokoro:<name>` for a Kokoro voice, otherwise a Piper voice name. See [models.md](models.md). |
| `speech_speed` | `1.0` | 0.5 to 3.0 | Speaking speed, for Kokoro and Piper voices: 2.0 says the same words in half the time. `jarvis speed` shows it, and `jarvis speed 1.5` sets it and restarts the voice service. Past about 2.0 words start to blur. |
| `kokoro_phoneme_map` | `false` | true or false | Kokoro voices only. Rewrite espeak's phonemes into the symbols Kokoro was trained on (`eɪ` to `A`, `aɪ` to `I`, `tʃ` to `ʧ`, `r` to `ɹ`, and the rest of the table in `src/jarvis/phonemes.py`) before they are spoken. Off, Kokoro gets espeak's raw output, as the kokoro-onnx package does. It is off until you have heard it help: set it, run `jarvis restart` and compare a few sentences. |
| `transcribe_engine` | empty | text | Voxtype `--engine`. Empty means Voxtype's own. Jarvis supports `whisper`, `parakeet`, `cohere` and `gguf` (Qwen3-ASR and other checkpoints Voxtype runs natively). |
| `transcribe_model` | `small.en` | text | Voxtype `--model`: a model name or a full path to a model file. Empty means Voxtype's own. For `parakeet`, `cohere` and `gguf`, a full path to a `.gguf` file goes in Jarvis's own Voxtype config instead, in the section Voxtype reads it from; the `gguf` engine is also told the architecture the file names. The vocabulary hint goes only to Whisper, the one engine that reads it. The model must already be downloaded; `jarvis doctor` checks Whisper models and model files. Voxtype's CPU build runs every engine on the CPU; its GPU build is chosen system-wide (`voxtype setup gpu`). |
| `transcribe_threads` | `8` | 1 to 256 | CPU threads for speech-to-text, written to Jarvis's own Voxtype settings file for Whisper and for the GGUF engine (`transcribe_engine = "gguf"`). Fewer threads leave more of the CPU to a game but transcribe more slowly; Voxtype's GPU build needs few either way. |
| `transcribe_backend` | `""` | `""`, `auto`, `cpu`, `cpu_accel`, `vulkan` | Where the GGUF engine runs, written as its `gguf_backend`; empty leaves it to Voxtype (its GPU build uses the graphics card). `cpu` keeps speech-to-text off the card on the GPU build, for a game that fills it: measured on the reference machine, no VRAM instead of about 3 GB for each request, but 7 to 8 CPU-seconds instead of 0.35 and about a second longer, with the same transcript. Other engines ignore it. |
| `transcribe_short_window` | `true` | true or false | Size Whisper's window to the clip (`context_window_optimization`); much faster for short commands. |
| `vocabulary_hint` | `true` | true or false | Prime speech-to-text with Jarvis's name, its actions and your artist names. |
| `hint_names` | `40` | 0 to 200 | Most artist names put in the hint (the hint also stops at 800 characters). |
| `listen_to_pc` | `false` | true or false | Also take requests heard in PC audio. `jarvis pc-audio on`. The voice prompt tells the model either way whether it hears PC audio, whose (`pc_audio_apps`) and how such a request is treated; the chat never hears it and its prompt leaves it out. On, the voice service also has `pc_audio_listen` (`safe`): asked what is said or played there, it listens (see `pc_audio_recall_s`) and gets the words as PC-audio data, so with `bypass_consent` off a later action or a search waits for your yes, as after a file's text; a stopped program or a shell command after them is refused whatever it is set to. Only your own request can use it, never a scheduled one, and only the voice service hears: in `jarvis ask` every listen is refused. A stop or speaking over Jarvis ends a listen at once. |
| `pc_audio_apps` | `["discord"]` | list of app or process names | Whose audio to listen to: every stream they play is recorded and heard mixed. An empty list means all PC output. `--apps`, `--all-output`. |
| `pc_audio_grant_s` | `600.0` | 0 to 86400 | After you say yes to a PC-audio request, how long further PC-audio requests run without asking. 0 asks every time. Time the PC spends asleep counts. |
| `confirm_blocks_on_pc_speech` | `true` | true or false | A yes counts as no if speech was heard in PC audio while Jarvis waited for it. |
| `pc_audio_cooldown_s` | `5.0` | 0 to 600 | Pause after a PC-audio request during which another is ignored. |
| `pc_audio_recall_s` | `0.0` | 0, or 3 to 60 | With `listen_to_pc` on, how many of the last seconds of PC audio Jarvis keeps in memory, so that `pc_audio_listen` can tell you what was just said. 0 keeps none: it then listens only from when you ask, for 3 to 60 seconds. Nothing is kept while Jarvis is paused or speaks, and a pause or the end of the stream drops it; it reaches the disk only while Voxtype transcribes it ([privacy](privacy.md)). |

See [the PC-audio ADR](adr/0003-pc-audio-is-not-the-owner.md) for why the last four exist.

### Dictation and meetings

Jarvis follows Voxtype's own status with one long-lived `voxtype status --follow --format json --extended`
(read-only; it costs nothing while the status is the same) and reads what it last said, so asking never waits
for Voxtype. While your dictation is recording or transcribing, a wake word is ignored: no request, no
chime, and one `wake_ignored` event in the audit log. Dictation that starts while Jarvis is capturing a
request drops that request, but not the chime: it already played when the wake word was heard. Nothing is
transcribed, Jarvis says nothing more, one `wake_ignored` event with the reason `dictation started mid-request` is
recorded, and Voxtype's recording is left alone. While a Voxtype meeting runs (any `meeting_status` but
`idle`), or while dictation runs, no follow-up window opens. A status that is not known yet or cannot be read (an
older Voxtype without those options, a line that is not JSON, a status command that ended) never keeps Jarvis
from taking the wake word, but it does keep follow-ups off. A Voxtype daemon that is not running counts as no
dictation: the status command says so itself. Measured with
`tools/idle_check.py` on 2026-11-22 (Voxtype 1.3.0, daemon idle, 20 s): the follower and its Python parent use 0.0 % of
one core, against 2 to 3 ms of CPU for each plain `voxtype status` call and about 40 ms of CPU (0.2 to 0.4 s of
wall time) for each `--extended` one that polling would make.

### Speakers and other apps' speech

Jarvis knows only the apps in `pc_audio_apps` as PC audio. On speakers, speech from any other app (a video
in the browser, a game's dialogue) can reach the microphone, and Jarvis hears it as you: it can wake Jarvis,
and a "yes" in it can answer a question. With headphones nothing reaches the microphone. On speakers,
PipeWire's echo cancellation removes everything the PC plays from the microphone before Jarvis hears it. It
is a change to your PipeWire setup, so Jarvis does not make it; to use it, save this as
`~/.config/pipewire/pipewire.conf.d/60-jarvis-echo-cancel.conf`:

```
context.modules = [
  { name = libpipewire-module-echo-cancel
    args = {
      library.name = aec/libspa-aec-webrtc
      monitor.mode = true
      aec.args = { webrtc.noise_suppression = false webrtc.gain_control = false }
      capture.props = { node.name = "jarvis-aec-capture" }
      source.props = { node.name = "jarvis-aec-source" node.description = "Microphone for Jarvis" }
    }
  }
]
```

Then run `systemctl --user restart pipewire`, set `audio.source = "jarvis-aec-source"` and run
`jarvis restart`. `monitor.mode` takes what the default output plays as the sound to remove, so no new output device
appears and every app keeps playing where it did. It removes Jarvis's own voice from the microphone too, so a
wake word heard while Jarvis speaks (`barge_in`) is more likely to be yours. `aec.args` turns off WebRTC's noise suppression and
gain control: noise suppression is on by default, and one practitioner found that noise suppression and gain
control together stopped the wake word from being heard at all
([FINDINGS.md](https://github.com/bluetooth47/agent-voice/blob/main/bench/FINDINGS.md)). Use a wired or
built-in microphone: after a WirePlumber restart the module can fall back to a Bluetooth headset's
microphone, which drops the headset to its low-quality call profile. Check the wake word, idle and while
Jarvis speaks, before and after the change. This setup was checked to load in PipeWire 1.6.9, not yet
measured on the owner's microphone.

**Warning:** when echo cancellation was applied on the maintainer's PC on 2026-11-21, the PC's sound output
broke and the change was reverted; the cause is not known yet
([#13](https://github.com/NecturaLabs/Jarvis/issues/13)). Before trying it, keep the way back at hand: delete
`60-jarvis-echo-cancel.conf`, run `systemctl --user restart pipewire`, set `audio.source` back to `""` and run
`jarvis restart`.

### Azure wake word

Jarvis can listen with an Azure AI Speech custom keyword model instead of LiveKit's `hey_jarvis`
([ADR 0010](adr/0010-azure-keyword-wake-word.md)). The model is a `.table` file you train in Speech Studio
(Custom Keyword) and download from the project's Models page. It runs on your PC: Microsoft's Speech SDK is
loaded only by a small helper that Jarvis starts with no network, so it cannot send anything to Microsoft.
The helper still runs as you, with your files and local sockets.

1. Put the SDK in its own environment (it is closed source, so it is never one of Jarvis's packages):

   ```sh
   uv venv --python 3.14 <data_dir>/azure-wake/venv
   uv pip install --python <data_dir>/azure-wake/venv/bin/python azure-cognitiveservices-speech==1.53.1
   ```

2. Keep the `.table` file outside the repository (for example under `<data_dir>/azure-wake/`) and switch:
   `jarvis wake azure --model <file>.table`. `jarvis wake livekit` switches back; `jarvis wake` shows which
   one listens. Each switch restarts the voice service if it is running.
3. With `audio.wake_engine = "azure"`, `jarvis doctor` checks the model file, the Python and that a namespace with no network can be made.

The keyword engine documents that a keyword is heard after silence ("at least 1.5 s" before the first one,
read from Microsoft Learn's custom keyword quickstart; not fetched again when this was written), and not in
the middle of a sentence. So the helper writes 1.5 s of silence into the engine's stream before the first
sample and again after each keyword, and the samples it reports leave that silence out. A keyword said
within 1.5 s of Jarvis ending its own speech or of the last wake word, or in the middle of a sentence, is
outside what the engine promises and may be missed; how often it is, with your models, is not measured yet.

If the Azure model cannot start, Jarvis says so in the log and listens with LiveKit's instead. The model's own
threshold decides, so neither `wake_threshold` nor `wake_thresholds` applies to it. An Azure model trained with "Containing single
keyword" also wakes on "Jarvis" alone; while one listens, a sentence in which Jarvis says its own name cannot
be spoken over.

## `[persona]`

| Key | Default | What it does |
|---|---|---|
| `name` | `Jarvis` | What the prompt calls the assistant. It is also dropped from the start or end of a request ("Jarvis, pause"). The wake phrase is fixed by the wake word model and is not changed by this. |
| `tone` | calm, warm, to the point, with a light touch of dry wit | How it speaks. |
| `addendum` | empty | Your own extra instructions, added to the system prompt. |

## `[sing]`: how Jarvis sings

Ask Jarvis to sing, in any words: the model writes the song as notes (words or syllables, pitches and lengths)
and the `sing` tool checks it in code before a sound is made: at most 64 notes and 30 seconds, pitches from C2
to C6. The song plays like a reply, after what Jarvis said before it: say the wake word over it to stop it,
`jarvis hush` ends it, and nothing is sung while Jarvis is paused. The song is checked for its own wake word in
chunks of about 2.5 seconds, so a stretch that sounds like it is the only part that cannot be spoken over. It is sung by voice only; the chat does not
offer it. The tool's class is `safe` and it counts as an action: after outside text (a web page) it needs a
yes, unless `boundaries.bypass_consent` is on, and the question reads out the song's length and all its words.
`sing = "deny"` under `[boundaries.classes]` switches it off.

| Key | Default | Allowed | What it does |
|---|---|---|---|
| `engine` | `auto` | `auto`, `speech`, `soulx` | `speech` sings with the Kokoro voice: its words stretched to each note and moved to its pitch on the CPU (Praat), in a fraction of the song's length. `soulx` sings with SoulX-Singer, a singing-voice model, on the GPU for about two seconds a song, in the same voice; it falls back to `speech` when its files are missing or not the pinned release, it fails, it runs past 20 seconds or the request's time left, or less than 5 seconds are left to start it, and it stops at once when the song is stopped (the wake word, `jarvis hush`, a pause). `auto` is `soulx` when its files are installed ([models.md](models.md#singing)), else `speech`. |
| `device` | `0` | `0` to `15` | The Vulkan device SoulX-Singer runs on (`audiocpp_cli --list-devices` lists them). |

Jarvis sings in a Kokoro voice: the speaking voice when `audio.voice` is a Kokoro one, else `bm_george`. The song
is moved by whole octaves to suit that voice. Restart Jarvis (`jarvis restart`) after changing these.

## `[format]`

| Key | Default | Allowed | What it does |
|---|---|---|---|
| `thousands_separator` | `","` | `","`, `" "`, `"'"`, `""` | What Jarvis puts between groups of three digits of a number's whole part, in every reply (the system prompt says so) and in the calculator's answers: 48 times 31 is 1,488. `""` writes none. A calculator answer of more than 20 digits is given in powers of ten without groups. |

## `[clock]`

| Key | Default | Allowed | What it does |
|---|---|---|---|
| `twelve_hour` | `true` | `true`, `false` | Say the time as "3:42 PM"; `false` says "15:42". |

The time is this PC's own clock and time zone; only the time in a named place asks Open-Meteo for that
place's time zone.

## `[weather]`: your home, for weather that names no place

Weather comes from Open-Meteo, which needs no account or key ([ADR 0007](adr/0007-outside-services-for-answers.md)).
A weather question that names no place uses your home; one that names a place looks that place up.

| Key | Default | Allowed | What it does |
|---|---|---|---|
| `place` | empty | text | Your home's name, said back to you and looked up once per run when `coordinates` is empty. |
| `coordinates` | empty | empty, or `[latitude, longitude]` within 90 and 180 degrees | Your home's exact point; no place lookup is sent when it is set. |
| `temperature_unit` | `celsius` | `celsius`, `fahrenheit` | Units of every temperature. |
| `wind_unit` | `kmh` | `kmh`, `mph`, `ms`, `kn` | Units of the wind speed. |

With neither set, Jarvis asks you to name a place.

## `[research]`: web search, pages and Reddit

| Key | Default | Allowed | What it does |
|---|---|---|---|
| `searxng_url` | `http://127.0.0.1:8888` | an `http://` or `https://` URL on `127.0.0.1`, `localhost` or `::1` | The local SearXNG Jarvis searches through. |
| `results` | `5` | 1 to 8 | Results per search when the model does not ask for a number. |
| `page_chars` | `2800` | 500 to 50000 | Most characters of a page or Reddit thread passed to the model for a spoken request, within `agent.tool_text_chars`, counting each `<` or `>` as the four characters it is sent as. The chat reads up to its own `tool_text_chars`. |
| `reddit_comments` | `6` | 1 to 10 | Top comments read from a Reddit thread. |

Search needs the SearXNG container running; without it Jarvis says search isn't running, and everything
else keeps working. To start it:

```sh
(umask 077; printf 'SEARXNG_SECRET=%s\n' "$(openssl rand -hex 32)" > deploy/searxng/.env)
docker compose -f deploy/searxng/compose.yaml up -d
```

It listens on `127.0.0.1:8888` only (`SEARXNG_PORT` in `.env` changes the port; change `searxng_url`
with it) and restarts with Docker unless you stop it (`docker compose -f deploy/searxng/compose.yaml
down`). The `.env` file is git-ignored. Reddit threads are found through SearXNG and read from the
Arctic Shift archive, a free copy of public Reddit that can lag behind it by a little.

## `[browser]`: the headless browser

| Key | Default | Allowed | What it does |
|---|---|---|---|
| `idle_close_s` | `60.0` | 10 to 900 | Seconds after its last use that the browser closes, with its profile and anything a page saved. |

For pages `web_read` can't read (built by scripts) or that must be used (a search box, a button), Jarvis
drives a real Chrome through [agent-browser](https://github.com/vercel-labs/agent-browser) (Apache-2.0),
when the `agent-browser` program is on `PATH` and has its browser (`agent-browser install`). `chat.browser`
offers its tools in the chat (on by default) and `agent.browser` to speech and `jarvis ask` (off by
default). The installer requires the program and has it download its Chrome; while either setting is on,
`jarvis doctor` fails when the program or its Chrome is missing (without the program the tools are not
offered). The tools:

- `browser_open` opens an address a search in the conversation returned or you wrote (the same rule as
  `web_read`), follows a link the open page shows, or reads on in the page. It asks nothing.
- `browser_screenshot` (the chat only) shows you one screen of the page. It asks nothing.
- `browser_act` clicks something on the page, or types up to 200 characters into a text box (never a
  password field) and presses Enter. It always asks first, naming the element, the site and the words (and
  where a link goes), even with `bypass_consent` on. With `jarvis ask --yes`, browser actions require
  an answer on an interactive terminal. Jarvis refuses actions when it cannot identify the link
  or fit the full host in the question. `[boundaries.classes]` can switch it off (`deny`) or
  ask for the confirm key (`strong`), never set it to run unasked.

The browser runs headless and muted, opens no window, and uses a throwaway profile and home folder in
`$XDG_RUNTIME_DIR/jarvis/browser/`: none of your browsers, cookies, logins or Downloads. Every connection it makes goes
through a proxy Jarvis runs on loopback, which reaches only public addresses on ports 80 and 443, so a
page cannot reach this PC (Strata, SearXNG) or your home network. The browser and `web_read` both
refuse connections when they cannot check this PC's own networks. Downloads, uploads, page scripts run by
Jarvis, cookies and other tabs are refused. It closes after `idle_close_s`, and when the conversation is
forgotten, so it uses nothing between requests. A yes given after it closed (a question left open longer
than `idle_close_s`) does nothing; ask again and Jarvis opens the page anew.
[ADR 0016](adr/0016-headless-browser.md) has the details.

## `[files]`: the folders Jarvis may read and write

| Key | Default | Allowed | What it does |
|---|---|---|---|
| `read_roots` | `["~"]` | `~`, paths under `~/`, absolute paths | Folders whose files Jarvis may list and read (`list_dir`, `read_file`). |
| `write_roots` | `["~/Documents"]` | the same | Folders Jarvis may create files in, or replace or add to them (`write_file`, asked first unless `bypass_consent` is on). |
| `deny` | `[]` | shell-style patterns | More names (`diary*`), paths under home (`.config/app`, `~/Documents/private`) or absolute paths (`/data/finance`) never read or written, with what is inside them. A trailing slash or a leading `./` makes no difference. An entry that could never match (`..` in it, `~` alone, another user's `~name`) is refused when the config loads. |
| `hidden_allow` | the desktop's settings: `.config/hypr`, `omarchy`, `waybar`, `walker`, `mako`, `swayosd`, `kitty`, `alacritty`, `ghostty`, `foot`, `btop`, `fastfetch`, `jarvis` | paths under home, shell-style | The hidden files and folders directly under home Jarvis may read (`.config/app` and the like). Everything else hidden there (shell history, app logins, cookies, caches) stays shut; `[]` shuts them all. A hidden folder deeper down (`~/Projects/site/.git`) is not shut by this; secret names still are. |

Whatever the roots say, Jarvis never reads or writes keys, tokens, password stores, browser profiles
and its own state: `.ssh`, `.gnupg`, `.aws`, `.kube`, `.docker`, `.netrc`, `.env` files, `*.pem`,
`*.key`, `id_rsa*` and the like, any name with `secret`, `password` or `credential` in it, shell
history files and cookie stores, `~/.config/gh`, keyrings and browser profiles (the full lists are `SECRET_NAMES` and `SECRET_PATHS`
in `src/jarvis/pc/files.py`). Paths are resolved through links first, so a link cannot lead out of a
root or to a secret. The checked path is then opened one folder at a time without following a link, so a
folder swapped for a link in between is refused, not followed. A file under home with a second name on
disk (a hard link) is never read or added to, since that name could be a secret's; outside home hard
links are read as usual. Writing never touches a hidden file or folder (no `~/.bashrc`, no `~/.config`),
never overwrites unless asked to replace, and never makes a file executable.

Jarvis's own configuration files are read by `read_jarvis_file`, from a fixed list whatever the read roots
say: `config` (this file), `strata_run_config`, `strata_profile`, `voxtype` (its own Voxtype settings) and
the two units (`voice_unit`, `strata_unit`). `files.deny` and the lists above still apply to them, a value
under a name such as `api_key`, `token` or `password` is hidden, and nothing is ever written: Jarvis does not
change its own configuration. `about_jarvis` tells it its version, model and settings without reading a
file.

The question before a write names the file, how much text, the folders it would make ("making the new
folder Documents/work") and, for a replace, the old file's size ("replace todo.md in Documents (2.1 KB
now) with 400 characters"). Each name is shown on one line: a newline or another control character in it
becomes a � mark. The write is refused if any of that no longer holds when it runs: a folder it would
have to make that the question did not name, or an old file that is not the one you were asked about (a
different size, inode, modification or change time, so a file swapped for another of the same size is
caught). A replace keeps the old file's read and write permissions, so a private file stays private; it
never carries an execute bit over. A file you made read-only is neither replaced nor added to, and you
are not asked about it.

A file's text is data, like a web page (a shell command's output counts as one): while it is in the conversation, side effects need your yes,
and so do the operations that send words off the PC (`web_search`, `reddit_search`, and `weather` or
`time_now` for a named place), so a file's contents cannot leave in a search without you seeing it.
With `boundaries.bypass_consent` on (the default) none of these questions is asked, which is why the
hidden files under home stay shut unless you open them.

## `[boundaries]`: whether Jarvis asks first

| Key | Default | Allowed | What it does |
|---|---|---|---|
| `bypass_consent` | `true` | `true`, `false` | `true`: Jarvis never asks for a yes. Every action that would need one runs at once and is audited as run with consent bypassed: `confirm` actions, side effects after outside or file text, searches while file text is in the conversation, and requests heard in PC audio. A `strong` action still waits for the confirm key, and an operation set to `deny` is still refused. `false`: Jarvis asks as described below. `jarvis consent ask` or `jarvis consent bypass` switches it and restarts the voice service. |
| `max_actions` | `30` | 1 to 10000 | The most side-effecting actions (playback changes, a file write) that run in any `actions_window_s`, counted across the voice service, the chat and `jarvis ask` together, in both consent modes. One past it is refused, audited as `action_limit` and Jarvis says why: with the window already full, before you are asked. Only actions that run count, taking their place once you said yes (or no yes was needed); one you decline, or one declined unasked after a no, takes none. Reads, searches and answers do not count. When the shared count file stays locked for 2 seconds, that process counts only its own actions for that one, and logs it. Each process's own count holds too, so an emptied or damaged count file does not lift the limit. |
| `actions_window_s` | `600.0` | 1 to 86400 | The window `max_actions` is counted over, in seconds. |

With consent bypassed, text from a web page, a file, a picture or a call's audio can steer actions with no
question in between. Turn asking on (`jarvis consent ask`) before letting Jarvis read pages or listen to PC
audio you do not trust.

## `[boundaries.classes]`

A table from operation name to a class: `safe` (runs), `confirm` (needs your spoken yes), `strong`
(needs the confirm key, below; never a spoken yes, and not skipped by `bypass_consent`) or `deny`.
A name that is no operation of Jarvis's stops `jarvis run`, `ask` and `chat` with an error, so a typo
cannot leave an operation you meant to switch off at its shipped class. An operation not listed keeps
its shipped class, which is `safe` for every operation but `write_file`, `browser_act` and `pc_window_close` (`confirm`) and `stop_process`
(`strong`): `spotify_*`, `time_now`, `weather`, `web_search`, `web_read`, `reddit_search`, `reddit_read`,
`pc_stats`, `pc_volume`, `pc_media`, `pc_audio_listen`, `pc_windows`, `pc_window`, `pc_screenshot`, `calculate`, `list_dir`, `read_file`, `find_processes`, `open_app`,
`sing`, `browser_open`, `browser_screenshot`; set one to `deny` to switch it off: it is then
not offered to the model at all. `no_action`, `think_harder` and `stay_quiet` are how Jarvis answers, not tools, and
cannot be switched off. `stop_process`
may be set only to `strong` or `deny`: a program is stopped by your command and your key alone, so any
other class stops `jarvis run`, `ask` and `chat` with an error. `browser_act` may not be set to `safe`. Whatever the class, side effects are raised to `confirm` while text from outside
your own speech is in the request, and so are the operations that send words off the PC while a file's text or a shell command's output is. Example:

```toml
[boundaries.classes]
spotify_like = "confirm"
spotify_queue = "deny"
reddit_search = "deny"
```

Without editing the file: `jarvis tools` lists every tool with its setting, and `jarvis tools NAME on|ask|key|off`
sets one (`default` puts it back as shipped); in `jarvis chat`, `/config` has a Tools list where Enter moves a tool
on to its next setting, from as shipped to stricter and back, never below how it ships (that takes `jarvis tools
NAME on`). Either way the change is audited (`tool_class_changed`, with the old setting and where it was made).
The command restarts the voice service at once; the chat restarts it once, when its settings close. An open chat
takes the change the next time it opens.

`/config` in the chat also sets the speaking speed (`audio.speech_speed`), whether Jarvis acts without asking
first (`boundaries.bypass_consent`), whether it takes requests from PC audio (`audio.listen_to_pc`), whether
spoken requests get the headless browser (`agent.browser`) and when the browser closes
(`browser.idle_close_s`), marked "voice too": each change is saved and audited (`setting_changed`; acting
without asking as `consent_changed`, as `jarvis consent` records it), and the voice service takes them when
the settings close. Acting without asking also applies to the open chat at once, turned on or off.

## The confirm key (`jarvis confirm`)

A `strong` action runs only after you press a key. Jarvis says what it is about to do, then "Press the
confirm key to allow it", and waits `audio.confirm_key_s` for `jarvis confirm`. Bind that command to a key
in `~/.config/hypr/bindings.lua`, for example:

```lua
o.bind("SUPER + CTRL + Y", "Jarvis confirm", "jarvis confirm")
```

- One press allows the one action Jarvis asked about, once. Not pressing in time, saying "Hey Jarvis",
  `jarvis hush` or `jarvis pause` is no; a press after a timeout, `hush` or `pause` finds nothing
  waiting. A spoken yes
  never counts. The wait is timed on the boot clock, so a suspend does not leave it open.
- Press once the question has ended. A press while nothing waits, or one sooner than 0.3 s after the
  question ended, allows nothing and holds the key off for 30 s, so a program pressing it over and over
  cannot catch a question. Jarvis says so if it was about to ask.
- The key works only through the control socket of the voice service. As a best effort, a press from a
  program Jarvis started is refused and audited. It is no guarantee: any program running as you can run
  `jarvis confirm`, so the key keeps the model (and what it reads) from acting alone, not malware already
  on your account. The chat window and `jarvis ask` (even with `--yes`) have no confirm key, so they
  refuse a `strong` action.
- Each press, refused or not, is in the audit log (`confirm_key`), and the answer is a `confirmation`
  event with `"method": "key"`.

## Opening a program

Jarvis opens a program only when you ask it to, in any words, and only one listed in `[apps]`. It
needs to know where: a workspace from 1 to 10, or the background (Hyprland's special workspace
`jarvis`: running, out of sight until you show it). When you did not say where, it asks; it never
picks one itself. Nothing else opens a program: with Spotify closed, a request to play says so and
asks whether to open it.

Hyprland itself starts the program, with a `[workspace N silent]` rule so it does not take focus.
Jarvis then reads Hyprland's windows until one of the program's shows, up to 15 seconds, moves it if
it landed on another workspace, looks again a second later (some programs replace their first
window) and only then says where it opened. A program already open is not opened again; Jarvis says
where it is. A request heard in PC audio opens nothing. `open_app` is `safe`: your own request needs
no yes, while outside text in the request raises it to `confirm` as for any action.

```toml
[apps.programs.spotify]
command = ["uwsm-app", "--", "spotify"]  # run as written, no shell; the Omarchy way to start it
window = "\\bspotify\\b"                 # matched, case-blind, with a window's class
```

| Key | Default | Allowed | What it does |
|---|---|---|---|
| `programs` | Spotify only | names of lowercase letters, digits and `-` (up to 32) | The programs `open_app` may start. An empty table removes `open_app`. |
| `programs.<name>.command` | | 1 to 32 strings | The program and its arguments. |
| `programs.<name>.window` | | a regular expression | How the program's window is known: by its class, never its title, so a browser tab named after it is not taken for it. Only windows that were not open before it was started are moved. |

## Stopping a program

Ask in any words to close, end or kill a program. Jarvis first
looks for the programs of yours that match the name you said (`find_processes`, `safe`): by the program's
name, its executable, the name it was started as (a Wine or Proton game's `.exe`), or its window's
class or title. It lists at most 8, folds a program's own child processes into it, and, when more than one
matches, reads them out and asks which; when none matches it says so and never tries a wider name. It then
stops the one you chose (`stop_process`, `strong`): "Should I stop <program> (PID <number>)?", and only your
confirm key allows it. That allows that exact process: if it ended and its PID went to another program
before the press, nothing is stopped. It is asked to close (SIGTERM); a forced stop (SIGKILL) only when you
ask for one. Jarvis says whether it ended within 3 seconds.

In code, whatever the model says:

- `stop_process` takes only a PID that `find_processes` listed in the same request, never a name or pattern.
- A request heard in PC audio stops nothing, even with the key pressed and questions bypassed. Nor does
  a request that also read outside text (a web page, a file, a picture, a long paste, Spotify names, a
  shell command's output, or such text remembered from earlier): ask again in a request of its own.
- A program matched only by one window's title (a browser tab's, say) is marked so, and the question
  names that title and says all its windows close.
- Only your own processes are listed. Never listed or stopped: PID 1, kernel threads, other users'
  processes and set-user-ID programs, processes another user started (or whose parent cannot be seen) and
  those in the login session's own scope (`session-N.scope`), Jarvis and the terminal or shell it runs in,
  the processes that started the desktop session, login shells, the `jarvis` and `jarvis-strata`
  services, and `systemd`, D-Bus, PipeWire, WirePlumber, Hyprland, Xwayland, `uwsm`, the bar (`waybar`,
  Omarchy's `quickshell`), `hyprlock`, `hypridle`, the desktop portals, the keyring, the polkit agent,
  `dconf-service`, the accessibility bus, `gvfs*`, `mako`, `swayosd-server`, `hyprsunset`, `walker`,
  `elephant`, `fcitx5`, Omarchy's own `omarchy-*` helpers, Voxtype and Strata. A name is checked in full
  (a program's executable, or a script's file name), not only its first 15 characters. The list, with the
  reason for each, is in `src/jarvis/pc/processes.py`.

## Reminders and scheduled requests

There are no keys to set. Ask by voice or in the chat: a reminder ("remind me to stretch in 20 minutes",
"every weekday at 9:00") is said when due, after the chime; a scheduled request ("pause the music at
midnight") is carried out then as if you had just asked, with every question asked aloud and nobody
answering counted as a no. A time is minutes from now (up to a week) or a time of day today, tomorrow or on a
weekday; a job is once, daily, on weekdays, at weekends, weekly, or every 5 to 1,440 minutes. Ask what is set
to hear the list by number, and cancel one by its number or all of them. Times are said as `[clock]` sets.

Only the voice service (`jarvis run`) says them, between requests; the chat and `jarvis ask` say so when it
is not running. Due jobs wait while your Voxtype dictation or a meeting runs. A one-off reminder is still
said up to six hours late, and says how late; a scheduled request or a repeating reminder more than ten
minutes late (the PC slept, Jarvis was stopped or paused) is not done but shown as missed, and a repeating one
moves on to its next time. The jobs live in `~/.local/state/jarvis/schedule.json` ([privacy](privacy.md)); at
most 50, 200 characters each. A file written by a newer version of Jarvis is left as it is: this one sees no
jobs in it and refuses to add one, rather than writing over the newer jobs. Why it works this way is in
[ADR 0011](adr/0011-reminders-and-scheduled-requests.md).

## `[shell]`: an opt-in shell command (off)

`pc_shell` runs one shell command you say, as you, and only when you turn it on:

```toml
[shell]
enabled = true       # off by default
timeout_s = 30       # stopped after this (1 to 600)
memory_max_mib = 1024
cpu_percent = 100    # of one core; 200 is two
tasks_max = 256      # processes and threads together
```

Every command is `strong`: only your confirm key runs it. No `[boundaries.classes]` entry lowers that
(`deny` still turns it off), and `bypass_consent` never skips it. Before the key, Jarvis reads the command
out character by character, so that two commands that run differently never sound the same: each symbol by
its name ("ls space star dot txt"), each space as "space", and capitals marked ("all capitals HOME",
"capitalised Notes"). A command with a tab, a line break or two spaces in a row, or one too long to read out
in full, is refused before anything is asked. A request heard in PC audio runs nothing, nor does a request
that also read outside text (a page, a file, a picture, a long paste, Spotify names, a program's output,
or such text remembered from earlier): say the command in a request of its own. A command's output counts
as file text as well: a search that sends text off the PC afterwards needs your yes.

Each command runs once in its own `systemd-run --user --scope` unit (`jarvis-shell-<random>`, described
only as "Jarvis shell command", so the journal never holds the command), in your home folder, with the
limits above and no swap. `systemd-run` is told not to fill in `$VARIABLES` itself and gets only what it
needs to reach your user manager. The command runs behind `setpriv --no-new-privs`, so `sudo` and every
other set-user-ID program cannot raise its rights. Its environment holds only `PATH`, `HOME`, `LANG` and
`TERM`: none of Jarvis's own, so no token or key is passed on. It has no input. When it returns or runs out
of time, every process in its scope is killed and Jarvis checks that the scope is gone; if it cannot see
that, it says something may still be running. Jarvis keeps the first part of the output (`[agent]
tool_text_chars`, the same cap as a file read) with the exit status, as text from a program, not your
words. The audit log keeps the command, but only a digest and the length of its output.

**What it cannot stop.** A scope limits resources only. An approved command has all of your user's power:

- it can read, change and delete whatever your user can, and reach the network;
- it can start programs outside its scope and its limits through your user's own services
  (`systemd-run --user`, D-Bus), which outlive it;
- it can press the confirm key itself (`jarvis confirm`) for a question Jarvis asks later.

Jarvis cannot check that the command is the one you said: speech-to-text does not write shell syntax, so
the model writes the command from your words, and the read-out before the key is the check. Whether a
request is yours is decided by where the wake word was heard: speech from the speakers after you said
the wake word counts as yours, and the confirm key at the PC is what guards it. Approve only a command
you would type yourself.

## The Omarchy bar module (#14)

Omarchy's bar is part of `omarchy-shell` (Quickshell), not Waybar. It takes a custom module with `"type":
"command"`: it runs the `exec` command through `bash -lc` every `interval` seconds, shows the `text` of the
Waybar-style JSON line the command prints, shows its `tooltip` on hover, highlights the module when its `class`
includes `active`, and runs `onClick`, `onRightClick` and `onMiddleClick` through bash. `tools/bar_status.py`
is that command for Jarvis. Nothing here edits your Omarchy files: you add the two things below. It is a command
module and not a Quickshell plugin because its click actions are static and a command module needs no plugin install;
the cost is that it cannot turn the clicks off while Jarvis is unavailable (the not-running tooltip leaves the click
hint out, but a click still runs `jarvis pause` or `jarvis resume`, which then says Jarvis is not running).

1. Link the script where Omarchy's bar documentation keeps its scripts (any path works; use your clone's):

   ```bash
   mkdir -p ~/.config/omarchy/bar/scripts
   ln -s <your Jarvis clone>/tools/bar_status.py ~/.config/omarchy/bar/scripts/jarvis-status
   ```

2. Add this entry to a section of `bar.layout` in `~/.config/omarchy/shell.json`, for example at the end of
   `right`. The shell reloads the file when you save it. If that file does not exist yet, copy Omarchy's default
   first, `cp /usr/share/omarchy/config/omarchy/shell.json ~/.config/omarchy/shell.json`: once your file is
   there it replaces the default layout, with no deep merge, so a file holding only this entry would leave the
   bar with this one module.

   ```json
   {
     "id": "jarvis",
     "type": "command",
     "exec": "~/.config/omarchy/bar/scripts/jarvis-status",
     "interval": 5,
     "tooltip": "Jarvis",
     "onClick": "jarvis pause",
     "onRightClick": "jarvis resume"
   }
   ```

Remove the entry (and the link) to take the module away. The only actions it offers are the two fixed ones:
a left click runs `jarvis pause` and a right click `jarvis resume`; the tooltip says so.

What the bar shows comes from one reading of Jarvis's control socket, the same `status` request `jarvis status`
sends, which never loads the model or generates a token. The script does not start `jarvis` (see the cost below).
With Jarvis not running it does not ask the model server how it is, because that
would mean reading Jarvis's config in the script: the bar says `not running` and leaves the model state out. Run
`jarvis status` for it.

| Shown (class) | When |
|---|---|
| `idle` | waiting for the wake word |
| `listening`, `thinking`, `speaking`, `model-loading` (highlighted) | the request in flight, or the model on its way into memory |
| `paused` | after `jarvis pause`, until `jarvis resume` |
| `unavailable` | Jarvis is running but cannot work (the microphone was lost or the model server is down), or refused or failed the request |
| `not-running` | nothing serves the control socket |
| `timeout` | Jarvis did not answer in 8 seconds (the script gives up) |
| `failed` | the connection broke while the answer was being read |
| `malformed` | its answer is not a status reading (not JSON, an unknown or missing state, over 4 KB) |

The tooltip is built from fixed texts; of the reading only the model's state (`loaded`, `unloaded`, `loading`,
`failed`, `unreachable`, `missing`) is shown, and only when it is a plain lowercase word, because Omarchy's guidance
treats tooltip text as a place where markup in outside data could be read.

The bar has no way to be told of a change, so the module polls: every 5 seconds the bar starts the script, which
sends one request and exits. Starting `jarvis status --json` for each poll costs about 0.2 s of CPU (measured
0.20 to 0.22 s: Python's start-up and the imports of `jarvis.cli`), about 4 % of one core. The script asks the
socket directly, runs under `python3 -I -S` (no `site` import, set by its shebang) and imports only
`collections.abc`, `json`, `os`, `re`, `socket`, `sys` and `time`. The script's own cost is 0.015 s of CPU a poll,
with Jarvis running or not, and nothing between polls. Measured on this PC (2026-11-22, Python 3.14): `RUSAGE_CHILDREN` user plus
system time over 20 runs, three batches each, before 0.20 to 0.22 s a run and after 0.013 to 0.017 s; a bare
`python3 -I -S -c pass` is 0.005 s of that, and the shell the bar starts the command through (`bash -lc`) adds
about 0.006 s the script cannot change. Importing `jarvis` would bring the 0.2 s back, so the script repeats the
socket's path and request from `src/jarvis/control.py`; `tests/test_bar_status.py` fails if either drifts, and
runs the script against a real control server.

Each poll also costs the running Jarvis and Strata something: Jarvis answers `status` by asking Strata for its
model state (`GET /models`, through a new HTTP client each time; no tokens are generated). Measured the same day by
calling that same request in a separate process, 3 batches of 500, against the running Strata server (model
unloaded; Jarvis itself was not running, so this is the request and not the whole service): 2.4 to 2.5 ms of CPU on
Jarvis's side and 0.1 to 0.2 ms in Strata's process, against 0.01 s of Strata's CPU in 5 idle seconds. A poll in all
is about 0.018 s of CPU, 0.4 % of one core at the 5 second default, so the default stays. A longer `interval` costs
proportionally less and shows a short state (`speaking`, `thinking`) less often, and a shorter one costs
proportionally more (1 second would be about 1.8 % of a core); 3 to 10 seconds is the range the tests allow. The
script gives up after 8 seconds, so a stuck Jarvis cannot hold a poll for long.
