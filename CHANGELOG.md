# Changelog

Notable changes to Jarvis, in the [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format.
Jarvis has no tagged releases yet; everything below is unreleased. Once releases start, this project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- `jarvis doctor` recognises the picture readers of Qwen3.8-Flash-Next and Inkling 2 Mini by their pinned
  sha256, and the docs say Qwen3.8-Flash-Next needs `strata.memory_max = "44G"` (at 40G the Strata unit sat
  at its memory mark).

- Jarvis serves whichever model Strata is set up for, from Strata's own run config
  (`strata.run_config`, the `strata-<model>.json` that `strata launch` writes): the model, its engine, its
  flags, tokenizer and picture reader come from that file, and Jarvis adds only its own limits (context,
  VRAM, one request at a time, lazy load). Nothing model-specific is hardcoded any more. The VRAM limit goes
  through whichever flag the model's engine takes. `strata.learn_experts` (on) lets an engine that keeps a
  routing profile learn which experts Jarvis's requests use and start with them on the card. Jarvis no longer
  has a model default of its own; the model set up and measured here is Qwen3.8-Flash-Next (measured beside a game's budget: 7.95 GiB of VRAM, 35.5 GiB of RAM,
  under a second for a two-tool request, and the prompt reused between requests); Inkling 2 Mini is set up
  to compare and is too slow with only 8 GiB of VRAM today ([docs/models.md](docs/models.md)).
  **Migration:** `strata.model_file`, `tokenizer_dir`, `model_id`, `prefill` and `vision_file` are gone,
  and a config that still sets one is refused with what to do: run `strata launch` for the model, then
  `jarvis setup --run-config <its strata-<model>.json>` (it replaces `--model-file` and `--tokenizer-dir`),
  and delete the old keys. `strata.pictures = false` replaces leaving `vision_file` unset. `jarvis setup`
  without a run config saves the config and says what comes next.

- The command line reads better. `jarvis help` lists the commands in groups (everyday, running Jarvis,
  settings, first-time setup, other) with what each does, instead of argparse's list of every name,
  twice. Every help page is plain text within 80 columns. `jarvis <command> --help` says what the command
  does and explains every option, including what `consent bypass`, `ask --yes` and `spotify-login
  --no-browser` do not do, and a mistyped command gets a suggestion when one is close. `jarvis` alone
  names the services as running or stopped and points at `jarvis help`. `jarvis doctor` lines its checks
  up and ends with how many failed.

- The chat remembers a little less: `chat.memory_tokens` defaults to 10,000 (was 10,500), to make room in
  the default context for the list of Omarchy's guides. A config that still sets 10,500 gets what the window
  leaves, about 10,000.

- From a practices audit (#30): at most 30 actions that change something (playback, a file write) run in
  any 10 minutes, the voice service and the chat counted together and even with questions bypassed; one
  more is refused, recorded as `action_limit`, and Jarvis says why (`boundaries.max_actions`,
  `boundaries.actions_window_s`). Only actions that run count: with the window full Jarvis refuses before
  asking, and an action you decline takes no place. A damaged count file starts the count again instead of
  failing every request, a failed write leaves the old count, and a lock held for 2 seconds falls back to
  the process's own count. Before a write Jarvis names the folders it would make and, for a
  replace, the old file's size; the chat also shows the first three lines of the new text. A name with a
  newline or another control character is shown with a mark, on the question's own line. A write is
  refused if a folder it was not asked about would be needed, or the old file is not the one asked about
  (its size, inode, modification or change time differ). A replace keeps the old file's read and write
  permissions instead of 0644. Files are
  opened one folder at a time without following a link, so a folder swapped for a link after the check is
  refused, and a file under home with a second hard link is not read or added to. The audit log keeps
  five older generations instead of one and holds only the length and SHA-256 of what a write puts in a
  file. While Jarvis asks first, chat pastes over 200 characters in all count as outside data, a request
  brought back with ↑ keeping what was pasted into it. Starting a new conversation (or one forgotten
  after the idle time, checked before every request, PC audio included) also forgets the pages and
  threads earlier searches returned.
- A wake word said while your Voxtype dictation runs is ignored: no request, no chime, one `wake_ignored`
  event in the audit log; and a running Voxtype meeting keeps follow-ups off like dictation does. Jarvis
  follows `voxtype status --follow --format json --extended` in the background, so asking never waits for
  Voxtype (#29). A status that cannot be read no longer counts as dictation: the wake word is taken, so a
  broken or older Voxtype does not leave Jarvis deaf, and follow-ups stay off until the status is known.
- The Azure keyword helper writes 1.5 s of silence into the engine's stream before the first sample and after
  each detection, which the engine is documented to want before a keyword (#29).
- Both services are safer and kinder to a game beside them (#33). Both set `NoNewPrivileges`, `LockPersonality`,
  `RestrictSUIDSGID` and `CPUWeight=50`; Strata also sets `UMask=0077`, and the voice service
  `SystemCallArchitectures=native`. A service that fails five times in 15 minutes now stays `failed` instead
  of restarting for ever; `jarvis start`, `jarvis restart`, `jarvis setup --enable` and the commands that
  restart the voice service after a change (`consent`, `pc-audio`, `wake`) clear that first. Strata's
  `MemoryHigh` is `strata.memory_max` and its `MemoryMax` is 10% above it (44G for the default 40G).
  `strata.prefill` also accepts `"auto"`, and the new `strata.vram_reserve_mib` (unset by default) passes
  `--vram-reserve-mib`. Run `jarvis setup` to rewrite the units. Not measured yet: the effect on time to
  first word, VRAM and a running game (playbook).
- `boundaries.bypass_consent` no longer runs `strong` actions unasked: they wait for the confirm key.
  No operation Jarvis ships is `strong` yet; this matters for one you set to `strong` yourself.
- `strata.idle_unload_s` defaults to 86400 (24 hours) instead of 300, so a fresh install keeps the model
  loaded, as the owner decided (#34). Set a lower value to free the RAM sooner. A saved config or a Strata
  profile already written keeps its value until you change it or run `jarvis setup` again.
- `jarvis doctor` asks Voxtype which speech models are installed (`voxtype info models`), which also checks
  each file's size and header, and falls back to looking for the file when Voxtype cannot say.
  `stop_process` is the one shipped `strong` operation, and `boundaries.classes` may set it only to
  `strong` or `deny`; any other class stops Jarvis with an error.
- The chat remembers a little less: `chat.memory_tokens` defaults to 10,500 (was 11,500), to make room in
  the model's context for `find_processes`, `stop_process`, `calculate` and `open_app`.
- Jarvis never opens Spotify, or any program, unasked: with no player it says Spotify is closed and asks
  whether to open it and where (#36). Before, playing or resuming started the Spotify app.
- From a practices audit: a name in `boundaries.classes` that is no operation (a typo, say) stops Jarvis
  with an error instead of being ignored; turning questions on or off with `jarvis consent`, and the mode
  the voice service starts in, are recorded in the audit log and shown in the chat; words heard from the PC
  reach the model as marked outside data, so someone in a call cannot pose as you; a saved chat
  conversation is read only if it is your own file, readable by nobody else; only web links in a chat
  reply open when clicked; a folder with more than 10,000 entries is listed from the first 10,000; an
  argument shorter than its minimum length is refused; and both units start as `Type=exec`, the voice
  service after WirePlumber too.
- Hidden files and folders under your home are no longer read unless `files.hidden_allow` opens them;
  by default it opens the desktop's own settings (`~/.config/hypr`, `omarchy`, `waybar`, `kitty` and the
  like). Shell history and cookie stores are never read anywhere. With questions bypassed, this keeps
  logins and tokens kept in hidden folders from reaching the model, and from leaving in a search.
- Under each chat reply: "✻ Worked for 4m 41s · done 7:37 PM · 50 TPS" (how long it took, when it was done,
  and how fast the model wrote, from Strata's own timings), a blank line below the reply and in the column of
  its dot. Replies are set as Claude Code sets them: headings bold (the first level also italic and
  underlined) with a blank line above and below, lists marked with a dash and numbers at the text's edge,
  code with no side bar, quotes in italics behind a dim bar, table headers in plain weight. Typing `/` opens
  the command menu over the transcript's last lines, right above the input, as in Claude Code, so nothing
  moves while it is open; the letters typed are in bold and the chosen row is lit whole. `/exit` leaves
  (`/quit` still works).
- Jarvis remembers the conversation. Instead of the last 4 turns, forgotten after 5 minutes, it keeps as
  much as fits a token budget (`agent.memory_tokens` 6,000 for speech, `chat.memory_tokens` 10,500 for the
  chat) and has the model summarize the oldest turns when it outgrows it. Speech forgets after 30 idle
  minutes; the chat never does and resumes its conversation when it opens again (`chat.resume`).
- Replies are no longer cut. The model is told its length and capped in tokens (`agent.reply_tokens` 400,
  `chat.reply_tokens` 2,048, was `model.max_tokens` 200); a reply that reaches the cap anyway is kept and
  says so. A request may take more steps (speech 8 model calls and 12 tool calls, the chat 30 and 40 within
  10 minutes; all in `[agent]` and `[chat]`), and a long one clears its oldest tool results to stay within
  the context window.
- The chat gets its own prompt: written replies in Markdown, sources as links. Speech keeps short spoken
  replies.
- `jarvis chat` is rebuilt closer to Claude Code: replies are rendered as Markdown (tables, lists, code,
  links) while they are written, and never cut; by default it shows only requests and replies, with a working
  line that says what Jarvis is doing ("Searching the web… (8s · esc to interrupt)"); Ctrl+O or
  `/verbosity` shows each step and its result, then every detail. New commands: `/clear` (a new
  conversation), `/compact`, `/context`, `/verbosity`, `/config`, `/status`, `/copy` and `/export`; the menu
  narrows as you type and offers a command's arguments. The input takes several lines (Shift+Enter), ↑ and ↓
  bring back requests from earlier sessions, Ctrl+C clears the input and twice leaves, and a toolbar shows
  the PC's CPU, GPU, VRAM, RAM and temperatures and how full the model's context is.

- `jarvis chat` is redesigned after Claude Code's terminal interface: a welcome card, each request grouped
  with its steps (`●`) and their results (`⎿`), a working line with Esc to stop the request, a yes-or-no
  choice for questions (1 or y, 2, n or Esc), a `/` command menu, ↑ and ↓ for earlier requests, `?` for the
  shortcuts, Ctrl+O for every step, and the terminal's own colours, so it follows the Omarchy theme. A typed
  request shows at once.
- `jarvis chat` reads well on every terminal theme. Omarchy sets the terminal's black to its background, so
  the shaded bands behind requests and code were invisible; requests now start with a cyan `❯` and say how
  and when they came in at the right, code blocks and quotes have a quiet rule at their left, and tables,
  headings, bullets and numbers use the theme's text and grey instead of four different colours. The
  command menu, `/config` and the yes-or-no choice point at the highlighted row with `❯`, the working line
  says "Waiting for your answer" while a question is open, the footer and toolbar shorten themselves to
  stay on one line in a narrow window, and the input says when typed requests are off.
- Jarvis says the time on a 12-hour clock ("3:42 PM") by default; `clock.twelve_hour = false` keeps "15:42".
- Jarvis starts speaking sooner. Each sentence of a reply is spoken as soon as the model has written it and
  synthesized while the one before plays. The silence before the first word is cut from 0.23 s to 0.03 s. A
  sentence claiming an action is not spoken before an action succeeded, and a sentence said before a tool
  call plays to its end instead of being cut off mid-word
  ([ADR 0008](docs/adr/0008-speak-while-writing-and-barge-in.md)).
- The service log's timing line now shows when the first words were ready.
- The default Strata context window is 24,576 tokens (was 8,192), about 8.1 GiB of VRAM with the default model, and the model sees up to 3,000 characters
  of each tool result (was 1,500), so a search and two page reads fit in one request. A saved
  `strata.context` keeps its value; to change it, edit it and run `jarvis setup`.

### Added

- `strata.kv_resident` keeps that many context cells in VRAM and the whole context in pinned RAM
  (`--kv-resident`, Strata's KV streaming), so `strata.context` can grow without more VRAM. It is off
  (`0`) by default until a bigger context has been measured on this PC.

- Jarvis answers questions about Omarchy from the guides Omarchy ships for agents
  (`/usr/share/omarchy/default/agents/skills/`, part of #9). The prompt lists each guide by its first
  sentence, read when Jarvis starts, so an Omarchy update is picked up at the next start. The safe
  `os_skill` operation reads one file of one guide, in parts that fit a tool result, and only a file
  listed at start. Only folders and files that root owns and nobody else can write are listed or read,
  each opened from `/` without following a link, and the list keeps to a fixed size so it always fits the
  context. The text counts as outside text, like a web page, and Jarvis is told never to run a command
  from it. Every guide's SKILL.md is listed before any other file, and `boundaries.classes` may name
  `os_skill` on a PC without the guides.

- An opt-in shell command, `pc_shell` (#19), off unless `[shell] enabled` is set. Each command needs the
  confirm key, which no class setting lowers, and runs once in its own user scope with time, memory, CPU
  and task limits. It runs with no new privileges, so `sudo` cannot work, and with only `PATH`, `HOME`,
  `LANG` and `TERM` set. Before the key, the command is read out character by character, so two
  commands that run differently never sound the same. A request heard in PC audio, or one that read
  outside text, runs nothing. The audit log keeps the command but only a digest of its output.
- Wake word training and benchmark tools: `tools/wake_train.sh` trains and scores a phrase with a chosen
  LiveKit project's key, `tools/wake_ungated.py` runs the trainer without the gated test sets (#37), and
  `tools/wake_bench.py` scores models on 55 local voices and about 24 hours of real media, alone and
  together as Jarvis runs them. [docs/playbook/wake-training.md](docs/playbook/wake-training.md) says how.
- `jarvis doctor` now loads each wake word file that is on and fails one that does not load, and notes a
  `wake_thresholds` name that is not in `wake_words`. `jarvis wake` shows each model's own threshold in its
  first line, and the log names every wake word that passed in a frame, not only the highest.
- Each wake word can have its own threshold (`[audio.wake_thresholds]`, such as `jarvis = 0.85`); the
  others keep `audio.wake_threshold`. Models score on different scales: on the trainer's reduced test set
  the trained `jarvis` falsely woke 5 times in about an hour at 0.5. `jarvis wake` shows a model's own
  threshold beside its name. `docs/models.md` records the measured recall and false wakes of the trained
  `hey_jarvis`, `yo_jarvis` and `jarvis` models (#15, #37).

- `tools/first_audio.py` times, on a loaded model, how long the first sentence of a reply takes to reach the
  speaker and how long the whole reply takes to write (#12). Fake-clock tests show the first sentence goes to the
  speaker while the model is still writing, and each sentence goes once, in order.
- A state indicator for the Omarchy bar (#14): `tools/bar_status.py` asks the running Jarvis for its status over
  the control socket and turns it into the bar's text, tooltip and class for idle, listening, thinking,
  speaking, paused, model loading and unavailable, and into a state of its own for a Jarvis that is not
  running, a timeout, a broken connection and an unreadable answer. The module offers only a left click for
  `jarvis pause` and a right click for `jarvis resume`, polls every 5 seconds and never loads the model. The
  script's own cost is 0.015 s of CPU a poll: it does not start `jarvis` (about 0.2 s a poll) and runs without
  Python's `site` import. Each poll also makes Jarvis ask Strata for its model state (no tokens), about 0.003 s more
  across both, 0.018 s in all or 0.4 % of one core. With Jarvis not running it shows `not running` without the
  model's state or the click hint, and `jarvis status` still gives the state. The entry to add to
  `~/.config/omarchy/shell.json` is in
  [configuration](docs/configuration.md#the-omarchy-bar-module-14); nothing is installed for you.
- `audio.wake_words` picks which LiveKit wake word models listen, side by side, each
  `<data_dir>/wake/<name>.onnx`; `hey_jarvis` and `yo_jarvis` are on by default. `jarvis wake --words
  hey_jarvis,yo_jarvis` sets the list and `jarvis wake` shows every model in the folder as on or off. A
  model that is missing or cannot be loaded is left out with a warning and the others still listen.
  `jarvis doctor` fails only when no model has a file, and lists each missing one. The log names the
  phrase each wake came from, and they share `wake_threshold` unless
  `wake_thresholds` names one. The check that stops Jarvis
  hearing itself during barge-in uses the same models.
- `open_app` (#36): ask Jarvis, in any words, to open a program from the new `[apps]` list (Spotify by
  default) on a workspace from 1 to 10 or in the background. Jarvis asks where when you did not say.
  Hyprland starts it there without taking focus, and Jarvis says where it opened only once it saw the
  window there, moving it when it landed elsewhere. A request heard in PC audio opens nothing, and a
  program already open is not opened again ([Opening a program](docs/configuration.md#opening-a-program)).
- `calculate`: Jarvis works out sums in code, exactly, however you ask (products, percentages, powers,
  roots, logarithms, factorials), and says the
  answer; the model never does the arithmetic itself. Numbers in replies and answers come in
  groups of three (1,488); the new `format.thousands_separator` sets the separator, or none.
- `audio.vad_neg_threshold`: once speech has started it goes on while the probability stays above this
  (default: `vad_threshold` less 0.15, Silero's own margin), so a breath inside a sentence no longer ends a
  request early. A request after the wake word, from the mic or PC audio, needs 0.15 s of speech, so a click
  after "Hey Jarvis" is dropped untranscribed (#29). With the Azure wake word, which is reported up to 0.6 s
  after the keyword ends, only the keyword's last quarter second is left out of that count, so a short
  request said before the report still counts.
- `audio.kokoro_phoneme_map` (off by default): rewrite espeak's phonemes into the symbols Kokoro was trained
  on before they are spoken, to compare by ear (#29).
- `jarvis doctor` checks that `audio.source` names a microphone PipeWire lists, and says so when PipeWire
  cannot be read rather than blaming the name (#29).
- Stop a program by voice, asked in any words:
  `find_processes` lists your own programs that match the name you said (at most 8, by program name,
  executable, a Wine game's `.exe`, or window title), Jarvis asks which one when several match, and
  `stop_process` stops only a PID it listed in the same request, after the confirm key. The PID is bound
  to its program and start time, so a PID reused in between is never signalled. A request heard in PC
  audio, or one that also read a web page, a file or a picture, stops nothing; a match by one window's
  title alone says that all the program's windows close. The desktop session, its helpers, the sound
  server, Voxtype, Strata and Jarvis itself are never listed or stopped
  ([Stopping a program](docs/configuration.md#stopping-a-program)).
- `jarvis confirm`, the confirm key: bind it to a key, and a `strong` action runs only after you press it.
  Jarvis says what it is about to do, then "Press the confirm key to allow it", and waits
  `audio.confirm_key_s` (30 s). One press allows that one action, once; a spoken yes never does. A press
  with nothing waiting holds the key off for 30 s, and, as a best effort, a press from a program Jarvis
  started is refused.
  Only spoken requests can use it: the chat window and `jarvis ask` still refuse `strong` actions.

- `jarvis pause` lets go of the microphone and, once the request in flight has ended, frees the model;
  `jarvis resume` listens again and reloads it. `jarvis hush` stops the speech. `jarvis status` says what
  the running Jarvis is doing (idle, listening, thinking, speaking, paused, model loading, unavailable), and
  `--json` prints it for a status bar (#14). These reach `jarvis run` through a socket only your user can
  open, in `$XDG_RUNTIME_DIR/jarvis/`, and each is recorded in the audit log. A pause lasts until `resume`
  or until the voice service restarts. `jarvis stop` still stops the services; `hush` is the speech
  interrupt #22 asked `stop` to be.
- Jarvis carries on across desktop session changes (#22). A microphone whose capture ends (PipeWire
  restarting, the microphone unplugged) or goes silent for 5 s is opened again: after 1 s, then twice as
  long after each try that brings no frame, up to 30 s. `jarvis run` used to end with "the microphone
  stream ended" and wait for systemd to restart it. Meanwhile `jarvis status` says `unavailable`, and a
  request the lost microphone cuts short ends without an apology. After a suspend, Jarvis starts listening
  afresh and checks the model server, loading the model again only if its load failed. A model server that is
  down because its model file is missing (its drive not mounted) is no longer waited for: Jarvis says so at
  once, and `jarvis status` names the model `missing`. A new `audio.listen_when_locked` (on by default)
  can make Jarvis take no request while the Omarchy lock screen is up (the wake word still stops its speech);
  `jarvis doctor` then checks the lock state can be read.

- `boundaries.bypass_consent`, on by default (the owner's choice): Jarvis acts without asking for a yes,
  including after outside text and for requests heard in PC audio; an operation set to `deny` is still
  refused, and each bypassed question is audited. `jarvis consent ask` turns asking back on, `jarvis consent
  bypass` off again.
- `jarvis chat` shows pictures: ask to see something and Jarvis searches SearXNG's image category
  (`image_search`) and draws the best result in the window (`show_image`), with kitty's graphics protocol
  through textual-image 0.14.2 (new dependency, LGPL-3.0-or-later). A read page offers its own picture too.
  Only a picture a search or page in the same conversation offered is fetched, through the same address
  checks as pages, then checked and shrunk; the newest 50 are kept 0600 in `chat-pictures`.
- Pictures in `jarvis chat`, as in Claude Code: Ctrl+V pastes a copied picture as an `[Image #1]` tag (or
  copied text where the cursor is), a picture file dropped on the terminal or `/image <path>` attaches that
  file, and `/image` alone the newest screenshot. Pictures are checked and shrunk with Pillow 13.0.0 (new
  dependency, MIT-CMU) and go only to the local model, which needs `strata.vision_file`. What a picture
  shows counts as outside text, so an action the model chooses in that request needs your yes.
- Pictures for the model: `strata.vision_file` names the model's picture reader (`mmproj-F16.gguf`, pinned
  in `jarvis doctor`), and the Strata profile then reads pictures with Strata's processor reader, started by
  the first picture and stopped after `strata.vision_idle_s` (120 s), so it costs no VRAM and no RAM while
  unused.
- PC operations: `pc_stats` ("how's my PC doing": CPU, graphics card, VRAM, memory, temperatures, disks,
  uptime), `list_dir` and `read_file` within `files.read_roots` (home by default), and `write_file` within
  `files.write_roots` (`~/Documents` by default), which always asks first. Keys, tokens, password stores,
  browser profiles and Jarvis's own state are never read or written; writes never touch hidden files.
  Jarvis's own `files` and `pc` skills say how to use them.
- While text read from a file is in the conversation, operations that send words off the PC (web and Reddit
  searches, a named place for the weather or time) need a yes, so a file's contents cannot leave in a search
  unseen.
- Skills: Jarvis's own instructions for web research and game look-ups, listed in its prompt and read
  through the safe `read_skill` operation before such a task.

- Follow-up requests: the next request within `audio.follow_up_s` (6 s; 0 turns it off) of a spoken reply
  needs no wake word. It never opens while Voxtype dictation is recording or transcribing; a sound too
  short for a request (a cough, a click) is ignored, and unclear speech is dropped without a word.
- Pull requests check every commit's subject against the Conventional Commits types in `CONTRIBUTING.md`,
  and, from a fork, each commit's DCO sign-off (`tools/check_pr_commits.py`).
- CI lints the workflow files with actionlint (and shellcheck over their `run:` steps); `tools/check.sh` runs
  it too when `actionlint` is installed.
- An Azure custom keyword model can be the wake word instead of LiveKit's: `audio.wake_engine = "azure"`
  or `jarvis wake azure --model FILE`, and `jarvis wake livekit` to switch back. Microsoft's Speech SDK runs
  only in a separate MIT-licensed helper with no network; an Azure model that cannot start falls back to
  LiveKit's, and `jarvis doctor` checks the setup
  ([ADR 0010](docs/adr/0010-azure-keyword-wake-word.md)). While an Azure model listens, a sentence that
  says "Jarvis" cannot be spoken over.
- The service log shows the score of each LiveKit wake, to tune `audio.wake_threshold` against false wakes.
- Speaking over Jarvis: say the wake word while it talks and it stops and listens to the new request; a
  question spoken over counts as no. `audio.barge_in = false` turns it off. A sentence that would trigger the
  wake word itself cannot be spoken over.
- `config.example.toml`: every setting at its default, with no personal values, to copy to
  `~/.config/jarvis/config.toml`.
- The time and date, here or in a named place, and the weather now, today, tomorrow or for the week, at home
  (`[weather]`) or wherever you name, from Open-Meteo. "What time is it" and "what's the weather" skip the
  model.
- Research: web search through a local SearXNG (`deploy/searxng/`), reading the pages a search returned, and
  Reddit threads found through SearXNG and read from the Arctic Shift archive. Pages are opened only on public
  addresses, checked at every connection and redirect, and everything they say is kept as outside data
  ([ADR 0007](docs/adr/0007-outside-services-for-answers.md)).
- The first public version of Jarvis, a local, always-on voice agent for Omarchy, open source under
  GPL-3.0-or-later.
- A "Hey Jarvis" wake word and end-of-speech detection, with speech to text through Voxtype and Jarvis's own
  settings file.
- One agent on a local model served by Strata, with plain commands such as "pause" or "volume 40" answered
  without the model.
- Spoken replies through Kokoro, or a Piper voice; no desktop notifications.
- Spotify by voice: play a track, album, artist, playlist or liked songs, pause, resume, skip, volume,
  shuffle, repeat, queue, like and what is playing, with PKCE login.
- Boundaries in code: typed operations with default deny, provenance on every message part, confirmation
  bound to one exact action and an audit log.
- Optional listening to PC audio, treated as someone else's speech that needs the owner's yes.
- `jarvis setup`, `doctor`, `start`, `stop`, `restart`, `status`, `unload`, `logs`, `ask`, `chat`,
  `pc-audio` and `spotify-login`, and a versioned TOML configuration.
- Community files: contributing guide, code of conduct, support, governance, accessibility, security policy
  and roadmap.
- `tools/check.sh`, one command for everything CI runs, and checks of the dependency licences and lock
  sources (`tools/check_licences.py`), the docs (`tools/check_docs.py`) and the workflow files.
- Tests that cannot reach the network, audio, the real programs Jarvis drives or your home folder
  (`tests/conftest.py`), and the product rules checked in code (`tests/test_product_rules.py`, Ruff's
  banned imports).
- Opt-in git hooks (`tools/install-hooks.sh`) and a read-only idle check (`tools/idle_check.py`).
- A docs map, an architecture overview, a statement of what Jarvis keeps and sends (`docs/privacy.md`), and
  recipes and agents for coding agents (`.claude/`, `.codex/`, `.agents/`).

### Fixed

- Emptying or damaging the shared action count (`actions.json`) no longer lifts `boundaries.max_actions`:
  each process's own count must allow an action too. The shared file's answer used to replace it.

- "Play my liked songs" draws from the whole library: 50 songs picked at random from the newest page and
  three more pages at random places in it. It always played the newest 50 shuffled and said "your liked
  songs"; with more than 50 it now says "50 of your liked songs".

- Saving a setting (`jarvis setup`, `consent`, `pc-audio`, `wake`, the chat's `/config`) changes only that
  setting in `config.toml` and keeps the rest of the file, comments included. It wrote every setting with
  its value at the time, about 140 lines, so a default changed in a later version never reached the install
  and comments were lost. A setting set back to its default is taken out of the file. Defaults written by
  an earlier save stay until they are changed or removed by hand. Jarvis now depends on tomlkit (MIT).

- A question in the chat no longer takes words typed ahead as a yes. It opened with "Yes" chosen and took
  Enter, `1` or `y` for yes after 0.4 s, so the owner typing the next request could answer it. "No" is now
  chosen at first (Enter says no), `y` no longer means yes, and a `1` typed among other letters waits until
  the typing stops; `1`, or moving up to "Yes" and Enter, says yes.

- The calculator's `log` with one number is to base 10, as on a calculator: `log(100)` is 2. It was the
  natural logarithm (4.6), which `ln` still gives.

- The calculator bounds a logarithm like every other step: a 47-character `log` of a huge exact base ran for
  more than 30 s while it checked for a whole answer, and now answers at once. A logarithm to a base just
  above 1 (`log(5, 1 + 1/10**30)`) is worked out instead of failing with an internal error, and one too
  close to 1 to work out says so. A number with a leading zero is refused, so `min(1,000, 2,500)` says to
  write 1000 instead of answering 0.

- The calculator keeps a floor division exact: `(7 // 2)!` is 6 and `10**18 // 7` is 142857142857142857. A
  floor division of two exact numbers gave a plain integer, which a factorial or `round`'s digits refused
  and which was shown through a float (142857142857142864).

- A request that finds the model unloaded gives its first model call the 180 s the load may take on top of
  `model.request_timeout_s`. The request was allowed 270 s, but the call was cut at 120 s, leaving about 13 s
  for the answer after a load measured at 107 s while gaming; a slower load ended in "My model isn't
  answering right now."

- No follow-up window opens while another program plays audio (music, a video, a game, a call): anything the
  microphone heard for 6 s after a reply was taken as your next request, so sung or played lyrics started
  requests, and each reply opened the window again. Speech heard in the window is dropped too if such audio
  started meanwhile. `audio.follow_up_over_audio = true` keeps the window open over other audio.

- The chat no longer closes on a picture whose header claims more than about 179 million pixels (a big scan
  or map, or a crafted file): Pillow refused it with an error the picture check did not catch, and the
  window ended. It is now refused as too large to read, from the chat and from `show_image`, and any other
  failure while loading a picture leaves a note instead of ending the window.

- `files.deny` closes a folder however its path is written: `~/Documents/private`, an absolute path, a
  trailing slash or a leading `./`. Only `Documents/private` matched before; the other forms were accepted
  and matched nothing, so the folder stayed readable. An absolute path now also closes a folder under a
  read root outside home, and an entry that could never match is refused when the config loads.

- The calculator reads a percentage followed by a plus or minus as a percentage: `80 * 15% + 80` is 92 and
  `100 - 20% - 5` is 94.8. It read the `%` as a remainder and answered 0 and 100. A `%` with space on both
  sides (`17 % 5`) or between two numbers (`17%5`) is still the remainder, and so is `mod`.

- The wait for your answer no longer counts against a request's time limit. Reading a question out, waiting
  for a spoken yes or for the confirm key came out of `agent.max_seconds`, so an approved action could run
  and the reply still said "That took too long, so I stopped." or, with a few seconds left, "My model isn't
  answering right now." Once an action has run, the model gets at least 15 s to say how it went, and a
  request that runs out of time after acting says what it did instead of that it stopped.

- "Skip two songs" skips two: `spotify_next` and `spotify_previous` take a `count` (1 to 10), since a second
  identical call in one request is refused as a repeat. It skipped one and said two. A skip that fails
  part-way says how many tracks it moved by. To make room for the count in the default context,
  `chat.memory_tokens` defaults to 9,900 (was 10,000); a config that still sets 10,000 gets what the
  window leaves.

- An action that ran but did not end cleanly is no longer reported as not done: a shell command that
  exits non-zero, a program asked to stop that is still running, and a program that started without its
  window showing where asked. A reply may say what happened, instead of being replaced by "Sorry, I didn't
  manage to do that", and the chat's steps show "done, not cleanly" rather than "not done".

- A reply that says something was done is checked against what ran: only an action that succeeded
  counts, so after Jarvis has only looked something up (what is playing, a process list, a folder, a
  search) "Paused it for you" is challenged like any claim with nothing behind it. The check also knows
  the PC verbs (opened, launched, wrote, created, appended, ran, closed, killed, deleted, sent and the
  like), and a player command with nothing playing counts as not done.

- From the review of the run-config change: a model whose drive is not mounted when Jarvis starts is known
  again by its path, so a request says the model file is missing at once instead of waiting for the server,
  and pictures stay on for when the drive is back. `jarvis setup` refuses a run config whose engine would
  fill the card with its expert cache unless `strata.vram_reserve_mib` is set, and `jarvis doctor` fails
  "VRAM left for games" then; it also names a missing model drive as such. The learned expert profile is
  kept per model, a run config that sets its own `--host` or `--port` is refused, a relative routing
  profile is a setup error, and a bare `--vision` without a picture reader is dropped.

- The chat's working line says "Checking the time" while Jarvis reads the clock; it said "Running Time now".

- The calculator works out a negative number that is not exact (`cos(pi)`, `sin(4)`) to a whole power;
  it refused it as a fractional power.

- The calculator rounds a half away from zero, as calculators do: `round(2.5)` is 3 and `round(0.125, 2)`
  is 0.13. It used Python's rounding to even, which gave 2 and 0.12.

- A plain command that needs the model after all (it failed, or its result needs wording) no longer sends
  the model a remembered conversation larger than its budget: the conversation is summarized or trimmed
  first, as for any other request.

- `open_app` builds a valid Hyprland command for a program whose command or window name holds an invisible
  character beyond Latin-1 (a zero-width space, a line separator). Such a character was escaped in a form Lua
  refuses, so the program did not open.

- Reading in parts no longer skips text full of angle brackets (HTML, XML, code). `read_file`, `list_dir`
  and `os_skill` sized each part in plain characters, but each `<` and `>` reaches the model spelled out
  in four, and the result is then cut to its cap: the cut took the line that says where the next part
  starts, so what followed was never read. Parts are now sized as they are sent. A cut `pc_shell` output
  keeps its note that it was cut, and a page that has a picture names the picture before the text, so a cut
  never takes either.

- `read_file` no longer loses most of a line longer than one part (a one-line JSON or minified file). It
  cut the line at about 2,900 characters and reported it as read; now it says which characters it gave,
  and `from_char` reads on inside the line.

- A `[boundaries.classes]` entry that sets `pc_shell` below `strong` is refused with a message that names
  what `pc_shell` takes; it named only `stop_process` before.

- A shell command's output (`pc_shell`) is now outside text of its own kind, `shell`. It was marked as a
  process listing, which `stop_process` lets through. So a remembered output (a file shown with `cat`, a page
  fetched with `curl`) did not block a later shell command in the chat, and within one request it did not
  block `stop_process`. The output also counts as file text now: a search that sends text off the
  PC afterwards needs your yes.

- With the Azure wake word, a pause after "Hey Jarvis" no longer loses the request. The keyword's own last
  quarter second started the capture, so a pause longer than `end_silence_s` ended it before the request
  was said, and it was dropped as too little speech. Now only speech after the keyword starts a request,
  and Jarvis waits up to `no_speech_timeout_s` for it, as with the built-in wake word.

- Reading an Omarchy guide in parts: each part names the guide's version, and a later part asked for
  after an Omarchy update changed the guide is refused with a note to read it again from part 1, instead
  of mixing two versions.

- A model server that answers but lists no model (Strata while its engine is down or restarting) is
  shown as `unloaded`, not as unreachable, and a wake word then loads the model.

- A pause during the start-up load, or right after a wake word, now frees the model once the load
  is over. Both loads skipped the turn-taking with a pause, so the pause's unload met a busy server and
  Jarvis stayed paused with the model loaded.

- Outside text in a tool result (a web page, a file) now fits the size planned for it: its angle
  brackets are spelled out before it is cut, not after, which could make it four times as long. A long
  request also keeps its tool results whole while they fit: the system prompt was counted twice when
  making room.

- `jarvis status`, `status --json` and the status bar say the model is loading while a load Jarvis asked
  for is under way (at start, after a resume, on a wake word). Strata reports only loaded or unloaded, so a
  2-minute load after `jarvis restart` showed `model: unloaded`. Each load request now waits as long as
  Strata may take to answer it (three engine starts of 15 minutes), the start-up load is sent even while
  Strata lists no model, a load a resume asked for stops retrying once you pause again, and a wake word during a load joins it
  instead of queueing a second load.
  A running Jarvis whose model server is down now says `server unreachable` or `file missing`, as it
  does when Jarvis is not running.

- `jarvis start` and `jarvis restart` no longer print "Failed to reset failed state of unit jarvis.service:
  Unit jarvis.service not loaded" when Jarvis was stopped: only a unit in the `failed` state is reset.
  `jarvis help` now lists the commands, as `jarvis --help` does.
- The audit log now says why a confirmation ended as no (#38). Before, the PC sleeping with the question
  open, no answer in time, a lost microphone and a spoken no all wrote the same line. Each no now carries a
  `reason` from a fixed list, and the chat's activity pane shows it. Only a clear yes approves, as before.
- A yes no longer approves a question that was hushed, paused or could not be played before it was heard
  out: that question is a no (`not_heard`), and no answer is listened for.
- A yes spoken after the PC woke from a suspend no longer answers a question asked before it. The question is
  now counted as a no, and Jarvis listens afresh (#22). A sleep also ends the follow-up window, so speech
  after waking is no longer taken as a request without the wake word, and drops a request begun after the
  wake word; speech heard before the sleep is not answered either (#22).
- Your Voxtype dictation starting while Jarvis is capturing a request now drops that request: before, what you
  dictated could be transcribed and answered as a request. Jarvis stops capturing, says nothing, records one
  `wake_ignored` event and never touches Voxtype's recording (#3). A pause that lands after the last
  frame of a reply's follow-up window was read, and before the request heard in it is taken, no longer lets
  that request through (#11).
- `jarvis pause`, `resume` and every other stop of the microphone, and a restart of the Voxtype status
  follower, no longer leak the capture's pipe (#32). A stop waits at most half a second for the reader of a
  capture whose command left a child holding the pipe, and keeps that pipe instead of closing it under the
  reader; a stop that lands while a start is under way no longer closes the new capture's pipe; and a
  status line that cannot be handled ends the Voxtype command instead of waiting for it for ever. The
  tests now fail on any warning, so an unclosed pipe or file shows up in CI.
- The chat no longer shows a reply twice for a moment: the draft is gone before the whole reply
  appears.
- Web research is safer and steadier. Text a browser hides (the `hidden` attribute, `aria-hidden`, and
  inline `display:none`, `visibility:hidden`, `opacity:0` or `font-size:0`) no longer reaches the model, a page
  with more than 20,000 elements is read from its start so it cannot hold up a request, the weather and
  Reddit archive answers must arrive within 15 seconds, the SearXNG container is limited to 200 processes, and
  the documented command now creates its `.env` file readable only by you. After Spotify says "too many
  requests", Jarvis sends nothing to Spotify until the wait it asked for is over.
- Voice and speech to text, from a practices audit: the check that asks whether a sentence would wake Jarvis
  itself now low-pass filters before it drops Kokoro's and Piper's speech to 16 kHz, so sounds above 8 kHz
  no longer fold into the band; a sentence of more than 510 phonemes is split between words and all of it
  is spoken, instead of cut mid-word; Kokoro refuses a model file it cannot run when it loads, not at the
  first sentence, and accepts the older build's `tokens` input; Voxtype runs quietly and keeps no audio of a
  failed transcription from Jarvis's runs; and Jarvis waits as long as Strata's `Retry-After` says while it loads the model at start.
- `jarvis chat` could end with a `NoMatches` traceback when it was closed: a timer that redraws the
  window, reads the log or draws the toolbar could fire after the window's parts were taken down.
- In `jarvis chat`, a request from the log without a time no longer ends in a lone "·", a reply that
  starts with a heading no longer leaves its `●` on a line of its own, the details after a step's name are
  no longer bold, and the transcript fills the window's width instead of stopping six columns short.
- A web page read a second time in a session (or one much like a page read before) came back as "no
  readable text": trafilatura's duplicate check remembers text across calls. It is off now.
- A request that reaches its step limit no longer ends with "That needed too many steps, so I stopped.":
  its last model call gets no tools and answers from what the earlier steps found. Research questions
  asked by voice that kept searching and re-reading pages now get an answer.
- `docs/configuration.md` warns that its echo-cancel setup broke the PC's sound output when it was applied
  on 2026-11-21, and says how to undo it.
- Asked the time, Jarvis says only the time, and asked the date, only the date: the time tool no longer
  hands the model both, and the core prompt tells it to answer only what was asked.
- The check that Jarvis does not claim an action no tool did now also catches "I set the volume to
  thirty", "I've lowered the volume", "I muted it", "The volume is now at 20" and "Sure — I muted it".
  A question that asks ("Shall I put it on shuffle?", "Muted, or just quieter?") is not a claim, but
  "Done, anything else?" still is.

[Unreleased]: https://github.com/NecturaLabs/Jarvis/commits/dev
