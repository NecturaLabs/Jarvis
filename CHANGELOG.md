# Changelog

Notable changes to Jarvis, in the [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format.
Jarvis has no tagged releases yet; everything below is unreleased. Once releases start, this project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

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
  much as fits a token budget (`agent.memory_tokens` 6,000 for speech, `chat.memory_tokens` 11,500 for the
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

- `jarvis confirm`, the confirm key: bind it to a key, and a `strong` action runs only after you press it.
  Jarvis says what it is about to do, then "Press the confirm key to allow it", and waits
  `audio.confirm_key_s` (30 s). One press allows that one action, once; a spoken yes never does. A press
  with nothing waiting holds the key off for 30 s, and, as a best effort, a press from a program Jarvis
  started is refused.
  Only spoken requests can use it: the chat window and `jarvis ask` still refuse `strong` actions.

- `jarvis pause` lets go of the microphone and, once the request in flight has ended, frees the model;
  `jarvis resume` listens again and reloads it. `jarvis hush` stops the speech. `jarvis status` says what
  the running Jarvis is doing (idle, listening, thinking, speaking, paused, model loading, unavailable), and
  `--json` prints it for the Omarchy bar (#14). These reach `jarvis run` through a socket only your user can
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
