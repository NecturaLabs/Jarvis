# Changelog

Notable changes to Jarvis, in the [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format.
Jarvis has no tagged releases yet; everything below is unreleased. Once releases start, this project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Jarvis remembers the conversation. Instead of the last 4 turns, forgotten after 5 minutes, it keeps as
  much as fits a token budget (`agent.memory_tokens` 6,000 for speech, `chat.memory_tokens` 12,000 for the
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

- `docs/configuration.md` warns that its echo-cancel setup broke the PC's sound output when it was applied
  on 2026-11-21, and says how to undo it.
- Asked the time, Jarvis says only the time, and asked the date, only the date: the time tool no longer
  hands the model both, and the core prompt tells it to answer only what was asked.
- The check that Jarvis does not claim an action no tool did now also catches "I set the volume to
  thirty", "I've lowered the volume", "I muted it", "The volume is now at 20" and "Sure — I muted it".
  A question that asks ("Shall I put it on shuffle?", "Muted, or just quieter?") is not a claim, but
  "Done, anything else?" still is.

[Unreleased]: https://github.com/NecturaLabs/Jarvis/commits/dev
