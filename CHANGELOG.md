# Changelog

Notable changes to Jarvis, in the [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format.
Jarvis has no tagged releases yet; everything below is unreleased. Once releases start, this project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Jarvis starts speaking sooner. Each sentence of a reply is spoken as soon as the model has written it and
  synthesized while the one before plays. The silence before the first word is cut from 0.23 s to 0.03 s. A
  sentence claiming an action is not spoken before an action succeeded
  ([ADR 0008](docs/adr/0008-speak-while-writing-and-barge-in.md), proposed).
- The service log's timing line now shows when the first words were ready.
- The default Strata context window is 24,576 tokens (was 8,192), about 8.1 GiB of VRAM with the default model, and the model sees up to 3,000 characters
  of each tool result (was 1,500), so a search and two page reads fit in one request. A saved
  `strata.context` keeps its value; to change it, edit it and run `jarvis setup`.

### Added

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

[Unreleased]: https://github.com/NecturaLabs/Jarvis/commits/dev
