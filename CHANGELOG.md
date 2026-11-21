# Changelog

Notable changes to Jarvis, in the [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format.
Jarvis has no tagged releases yet; everything below is unreleased. Once releases start, this project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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

[Unreleased]: https://github.com/NecturaLabs/Jarvis/commits/dev
