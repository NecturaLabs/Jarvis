# Security policy

Jarvis acts on your PC and accounts by voice, so security issues matter. The voice runs locally; nothing is
sent to a hosted model.

## Reporting a vulnerability

Report security problems privately, never in a public issue or pull request:

- on GitHub, through private vulnerability reporting:
  <https://github.com/NecturaLabs/jarvis/security/advisories/new> (**Security → Report a
  vulnerability**), or
- by e-mail to support@necturalabs.com, with "Security" in the subject.

Say what is affected, the version or commit, how to reproduce it and what an attacker gains. Never include
real tokens (Spotify or Strata), recordings or personal data; redact them from logs.

## What counts as a vulnerability

A bug is a vulnerability when it crosses a security boundary: something untrusted, such as speech heard in PC
audio, text from the web, a file or Spotify metadata, causes an action the owner did not confirm, or gains
access or control it should not have. Examples: bypassing the policy engine or a confirmation, an action that
runs without its spoken yes, a confirmation bound to a different action than the one asked, a Spotify or
Strata credential readable by another user or sent off the machine, or a request that leaves this PC for a
host that is not a local one. Code that could be more robust without crossing a boundary is a normal bug, for
example a misheard command that the owner can see in the audit log, or a crash: report it in the
[bug form](https://github.com/NecturaLabs/jarvis/issues/new/choose).

## Research in good faith

Test only your own computer and accounts, or ones you have permission to test. Do not play audio into
anyone else's room or call. Avoid privacy violations, data destruction and service disruption, including to
Spotify. Do not exploit a problem beyond what shows it, and give us the time described below before
publishing.

## What happens next

1. We acknowledge the report within 7 days.
2. We confirm or rule out the vulnerability and tell you which, within 14 days of the report where we can.
3. We fix it privately in a GitHub security advisory for this repository, asking GitHub for a CVE when the
   issue affects a released version.
4. We publish the fix, then the advisory (GHSA, and the CVE where one was issued). The release notes list the
   fix under Security, and the advisory and notes credit you unless you ask us not to.

We follow coordinated disclosure: please keep the details private until the advisory is published, or for 90
days from your report if that comes first. If a fix needs longer, we agree a date with you.

## Supported versions

Only the latest release on `main` and the current `dev` branch are supported.

## Scope

Of particular interest: anything that lets speech, outside text or Spotify metadata cause an action the owner
did not confirm. What exists today and what is in scope:

- **Whose speech is whose.** Speech from the microphone is the owner's. Speech heard in PC audio (a call, a
  video) is someone else's: it is only listened to when the owner turns `listen_to_pc` on, its side effects
  need the owner's spoken yes, and it stays out of the owner's conversation history. With
  `audio.confirm_blocks_on_pc_speech` on (the default), a yes does not count if voices were heard in PC
  audio while Jarvis waited for it. See
  [ADR 0003](docs/adr/0003-pc-audio-is-not-the-owner.md). Known limit: only the apps in `pc_audio_apps` are
  known as PC audio. On speakers, speech from another app that reaches the microphone is heard as the owner's;
  headphones or PipeWire echo cancellation ([docs/configuration.md](docs/configuration.md), "Speakers and
  other apps' speech") close that gap.
- **Outside text is data.** Song, artist and playlist names from Spotify are wrapped as data and marked with
  their origin; while such text is in a request, every side-effecting operation needs a spoken yes.
- **The chat window** (`jarvis chat`) shows outside text, such as Spotify names and PC speech, as plain
  characters only: control characters are removed and long text is cut (`activity.py`).
- **Policy and confirmation.** Operations are typed with validated arguments and default deny; unknown
  operations are refused. A confirmation is bound to the hash of one exact action and starts only after
  Jarvis has finished speaking. Anything but a clear yes is a no, and after one no in a request the owner is
  not asked again. Every request, decision and result is written to `audit.jsonl` (mode 0600, kept to 5 MiB
  plus one rotated copy). The `strong` class (keybinding confirmation) is not built: such a call is refused.
- **Spotify.** Login is OAuth Authorization Code with PKCE (S256), a validated `state` and a callback bound to
  `127.0.0.1`. There is no client secret. Tokens are kept in a mode 0600 file under `~/.local/state/jarvis/`
  and a token file that is a symbolic link, belongs to another user or is readable by others is refused.
- **Local services.** Strata listens on `127.0.0.1` and requires an API key held in a mode 0600 file. A custom
  `model.base_url` must be an `http` or `https` URL on this machine (`127.0.0.1`, `localhost` or `::1`), so a
  request and its key never leave it, and may not carry credentials. Nothing is sent to a hosted model; see
  [ADR 0005](docs/adr/0005-local-only.md).
- **Recordings.** Jarvis keeps no audio. Setting `JARVIS_DEBUG_AUDIO_DIR` saves the owner's requests (never PC
  audio) to that folder, mode 0600, for debugging; leave it unset otherwise.

Planned, not built yet (reports about them are out of scope until they exist): the web tools and the local
SearXNG container, PC operations other than Spotify, and keybinding confirmation.
[AGENTS.md](AGENTS.md) ("Boundaries") lists the security-sensitive code.
