# Security policy

Jarvis is a private NecturaLabs project that acts on its owner's PC and accounts by voice, so security
issues matter even though it has no public users.

## Reporting a vulnerability

Open a GitHub security advisory on this repository (Security > Advisories > New draft advisory), or tell a
NecturaLabs maintainer directly. Do not file a public issue for a vulnerability, and never paste tokens,
recordings or personal data into a report.

## Supported versions

Only the latest release on `main` and the current `dev` branch are supported.

## Scope

Of particular interest: anything that lets speech, outside text or Spotify metadata cause an action the owner
did not confirm. What exists today and what is in scope:

- **Whose speech is whose.** Speech from the microphone is the owner's. Speech heard in PC audio (a call, a
  video) is someone else's: it is only listened to when the owner turns `listen_to_pc` on, its side effects
  need the owner's spoken yes, and it stays out of the owner's conversation history. A yes does not count if
  voices were heard in PC audio while Jarvis waited for it. See `docs/adr/0003-pc-audio-is-not-the-owner.md`.
- **Outside text is data.** Song, artist and playlist names from Spotify are wrapped as data and marked with
  their origin; while such text is in a request, every side-effecting operation needs a spoken yes.
- **Policy and confirmation.** Operations are typed with validated arguments and default deny; unknown
  operations are refused. A confirmation is bound to the hash of one exact action and starts only after
  Jarvis has finished speaking. Anything but a clear yes is a no, and after one no in a request the owner is
  not asked again. Every request, decision and result is written to `audit.jsonl` (mode 0600, kept to 5 MiB
  plus one rotated copy). The `strong` class (keybinding confirmation) is not built: such a call is refused.
- **Spotify.** Login is OAuth Authorization Code with PKCE (S256), a validated `state` and a callback bound to
  `127.0.0.1`. There is no client secret. Tokens are kept in a mode 0600 file under `~/.local/state/jarvis/`
  and a token file that is a symbolic link, belongs to another user or is readable by others is refused.
- **Local services.** Strata listens on `127.0.0.1` and requires an API key held in a mode 0600 file. A custom
  `model.base_url` must be `https`, or `http` to this machine, and may not carry credentials. Nothing is
  sent to a hosted model; see `docs/adr/0005-local-only.md`.
- **Recordings.** Jarvis keeps no audio. Setting `JARVIS_DEBUG_AUDIO_DIR` saves the owner's requests (never PC
  audio) to that folder, mode 0600, for debugging; leave it unset otherwise.

Planned, not built yet (reports about them are out of scope until they exist): the web tools and the local
SearXNG container, PC operations other than Spotify, and keybinding confirmation.
