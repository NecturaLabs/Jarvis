# What Jarvis keeps and sends

Jarvis runs on your PC. This page lists what it writes to disk, what leaves the machine and how to remove
it. It is a plain statement of what the code does, not legal terms; the code wins if they differ.

## What stays on your PC

| What | Where | What it holds |
|---|---|---|
| Config | `~/.config/jarvis/config.toml` | Your settings, including your Spotify app's client ID (there is no client secret). |
| Strata API key | `~/.local/state/jarvis/strata.env`, mode 0600 | A random key that only Jarvis and its Strata service use. |
| Spotify tokens | `~/.local/state/jarvis/spotify-token.json`, mode 0600 | The access and refresh tokens from `jarvis spotify-login`. |
| Audit log | `~/.local/state/jarvis/audit.jsonl`, mode 0600 in a 0700 folder | Every request as transcribed or typed (a paste in the chat included), the model's tool calls, each policy decision and each result (with the start of each page or file Jarvis read, its own configuration files included, and of the programs and window titles `find_processes` listed). What `write_file` puts in a file is not in it: only its length and SHA-256. With PC audio on, it also holds what other people said in it. At 5 MiB it moves to `audit.jsonl.1` and a new one starts; the five older logs are kept (`.1` the newest, `.5` the oldest) and the one before them is deleted, so at most about 30 MiB. |
| Reminders and scheduled requests | `~/.local/state/jarvis/schedule.json`, mode 0600, with `schedule.json.lock` beside it | The words of each reminder and scheduled request as the model wrote them from your request (at most 50, 200 characters each), with when it is due and how it repeats. A one-off job is removed once it is said or done, a repeating one when you cancel it. The audit log also holds the words, in the `schedule_add` call that set the job, and records each job coming due or missed by its number only. |
| Action count | `~/.local/state/jarvis/actions.json`, mode 0600 | The times of the side-effecting actions of the last 10 minutes (`boundaries.actions_window_s`), shared by the voice service and the chat to cap how many run. No names or words. |
| Chat input history | `~/.local/state/jarvis/chat-history`, mode 0600 | The last `chat.history_size` (1000) lines you sent in the chat, for ↑, with the text you pasted into each kept beside it so a long paste brought back still counts as pasted. |
| Service log | The systemd journal (`jarvis logs`) and `~/.local/state/jarvis/strata.log` | What Jarvis heard, the answers to its questions and any errors; for each request, its sound levels as numbers (`levels:`), never the sound. |
| Speech test set | The folder `jarvis speech-test` was given (`speech-test` in the data folder by default), mode 0700, files 0600 | Only when you run `jarvis speech-test record`: each request you said as a WAV file, with the card, the words you confirmed or typed and its sound levels (`set.jsonl`). `score --out` writes what was heard to the file you name, mode 0600. Nothing is sent anywhere; delete the folder to remove it. |

Audio is held in memory. Each request is written to a private temporary file for Voxtype to transcribe and
deleted straight after. Jarvis keeps no recordings unless you set `JARVIS_DEBUG_AUDIO_DIR`, which saves each
request there (mode 0600) for debugging.

The model runs in the `jarvis-strata` service on `127.0.0.1`; your requests never leave the PC to reach it.
With the Azure wake word on, the audio Jarvis listens to goes to a helper process on the PC that runs
Microsoft's Speech SDK in a namespace with no network, so neither the audio nor the SDK's telemetry can go
out over the network. It still runs as you, so it can read your files and local sockets; it only cannot
reach the network ([ADR 0010](adr/0010-azure-keyword-wake-word.md)).
Jarvis sends no telemetry and checks for no updates.

## What leaves your PC

Nothing is sent until you ask for something that needs it. Spotify, once you have run
`jarvis spotify-login` with your own Spotify developer app ([ADR 0005](adr/0005-local-only.md)):

- `accounts.spotify.com`: the login (PKCE) and the token refresh.
- `api.spotify.com`: playback commands, the volume, the device list, searches built from your words ("play
  Glass Harbor" searches for "Glass Harbor"), saving a song when you ask Jarvis to like it, and reads of your
  playlists, liked songs, top artists and tracks, followed artists and recently played tracks, used to match
  what you ask for and to teach speech to text your artist names.

What Spotify does with that is set by Spotify's own privacy policy.

Questions that need the outside world ([ADR 0007](adr/0007-outside-services-for-answers.md)):

- **Weather and the time abroad**: `api.open-meteo.com` gets your home's coordinates (`[weather]`) or the
  place you name; `geocoding-api.open-meteo.com` gets a place name to look up (your home's name once per
  run when no coordinates are set). The time here is read from this PC's clock and sends nothing.
- **Web search**: your search goes to the SearXNG container on this PC, which passes it on, with no cookies
  or account, to the public search engines in SearXNG's default set (in a test on 2026-11-21 only Google's
  answered; others were rate-limited). They see the search and your public IP address.
- **Web pages**: Jarvis opens only a page a search in the same conversation returned, and only on a public
  address. The site sees your IP address and a generic user agent.
- **Pictures in the chat**: a picture search goes through SearXNG like a web search. A picture Jarvis shows
  is fetched from the site that holds it, which sees your IP address, and only when an image search or a
  read page in the same conversation offered it. The chat keeps the newest 50 shown pictures, 0600, in
  `~/.local/state/jarvis/chat-pictures/`; delete the folder to remove them.
- **Reddit**: thread searches go through SearXNG the same way, with `site:reddit.com` added; the
  `arctic-shift.photon-reddit.com` archive gets the ids of the threads Jarvis reads.

Files Jarvis reads stay on the PC: their text goes only to the local model. While a file's text, or a shell
command's output, is in the conversation, a search or a place lookup that could carry it out needs your yes
first.

Pictures you attach in the chat go only to the local model, shrunk; they are not written to disk, and the
conversation keeps only their names. The clipboard is read only when you press Ctrl+V in the chat.

No request names you, the PC or Jarvis. Set an operation to `deny` in `[boundaries.classes]` to keep it
from ever running.

## Removing it

- Delete `~/.local/state/jarvis/` to remove the audit log, the Strata log, the tokens and the key.
- The service log is in the systemd journal, which keeps and drops it by its own limits (`journald.conf`, see
  `man journald.conf`). There is no way to remove only Jarvis's lines from it.
- Remove Jarvis's access to your Spotify account on Spotify's account page, under apps.
- `jarvis stop` stops both services; the units are in `~/.config/systemd/user/`.
- `docker compose -f deploy/searxng/compose.yaml down` stops and removes the search container. While it runs
  it keeps engine caches in memory (`/tmp` in the container); they go with it.
