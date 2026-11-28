# What Jarvis keeps and sends

Jarvis runs on your PC and can use Strata on another machine you own. This page lists what Jarvis
writes to disk, what leaves the PC and how to remove it.

## What stays on your PC

| What | Where | What it holds |
|---|---|---|
| Config | `~/.config/jarvis/config.toml` | Your settings, including your Spotify app's client ID (there is no client secret). |
| Strata API key | `~/.local/state/jarvis/`, mode 0600 | `strata.env` for local mode. With a remote model, `strata-backend.env` holds the backend key and `strata-remote.json` holds the client's copy bound to its URL. |
| Spotify tokens | `~/.local/state/jarvis/spotify-token.json`, mode 0600 | The access and refresh tokens from `jarvis spotify-login`. |
| Audit log | `~/.local/state/jarvis/audit.jsonl`, mode 0600 in a 0700 folder | Every request as transcribed or typed (a paste in the chat included), the model's tool calls, each policy decision and each result (with the start of each page or file Jarvis read, its own configuration files included, and of the programs and window titles `find_processes` listed). What `write_file` puts in a file is not in it: only its length and SHA-256. With PC audio on, it also holds what other people said in it to wake Jarvis; what `pc_audio_listen` heard for you is in its tool result only as a length and SHA-256, but the reply Jarvis gives about it is recorded as written, like every reply. At 5 MiB it moves to `audit.jsonl.1` and a new one starts; the five older logs are kept (`.1` the newest, `.5` the oldest) and the one before them is deleted, so at most about 30 MiB. |
| Reminders and scheduled requests | `~/.local/state/jarvis/schedule.json`, mode 0600, with `schedule.json.lock` beside it | The words of each reminder and scheduled request as the model wrote them from your request (at most 50, 200 characters each), with when it is due and how it repeats. A one-off job is removed once it is said or done, a repeating one when you cancel it. The audit log also holds the words, in the `schedule_add` call that set the job, and records each job coming due or missed by its number only. |
| Action count | `~/.local/state/jarvis/actions.json`, mode 0600 | The times of the side-effecting actions of the last 10 minutes (`boundaries.actions_window_s`), shared by the voice service and the chat to cap how many run. No names or words. |
| Restart note | `~/.local/state/jarvis/voice-restart-pending`, mode 0600 | An empty marker the chat leaves when a setting changed while a spoken request was under way; the voice service restarts itself once it is idle and deletes it. It holds no text. |
| Spoken hand-off to the chat | the audit log, as `chat_handoff_received` | When you ask by voice for something to be done in the open chat, the audit log records the words you said (`chat_handoff_received` holds them, and the chat's request that follows is marked "spoken") and each refused hand-off (`chat_handoff_refused`). The words travel only over the chat window's own socket in the runtime folder (`$XDG_RUNTIME_DIR/jarvis/chat/`, mode 0600) and are kept nowhere else. |
| Chat conversation | `~/.local/state/jarvis/chat-conversation.json`, mode 0600 | What the chat remembers between sessions (`chat.resume`): each request and reply, what was done and looked up, a summary of older turns, and with `chat.memory_detail = "full"` (the default) each request's tool calls and results as the model read them: whole pages, file text, command output and text written to files, within `chat.memory_tokens`. `/clear` empties it. Speech keeps its conversation in memory only. |
| Chat input history | `~/.local/state/jarvis/chat-history`, mode 0600 | The last `chat.history_size` (1000) lines you sent in the chat, for ↑, with the text you pasted into each kept beside it so a long paste brought back still counts as pasted. |
| Chat archive | `~/.local/state/jarvis/chat-archive/`, folder 0700, files 0600 | Every chat conversation, for `/resume` and for reading back: three files each, the conversation as the model remembers it (with which parts came from outside text or your files), a log of the window's own events (your requests, the steps and tool results with their text, the replies, the times) and its name, times and number of requests. With `chat.memory_detail = "full"` (the default) the conversation file also holds each request's tool calls and results: whole pages, file text, command output and text written to files, in each of up to `chat.archive_keep` conversations. Written after each request, and on `/clear`, `/resume` and when the chat closes if something changed; opening and closing the chat with nothing new writes no copy. `/clear` starts a new conversation and leaves the archived one in place. Only the newest `chat.archive_keep` (50) are kept, older ones are deleted; with `0` the chat keeps none and deletes those kept before when it opens. `/forget all` in the chat deletes them all and forgets the conversation in the window too, as `/clear` does, so none of it is archived again; it waits while a request runs. Deleting the folder removes them as well. |
| T3 Code sessions | `~/.local/state/jarvis/acp-sessions/`, folder 0700, files 0600 | Each `jarvis acp` session's conversation as the model remembers it (one file per mode it used, with the same provenance flags as the chat archive), beside a small file with its mode and effort, so a thread in T3 Code goes on after Jarvis or T3 Code restarts. Written after each reply. Only the newest 200 sessions are kept. `chat.archive_keep = 0` keeps none and deletes those kept before; `/forget all` in the chat and deleting the folder remove them; `/clear` in a thread forgets that one. |
| Service log | The systemd journal (`jarvis logs`) and `~/.local/state/jarvis/strata.log` | What Jarvis heard, the answers to its questions and any errors; for each request, its sound levels as numbers (`levels:`), never the sound. |
| Output recordings | `~/.local/state/jarvis/recordings/`, folder 0700, WAV files 0600 | Up to five minutes of audio Jarvis sends to its player, after you ask it to record. The voice service saves it when you ask it to stop or it shuts down normally. The file contains the player input with idle time omitted. |
| Plugins | `<data_dir>/plugins/` (`data_dir` is in the config; never read or written by the file operations) | Only after you enable a plugin: the Git repositories of its marketplace and, for a plugin kept elsewhere, its own, as downloaded ([ADR 0027](adr/0027-skill-plugins.md)). Jarvis reads only their skills; the copy before the latest update stays beside as a hidden `.old` folder until the next one. The folder is not private data of yours; remove it with `jarvis plugins uninstall` and deleting the folder. |
| Speech test set | The folder `jarvis speech-test` was given (`speech-test` in the data folder by default), mode 0700, files 0600 | Only when you run `jarvis speech-test record`: each request you said as a WAV file, with the card, the words you confirmed or typed and its sound levels (`set.jsonl`). `score --out` writes what was heard to the file you name, mode 0600. Nothing is sent anywhere; delete the folder to remove it. |

Audio is held in memory. Each request is written to a private temporary file for Voxtype to transcribe and
deleted straight after. With PC audio on, what `pc_audio_listen` hears for you (up to a minute) is held in
memory until it is transcribed, and written to such a temporary file only while Voxtype transcribes it. With
`audio.pc_audio_recall_s` set (0, off, by default), the last that many seconds of PC audio (60 at most) are
also held in memory while a stream plays, each frame dropped as a newer one comes, so Jarvis can tell you what
was just said. Nothing is held while Jarvis is paused or speaks, and a pause or the end of the stream drops it
([ADR 0015](adr/0015-listen-to-pc-audio.md)). The words heard are not kept in the remembered conversation after
the request that heard them, even with `agent.memory_detail = "full"`; Jarvis's reply about them is remembered
like every reply. `JARVIS_DEBUG_AUDIO_DIR` saves the owner's microphone requests there
(mode 0600) for debugging. Output recordings are independent of this setting.

By default the model runs in the `jarvis-strata` service on `127.0.0.1` and model requests stay on this PC.
With the Azure wake word on, the audio Jarvis listens to goes to a helper process on the PC that runs
Microsoft's Speech SDK in a namespace with no network, so neither the audio nor the SDK's telemetry can go
out over the network. It still runs as you, so it can read your files and local sockets; it only cannot
reach the network ([ADR 0010](adr/0010-azure-keyword-wake-word.md)).
Jarvis sends no telemetry and checks for no updates.

## What leaves your PC

With [a remote Strata model](configuration.md#model-on-another-machine), the model request goes to
your backend. It can include the system prompt, transcribed or typed requests, remembered conversation,
tool descriptions, tool results such as file excerpts or command output, and pictures you attach.
Model replies and tool calls travel back to the client. The Strata bearer key goes with model requests
and with health and capability checks, including at client startup. Microphone recordings, wake-word
processing, Spotify credentials and local audit files stay on the client. The backend and any HTTPS
proxy you set up can read the request; check how those services log and retain data. A forwarded
loopback connection travels through SSH. Direct HTTP requires explicit opt-in and relies on your
network for confidentiality.

Installation downloads code from GitHub and locked packages from PyPI. `tools/get_models.py` (run by the installer) fetches
the pinned GitHub model assets. `jarvis update` contacts the configured Git remote and applying an
update may download locked packages. The daily timer does this after you opt in. These operations
send their normal download requests and do not upload your Jarvis settings, audit log, audio or
requests.

Plugins: while `plugins.enabled` is empty (the default) Jarvis contacts nothing for them. Once you enable
one, `jarvis plugins install`, `jarvis plugins update` and `jarvis update --apply` (so the daily timer, when
`plugins.auto_update` is on) run `git clone` and `git fetch` over https against the marketplace repositories
named in `plugins.marketplaces` (`github.com` for an `owner/repo`) and the repositories their entries name.
Git sends nothing of your settings, audit log, audio or requests, and reads none of your own Git
configuration; the host sees your address and the repository asked for.

Spotify, once you have run `jarvis spotify-login` with your own Spotify developer app
([ADR 0005](adr/0005-local-only.md)):

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
- **Web pages**: Jarvis opens only a page a search in the same conversation returned, or an address you wrote
  yourself in a request of the conversation, exactly as written (never one in a page, a file, or a long paste while Jarvis asks before acting),
  and only on a public address. Jarvis checks the PC's own addresses and connected networks before
  opening a connection. If it cannot read that information, it refuses the connection. The site sees
  your IP address and a generic user agent.
- **Pictures in the chat**: a picture search goes through SearXNG like a web search. A picture Jarvis shows
  is fetched from the site that holds it, which sees your IP address, and only when an image search or a
  read page in the same conversation offered it. The chat keeps the newest 50 shown pictures, 0600, in
  `~/.local/state/jarvis/chat-pictures/`; delete the folder to remove them.
- **The headless browser** (when agent-browser is installed; [ADR 0016](adr/0016-headless-browser.md)): it
  opens the same pages `web_read` may, and the links they show. Each site sees what any visit by Chrome shows
  it (your public IP address, the browser and what the page itself loads), and gets words only when you said
  yes to typing them there. An interactive terminal is required to confirm browser actions under
  `jarvis ask --yes`. It carries none of your browsers' cookies or logins, uses a home folder of its
  own (never your Downloads), its own profile is deleted once it has ended, and it reaches nothing on this PC
  or your network.
- **Reddit**: thread searches go through SearXNG the same way, with `site:reddit.com` added; the
  `arctic-shift.photon-reddit.com` archive gets the ids of the threads Jarvis reads.

Files Jarvis reads stay on the PC. Their text can go to the configured model. While a file's text, or a shell
command's output, is in the conversation, a search or a place lookup that could carry it out needs your yes
first.

Pictures you attach in the chat go to the configured model, shrunk; they are not written to disk, and the
conversation keeps only their names. The clipboard is read only when you press Ctrl+V in the chat.

No request names you, the PC or Jarvis. Set an operation to `deny` in `[boundaries.classes]` to keep it
from ever running.

## Removing it

- Disable the update timer with `jarvis update --auto off` and stop Jarvis before removing the
  installation. The default installer puts the checkout and model assets under
  `~/.local/state/jarvis/`, alongside the saved state. Remove only the folders you want to delete.
- The service log is in the systemd journal, which keeps and drops it by its own limits (`journald.conf`, see
  `man journald.conf`). There is no way to remove only Jarvis's lines from it.
- Remove Jarvis's access to your Spotify account on Spotify's account page, under apps.
- `jarvis stop` stops both services; the units are in `~/.config/systemd/user/`.
- `docker compose -f deploy/searxng/compose.yaml down` stops and removes the search container. While it runs
  it keeps engine caches in memory (`/tmp` in the container); they go with it.
