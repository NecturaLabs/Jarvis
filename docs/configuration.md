# Configuration

Every setting lives in one TOML file, validated when Jarvis reads it. This page is written from
`src/jarvis/config.py`; the code wins if they differ.

## Where things live

| What | Where (default) | Set by |
|---|---|---|
| Config file | `~/.config/jarvis/config.toml` (`$XDG_CONFIG_HOME/jarvis/`) | `jarvis setup`, `jarvis pc-audio`, or by hand |
| Strata profile (generated) | `~/.config/jarvis/strata-profile.json` | `jarvis setup` |
| Jarvis's own Voxtype settings (generated) | `~/.config/jarvis/voxtype.toml` | `jarvis run`, at each start |
| User units | `~/.config/systemd/user/jarvis.service` and `jarvis-strata.service` | `jarvis setup` |
| Strata API key, Spotify tokens, audit log, Strata log | `~/.local/state/jarvis/` (`$XDG_STATE_HOME/jarvis/`): `strata.env`, `spotify-token.json`, `audit.jsonl`, `strata.log` | Jarvis; the key and token files are mode 0600 |
| Wake, VAD and voice model files | `data_dir`, default `~/.local/share/jarvis/` | you; see [models.md](models.md) |

A missing config file means all defaults. An unknown key, an out-of-range value, a malformed file or a
`schema_version` other than `1` is an error that names the file and key; `jarvis` exits with status 2.
Paths and names may not contain control characters, because they are written into systemd units.

## Which changes need a restart

Jarvis reads the file when a command starts, so `jarvis ask`, `status`, `unload` and `doctor` always see the
current file. The running voice service reads it once, when it starts.

- `audio`, `persona`, `boundaries`, `spotify`, `model`, `clock`, `weather` and `research` keys: run
  `jarvis restart`.
- `strata` keys: run `jarvis setup` to rewrite the profile and unit (it keeps every other value), then
  `jarvis restart`. It restarts Strata too when its profile or unit was rewritten after Strata started, and
  otherwise leaves a running Strata, and its loaded model, alone. An empty `model.base_url` follows
  Strata's port; a saved URL that only repeated that port is dropped when the file is read, and
  `jarvis doctor` fails a saved URL on another port than Strata's.
- `jarvis pc-audio on|off` rewrites the file and restarts the voice service itself when it is running.

## Top level

| Key | Default | Allowed | What it does |
|---|---|---|---|
| `schema_version` | `1` | `1` | File format version. Any other value is refused. |
| `data_dir` | `~/.local/share/jarvis` | path | Where the wake word, VAD and voice model files live. `jarvis setup --data-dir`. |

## `[model]`: how Jarvis talks to Strata

| Key | Default | Allowed | What it does |
|---|---|---|---|
| `base_url` | empty | empty, or an `http://` or `https://` URL on `127.0.0.1`, `localhost` or `::1`; no user name or password | Empty means `http://127.0.0.1:<strata.port>/v1`. |
| `model` | `jarvis` | text | The model name sent in each request. |
| `request_timeout_s` | `120.0` | above 0, up to 600 | Longest one model call may wait. A request also stops after 90 s in total. |
| `max_tokens` | `200` | 1 to 4096 | Reply length cap per model call. |
| `temperature` | `0.2` | 0 to 2 | Sampling temperature. |

Jarvis sends every request with thinking switched off.

## `[strata]`: how the `jarvis-strata` service runs the model

These are written into the Strata profile and unit by `jarvis setup`. Strata serves only Jarvis and listens on
`127.0.0.1`.

| Key | Default | Allowed | What it does |
|---|---|---|---|
| `home` | `~/.local/share/strata` | path | Strata's data folder (`STRATA_DATA_DIR`). `--strata-home`. |
| `components_parent` | `~/.local/share` | path | The `XDG_DATA_HOME` Strata runs with. `--strata-components-parent`. |
| `model_file` | unset | path | The model file. Once set, the Strata unit starts only if the file exists; `jarvis doctor` fails until it is set. `--model-file`. |
| `tokenizer_dir` | unset | path | The model's tokenizer folder. `--tokenizer-dir`. |
| `model_id` | `qwen3.6-35b-a3b-ud-q4_k_xl` | text | The model's name inside Strata's profile. |
| `port` | `8080` | 1024 to 65535 | Strata's port on `127.0.0.1`. |
| `context` | `8192` | 2048 to 262144 | Context window in tokens (`--max-context`). |
| `expert_vram_gib` | `1.0` | 0 to 64 | VRAM budget for expert weights kept on the GPU, passed as `--expert-vram`; the rest of the experts stay in host memory. |
| `prefill` | `512` | 128 to 8192 | Prompt-reading chunk size (`--prefill`). |
| `idle_unload_s` | `300` | 30 to 86400 | Seconds without a request before Strata frees the model. |
| `reasoning_budget_tokens` | `512` | 1 to 8192 | Thinking-token cap in the profile. |
| `memory_max` | `40G` | digits and optional `K`, `M`, `G` or `T` | `MemoryHigh` and `MemoryMax` of the Strata unit. |

Fixed by Jarvis, not configurable: lazy loading on, one parallel slot, the HIP backend on GPU 0, `--mtp 2`.

### Keeping the model loaded

`idle_unload_s` accepts up to 86400 (24 hours), which keeps the model loaded through the day. Measured on the
reference machine (RX 7900 XT, 64 GB RAM), a loaded idle model holds about 38 GB of RAM and 7.3 GiB of VRAM and
uses no GPU or CPU while idle. A cold load took about 107 s with a game running; Jarvis says "One moment,
waking up" while that happens. At the default 300 s the model is freed after five idle minutes and loaded
again on the next wake word. The default `memory_max` of 40G leaves about 2 GB above the 38 GB measured, so
raise it before holding a larger model. `jarvis unload` frees the model at once.

## `[spotify]`: your own Spotify developer app

| Key | Default | Allowed | What it does |
|---|---|---|---|
| `client_id` | empty | text | Your app's client ID. There is no client secret. `--spotify-client-id`. |
| `redirect_port` | `8765` | 1024 to 65535 | Port of the login callback, `http://127.0.0.1:<port>/callback`; your app must list exactly this redirect URI. |
| `device_name` | empty | text | Preferred player, matched by name ignoring case. Empty means whichever player is active or listed. |
| `close_match` | `0.8` | above 0, up to 1 | How similar a search result must be to your words to be taken as the answer. |
| `loose_match` | `0.6` | above 0, up to 1, and not above `close_match` | The weaker similarity used to accept an artist or a library item. |
| `queue_max` | `20` | 1 to 100 | Most songs one queue request may add. |
| `search_pages` | `8` | 1 to 20 | Search result pages read while looking for a match. |
| `library_refresh_s` | `21600` | 60 or more | How often the playlist and liked-songs cache is refreshed in the background. |

## `[audio]`: microphone, wake word, speech and voice

| Key | Default | Allowed | What it does |
|---|---|---|---|
| `source` | empty | PipeWire source node name | Microphone. Empty means the default source. `--mic`. |
| `wake_threshold` | `0.5` | above 0, below 1 | Wake word score needed. Higher means fewer false wakes and more misses. |
| `vad_threshold` | `0.5` | above 0, below 1 | Speech probability that counts as speech. |
| `no_speech_timeout_s` | `4.0` | above 0, up to 30 | Silence after the wake word before Jarvis gives up. |
| `end_silence_s` | `0.7` | above 0, up to 5 | Trailing silence that ends a request. |
| `max_utterance_s` | `20.0` | above 1, up to 120 | Longest request. |
| `confirm_window_s` | `8.0` | above 1, up to 30 | How long Jarvis waits for the answer to a confirmation question. |
| `voice` | `kokoro:bm_george` | text | `kokoro:<name>` for a Kokoro voice, otherwise a Piper voice name. See [models.md](models.md). |
| `speech_speed` | `1.0` | 0.5 to 2.0 | Speaking speed, for Kokoro and Piper voices. |
| `transcribe_engine` | empty | text | Voxtype `--engine`. Empty means Voxtype's own. |
| `transcribe_model` | `small.en` | text | Voxtype `--model`: a model name or a full path to a model file. Empty means Voxtype's own. The model must already be downloaded; `jarvis doctor` checks Whisper models. |
| `transcribe_threads` | `8` | 1 to 256 | Whisper threads, written to Jarvis's own Voxtype settings file. |
| `transcribe_short_window` | `true` | true or false | Size Whisper's window to the clip (`context_window_optimization`); much faster for short commands. |
| `vocabulary_hint` | `true` | true or false | Prime speech-to-text with Jarvis's name, its actions and your artist names. |
| `hint_names` | `40` | 0 to 200 | Most artist names put in the hint (the hint also stops at 800 characters). |
| `listen_to_pc` | `false` | true or false | Also take requests heard in PC audio. `jarvis pc-audio on`. |
| `pc_audio_apps` | `["discord"]` | list of app or process names | Whose audio to listen to. An empty list means all PC output. `--apps`, `--all-output`. |
| `pc_audio_grant_s` | `600.0` | 0 to 86400 | After you say yes to a PC-audio request, how long further PC-audio requests run without asking. 0 asks every time. |
| `confirm_blocks_on_pc_speech` | `true` | true or false | A yes counts as no if speech was heard in PC audio while Jarvis waited for it. |
| `pc_audio_cooldown_s` | `5.0` | 0 to 600 | Pause after a PC-audio request during which another is ignored. |

See [the PC-audio ADR](adr/0003-pc-audio-is-not-the-owner.md) for why the last four exist.

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
      capture.props = { node.name = "jarvis-aec-capture" }
      source.props = { node.name = "jarvis-aec-source" node.description = "Microphone for Jarvis" }
    }
  }
]
```

Then run `systemctl --user restart pipewire`, set `audio.source = "jarvis-aec-source"` and run
`jarvis restart`. `monitor.mode` takes what the default output plays as the sound to remove, so no new output device
appears and every app keeps playing where it did.

## `[persona]`

| Key | Default | What it does |
|---|---|---|
| `name` | `Jarvis` | What the prompt calls the assistant. It is also dropped from the start or end of a request ("Jarvis, pause"). The wake phrase is fixed by the wake word model and is not changed by this. |
| `tone` | calm, warm, to the point, with a light touch of dry wit | How it speaks. |
| `addendum` | empty | Your own extra instructions, added to the system prompt. |

## `[clock]`

| Key | Default | Allowed | What it does |
|---|---|---|---|
| `twelve_hour` | `false` | `true`, `false` | Say the time as "3:42 PM" instead of "15:42". |

The time is this PC's own clock and time zone; only the time in a named place asks Open-Meteo for that
place's time zone.

## `[weather]`: your home, for weather that names no place

Weather comes from Open-Meteo, which needs no account or key ([ADR 0007](adr/0007-outside-services-for-answers.md)).
"What's the weather" uses your home; "what's the weather in Lisbon tomorrow" looks Lisbon up.

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
| `page_chars` | `2800` | 500 to 6000 | Most characters of a page or Reddit thread passed to the model. |
| `reddit_comments` | `6` | 1 to 10 | Top comments read from a Reddit thread. |

Search needs the SearXNG container running; without it Jarvis says search isn't running, and everything
else keeps working. To start it:

```sh
printf 'SEARXNG_SECRET=%s\n' "$(openssl rand -hex 32)" > deploy/searxng/.env
docker compose -f deploy/searxng/compose.yaml up -d
```

It listens on `127.0.0.1:8888` only (`SEARXNG_PORT` in `.env` changes the port; change `searxng_url`
with it) and restarts with Docker unless you stop it (`docker compose -f deploy/searxng/compose.yaml
down`). The `.env` file is git-ignored. Reddit threads are found through SearXNG and read from the
Arctic Shift archive, a free copy of public Reddit that can lag behind it by a little.

## `[boundaries.classes]`

A table from operation name to a class: `safe` (runs), `confirm` (needs your spoken yes), `strong`
(keybinding confirmation, not built yet, so such a call is refused) or `deny`. An operation not listed keeps
its shipped class, which is `safe` for every operation Jarvis has today (`spotify_*`, `time_now`, `weather`,
`web_search`, `web_read`, `reddit_search`, `reddit_read`); set one to `deny` to switch it off. Whatever the class, side effects are raised to
`confirm` while text from outside your own speech is in the request. Example:

```toml
[boundaries.classes]
spotify_like = "confirm"
spotify_queue = "deny"
reddit_search = "deny"
```
