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

[`config.example.toml`](../config.example.toml) in the repository lists every key at its default, with no
personal values: copy it to `~/.config/jarvis/config.toml` and change what you need. A missing config file
means all defaults. An unknown key, an out-of-range value, a malformed file or a
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
| `request_timeout_s` | `120.0` | above 0, up to 600 | Longest one model call may wait. A request also stops after `agent.max_seconds` (`chat.max_seconds` in the chat) in total. |
| `temperature` | `0.2` | 0 to 2 | Sampling temperature. |

Jarvis sends every request with thinking switched off. Older versions saved `max_tokens` here; a file
that still holds it loads, and the key is ignored: the reply cap is now `reply_tokens` in `[agent]` and
`[chat]`.

## `[agent]`: what one spoken request may do, and what Jarvis remembers

| Key | Default | Allowed | What it does |
|---|---|---|---|
| `max_steps` | `8` | 1 to 100 | Most model calls in one request; the last one may only answer, from what the others found. |
| `max_tool_calls` | `12` | 1 to 200 | Most tool calls in one request. |
| `max_seconds` | `90.0` | 10 to 3600 | Longest one request may take (180 s more when the model must load first). |
| `reply_tokens` | `400` | 64 to 8192 | Most tokens the model writes in one answer. The prompt asks for replies that fit; a reply the cap ends anyway is kept and says it was cut short. |
| `tool_text_chars` | `3000` | 500 to 50000 | Most characters of one tool result the model reads. |
| `memory_tokens` | `6000` | 0 to 200000 | Room the remembered conversation may take of the context window. When it outgrows it, the model summarizes the oldest turns (the latest two stay word for word); if that fails, the oldest are dropped. Jarvis uses less when the window cannot hold it beside the prompt, a reply and two tool results. |
| `memory_idle_s` | `1800.0` | 0 or more | Forget the conversation after this long without a request; `0` never forgets. |

A spoken reply that drew on outside text (a web page, a song name) is left out of later requests, as before;
the chat keeps it (see below).

## `[chat]`: the same for typed requests in `jarvis chat`

The same keys as `[agent]`, with room for longer written work: `max_steps` `30`, `max_tool_calls` `40`,
`max_seconds` `600.0`, `reply_tokens` `2048`, `tool_text_chars` `8000` (a read page fills it),
`memory_tokens` `12000` and `memory_idle_s` `0.0` (never forget), plus:

| Key | Default | Allowed | What it does |
|---|---|---|---|
| `resume` | `true` | `true`, `false` | Open the chat on the conversation it had last time, kept 0600 in `~/.local/state/jarvis/chat-conversation.json`. `false` starts each session fresh and deletes the file. |
| `verbosity` | `replies` | `replies`, `steps`, `everything` | How much of Jarvis's work the chat shows: requests and replies only, plus each step and its result, or plus every model call and detail. Ctrl+O and `/verbosity` change it and save it here. |
| `toolbar` | `true` | `true`, `false` | The line under the input with the PC's CPU, GPU, VRAM and RAM use, temperatures, and how full the model's context is. |
| `history_size` | `1000` | 0 to 100000 | Typed lines kept for ↑ between sessions, 0600 in `~/.local/state/jarvis/chat-history`; `0` keeps none. |

`/config` in the chat changes `verbosity`, `toolbar`, `resume`, `reply_tokens` and `max_steps` and writes
them here; the last three apply the next time the chat opens.

The chat remembers replies that drew on outside text, as data. While one is remembered, an action with a side
effect that the model chooses needs your yes; a plain command (`pause`, `volume 40`) does not, since the
model plays no part in it.

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
| `context` | `24576` | 2048 to 262144 | Context window in tokens (`--max-context`). Each 1,000 tokens more costs about 0.05 GiB of VRAM ([models.md](models.md)). It holds the prompt and the tool schemas (about 5,000 tokens together), the remembered conversation (`agent.memory_tokens` or `chat.memory_tokens`), one reply and the request's tool results; when a long request outgrows it, its oldest tool results are cleared down to their start. |
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
| `wake_threshold` | `0.5` | above 0, below 1 | LiveKit's wake word score needed. Higher means fewer false wakes and more misses. The service log shows each wake's score. |
| `wake_engine` | `livekit` | `livekit` or `azure` | Which wake word model listens. `azure` uses your Azure custom keyword model; see [Azure wake word](#azure-wake-word). `jarvis wake livekit\|azure`. |
| `azure_keyword_model` | empty | path to a `.table` file | The Azure custom keyword model. `jarvis wake azure --model FILE`. |
| `azure_python` | empty | path to a Python | The Python that has the Speech SDK. Empty means `<data_dir>/azure-wake/venv/bin/python`. |
| `vad_threshold` | `0.5` | above 0, below 1 | Speech probability that counts as speech. |
| `no_speech_timeout_s` | `4.0` | above 0, up to 30 | Silence after the wake word before Jarvis gives up. |
| `end_silence_s` | `0.7` | above 0, up to 5 | Trailing silence that ends a request. |
| `max_utterance_s` | `20.0` | above 1, up to 120 | Longest request. |
| `confirm_window_s` | `8.0` | above 1, up to 30 | How long Jarvis waits for the answer to a confirmation question. |
| `barge_in` | `true` | true or false | Listen for the wake word while Jarvis speaks: saying it stops Jarvis and starts a new request (a question spoken over counts as no). See [ADR 0008](adr/0008-speak-while-writing-and-barge-in.md). |
| `follow_up_s` | `6.0` | 0 to 15 | Seconds after a spoken reply in which the next request needs no wake word; 0 turns it off. The window opens once the reply has finished playing, only after a request in your own voice, and never while Voxtype dictation is recording or transcribing (`voxtype status`); with `listen_to_pc` on, speech heard while PC audio also held speech is dropped; speech shorter than 0.3 s (a cough, a click) is ignored and the window listens on; speech in it that cannot be made out is dropped without a word. The wake word inside the window starts a request as usual. Background speech in the room can start a request too: if it does, shorten the window or set 0. |
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
      aec.args = { webrtc.noise_suppression = false webrtc.gain_control = false }
      capture.props = { node.name = "jarvis-aec-capture" }
      source.props = { node.name = "jarvis-aec-source" node.description = "Microphone for Jarvis" }
    }
  }
]
```

Then run `systemctl --user restart pipewire`, set `audio.source = "jarvis-aec-source"` and run
`jarvis restart`. `monitor.mode` takes what the default output plays as the sound to remove, so no new output device
appears and every app keeps playing where it did. It removes Jarvis's own voice from the microphone too, so a
wake word heard while Jarvis speaks (`barge_in`) is more likely to be yours. `aec.args` turns off WebRTC's noise suppression and
gain control: noise suppression is on by default, and one practitioner found that noise suppression and gain
control together stopped the wake word from being heard at all
([FINDINGS.md](https://github.com/bluetooth47/agent-voice/blob/main/bench/FINDINGS.md)). Use a wired or
built-in microphone: after a WirePlumber restart the module can fall back to a Bluetooth headset's
microphone, which drops the headset to its low-quality call profile. Check the wake word, idle and while
Jarvis speaks, before and after the change. This setup was checked to load in PipeWire 1.6.9, not yet
measured on the owner's microphone.

**Warning:** when echo cancellation was applied on the maintainer's PC on 2026-11-21, the PC's sound output
broke and the change was reverted; the cause is not known yet
([#13](https://github.com/NecturaLabs/Jarvis/issues/13)). Before trying it, keep the way back at hand: delete
`60-jarvis-echo-cancel.conf`, run `systemctl --user restart pipewire`, set `audio.source` back to `""` and run
`jarvis restart`.

### Azure wake word

Jarvis can listen with an Azure AI Speech custom keyword model instead of LiveKit's `hey_jarvis`
([ADR 0010](adr/0010-azure-keyword-wake-word.md)). The model is a `.table` file you train in Speech Studio
(Custom Keyword) and download from the project's Models page. It runs on your PC: Microsoft's Speech SDK is
loaded only by a small helper that Jarvis starts with no network, so it cannot send anything to Microsoft.
The helper still runs as you, with your files and local sockets.

1. Put the SDK in its own environment (it is closed source, so it is never one of Jarvis's packages):

   ```sh
   uv venv --python 3.14 <data_dir>/azure-wake/venv
   uv pip install --python <data_dir>/azure-wake/venv/bin/python azure-cognitiveservices-speech==1.53.1
   ```

2. Keep the `.table` file outside the repository (for example under `<data_dir>/azure-wake/`) and switch:
   `jarvis wake azure --model <file>.table`. `jarvis wake livekit` switches back; `jarvis wake` shows which
   one listens. Each switch restarts the voice service if it is running.
3. With `audio.wake_engine = "azure"`, `jarvis doctor` checks the model file, the Python and that a namespace with no network can be made.

If the Azure model cannot start, Jarvis says so in the log and listens with LiveKit's instead. The model's own
threshold decides, so `wake_threshold` does not apply to it. An Azure model trained with "Containing single
keyword" also wakes on "Jarvis" alone; while one listens, a sentence in which Jarvis says its own name cannot
be spoken over.

## `[persona]`

| Key | Default | What it does |
|---|---|---|
| `name` | `Jarvis` | What the prompt calls the assistant. It is also dropped from the start or end of a request ("Jarvis, pause"). The wake phrase is fixed by the wake word model and is not changed by this. |
| `tone` | calm, warm, to the point, with a light touch of dry wit | How it speaks. |
| `addendum` | empty | Your own extra instructions, added to the system prompt. |

## `[clock]`

| Key | Default | Allowed | What it does |
|---|---|---|---|
| `twelve_hour` | `true` | `true`, `false` | Say the time as "3:42 PM"; `false` says "15:42". |

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
| `page_chars` | `2800` | 500 to 50000 | Most characters of a page or Reddit thread passed to the model for a spoken request, within `agent.tool_text_chars`. The chat reads up to its own `tool_text_chars`. |
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
