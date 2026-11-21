# Jarvis

A local, always-on voice agent for Omarchy. Say "Hey Jarvis" and it transcribes you with Omarchy's local
dictation (Voxtype), reasons with a local model served by [Strata](https://github.com/niko1221/strata), acts within the boundaries
you set and answers by voice. It runs on your PC and is free to use: no hosted model, no paid API.

Jarvis is pre-1.0 software; see [Status](#status).

## What it does today

- **Wake word:** a local "Hey Jarvis" model (LiveKit wake word) listens on your microphone. Silero VAD finds
  the end of your request.
- **Speech to text:** Voxtype, run with Jarvis's own settings file. Your own Voxtype configuration is never
  edited.
- **Reasoning:** one agent on a local Qwen3.6-35B-A3B model served by Strata. Plain commands such as "pause",
  "next song" or "volume 40" skip the model and answer in about 0.3 s.
- **Voice:** Kokoro by default (`kokoro:bm_george`); a Piper voice can be set instead. Jarvis replies by voice
  only, with no desktop notifications.
- **Spotify:** play a track, album, artist, playlist or your liked songs; pause, resume, next, previous; set or
  change the volume; shuffle; repeat; queue songs; like the current song; say what is playing. It can start the
  Spotify desktop app when you ask it to play or resume and no player is open.
- **Boundaries:** every action is a typed operation checked by a policy engine in code before it runs, and
  every request, decision and result goes to an audit log. See [SECURITY.md](SECURITY.md).
- **Listening to PC audio (off by default):** Jarvis can also take requests heard in what the PC plays, such as
  a Discord call. That speech is treated as someone else's, so you are asked to confirm before it changes
  anything. See [the ADR](docs/adr/0003-pc-audio-is-not-the-owner.md).

Not built yet: web research, PC operations other than Spotify, keybinding confirmation, voice barge-in. The
plan marks each as planned.

## Requirements

- Omarchy on Arch Linux with PipeWire (`pw-record` and `pw-play`; `pw-dump` too for PC audio) and systemd user services.
- Voxtype as the dictation backend, with the Whisper model you set in `audio.transcribe_model` downloaded.
- Python 3.14 and uv.
- Strata and a Qwen3.6-35B-A3B UD-Q4_K_XL model file with its tokenizer folder. Built and measured on an RX
  7900 XT with 64 GB of RAM; the loaded model holds about 38 GB of RAM and 7.3 GiB of VRAM there (see
  [docs/configuration.md](docs/configuration.md)).
- Spotify Premium and your own Spotify developer app (a client ID; there is no client secret).
- The wake word, VAD and voice model files, placed under the data folder ([docs/models.md](docs/models.md)).

## Set up

```sh
uv sync
uv run jarvis setup --model-file <model file> --tokenizer-dir <tokenizer folder> \
    --spotify-client-id <your app's client ID>
uv run jarvis spotify-login     # prints the redirect URI your Spotify app must list, then opens the login
uv run jarvis setup --enable    # writes and starts the user units
uv run jarvis doctor            # one line per check, with its fix; never loads the model
```

`jarvis setup` writes the config, the Strata profile, the Strata API key and two systemd user units
(`jarvis.service` and `jarvis-strata.service`), and keeps every value already set unless you pass it again.
Use `--data-dir`, `--strata-home`, `--strata-components-parent` and `--mic` for the other locations and the
microphone. The config file and every key are in [docs/configuration.md](docs/configuration.md).

## Everyday commands

These are the subcommands `jarvis --help` lists. `jarvis` on its own shows what is running and follows the log.

| Command | What it does |
|---|---|
| `jarvis setup` | Write the config, the Strata profile and the user units (`--enable` also starts them). |
| `jarvis spotify-login` | Connect Jarvis to your Spotify account with PKCE (`--no-browser` only prints the link). |
| `jarvis start` / `stop` / `restart` | Start Jarvis and its model server, stop both (which frees the model), or restart the voice service after a config change. |
| `jarvis status` | Show whether the model is loaded. |
| `jarvis unload` | Free the model's memory now, for example before a game. |
| `jarvis doctor` | Check every part without loading the model. |
| `jarvis logs` | Follow the live log (Ctrl+C to leave). |
| `jarvis ask <text>` | Send one typed request as if spoken; `--yes` answers yes to confirmations. |
| `jarvis chat` | Open a Textual window where you type requests and watch what Jarvis does, live. |
| `jarvis pc-audio [on\|off]` | Show or set listening to PC audio; `--apps discord,firefox` names the apps, `--all-output` listens to everything the PC plays. |
| `jarvis run` | Run the voice loop in the foreground. This is what `jarvis.service` runs. |
| `jarvis --version` | Print the version. |

## Docs

- [docs/configuration.md](docs/configuration.md): every config key, its range and what needs a restart
- [docs/models.md](docs/models.md): the model, speech to text, voices, the wake word, latency and licences
- [docs/plan.md](docs/plan.md): architecture, design and phases, with what is built
- [docs/roadmap.md](docs/roadmap.md): what is planned, and what Jarvis will not do
- [docs/adr/](docs/adr/): decisions
- [docs/playbook/](docs/playbook/): how to work on the project, and its traps
- [SECURITY.md](SECURITY.md): scope and how to report a vulnerability
- [ACCESSIBILITY.md](ACCESSIBILITY.md): voice, the chat window and keyboard-only use

## Status

Jarvis is pre-1.0 and has no tagged releases yet. It was built and measured on one machine (Omarchy, an RX
7900 XT and 64 GB of RAM); other hardware is untested. The configuration file has one schema version and no
migrations, so a later version may ask you to edit it. Web research, PC operations other than Spotify and
keybinding confirmation are not built; [docs/roadmap.md](docs/roadmap.md) lists what is planned and
[docs/plan.md](docs/plan.md) says what is built. Changes are in [CHANGELOG.md](CHANGELOG.md).

## Licence

Jarvis is licensed under the GNU General Public License, version 3 or (at your option) any later version
(GPL-3.0-or-later); see [LICENSE](LICENSE). It imports `piper-tts`, which is GPL-3.0-or-later, in the same
process for both of its voices, so Jarvis as a whole has to be GPL-compatible, and GPL-3.0-or-later is the
licence the project uses for its own code too. Contributions are accepted under the same licence (inbound =
outbound). Every dependency, model and voice is listed in [docs/models.md](docs/models.md) and
[docs/configuration.md](docs/configuration.md), with its licence where it has been checked.

## Contributing

Contributions are welcome: see [CONTRIBUTING.md](CONTRIBUTING.md) for what fits, the checks and the sign-off
(`git commit -s`). To work on the code:

```sh
uv sync                      # install the locked environment
uv run .lint/check.sh        # lint, format and type checks (tracked files only)
uv run pytest -q             # unit tests
```

Tests marked `local` need real hardware and services and never run in CI (`uv run pytest -m local`).

## Community

- [CONTRIBUTING.md](CONTRIBUTING.md): how to contribute
- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md): how we treat each other (Contributor Covenant 3.0)
- [SUPPORT.md](SUPPORT.md): where to get help; questions go to
  [Discussions](https://github.com/NecturaLabs/Jarvis/discussions)
- [SECURITY.md](SECURITY.md): report a vulnerability privately
- [GOVERNANCE.md](GOVERNANCE.md): who decides what
- [ACCESSIBILITY.md](ACCESSIBILITY.md): what Jarvis offers today, and its gaps
- [CHANGELOG.md](CHANGELOG.md) and [docs/roadmap.md](docs/roadmap.md): what changed and what is planned
- [Issues](https://github.com/NecturaLabs/Jarvis/issues): the backlog
