# Jarvis

[![CI](https://github.com/NecturaLabs/Jarvis/actions/workflows/ci.yml/badge.svg?branch=dev)](https://github.com/NecturaLabs/Jarvis/actions/workflows/ci.yml)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/NecturaLabs/Jarvis/badge)](https://scorecard.dev/viewer/?uri=github.com/NecturaLabs/Jarvis)
[![Licence: GPL-3.0-or-later](https://img.shields.io/badge/licence-GPL--3.0--or--later-blue)](#licence)

A local, always-on voice agent for Omarchy. Say "Hey Jarvis" and it transcribes you with Omarchy's local
dictation (Voxtype), reasons with a local model served by [Strata](https://github.com/niko1221/strata), acts within the boundaries
you set and answers by voice. It runs on your PC and is free to use: no hosted model, no paid API.

Jarvis is pre-1.0 software; see [Status](#status).

## What it does today

- **Wake word:** local LiveKit wake word models listen on your microphone: "Hey Jarvis", plus any phrase you
  trained yourself, each turned on or off with `jarvis wake --words`. Your own Azure
  custom keyword model can listen instead, still on the PC with no network (`jarvis wake azure`; setup in
  [Azure wake word](docs/configuration.md#azure-wake-word)). Silero VAD finds the end of your request.
- **Speech to text:** Voxtype, run with Jarvis's own settings file. Your own Voxtype configuration is never
  edited.
- **Reasoning:** one agent on a local Qwen3.6-35B-A3B model served by Strata. Plain commands such as "pause",
  "next song" or "volume 40" skip the model and answer in about 0.3 s.
- **Voice:** Kokoro by default (`kokoro:bm_george`); a Piper voice can be set instead. Jarvis replies by voice
  only, with no desktop notifications.
- **Spotify:** play a track, album, artist, playlist or your liked songs; pause, resume, next, previous; set or
  change the volume; shuffle; repeat; queue songs; like the current song; say what is playing. When Spotify is
  closed, Jarvis says so and asks whether to open it, and where; it never opens a program unasked.
- **Your PC:** ask how the PC is doing, in any words, for its CPU, graphics card, VRAM, memory,
  temperatures and disks. Jarvis can list and read your files (your home folder by default, hidden ones
  only for the desktop's own settings) and create or change text files in `~/Documents`, asking first unless you turned questions
  off. Keys, tokens, shell history and browser profiles are never read ([`[files]`](docs/configuration.md)).
  Ask it in any words to close, end or kill a program: Jarvis lists the programs of yours that
  match, asks which one when there are several, and stops only that one once you press the confirm key. It
  never stops the desktop, the sound server, dictation or itself, and nothing heard from PC audio can stop a
  program ([Stopping a program](docs/configuration.md#stopping-a-program)).
  A shell command you say can run too, only if you turn it on (`[shell]`, off by default) and only after
  your confirm key, with time and memory limits and no sudo ([`[shell]`](docs/configuration.md)).
- **Omarchy questions:** Jarvis answers questions about Omarchy (settings, themes, key bindings, crashes,
  building an app) from the guides Omarchy ships in `/usr/share/omarchy/default/agents/skills/`. It reads
  them as reference text and never runs a command they contain.
- **Sums:** arithmetic, however you ask it (products, percentages, square roots, powers and the like), is
  worked out exactly by a calculator in code, never guessed by the model.
- **Boundaries:** every action is a typed operation checked by a policy engine in code before it runs, and
  every request, decision and result goes to an audit log. See [SECURITY.md](SECURITY.md).
- **Time and weather:** the time, or the weather now, later or in a named place, asked in any words.
  The time comes from this PC; the weather from Open-Meteo, free and without an account. Set your home in
  [`[weather]`](docs/configuration.md).
- **Research:** questions it can't answer on its own go to a web search through a local SearXNG; it reads
  the best pages and says where the answer came from. Asked what Reddit thinks, it finds Reddit threads and
  reads their top comments from the Arctic Shift archive. Start the search container once as
  [`[research]`](docs/configuration.md) says. What these send, and to whom, is in
  [docs/privacy.md](docs/privacy.md).
- **Listening to PC audio (off by default):** Jarvis can also take requests heard in what the PC plays, such as
  a Discord call. That speech is treated as someone else's, so you are asked to confirm before it changes
  anything. See [the ADR](docs/adr/0003-pc-audio-is-not-the-owner.md).
- **Talking:** Jarvis starts speaking as soon as the first sentence of its reply is written. Say "Hey Jarvis"
  while it talks and it stops and listens to your new request. See
  [the ADR](docs/adr/0008-speak-while-writing-and-barge-in.md).

Not built yet: PC operations other than Spotify and files. The plan marks each as planned.

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
microphone. The config file and every key are in [docs/configuration.md](docs/configuration.md);
[`config.example.toml`](config.example.toml) shows them all at their defaults. Your own config stays in
`~/.config/jarvis/`, outside the repository.

The units start with your desktop session and stop with it. Jarvis keeps listening at the lock screen
(`audio.listen_when_locked` turns that off), opens the microphone again after PipeWire restarts or the
microphone is unplugged and plugged back, and listens afresh after a suspend. What each change does is in
[docs/playbook/session-transitions.md](docs/playbook/session-transitions.md).

## Everyday commands

These are the subcommands `jarvis --help` lists. `jarvis` on its own shows what is running and follows the log.

| Command | What it does |
|---|---|
| `jarvis setup` | Write the config, the Strata profile and the user units (`--enable` also starts them). |
| `jarvis spotify-login` | Connect Jarvis to your Spotify account with PKCE (`--no-browser` only prints the link). |
| `jarvis start` / `stop` / `restart` | Start Jarvis and its model server, stop both (which frees the model), or restart the voice service after a config change. |
| `jarvis status` | Show what Jarvis is doing (idle, listening, thinking, speaking, paused, model loading or unavailable) and the model's state; `--json` prints one JSON object for a status bar. |
| `jarvis pause` / `resume` | Let go of the microphone and free the model, until `resume` opens the microphone and loads the model again (or the voice service restarts). |
| `jarvis hush` | Stop Jarvis speaking now. |
| `jarvis confirm` | Allow the one action Jarvis is waiting on with "Press the confirm key". Bind it to a key ([configuration](docs/configuration.md#the-confirm-key-jarvis-confirm)). |
| `jarvis unload` | Free the model's memory now, for example before a game. |
| `jarvis doctor` | Check every part without loading the model. |
| `jarvis logs` | Follow the live log (Ctrl+C to leave). |
| `jarvis help` | List the commands (the same as `jarvis --help`). |
| `jarvis ask <text>` | Send one typed request as if spoken; `--yes` answers yes to confirmations. |
| `jarvis chat` | Open a Textual window where you type requests and watch what Jarvis does, live. |
| `jarvis wake [livekit\|azure]` | Show or switch the wake word model; `--model FILE` names the Azure `.table` file; `--words hey_jarvis,yo_jarvis` picks the LiveKit models that listen. |
| `jarvis pc-audio [on\|off]` | Show or set listening to PC audio; `--apps discord,firefox` names the apps, `--all-output` listens to everything the PC plays. |
| `jarvis run` | Run the voice loop in the foreground. This is what `jarvis.service` runs. |
| `jarvis --version` | Print the version. |

## Docs

- [docs/README.md](docs/README.md): the map of every document, and how far to trust each kind
- [docs/configuration.md](docs/configuration.md): every config key, its range and what needs a restart
- [docs/models.md](docs/models.md): the model, speech to text, voices, the wake word, latency and licences
- [docs/architecture.md](docs/architecture.md): the modules, the voice loop and one request end to end
- [docs/plan.md](docs/plan.md): architecture, design and phases, with what is built
- [docs/roadmap.md](docs/roadmap.md): what is planned, and what Jarvis will not do
- [docs/privacy.md](docs/privacy.md): what Jarvis keeps on disk and what leaves the PC
- [docs/adr/](docs/adr/README.md): decisions
- [docs/playbook/](docs/playbook/): how to work on the project, and its traps
- [SECURITY.md](SECURITY.md): scope and how to report a vulnerability
- [ACCESSIBILITY.md](ACCESSIBILITY.md): voice, the chat window and keyboard-only use

## Status

Jarvis is pre-1.0 and has no tagged releases yet. It was built and measured on one machine (Omarchy, an RX
7900 XT and 64 GB of RAM); other hardware is untested. The configuration file has one schema version and no
migrations, so a later version may ask you to edit it. PC operations other than Spotify and files are
not built; [docs/roadmap.md](docs/roadmap.md) lists what is planned and
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
uv run pytest -q -m "not local"   # unit tests
```

Tests marked `local` need real hardware and services and never run in CI (`uv run pytest -m local`).
`tools/check.sh` runs everything CI runs, in one command; `tools/install-hooks.sh` adds an opt-in pre-commit check.

## Project layout

```text
src/jarvis/          the package: voice loop, agent, policy, operations, CLI (map: docs/architecture.md)
  spotify/           the Spotify login, Web API client and operations
  prompts/           the system prompt
tests/               pytest; tests/conftest.py keeps every test off the network, audio and your home folder
tools/               the checks CI runs (tools/README.md lists them)
docs/                configuration, models, architecture, plan, decisions (map: docs/README.md)
.github/             CI, issue forms, the pull request template, labels
.claude/ .codex/    agents and recipes for coding agents (see below);
.agents/            the recipes again, for Codex and other harnesses
```

## Working with AI agents

The repository is set up for coding agents. [AGENTS.md](AGENTS.md) (loaded by Claude Code through `CLAUDE.md`)
holds the product rules, commands, workflow and boundaries.

- **Agents** (`.claude/agents/`, mirrored for Codex in `.codex/agents/`): `jarvis-reviewer` (read-only review of a
  finished change) and `jarvis-slop-check` (the words of a pull request, docs and spoken replies).
- **Skills** (`.claude/skills/`, mirrored in `.agents/skills/`): `add-operation` (a new thing Jarvis can do),
  `add-dependency-or-voice` (licence and pin checks) and `run-silent` (running Jarvis without sound or windows).
- **Tools** (`tools/`): `tools/check.sh` is the one command an agent runs before calling work done.

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
