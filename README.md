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
- **Reasoning:** one agent on a local model served by Strata, whichever Strata's run config names
  (Qwen3.8-Flash-Next on the reference PC; [docs/models.md](docs/models.md) compares the models measured).
  Plain commands such as "pause", "next song" or "volume 40" skip the model and answer in about 0.3 s.
  The model answers without thinking first, and thinks harder, up to a ceiling you set, only for a request
  that needs it ([thinking](docs/configuration.md)).
- **Voice:** Kokoro by default (`kokoro:bm_george`); a Piper voice can be set instead. Jarvis replies by voice
  only, with no desktop notifications.
  Tell Jarvis to be quiet, in any words, and it stops without a word. Say the wake word while it speaks
  to stop it and ask something else.
- **Singing:** ask Jarvis to sing, in any words, and the model writes the song as notes, up to 30 seconds,
  that Jarvis sings in its own voice: with Kokoro's voice moved to each note, or with SoulX-Singer, a local
  singing-voice model on the GPU, once you install it ([Singing](docs/models.md#singing)).
- **Spotify:** play a track, album, artist, playlist or your liked songs; pause, resume, next, previous; jump
  within a song or restart it; move playback to another device; set or change the volume; shuffle; repeat; queue
  songs; like or unlike the current song, or add it to one of your playlists (after a yes); say what is playing,
  what is queued next and which devices there are. When Spotify is
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
  It can read, set, turn up or down and mute the PC's own volume, and play, pause or skip a video or
  another media player (`playerctl` needed), leaving Spotify to its own tools unless you mean everything.
- **Omarchy questions:** Jarvis answers questions about Omarchy (settings, themes, key bindings, crashes,
  building an app) from the guides Omarchy ships in `/usr/share/omarchy/default/agents/skills/`. It reads
  them as reference text and never runs a command they contain.
- **Sums:** arithmetic, however you ask it (products, percentages, square roots, powers and the like), is
  worked out exactly by a calculator in code, never guessed by the model.
- **Boundaries:** every action is a typed operation checked by a policy engine in code before it runs, and
  every request, decision and result goes to an audit log. See [SECURITY.md](SECURITY.md).
- **Reminders and scheduled requests:** "remind me in 20 minutes", every weekday at a time, or every so
  many minutes; or a request Jarvis carries out later, such as pausing the music at midnight. Set them by
  voice or in the chat; the voice service says them when due. See
  [the ADR](docs/adr/0011-reminders-and-scheduled-requests.md).
- **About itself:** ask which model it runs, its version, its voice or how it is set up; it answers from its
  config and Strata's run config, and reads out its own config files, secrets hidden. It never changes them.
- **Time and weather:** the time, or the weather now, at an hour, later or in a named place, asked in any words.
  The time comes from this PC, for a city, a state or province, or a whole country. A country with several
  time zones (the United States, Russia, Australia) gets the time on each of its clocks. The weather comes
  from Open-Meteo, free and without an account. Set your home in [`[weather]`](docs/configuration.md).
- **Research:** questions it can't answer on its own go to a web search through a local SearXNG; it reads
  the best pages and says where the answer came from. Asked what Reddit thinks, it finds Reddit threads and
  reads their top comments from the Arctic Shift archive. Start the search container once as
  [`[research]`](docs/configuration.md) says. What these send, and to whom, is in
  [docs/privacy.md](docs/privacy.md).
- **Browser:** through [agent-browser](https://github.com/vercel-labs/agent-browser), which the installer requires, the chat opens
  pages that need a real browser (built by scripts, or used through a search box), reads them and shows you a
  picture of one. It runs headless with its own throwaway profile, reaches only public addresses, and asks
  before it clicks or types anything ([`[browser]`](docs/configuration.md#browser-the-headless-browser)).
- **Listening to PC audio (off by default):** Jarvis can also take requests heard in what the PC plays, such as
  a Discord call. That speech is treated as someone else's, so you are asked to confirm before it changes
  anything. See [the ADR](docs/adr/0003-pc-audio-is-not-the-owner.md). Ask it what is being said there and it
  listens for up to a minute and tells you; with `audio.pc_audio_recall_s` set it can also tell you what was
  just said. The audio is kept in memory and on disk only for the moment Voxtype transcribes it
  ([ADR 0015](docs/adr/0015-listen-to-pc-audio.md)).
- **Talking:** Jarvis starts speaking as soon as the first sentence of its reply is written. Say "Hey Jarvis"
  while it talks and it stops and listens to your new request. See
  [the ADR](docs/adr/0008-speak-while-writing-and-barge-in.md).

Not built yet: switching workspaces and running Omarchy's commands. The plan marks each
as planned.

## Install

Use an Omarchy desktop session on Arch Linux with PipeWire, WirePlumber, Voxtype, Git, uv,
system Python 3.14 and [agent-browser](https://github.com/vercel-labs/agent-browser) (`yay -S agent-browser`). Prepare the two models the installer leaves to their own tools:

1. Run `voxtype setup model` and download Whisper `small.en`, the default speech model.
2. Install [Strata](https://github.com/niko1221/strata), run `strata launch`, and choose a model your
   GPU, RAM and disk can handle. When Strata writes `strata-<model>.json`, stop the server it started
   with `strata launch --stop`. Jarvis starts its own. See [model sizes and memory](docs/models.md).

```sh
curl -fsSL https://raw.githubusercontent.com/NecturaLabs/Jarvis/main/install.sh | bash
```

The installer checks the required commands and prints how to install any that are missing. It clones
the `main` release branch, syncs the locked Python environment, and runs `jarvis setup`,
`uv run tools/get_models.py`, has agent-browser download its Chrome when it has none (`agent-browser
install`, into your home folder), and checks the installation with `jarvis doctor`. The tool downloads and
checks the SHA-256 of the default wake word, Silero VAD and Kokoro files (about 347 MiB combined).

The default checkout is `${XDG_STATE_HOME:-$HOME/.local/state}/jarvis/checkout`. Set
`JARVIS_INSTALL_DIR` or pass `--install-dir PATH` to choose another disk or use a clean existing checkout
of the same repository and branch. New installs put Jarvis's model assets beside it in
`${XDG_STATE_HOME:-$HOME/.local/state}/jarvis/models`. Use `--data-dir PATH` to change that location.
A re-run keeps saved settings unless you pass new ones, checks for a release update, and keeps model
files already present.

Jarvis keeps the model configured on a previous run. On first install, it uses the sole
`strata-*.json` it finds in Strata's usual `install/` folder. If there are several, or Strata is installed
elsewhere, use `--run-config PATH` or `JARVIS_RUN_CONFIG`. A prompt uses the terminal when one is
available. Pass options through `bash -s --` when piping. The same options work with `bash install.sh`
from a checkout.

```sh
curl -fsSL https://raw.githubusercontent.com/NecturaLabs/Jarvis/main/install.sh |
  bash -s -- --non-interactive --run-config /path/to/strata-model.json --enable
```

Use `--dry-run` to print the commands or `--help` for the full list of options and environment
variables. If Strata's model is set to use all available VRAM (`--expert-cache auto`), pass
`--strata-vram-reserve-mib MIB` to leave room for the desktop and other programs.
[Configuration](docs/configuration.md#strata-how-the-jarvis-strata-service-runs-the-model) explains how to choose it.

Pass `--enable` when you want both services to start now and with future desktop sessions. If you
skipped that flag, use the checkout path printed by the installer and run:

```sh
cd "${XDG_STATE_HOME:-$HOME/.local/state}/jarvis/checkout"
.venv/bin/jarvis setup --enable
.venv/bin/jarvis doctor --wait
```

Say “Hey Jarvis” and ask a question. Use `.venv/bin/jarvis status` to see whether it is listening
and `.venv/bin/jarvis logs` if it does not respond. The first model load may take a while.
[Configuration](docs/configuration.md#keeping-the-model-in-ram) covers the optional memory lock and
[session transitions](docs/playbook/session-transitions.md) covers sleep and microphone changes.

Spotify is optional. To connect it, use a Spotify Premium account and your own developer app.
Run `.venv/bin/jarvis setup --spotify-client-id ID`, then `.venv/bin/jarvis spotify-login` and follow
the printed redirect URI instructions. Run `.venv/bin/jarvis restart` to pick it up in the running
voice service. Other media players use `playerctl` (`sudo pacman -S playerctl` if wanted).

### Update

Run `.venv/bin/jarvis update` in the checkout for a preview and `.venv/bin/jarvis update --apply` to
install the update. The updater refuses local edits and divergent branches, defers while Jarvis is
handling a request, and restores the previous checkout if its checks or service restart fail.
Automatic daily updates are off by default. Turn them on or off with
`.venv/bin/jarvis update --auto on|off`. See [update settings](docs/configuration.md#update).

## Everyday commands

These are the subcommands `jarvis --help` lists. `jarvis` on its own shows what is running and follows the log.

| Command | What it does |
|---|---|
| `jarvis setup` | Write the config, the Strata profile and the user units (`--enable` also starts them). |
| `jarvis update` | Preview an update, install it with `--apply`, or manage the daily timer with `--auto on\|off`. |
| `jarvis spotify-login` | Connect Jarvis to your Spotify account with PKCE (`--no-browser` prints the link to open yourself instead of opening a browser, then waits for the login). |
| `jarvis start` / `stop` / `restart` | Start Jarvis and its model server, stop both (which frees the model), or restart Jarvis after a config change: the voice service restarts, a stopped model server starts, and a running one restarts only when its settings changed. |
| `jarvis status` | Show what Jarvis is doing (idle, listening, thinking, speaking, paused, model loading or unavailable) and the model's state; `--json` prints one JSON object for a status bar. |
| `jarvis pause` / `resume` | Let go of the microphone and free the model, until `resume` opens the microphone and loads the model again (or the voice service restarts). |
| `jarvis hush` | Stop Jarvis speaking now. |
| `jarvis confirm` | Allow the one action Jarvis is waiting on with "Press the confirm key". Bind it to a key ([configuration](docs/configuration.md#the-confirm-key-jarvis-confirm)). |
| `jarvis unload` | Free the model's memory now, for example before a game. |
| `jarvis model` | Show the mode (normal or coding), each mode's model and whether it is loaded; `jarvis model coding` and `jarvis model normal` switch ([configuration](docs/configuration.md#coding-coding-mode)). |
| `jarvis doctor` | Check every part without loading the model: one line per check (`ok`, `FAIL` with its fix, or `note`), then how many failed. |
| `jarvis logs` | Follow the live log (Ctrl+C to leave). |
| `jarvis help` | List the commands by group, each with what it does (the same as `jarvis --help`); `jarvis <command> --help` explains one command and its options, and a mistyped command gets a suggestion when one is close. |
| `jarvis ask <text>` | Send one typed request as if spoken; a question is answered `y` or `n` on the terminal, and `--yes` answers yes to confirmations, except a browser click or typing, which it still asks on the terminal (and refuses with no terminal to answer on). |
| `jarvis chat` | Open a Textual window where you type requests and watch what Jarvis does, live. |
| `jarvis consent [ask\|bypass]` | Show or set whether Jarvis asks for a yes before a risky action or one after outside text; a `strong` action still waits for the confirm key either way. |
| `jarvis wake [livekit\|azure]` | Show or switch the wake word model; `--model FILE` names the Azure `.table` file; `--words hey_jarvis,yo_jarvis` picks the LiveKit models that listen. |
| `jarvis pc-audio [on\|off]` | Show or set listening to PC audio; `--apps discord,firefox` names the apps, `--all-output` listens to everything the PC plays. |
| `jarvis run` | Run the voice loop in the foreground. This is what `jarvis.service` runs. |
| `jarvis speech-test record\|score\|compare` | Record a private set of your own requests, score how well speech-to-text hears them (word errors, names, time), and compare two scores for a difference beyond chance. |
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
migrations, so a later version may ask you to edit it. Some PC operations (workspaces, Omarchy's commands)
are not built; [docs/roadmap.md](docs/roadmap.md) lists what is planned and
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
`tools/check.sh` runs everything CI runs, in one command; `tools/install-hooks.sh` adds opt-in git hooks (lint,
tests, the commit subject and a private-details check).

## Project layout

```text
src/jarvis/          the package: voice loop, agent, policy, operations, CLI (map: docs/architecture.md)
  spotify/           the Spotify login, Web API client and operations
  prompts/           the system prompt
tests/               pytest; tests/conftest.py keeps every test off the network, audio and your home folder
tools/               the checks CI runs (tools/README.md lists them)
docs/                configuration, models, architecture, plan, decisions (map: docs/README.md)
.github/             CI, issue forms, the pull request template, labels
.claude/ .codex/    agents and recipes for coding agents (see below)
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
