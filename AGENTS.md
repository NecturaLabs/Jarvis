# Jarvis

A local, always-on voice agent for this Omarchy PC: a "Hey Jarvis" wake word, Omarchy's local dictation
(Voxtype) for speech-to-text, a local model served by Strata for reasoning, typed tools within user-set
boundaries, and Kokoro for speech (Piper is the alternative voice). Python 3.14 (system interpreter, managed by
uv), packaged as `jarvis`.
The build plan and its decisions are in `docs/plan.md`; read it before changing architecture.

## Commands

- Install the environment: `uv sync`
- All checks (Ruff with every rule, `ruff format --check`, pyright strict, shell and editorconfig checks):
  `uv run .lint/check.sh`. It lints **tracked** files only, so `git add` new files first.
- Tests: `uv run pytest -q`
- Single test: `uv run pytest tests/test_cli.py::test_version_prints_package_version -q`
- CLI: `uv run jarvis --version`
- Tests marked `local` need this PC's hardware or services and never run in CI: `uv run pytest -m local`.

## Product rules (from the owner; they shape the code)

- Jarvis's runtime loads no global agent instructions, skills, plugins or MCP servers. The only outside
  knowledge it may read is the Omarchy OS skills in `/usr/share/omarchy/default/agents/skills/`.
- One agent does all the work. Never add a subagent, delegation or second-agent tool to Jarvis.
- Boundaries are enforced in code before any tool runs, never only in the prompt: typed operations,
  default deny, provenance on every message part, confirmation bound to one exact action, an audit log.
- Voxtype is driven only through its integration contract (`voxtype record start --file …`,
  `voxtype record stop --wait --json`, `voxtype transcribe`). Never edit `~/.config/voxtype/` and never
  stop a recording Jarvis did not start.
- Strata serves only Jarvis. Its model lives outside the repository (`strata.model_file`). Health
  checks never generate tokens. The model is loaded when `jarvis run` starts and stays loaded (the owner's
  choice): it may hold RAM and VRAM, but must use no GPU or CPU while idle, and Jarvis runs beside games.
- Jarvis runs locally and is free to use and maintain: no hosted model backend, no paid API. Voice activation
  and speech-to-text stay local too.
- Jarvis replies by voice only; no desktop notifications.

## Working on the maintainer's PC

These hold on the maintainer's machine; elsewhere, use your own folders and keep audio silent the same way.

- Everything an agent runs is silent: start Jarvis, a voice or any audio test with
  `PIPEWIRE_REMOTE=/nonexistent PULSE_SERVER=unix:/nonexistent` (or the process's own mute). Tests that
  need the real mic or speakers are `local` and are run with the owner.
- Open nothing on the owner's screen: run windows and terminals hidden; only if that is impossible, on
  workspace 1. Stop only processes you started, by their recorded PID.
- Downloaded tools, voices and model files go on the development drive, never under `~/.local/share`;
  Strata's model is the owner's exception on `/mnt/models`.

## Git

- `feature/`, `bugfix/`, `docs/`, `chore/` … branches merge into `dev` (the working branch). `main` is the
  release branch: merge and push to it only when the owner asks. A pull request only when the owner asks for one.
- Conventional Commits; push every commit on a non-release branch at once.

## Open source

Jarvis is public under GPL-3.0-or-later (it imports `piper-tts`, GPL-3.0-or-later). Every commit is published:
no personal data, tokens, recordings or internal hostnames; owner- and machine-specific values live in config, not
code; commit under the public identity the repository is configured with. Check the licence of every new
dependency, model and voice against GPL-3.0-or-later before adding it, and keep `uv run reuse lint` clean. Outside
contributions follow `CONTRIBUTING.md` (DCO sign-off).

## Boundaries

- **Never** commit secrets, tokens or the Spotify refresh token; tokens live in `~/.local/state/jarvis/`
  (mode 0600). Never commit audio recordings of the owner except fixtures they approved.
- **Never** edit `.lint/` (managed by agentic-setup; update with `lint.py sync`); project lint changes go in
  `ruff.toml` and `pyrightconfig.json`.
- **Security-sensitive**: the policy engine, confirmation flow, the Spotify OAuth flow, the PC-operation
  tools and the web tools. Changes there get a security pass and a review before merging into `dev`.
