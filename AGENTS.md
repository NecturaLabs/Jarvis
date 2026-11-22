# Jarvis

A local, always-on voice agent for this Omarchy PC: a "Hey Jarvis" wake word, Omarchy's local dictation
(Voxtype) for speech-to-text, a local model served by Strata for reasoning, typed tools within user-set
boundaries, and Kokoro for speech (Piper is the alternative voice). Python 3.14 (system interpreter, managed by
uv), packaged as `jarvis`.
The build plan and its decisions are in `docs/plan.md`; read it before changing architecture.

## Commands

- Install the environment: `uv sync`
- Everything CI runs, in one command, before calling work done: `tools/check.sh` (lint, tests without `local`,
  `reuse lint`, the licence, docs and advisories checks). It lints **tracked** files only, so `git add` new files first.
- Lint only (Ruff with every rule, `ruff format --check`, pyright strict, shell and editorconfig):
  `uv run .lint/check.sh`
- Tests: `uv run pytest -q -m "not local"`; `tests/conftest.py` cuts audio, network and your home folder for each test.
- Single test: `uv run pytest tests/test_cli.py::test_version_prints_package_version -q`
- Opt-in git hooks (lint and tests before a commit, the subject format): `tools/install-hooks.sh`
- CLI: `uv run jarvis --version`
- Tests marked `local` need this PC's hardware or services and never run in CI: `uv run pytest -m local`.

## Product rules (from the owner; they shape the code)

- Jarvis's runtime loads no global agent instructions, skills, plugins or MCP servers. The only outside
  knowledge it may read is the Omarchy OS skills in `/usr/share/omarchy/default/agents/skills/`.
- One agent does all the work. Never add a subagent, delegation or second-agent tool to Jarvis.
- Boundaries are enforced in code before any tool runs, never only in the prompt: typed operations,
  default deny, provenance on every message part, confirmation bound to one exact action, an audit log.
- Voxtype is driven only through these commands: `voxtype transcribe` (with `-q`, `-c <Jarvis's own
  file>`, `--model`, `--engine`, `--initial-prompt`), `voxtype status --format json` (also with
  `--follow --extended`, one long-lived reader instead of polling), and, in
  `jarvis doctor`, `voxtype info models --json --engine …` (owner's decision, 2026-11-22, #34). The
  daemon's `record start --file …` and `record stop --wait --json` are its contract if that path is ever
  built. A new Voxtype command needs the owner's yes. Never edit `~/.config/voxtype/` and never stop a
  recording Jarvis did not start.
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
- Downloaded tools, voices and model files go where the maintainer's own environment rules say, never
  under `~/.local/share`; Strata's model stays where `strata.model_file` points.

## Workflow

- GitHub Issues is the only backlog; labels and issue rules are in `CONTRIBUTING.md`. Link the issue (`Closes #N`).
- A behavioural change gets the `jarvis-reviewer` agent before it merges, and every blocking finding is answered;
  pull request text and changed docs get `jarvis-slop-check`.
- Docs and `CHANGELOG.md` change in the same commit as the behaviour; `docs/README.md` says which docs are kept true.
- Recipes: the `add-operation` skill for anything Jarvis can do, `add-dependency-or-voice` before adding a package,
  model or voice, `run-silent` before starting Jarvis. Skills live in `.claude/skills/` with a byte-identical copy in
  `.agents/skills/`, and agents in `.claude/agents/` with a Codex copy in `.codex/agents/`; change both together.

## Git

- `feature/`, `bugfix/`, `docs/`, `chore/` … branches merge into `dev` (the working branch). `main` is the
  release branch: merge and push to it only when the owner asks. A pull request only when the owner asks for one.
- Conventional Commits; push every commit on a non-release branch at once. Commits are signed (the repository's
  local git config holds the SSH signing key).
- `dev` and `main` take only a commit that already passed CI (rulesets "CI for dev" and "CI for main"). Make the
  merge commit locally, push it to the topic branch first (`git push origin HEAD:<topic>`), wait for CI there,
  then push the same commit to `dev`.

## Open source

Jarvis is public under GPL-3.0-or-later (it imports `piper-tts`, GPL-3.0-or-later). Every commit is published:
no personal data, tokens, recordings or internal hostnames; owner- and machine-specific values live in config, not
code; commit under the public identity the repository is configured with. Keep everything generic: describe what
Jarvis does, asked in any words, never the owner's own phrases, titles, apps, drives or habits;
`tools/check_private.py` (in `check.sh`, CI and the hooks) enforces it, with the owner's terms in
`.git/info/private-terms`. Install the hooks (`tools/install-hooks.sh`) before pushing. Check the licence of every new
dependency, model and voice against GPL-3.0-or-later before adding it, and keep `uv run reuse lint` clean. Outside
contributions follow `CONTRIBUTING.md` (DCO sign-off).

## Boundaries

- **Never** commit secrets, tokens or the Spotify refresh token; tokens live in `~/.local/state/jarvis/`
  (mode 0600). Never commit audio recordings of the owner except fixtures they approved.
- **Never** edit `.lint/` (managed by agentic-setup; update with `lint.py sync`); project lint changes go in
  `ruff.toml` and `pyrightconfig.json`.
- **Security-sensitive**: the policy engine, confirmation flow, the Spotify OAuth flow, the PC-operation
  tools and the web tools; `.github/CODEOWNERS` lists their files. Changes there get a security pass and a review
  before merging into `dev`.
- **Never** weaken `tests/test_product_rules.py`, `tests/conftest.py` or the `banned-api` table in `ruff.toml`
  to get green: they enforce the product rules above.
