# Playbook

How to work on Jarvis from a cold start. Commands are in `AGENTS.md`; this file holds the procedures and the
traps found along the way. Add a trap the moment it costs time.

## Build and check

1. `uv sync` (uses the system Python 3.14; `uv.lock` pins every package with hashes).
2. `git add` new files, then `uv run .lint/check.sh` — the runner checks tracked files only.
3. `uv run pytest -q`; `uv run pytest -m local` only on this PC, with the owner when audio is involved.

## Launch and capture

- `uv run jarvis doctor` and `--help` are safe to run anywhere. `jarvis run`, `ask`, `start`, `stop`, `restart`
  and `spotify-login` touch the owner's services, network or Spotify account: agents start the voice loop silent
  (`PIPEWIRE_REMOTE=/nonexistent PULSE_SERVER=unix:/nonexistent`) and hidden, and only the owner runs it with
  real audio.

## Review and hand-off

- Self-review every diff against `docs/plan.md` and the issue before committing; security-sensitive areas
  (`AGENTS.md`, Boundaries) also get the security-audit skill.
- Work goes on a `feature/…` branch, merged into `dev`; `main` is released only when the owner asks.
- Before ending a session: commit and push the branch, update the issue with evidence, and leave the next step
  in the issue.

## Traps

- `.lint/check.sh` on an empty index passes without checking anything. Stage first.
- Ruff's base config ignores `PT009`/`PT027` for unittest scripts; `ruff.toml` re-selects them for pytest.
- pyright needs `venvPath`/`venv` in `pyrightconfig.json` to see the uv environment.
