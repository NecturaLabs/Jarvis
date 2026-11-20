# Jarvis

A local, always-on voice agent for Omarchy: say "Hey Jarvis", and it transcribes you with Omarchy's local
dictation (Voxtype), reasons with a local model served by Strata, acts within the boundaries you set, and
answers by voice. Its first core feature is Spotify control.

Status: phase 0 (project setup). The build plan is in [docs/plan.md](docs/plan.md).

## Requirements

Omarchy 4.5 on Arch Linux with PipeWire, Voxtype as the default dictation backend, Python 3.14, uv, and a
Strata-supported GPU (built on an RX 7900 XT).

## Develop

```sh
uv sync                      # install the locked environment
uv run .lint/check.sh        # lint, format and type checks (tracked files only)
uv run pytest -q             # unit tests
uv run jarvis --version
```

Install, setup and update commands (`./install.sh`, `jarvis setup`, `jarvis doctor`, `jarvis update`) arrive
with the phases that build them; see the plan.

## Docs

- [docs/plan.md](docs/plan.md) — architecture, design and phases
- [docs/adr/](docs/adr/) — decisions
- [docs/playbook/](docs/playbook/) — how to work on the project, and its traps

## Licence

Copyright NecturaLabs. All rights reserved.
