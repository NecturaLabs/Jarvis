---
name: run-silent
description: How an agent starts Jarvis, a voice or any audio test without making a sound or opening a window on the maintainer's PC - the environment that cuts audio, hidden windows, recorded PIDs, and which commands touch the owner's services. Use before running `jarvis run`, `jarvis ask`, `jarvis chat`, a voice, or a test marked `local`.
---

# Run silently

On the maintainer's PC everything an agent runs is silent and out of sight (`AGENTS.md`, "Working on the
maintainer's PC").

- Prefix any process that could reach audio with `PIPEWIRE_REMOTE=/nonexistent PULSE_SERVER=unix:/nonexistent`.
  The test suite does this for every test through `tests/conftest.py`; a script or a manual run must do it itself.
- Safe anywhere: `uv run jarvis --help`, `uv run jarvis doctor`, `uv run jarvis status`, the test suite, `tools/check.sh`.
- Touch the owner's services, Spotify account or screen: `jarvis start`, `stop`, `restart`, `run`, `ask`, `unload`,
  `spotify-login`, `chat`, `setup --enable`. Agents never run these against the owner's live setup; the owner does.
  To try the voice loop, run `jarvis run` with a temporary `XDG_CONFIG_HOME`/`XDG_STATE_HOME` and the audio cut.
- A Textual window (`jarvis chat`) runs headless in tests through Textual's `run_test`; never open it on screen.
- Record the PID of anything you start in the background and stop only that PID. Never stop a process by name.
- Tests marked `local` use the real microphone, speakers, Strata or Spotify: they run only with the owner
  (`uv run pytest -m local`). `tests/test_idle_local.py` only reads `/proc` and is the exception an agent may run.
