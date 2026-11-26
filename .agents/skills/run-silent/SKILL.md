---
name: run-silent
description: How an agent starts Jarvis, a voice or any audio test without making a sound or opening a window on the maintainer's PC - the environment that cuts audio, hidden windows, recorded PIDs, and which commands touch the owner's services. Use before running `jarvis run`, `jarvis ask`, `jarvis chat`, a voice, or a test marked `local`.
---

# Run silently

On the maintainer's PC everything an agent runs is silent and out of sight (`AGENTS.md`, "Working on the
maintainer's PC").

- Prefix any process that could reach audio with `PIPEWIRE_REMOTE=/nonexistent PULSE_SERVER=unix:/nonexistent`.
  The test suite does this for every test through `tests/conftest.py`; a script or a manual run must do it itself.
- Safe anywhere: `uv run jarvis --help`, `uv run jarvis doctor`, `uv run jarvis status`, `consent`, `wake` and
  `pc-audio` with no argument (they only show the setting), the test suite, `tools/check.sh`. `jarvis logs` is safe
  but never ends; read the journal with `journalctl --user -u jarvis -n 100 --no-pager` instead.
- Touch the owner's config, services, microphone, Spotify account or screen:
  - `jarvis setup` rewrites the config, the Strata profile, the Strata API key and the user units, and reloads
    systemd, even without `--enable`;
  - `plugins install`, `uninstall` and `update` rewrite the config or download from GitHub into the data folder
    (`plugins` with no argument only reads it);
  - `consent`, `wake` and `pc-audio` with any argument rewrite the config and restart a running voice service;
  - `start`, `stop`, `restart` and `unload` act on the services and the loaded model; `pause`, `resume`, `hush` and
    `confirm` act on the running Jarvis (`confirm` allows the action it is waiting on);
  - `run`, `ask`, `chat` and `speech-test` run requests or record the microphone; `spotify-login` opens a browser.

  Agents never run these against the owner's live setup; the owner does. To try one, point `XDG_CONFIG_HOME`,
  `XDG_STATE_HOME` and `XDG_DATA_HOME` at a temporary folder and cut the audio; `setup`, `consent`, `wake` and
  `pc-audio` still call `systemctl --user`, so do not run them even then.
- A Textual window (`jarvis chat`) runs headless in tests through Textual's `run_test`; never open it on screen.
- Record the PID of anything you start in the background and stop only that PID. Never stop a process by name.
- Tests marked `local` use this PC's real services or hardware: they run only with the owner
  (`uv run pytest -m local`).
