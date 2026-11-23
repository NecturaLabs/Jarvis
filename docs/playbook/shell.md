# Checking the shell operation's limits

`pc_shell` (`src/jarvis/pc/shell.py`, `[shell]` in [configuration.md](../configuration.md)) depends on the
user's systemd manager and on `setpriv` for its limits. The unit tests use a fake runner, so these checks
run the real one.

## On the owner's own user (`local` tests)

```sh
PIPEWIRE_REMOTE=/nonexistent PULSE_SERVER=unix:/nonexistent uv run pytest tests/test_pc_shell.py -m local -p no:randomly
```

They run harmless commands (`env`, `sleep`, a Python allocation) and check that:

- the environment holds no variable of Jarvis's, the no-new-privileges flag is set, and `sudo -n true` fails;
- a command past `timeout_s` is stopped and its scope, with the background `sleep`s it started, is gone;
- a command that returns at once is ended with what it left running in the background;
- a 512 MiB allocation under a 64 MiB `memory_max_mib` ends by signal 9, while a 16 MiB one runs.

Seen on 2026-11-23: all four passed (systemd's user manager, `setpriv` from util-linux).

## On a disposable user (owner-run)

This shows that cleanup and the limits hold for a user with nothing of the owner's. It needs `sudo` to
create and remove the user, so the owner runs it:

```sh
sudo useradd --create-home jarvis-shell-check
sudo loginctl enable-linger jarvis-shell-check      # gives it a user manager
sudo -iu jarvis-shell-check bash -c 'cd /path/to/Jarvis && uv run pytest tests/test_pc_shell.py -m local -p no:randomly'
sudo loginctl disable-linger jarvis-shell-check
sudo userdel --remove jarvis-shell-check
```

Record the date and the result here. Not yet run.
