# The confirm key: checks with the owner

`jarvis confirm` allows one `strong` action (`docs/configuration.md`, "The confirm key"). The fake-device
tests cover it (`tests/test_confirmkey.py`, and the confirm-key tests in `tests/test_voice.py`,
`tests/test_agent.py`, `tests/test_control.py` and `tests/test_cli.py`). What they cannot show is the key
press reaching the running voice service from Hyprland, so that check is the owner's.

## Owner run (not run yet)

This run makes `spotify_pause` `strong` for the test and puts it back afterwards; the run after it uses
`stop_process`, the one shipped `strong` operation.

1. Bind the key in `~/.config/hypr/bindings.lua` and reload Hyprland:
   `o.bind("SUPER + CTRL + Y", "Jarvis confirm", "jarvis confirm")`.
2. In `config.toml`, under `[boundaries.classes]`, set `spotify_pause = "strong"`, then `jarvis restart`.
3. With music playing, say "Hey Jarvis, pause the music". Jarvis asks "Should I pause the music? Press the
   confirm key to allow it." Press the key: `jarvis confirm` prints what it confirmed, and the music
   pauses.
4. Ask again and do not press. After 30 s Jarvis says it left it, and the music plays on.
5. Ask again and say "Hey Jarvis" while it waits: that is a no.
6. Press the key with nothing waiting (`jarvis confirm` says "nothing is waiting for the confirm key"),
   then ask within 30 s: Jarvis says the key was pressed too early and leaves it.
7. Put `spotify_pause` back (delete the line) and `jarvis restart`.

Record the date and what was heard here.

## Owner run: stopping a program (not run yet)

The fake-`/proc` tests cover the matching, the protected list and every refusal
(`tests/test_pc_processes.py`); this checks the real desktop. Use programs you can lose.

1. Open one program you can lose (a calculator, an image viewer). Say "Hey Jarvis, close the calculator".
   Jarvis asks "Should I stop <program> (PID <n>)? Press the confirm key to allow it." Press the key: the
   window closes and Jarvis says it stopped it.
2. Open two windows of it as two programs, then ask again in other words ("end the calculator"). Jarvis
   reads both, with their PIDs, and asks which one; it stops only the one you name, after the key.
3. Say "Hey Jarvis, kill pipewire": Jarvis says it never stops that, and asks nothing.
4. Name a program that is not running: Jarvis says it can't find it.
5. Ask for one and do not press the key: after 30 s it is left running.

Record the date and what was heard here.

## Traps

- A press while Jarvis is still saying "Press the confirm key" comes too soon: it holds the key off for
  30 s. Wait for the question to end.
- The socket refuses a press it cannot trace (a `/proc` it cannot read, for example under `hidepid=2`)
  as "started by Jarvis": it fails closed.
