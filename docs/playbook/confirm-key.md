# The confirm key: checks with the owner

`jarvis confirm` allows one `strong` action (`docs/configuration.md`, "The confirm key"). The fake-device
tests cover it (`tests/test_confirmkey.py`, and the confirm-key tests in `tests/test_voice.py`,
`tests/test_agent.py`, `tests/test_control.py` and `tests/test_cli.py`). What they cannot show is the key
press reaching the running voice service from Hyprland, so that check is the owner's.

## Owner run (not run yet)

No shipped operation is `strong`, so make one `strong` for the test and put it back afterwards.

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

## Traps

- A press while Jarvis is still saying "Press the confirm key" comes too soon: it holds the key off for
  30 s. Wait for the question to end.
- The socket refuses a press it cannot trace (a `/proc` it cannot read, for example under `hidepid=2`)
  as "started by Jarvis": it fails closed.
