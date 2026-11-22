# Session transitions

What Jarvis does when the desktop session changes under it, what was seen on the maintainer's PC, and
how to check each one. The code paths have fake-environment tests (`tests/test_voice.py`,
`tests/test_control.py`, `tests/test_audio.py`, `tests/test_model.py`); `tests/test_transitions_local.py`
checks the real pieces those fakes stand in for (`uv run pytest -m local tests/test_transitions_local.py`,
with the owner: one test opens the microphone for two seconds).

Read the outcome of each step in the voice log: `jarvis logs`, or
`journalctl --user -u jarvis.service -u jarvis-strata.service -f`.

## Desktop session start and stop

Both units are `PartOf=` and `WantedBy=graphical-session.target`: they start when you log in and stop
when the session ends. Stopping Strata frees the model's RAM and VRAM.

- Seen on 2026-11-21: the units stopped at 15:38 when the PC was shut down; it booted at 15:40, and the
  units started at 15:42, once the session started after login.

## Lock screen

The Omarchy shell draws the lock screen itself (`WlSessionLock`) and does not tell logind, so
`loginctl show-session` keeps `LockedHint=no` while locked. Jarvis is a user service and nothing in it
stops at a lock: by default it keeps listening, and anyone at the locked PC can talk to it.

With `audio.listen_when_locked = false`, Jarvis asks the shell (`omarchy-shell lock isLocked`, a few
milliseconds) before it takes a request: after the wake word, before a follow-up window, for a request
spoken over a reply, and for a PC-audio request. While locked it takes none; the wake word still stops its
speech. The log says `the wake word was ignored: the screen is locked`. An answer that cannot be read
counts as locked.

- Check: `uv run pytest -m local tests/test_transitions_local.py -k unlocked` at an unlocked screen
  (asks the shell; locks nothing). Seen passing on 2026-11-22.
- With the owner: set the key to `false`, `jarvis restart`, lock, say the wake word, and look for the log
  line; unlock and it answers again. Not run yet.

## PipeWire restart, microphone unplugged

`pw-record` ends when PipeWire restarts. Not verified on this PC: whether it ends or carries on silent
when its microphone is unplugged (WirePlumber may move it to another source). Jarvis handles both.

- Seen on 2026-11-21 at 15:13 and 15:15: PipeWire restarted, `pw-record` ended, and `jarvis run` exited
  with `AudioError: the microphone stream ended`; systemd started it again 5 s later.

Now the loop opens the microphone again: a capture that ended, or that sent nothing for 5 s
(`MIC_STALL_S`), is restarted after 1 s, then twice as long after each try that brings no frame, up to 30 s.
The log says `the microphone stopped (...); opening it again`. Meanwhile `jarvis status` says
`unavailable`; a pause still wins and shows `paused`. A request the lost microphone cuts short ends without
an apology. The stall time counts only while the loop waits for a frame, not while it answers.

- With the owner: `systemctl --user restart pipewire` while Jarvis runs, then say the wake word. Unplug
  the microphone, wait 10 s, plug it back, say the wake word. Not run yet.
- A microphone set by serial (`audio.source`) gets a new serial when it is plugged back in, so the
  reopened capture falls back to PipeWire's default. Set it by node name to keep it.

## Suspend and resume

Linux's monotonic clock stands still while the PC sleeps and its boot clock does not. When the gap
between them grows by 2 s or more between two frames, the loop takes it as a suspend: it drops the
frames from before it, resets the wake word, and asks the model server for its state: a model whose load
failed is loaded again (not while paused); an unloaded one stays unloaded, since `jarvis unload` before a
game or the idle unload put it there. The log says `back from sleep after N s: listening afresh`, then
`back from sleep: the model is <state>`. A stall check cannot fire during the sleep itself, since it
times with the monotonic clock.

- Not seen yet: no kept boot of this PC has suspended (`journalctl -k | grep 'PM: suspend'`).
- With the owner: suspend, wake, say the wake word; look for the log line and check
  `jarvis status --json` shows the model `loaded`.

## Model drive missing

`/mnt/models` is mounted `nofail`, so the desktop starts without it. `jarvis-strata.service` has
`ConditionPathExists=` on the model file, so systemd skips Strata. Nothing is left to wait for.

Jarvis then answers a request at once with "My model's file is missing, so I can't answer. Is its drive
mounted?" instead of waiting 30 s for the server. `jarvis status` shows the model `missing`, and
`jarvis doctor` fails the model line with "mount its drive, or set --model-file". Once the drive is
back, `jarvis start` starts Strata.

- Check: `uv run pytest -m local tests/test_transitions_local.py -k strata` (one status GET). Seen
  passing on 2026-11-22, with the drive mounted.
- Not run: booting without the drive.
