# ADR: Refuse to start or load the model when the memory available would be left below a margin

**Status:** proposed
**Date:** 2026-11-26

## Context and problem statement

Strata reads the model's experts into RAM when it loads (about 39 GB for the model on the reference
machine, locked in part for the GPU while `strata.keep_in_ram` is on). Nothing in Jarvis checked that the
load fits beside what already runs. With a game running and 13 to 20 GB of RAM available, the model server
was started, the kernel failed page allocations, and the game's process died. Jarvis runs beside games by
design (AGENTS.md), so a load that takes a running program's memory is a defect in Jarvis, not in the game.

Several paths start or load the model: `jarvis start`, `jarvis restart`, the voice service pulling the unit
in, a switch of modes that restarts the server, `jarvis update`, the load at `jarvis run`, and the wake word
or a request loading it again after an unload. Which of them are guarded, and how?

## Considered options

1. Check only in the commands (`start`, `restart`).
2. Check only in the systemd unit (`ExecStartPre=`).
3. Check in both, and in the client that asks the server to load.
4. Cap the unit's memory harder (`MemoryMax`) and let the kernel kill the server instead.

## Decision outcome

Chosen option 3.

- **The number.** The need is the size of the model file the profile serves (`--native`), which is what
  Strata's log calls "about 39 GB". This is an empirical estimate, not Strata's own figure: on the reference
  machine the loaded server's cgroup holds 35.6 GiB of anonymous memory against a 36.5 GiB file, and the
  second part of a split model (`--ple-gguf`) is not counted. It would change if Strata reported its need
  (`strata serve` or `strata launch` printing a planned RAM figure), if an engine's footprint stopped
  tracking the file (a long context, `strata.kv_resident`, the picture reader), or if the second part were
  shown to be read into RAM too. What is available is `MemAvailable` from `/proc/meminfo`: free memory plus
  the page cache the kernel can reclaim, as the kernel counts it. A load is refused when
  `available - need < strata.min_free_ram_mib`. A file size or meminfo that cannot be read blocks nothing.
- **A restart gives memory back.** A running server holds its loaded model, which `MemAvailable` leaves out.
  Where Jarvis stops the server first (`restart`, a mode switch, `doctor`), the anonymous memory of the
  unit's cgroup (`memory.stat`, `anon`) is added to what is available. The unit's own check runs after the
  stop, so it needs no credit.
- **The margin is 4 GiB** (`strata.min_free_ram_mib`). It covers what the file size does not: the context,
  the picture reader and pinned context RAM, the desktop and a game's growth during the load, and it keeps the
  load out of the kernel's emergency reserve, where page allocations fail. It is a setting because the right
  figure depends on the engine and the PC. The defect's case (13 to 20 GB available against a 36.5 GiB file)
  is refused at any margin.
- **The override is a config key**, `strata.allow_low_ram`, not a command-line flag: the unit's check reads the
  config when it runs, and a flag on `jarvis start` could not reach it. The owner sets it, starts, and turns it
  off again.
- **What counts as already loaded.** The client guard and `doctor` add what the server's cgroup holds, as the
  restart paths do, and the client asks nothing while a load of its own waits for its answer or the server
  lists the model as loaded: the model's memory is then already out of `MemAvailable`, and refusing would be
  wrong. A load under way in another process has taken part of its memory already, which the same credit
  adds back.
- **Where it is guarded.** `jarvis start` and `jarvis restart` (and the backend's) print the reason, change
  nothing and exit 1. The unit has `ExecStartPre=python -m jarvis.ramguard --config … --profile …`, which reads
  the config and the profile at run time (the profile is what the mode in force serves) and exits 75, so every
  other way of starting the unit is covered too. A switch of modes leaves the running server alone and says
  why. The client (`StrataModel.load`, `warm`, and a request that must load the model) refuses before
  asking the server, which covers the load at `jarvis run`, the wake word after an unload, and a request that
  would let the server load lazily, and one that finds the server down beside too little memory (its unit's
  check refused the start) answers with the reason and says to run `jarvis start`. `jarvis update` makes
  the check before it restarts the server: a refusal keeps the server running what it loaded, skips only that
  restart and prints the reason; the rest of the update goes ahead, and `jarvis restart` reloads when there is
  room. `jarvis doctor` shows the check; the update's health check only notes it. A refusal in the
  commands, the unit, the switch, the client and the update is audited as `ram_refused` with where, the
  need, what was available, the margin and the model file.
- **The unit's check.** It runs as `python -I -m jarvis.ramguard` (isolated: no current-directory or
  environment path; checked against the venv's editable install, which it still finds), and the unit has
  `RestartPreventExitStatus=75` so that a refused start is not retried into the start limit. systemd
  documents that setting for the main process; whether it also stops the restart after a failed
  `ExecStartPre=` was not exercised, and if it does not, the start limit bounds the retries.
- **A refused switch writes nothing.** The switch makes the check before it writes the server's files, so a
  refusal leaves their times alone and nothing later sees them as newer than the server.
- **Option 4 rejected:** `MemoryMax` ends a server that already took the memory, after the other program
  lacked it; and it is set above the model's size on purpose.

## Consequences

- With a game that holds more than about 19 GiB beside the 36.5 GiB model on a 64 GB PC, the model does not
  load until the game is closed or `allow_low_ram` is set. That is the intended trade.
- A refused unit start is retried by `Restart=on-failure` within the start limit, then stays `failed`.
  `jarvis start` clears it. This was not run against systemd with a refusing check (nothing here may start
  the owner's services).
- `MemAvailable` is an estimate, and the file size is not the engine's whole footprint; the margin absorbs
  the difference, and a model whose engine uses much more than its file needs a larger one.
- An update that rolls back puts the old unit text back, which has no memory check (the check belongs to the
  new version), and restarts the server to read the old files: a model the server held before the update, so
  it is not checked again.
- Strata's own load is not touched: a load started outside Jarvis (`strata` run by hand) is not guarded.
