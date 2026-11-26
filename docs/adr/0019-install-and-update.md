# 0019. How Jarvis installs and updates from a release branch

Status: proposed  
Date: 2026-11-26

## Context

Jarvis needs native audio tools, a local Strata model and several downloadable voice assets.
The voice service runs from a Git checkout's `.venv`. Replacing that environment while Jarvis
is answering can interrupt a request. A checkout may also contain a user's changes.

## Decision

Ship a Bash installer with the release branch. The installer checks prerequisites, uses system
Python with the uv lockfile, verifies the model assets it downloads by size and SHA-256, and asks
for a Strata run config only if it cannot reuse or find one. Omarchy packages, Strata and Voxtype
models follow their own setup procedures. Starting user services requires an explicit flag.

The headless browser is part of Jarvis, not an option (the owner's decision): the installer stops
without agent-browser, printing how to get it, and has it download its own Chrome into the home
folder (`agent-browser install`, never `--with-deps`, which needs root). While the browser is
switched on, `jarvis doctor` fails a missing agent-browser or Chrome, so an update to a release that
needs it rolls back on a PC without it.

Follow `main` by default. A checkout on a different release branch can select it in `[update]`.
Fetch only that branch from the checkout's configured upstream and require a clean fast-forward.
The daily user timer is opt-in. A process lock serializes update attempts. An apply holds the
mode and request locks while it changes code, dependencies and generated service files. An
active voice request, including capture or playback, defers the update. A running Strata service
keeps its model loaded when its settings stay the same.

Rollback restores the old commit with `git reset --keep`, synchronizes its locked environment,
and restores generated service files and prior running services. This covers a failed sync,
essential local checks and failed service startup. Concurrent checkout edits can prevent rollback.
A power loss or forced kill can leave an incomplete update. `jarvis doctor` reports the most
recent recorded result, and the old commit remains in the local Git history.

## Release trust

The installer trusts the HTTPS origin of the downloaded script and its public clone. The updater
trusts the configured Git remote and the user's Git transport configuration. Git checks object
integrity and Jarvis checks ancestry. `uv sync --locked` requires the tracked lockfile and uses its
artifact hashes. Jarvis checks the release pins of the model files it fetches itself. Its update
health check does a read-only Strata probe and checks the running voice service's control socket
when those services are expected to be up. It does not generate test tokens or play test audio.

Jarvis does not yet require valid SSH commit signatures. An allowed-signers file read from the
incoming commit could be replaced by that same commit. A key policy taken from the currently
trusted install could authenticate later updates, but the initial bootstrap, which keys can sign
a release, key rotation and the policy for merge commits need an owner decision. Fast-forward
checks alone do not protect against an authorized writer or a compromised upstream.

## Alternatives

A wheel or single binary would simplify downloads but would need a separate release and rollback
process, and still needs Strata, Voxtype and the user units. Unattended system packages or third-party
model selection would put hardware and privilege choices in the installer. Default-on updates
would let a new installation change code and restart without a separate choice.

## Owner decisions

- Confirm `main` as the default release branch.
- Confirm automatic updates start off.
- Confirm whether the release needs a maintained SSH signer policy and how users bootstrap it.
- Confirm `${XDG_STATE_HOME:-$HOME/.local/state}/jarvis/checkout` for the checkout and adjacent
  `jarvis/models` for new installations.
