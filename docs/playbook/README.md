# Playbook

How to work on Jarvis from a cold start. Commands are in `AGENTS.md`; this file holds the procedures and the
traps found along the way. Add a trap the moment it costs time.

## Build and check

1. `uv sync` (uses the system Python 3.14; `uv.lock` pins every package with hashes).
2. `git add` new files, then `tools/check.sh` — everything CI runs; the lint checks tracked files only.
3. `uv run pytest -m local` only on this PC, with the owner when audio is involved. The `run-silent` skill says
   what an agent may start.

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

## Upkeep

- Dependabot proposes package and action updates weekly, a week after each release (`cooldown`). The Python
  interpreter (`.python-version`, `requires-python`) has no Dependabot ecosystem: bump it by hand, with the CI
  run on the new version.
- [session-transitions.md](session-transitions.md): what Jarvis does at a lock, suspend, lost microphone
  or missing model drive, what was seen, and how to check each.
- `tools/idle_check.py --unit jarvis.service --unit jarvis-strata.service` measures the "no CPU or GPU while
  idle" rule on a running install; it only reads `/proc`.

## Traps

- `dev` and `main` refuse a commit that has not passed CI. A merge commit made on `dev` is new, so push it to
  the topic branch first, let CI pass on it there, then push it to `dev`: the check belongs to the commit.

- `.lint/check.sh` on an empty index passes without checking anything. Stage first.
- Ruff's base config ignores `PT009`/`PT027` for unittest scripts; `ruff.toml` re-selects them for pytest.
- A test fixture that wraps its process in `sh -c` must `exec` the real program. Otherwise stopping it kills the
  shell and leaves the child running; one such child spun a CPU core and slowed the whole suite about 8 times
  (commit `fb40b3d`).
- pyright needs `venvPath`/`venv` in `pyrightconfig.json` to see the uv environment.
- SearXNG (`deploy/searxng/`): `engines: remove: [brave]` under `use_default_settings` crashes it with
  `KeyError: 'brave'` (other engines refer to it); disable it in the `engines:` list instead. Brave's results come
  back as `http:///go/v1/…` links that cannot be opened. `cap_drop: ALL` needs `CHOWN`, `SETGID` and `SETUID`
  added back, and a read-only root breaks the image's certificate update at start. Without `default_lang: en`
  the engines answer in the public IP's language.
- Reddit serves no page to Jarvis, so a Reddit link from a web search is read from the Arctic Shift archive
  (`web/ops.py`); the archive's `/api/posts/search` takes one title word, so Reddit search goes through
  SearXNG.
- Trafilatura takes a page's `<h1>` as its title, not `<title>`.
- Agents never run `jarvis ask` or `run` against the owner's live setup (`run-silent`), so the model's choice
  of the new tools is checked by the owner by voice, not by an agent.
- The repository's git uses diff3 conflict markers (`|||||||` base section). A script that resolves
  conflicts by taking both sides must drop the base section too, or the old text comes back.
- Line-number edits (`sed -i '826s/…/'`) after `ruff format` hit the wrong line: the formatter moves
  lines. Edit by matching text.
- Voxtype is driven only through the commands AGENTS.md names; a new one (even read-only, such as
  `voxtype info models`) is the owner's decision, not an audit fix.
- `jarvis run` serves `$XDG_RUNTIME_DIR/jarvis/control.sock`; `jarvis status --json`, `pause`, `resume`
  and `hush` talk to it. Unit tests make their sockets under pytest's temp folder (`conftest.py` refuses
  any other Unix socket).
