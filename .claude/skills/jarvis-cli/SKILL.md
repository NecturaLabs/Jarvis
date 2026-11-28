---
name: jarvis-cli
description: How an agent runs the `jarvis` command and reads its answer - `--agent` (or `JARVIS_AGENT_FORMAT=1`) for one line of JSON with `command`, `ok` and `exit`, each command's own keys, the exit codes, the commands that refuse it - and how to keep both formats right when changing a command's output (`jarvis.cliout`). Use before running any `jarvis` command to read its result, and before adding or changing what a command prints.
---

# The `jarvis` command, for agents

Read `run-silent` first: it says which commands are safe to run and which act on the owner's setup.
This skill is about reading what a command answers.

## Ask for the agent format

Add `--agent` (or `--agent-format`) before or after the command, or set `JARVIS_AGENT_FORMAT=1` once
for every command. The answer is then exactly one line of JSON on stdout, nothing else, even when the
command fails, its arguments are wrong, or it crashes. Never parse the text a person sees: its wording,
alignment and colour change freely.

```sh
uv run jarvis doctor --agent --no-service
JARVIS_AGENT_FORMAT=1 uv run jarvis tools
```

Every answer has:

| Key | Meaning |
|---|---|
| `command` | the command that ran (`status` when none was named; `null` when argparse could not tell) |
| `ok` | `true` when `exit` is 0 |
| `exit` | the exit code, also the process's: 0 done, 1 failed (a failed check, a crash), 2 refused (bad arguments, an interactive command, a setting that needs a value) |

and then either the command's own keys or, for a command without them, `output` and `errors`: its
stdout and stderr lines, colour removed. A crash puts its traceback in `errors`.

| Command | Own keys |
|---|---|
| `status` | `state`, `paused`, `model`, `running` (as `jarvis status --json`) |
| `doctor` | `checks` (each `status` `ok` or `fail`, `what`, `fix`), `notes`, `summary` |
| `tools` | `tools` (each `name`, `setting`, `state`, `shipped`, `changed`, `description` in full); after `tools NAME SETTING`, `tool` |
| `speed` | `speech_speed` |
| `consent` | `consent` (`bypassed` or `asks`), `said` |
| `archive` | `path`, `transcripts`, `user_recordings`, `jarvis_recordings` |
| `plugins` | `marketplaces` (each `source`, `downloaded`, `name`, `commit`, `date`, `refused`, and `plugins` with `name`, `description`, `enabled`, `skills`, `ignored`) |

`acp`, `chat`, `logs`, `run`, `ask`, `speech-test` and `spotify-login` wait on a person or follow a live
stream: with `--agent` they do nothing and answer `exit` 2. Read the journal with
`journalctl --user -u jarvis -n 100 --no-pager` instead of `jarvis logs`.

## What a person sees

Checks and states are one line each, status first in its own column, then what it is about, then
for a failure the fix after an arrow; `jarvis doctor` ends with the count and the names of what failed:

```text
ok   config /home/user/.config/jarvis/config.toml
FAIL Strata server (unreachable)  -> jarvis start
ok   Spotify login (40 days left)
note Spotify login lacks playlist-read-private: run `jarvis spotify-login` once to …
12 checks, 1 failed: Strata server (unreachable)
```

Colour marks the status word only, and only on a terminal: never in a pipe, with `NO_COLOR` set,
with `TERM=dumb`, or in the agent format.

## Changing what a command prints

- Write check, note and closing lines with `jarvis.cliout` (`check`, `note`, `summary`), never with
  hand-made padding or escape codes, so both formats stay in step.
- A command whose answer an agent acts on gets its own keys: add a function to `AGENT_FIELDS` in
  `src/jarvis/cli.py` returning them, and keep existing keys stable (add, never rename). A new
  interactive command goes in `INTERACTIVE`.
- Test both formats in `tests/test_cli_agent.py` (agent keys, one line) and `tests/test_cli.py` (the
  lines people read); `tests/test_cliout.py` covers the module itself.
- Update the command table in `README.md` and this skill's tables, then run `tools/check.sh`.
