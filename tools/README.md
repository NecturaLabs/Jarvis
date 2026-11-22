# Tools

The scripts the checks and the maintainer use. Each runs from the repository root. `tools/check_docs.py` fails when
a file here is not listed.

| Script | What it does | Run by |
|---|---|---|
| `check.sh` | Everything CI runs, in one command, with audio cut. Exits 1 on a failure and 3 when a check could not run: the advisories check without PyPI, the workflow check without `actionlint`. | you, before calling work done |
| `check_licences.py` | Every installed package's licence is compatible with GPL-3.0-or-later, and `uv.lock` resolves only from PyPI. | `check.sh`, CI |
| `check_docs.py` | Every doc is in `docs/README.md`, every ADR in `docs/adr/README.md`, every module in `docs/architecture.md`, every script here; links and backticked paths resolve; skills match their `.agents/` copy. | `check.sh`, CI |
| `idle_check.py` | Reads `/proc` for a running unit or PID and fails on CPU or GPU use or memory growth while idle. Starts and stops nothing. | the maintainer, `tests/test_idle_local.py` |
| `first_audio.py` | Sends typed requests to the loaded model through the real agent and times when the first sentence reaches the speaker against when the whole reply is written (#12). Plays nothing, loads and starts nothing, and the agent has only `no_action`, so only the time saved compares with real use; refuses (exit 2) when the server is down or the model not loaded; exit 3 when a request failed. | the maintainer, `docs/playbook/README.md` |
| `bar_status.py` | Asks the running Jarvis for its status over the control socket and prints the line an Omarchy bar command module shows (glyph, tooltip, class), with a state for not running, a timeout, a broken connection and an unreadable answer. Standard library only, started without Python's `site` import. 0.015 s of CPU a poll. | the Omarchy bar, every 5 s, once you add it ([configuration](../docs/configuration.md#the-omarchy-bar-module-14)) |
| `wake_train.sh` | Trains and scores one LiveKit wake phrase on the training machine with a chosen LiveKit project's key file; stops before training when that project's speech credit ran out. | the maintainer, [wake training](../docs/playbook/wake-training.md) |
| `wake_ungated.py` | Runs the wake word trainer without its two gated test sets (ESC-50, GTZAN) until access is granted (#37). | `wake_train.sh --ungated` |
| `wake_bench.py` | Benchmarks wake word models on 55 local voices they never heard (clean, with TV and music at 10 to 0 dB, with echo) and on about 24 h of real media, and prints the tables: false wakes an hour, recall at a matched false-wake rate, per voice group, and all models together as Jarvis runs them. | the maintainer, [wake training](../docs/playbook/wake-training.md) |
| `install-hooks.sh` | Points this clone's `core.hooksPath` at `tools/hooks/` (`--uninstall` undoes it). | you, once, if you want the hooks |
| `hooks/` | `pre-commit` runs lint and tests when the commit stages code or docs; `commit-msg` checks the Conventional Commits subject and warns on a missing sign-off; all three, with `pre-push`, refuse private details. | git, after `install-hooks.sh` |
| `check_private.py` | Nothing about one person or machine: real e-mail addresses and home folders, this PC's host name and home, the global git identity, and the terms in the maintainer's private list (`.git/info/private-terms`, never committed). `--staged`, `--message` and `--push` check a commit or a push. | `check.sh`, CI, the hooks |
| `check_pr_commits.py` | Every non-merge commit in a range has a Conventional Commits subject with a listed type, and, with `--signoff required`, a sign-off naming its author. | CI (`pull-request-commits.yml`), you |
| `ci-tool-install.sh` | Pinned downloads of uv, the shell checkers and actionlint, verified by sha256. | CI |
