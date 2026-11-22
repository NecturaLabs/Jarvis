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
| `install-hooks.sh` | Points this clone's `core.hooksPath` at `tools/hooks/` (`--uninstall` undoes it). | you, once, if you want the hooks |
| `hooks/` | `pre-commit` runs lint and tests when the commit stages code or docs; `commit-msg` checks the Conventional Commits subject and warns on a missing sign-off; all three, with `pre-push`, refuse private details. | git, after `install-hooks.sh` |
| `check_private.py` | Nothing about one person or machine: real e-mail addresses and home folders, this PC's host name and home, the global git identity, and the terms in the maintainer's private list (`.git/info/private-terms`, never committed). `--staged`, `--message` and `--push` check a commit or a push. | `check.sh`, CI, the hooks |
| `check_pr_commits.py` | Every non-merge commit in a range has a Conventional Commits subject with a listed type, and, with `--signoff required`, a sign-off naming its author. | CI (`pull-request-commits.yml`), you |
| `ci-tool-install.sh` | Pinned downloads of uv, the shell checkers and actionlint, verified by sha256. | CI |
