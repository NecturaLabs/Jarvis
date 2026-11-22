# Contributing

Jarvis is an open-source NecturaLabs project, and contributions are welcome. These notes apply to everyone
working on it, human or agent; [AGENTS.md](AGENTS.md) has the full conventions. Report security problems
privately, as [SECURITY.md](SECURITY.md) says, never in a public issue.

## Your first contribution

1. Find an issue: [`good first issue`](https://github.com/NecturaLabs/Jarvis/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
   and [`help wanted`](https://github.com/NecturaLabs/Jarvis/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22)
   mark ones that are ready to pick up. Say in the issue that you are working on it. [docs/roadmap.md](docs/roadmap.md)
   lists what is planned.
2. To report a bug, use the [issue forms](https://github.com/NecturaLabs/Jarvis/issues/new/choose) after
   searching for a duplicate. To propose a feature or agree the approach for a large change, start a
   [discussion](https://github.com/NecturaLabs/Jarvis/discussions) first; a pull request that arrives
   unannounced may not fit.
3. Fork the repository, clone it and make a branch named `<type>/<topic>` (`feature/`, `bugfix/`, `docs/`,
   `chore/`, for example `bugfix/confirmation-timeout`).
4. Make the change with its tests, run the [checks](#checks), and commit with
   [Conventional Commits](#commit-messages) and a sign-off (`git commit -s`).
5. Open a pull request against `dev`, fill in the template and link the issue. `dev` is the default branch
   and takes all work; `main` only takes `dev` when a release is cut. Only the maintainer merges.

Questions about using Jarvis go to [Discussions](https://github.com/NecturaLabs/Jarvis/discussions)
([SUPPORT.md](SUPPORT.md)).

## What fits Jarvis

Jarvis is a local, always-on voice agent for Linux with Omarchy (Hyprland), PipeWire, Voxtype and Strata.
Contributions that keep to the product rules in [AGENTS.md](AGENTS.md#product-rules) are welcome. These will
be declined:

- a hosted model backend, a paid API, or sending what the owner says or what the PC plays to any service
  other than the ones a feature needs (Spotify's API is the only one today; see
  [ADR 0005](docs/adr/0005-local-only.md));
- subagents, delegation or a second-agent tool: one agent does all the work;
- loading global agent instructions, skills, plugins or MCP servers at runtime, apart from the Omarchy OS
  skills;
- a tool or code path that runs without the policy engine deciding first, or that lets a prompt stand in for
  a check in code ([ADR 0002](docs/adr/0002-typed-operation-policy.md));
- desktop notifications: Jarvis replies by voice;
- driving Voxtype with any command AGENTS.md does not list, or editing a user's `~/.config/voxtype/`;
- code, models or voices under a licence that is not compatible with GPL-3.0-or-later.

Support for other operating systems or desktops is not a goal at present; discuss it first. The plan and its
reasons are in [docs/plan.md](docs/plan.md).

## What to expect

Jarvis has one maintainer, so reviews happen when there is time. A new issue or pull request usually gets a
first reply within a week. A pull request is merged once its checks pass and the review's blocking points
are resolved; one that stalls for 30 days without a reply from its author may be closed, and can be reopened
at any time. How decisions are made: [GOVERNANCE.md](GOVERNANCE.md).

## Licence of contributions

Jarvis is licensed under the [GNU General Public License, version 3 or (at your option) any later
version](LICENSE). Contributions come in under the same terms: unless you explicitly state otherwise, any
contribution you intentionally submit for inclusion in Jarvis is licensed GPL-3.0-or-later, without any
additional terms or conditions. Please sign off each commit (`git commit -s`) to certify the
[Developer Certificate of Origin](https://developercertificate.org/): that you wrote the change or otherwise
have the right to submit it under this licence. GitHub signs off web commits only in a repository that turns
on "Require contributors to sign off on web-based commits", and your fork may not have it on. Where it is
off, use `git commit -s`, or `git rebase --signoff` for commits you already made. CI checks the sign-off on
pull requests from forks ([Commit messages](#commit-messages)).

A pull request from a fork runs CI with a read-only token and no secrets, once a maintainer has approved the
run.

## Setup

1. Install [uv](https://docs.astral.sh/uv/) and Python 3.14. `pyproject.toml` sets uv to use the system
   Python only, so install 3.14 with your package manager.
2. Install the environment: `uv sync`.
3. Running Jarvis itself needs Omarchy, PipeWire, Voxtype, Strata and model files; the unit tests and the
   checks need none of them. [README.md](README.md) and [docs/models.md](docs/models.md) list the rest.

Run anything that could make a sound silently, with
`PIPEWIRE_REMOTE=/nonexistent PULSE_SERVER=unix:/nonexistent`. Tests feed recorded fixtures instead of the
microphone and use no real audio device.

## Checks

```sh
uv sync                  # install the locked environment
tools/check.sh           # everything CI runs: lint, types, workflows, tests, REUSE, licences, docs, advisories
uv run .lint/check.sh    # lint only: Ruff (every rule), format check, pyright strict, shell and editorconfig
actionlint               # the workflow files in .github/workflows/, with shellcheck over their run: steps
uv run pytest -q -m "not local"   # tests only
tools/install-hooks.sh   # optional: run the checks before each commit and check the subject format
```

`.lint/check.sh` lints tracked files only, so `git add` new files first. CI runs [actionlint](https://github.com/rhysd/actionlint)
at the version pinned in `.github/workflows/ci.yml`; install it from your distribution (`pacman -S actionlint`
on Arch) or from its releases. Without it `tools/check.sh` reports the workflow check as not run and exits 3. `tests/conftest.py` gives every test a
temporary home folder and refuses the network, audio and the real programs Jarvis drives, so a test cannot reach
your own setup. [tools/README.md](tools/README.md) lists every script. Every check must pass before a
pull request can merge; fix findings rather than silencing them. Where a suppression is deliberate, put it on
the narrowest line and give the reason beside it, as the existing `# ruff: ignore[...] - reason` comments do.
Do not edit `.lint/`; project lint settings go in `ruff.toml` and `pyrightconfig.json`.

Tests marked `local` need this project's real hardware or services (Strata, Voxtype, audio, Spotify) and
never run in CI. You do not need to run them; if your change touches that code, say in the pull request that
they were not run. CI runs on GitHub-hosted runners.

A bug fix comes with a regression test that fails without the fix. Do not weaken a test to make it pass:
change it only when the requirement it covers has changed, and say so.

## Issues and pull requests

[GitHub Issues](https://github.com/NecturaLabs/Jarvis/issues) hold the backlog, defects and follow-ups. Before
starting, check for an existing issue and record partial progress there; close it only when its acceptance
criteria are met.

### Issue content

Use a specific, action-oriented title without type or priority prefixes. State the problem or deliverable,
relevant files and constraints, and checkable acceptance criteria. Bugs need observed and expected
behaviour, reproduction steps and environment details. Say when facts are unknown. Redact tokens, recordings
and personal data. Record each owner decision with its date, answer and alternatives, and keep requirements
that remain unresolved.

Write direct statements. Progress notes name the commit, the checks actually run and the remaining gaps.
Close compound issues only when every criterion is verified; put independent follow-ups in linked issues. For
an exact duplicate, link the canonical issue, add `duplicate` and close it.

### Labels and triage

[`.github/labels.json`](.github/labels.json) records label names, colours and descriptions. Every triaged
issue except a closed duplicate has exactly one type:

| Type | Use |
|---|---|
| `bug` | Incorrect existing behaviour or a failure |
| `enhancement` | A new or expanded user-facing capability |
| `maintenance` | Internal upkeep, refactoring, tooling or process work |
| `documentation` | Documentation changes without product behaviour changes |
| `verification` | Checks of implemented behaviour, with recorded evidence |
| `owner-action` | Maintainer decisions, legal information or external-account tasks |

Apply one `area:*` label for the main affected area. The form's Area answer maps to these labels:

| Form area | Label |
|---|---|
| Wake word, microphone and transcription | `area:audio` |
| Speech output and confirmation | `area:voice` |
| Agent loop and prompts | `area:agent` |
| Boundaries, provenance and audit log | `area:policy` |
| Model and Strata | `area:model` |
| Spotify | `area:spotify` |
| PC operations | `area:pc-ops` |
| Web research | `area:research` |
| Install, setup and configuration | `area:setup` |
| Tests | `area:tests` |
| Documentation | `area:docs` |
| CI, lint and supply chain | `area:ci` |

Every open issue has one priority (`priority:p0-critical`, `p1-high`, `p2-normal`, `p3-low`; P2 is the default
until triage) and one size (`size:small`, `medium`, `large`, or `needs-estimate` when the scope is unclear).
Priority is urgency; size is remaining work, not a delivery date.

Forms apply the type, P2 priority, `size:needs-estimate` and `needs-triage`; the verification form also applies
`needs-verification`. Dropdown answers do not apply labels automatically. During triage, the maintainer checks
for duplicates, confirms actionable scope and acceptance criteria, applies the area, reviews priority and
size, then removes `needs-triage`.

`good first issue` and `help wanted` mark open work that has checkable acceptance criteria and needs no
maintainer decision. `needs-verification` means live, hardware or owner acceptance checks remain, and
`security` marks credential, trust, privacy or command-execution risks. `dependencies`, `github_actions` and
`python` are automation labels for dependency pull requests, not issue types.

Blank web issues are disabled. See GitHub's
[issue-form syntax](https://docs.github.com/en/communities/using-templates-to-encourage-useful-issues-and-pull-requests/syntax-for-issue-forms).

### Pull requests

A pull request is one focused change (a refactor goes apart from a behaviour change), on a branch named
`<type>/<topic>`, titled like a commit subject, with the template filled in: summary, issue (`Closes #N`, or
`Refs #N` for part of one), review focus, verification and what was not verified. Keep independently
reviewable work in atomic commits. Contribute to the existing branch or pull request for the work; do not
open a duplicate. CI must pass on a branch that is up to date with `dev`. Keep **Allow edits by maintainers**
on so the maintainer can update it for you. Wait for the maintainer's approval before merging.

Security-sensitive areas (the policy engine, the confirmation flow, the Spotify login, the PC-operation tools
and the web tools) get a security pass and a review before they merge; [CODEOWNERS](.github/CODEOWNERS) lists
the paths.

## Commit messages

[Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/): `<type>(<scope>): <summary>`, summary
in the imperative and at most 72 characters.

- Types: `feat`, `fix`, `perf`, `refactor`, `docs`, `test`, `build`, `ci`, `chore`, `style`, `revert`.
- Scopes: a module or area (`cli`, `policy`, `spotify`, `audio`, `services`, `chat`, `config`, ...).
- Breaking changes: `!` after the scope and a `BREAKING CHANGE:` footer.
- Security fixes: the `security` scope or a `Security:` footer naming the advisory (`Security: GHSA-...`).
- End each commit with a `Signed-off-by:` trailer (`git commit -s`).

The "Pull request commits" check runs `tools/check_pr_commits.py` on every pull request: each commit except
merges needs a subject in this form, and in a pull request from a fork each commit needs a `Signed-off-by:`
trailer with its author's name and email. When it fails, it names each commit and how to fix it: `git commit
--amend` for the last commit's subject, `git rebase -i <base>` for an earlier one, and `git commit --amend
--no-edit --signoff` or `git rebase --signoff <base>` for a missing sign-off. Run it yourself with
`python3 -B tools/check_pr_commits.py origin/dev HEAD --signoff required`.

Example: `feat(spotify): queue several songs in one request`.

## Before merging

- [ ] Changes are reviewable as individual commits, based on current `dev`.
- [ ] `tools/check.sh` passes, and the required GitHub Actions checks pass.
- [ ] A new or changed tool operation has tests, including a refused and an unconfirmed case.
- [ ] Anything that can make a sound ran silently, and the pull request says how.
- [ ] Docs (`README.md`, `docs/configuration.md`, `docs/models.md`, `docs/plan.md`, an ADR where a decision
      changed) and `CHANGELOG.md` (Unreleased) are updated where the change touches them.
- [ ] A new dependency, model or voice has its licence checked and recorded in `docs/models.md`.
- [ ] No credentials, tokens, recordings or personal paths in code, tests, logs or screenshots.
