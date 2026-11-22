# Docs

Where each kind of fact about Jarvis lives, and how far to trust it. Every document under `docs/` is listed
here; `tools/check_docs.py` fails when one is not.

| You want to know | Read |
|---|---|
| Every config key, its range and what needs a restart | [configuration.md](configuration.md) |
| The model, speech to text, voices, the wake word, latency and licences | [models.md](models.md) |
| How the code is laid out: modules, threads, one request end to end | [architecture.md](architecture.md) |
| The build plan, its design and what is built | [plan.md](plan.md) |
| What is planned, and what Jarvis will not do | [roadmap.md](roadmap.md) |
| What Jarvis keeps on disk and what leaves the PC | [privacy.md](privacy.md) |
| Why a decision was made | [adr/](adr/README.md) |
| How to work on the project, and its traps | [playbook/](playbook/README.md) |
| When each project practice was last audited | [playbook/practices-audit.md](playbook/practices-audit.md) |
| What Jarvis does at a lock, suspend, lost microphone or missing model drive | [playbook/session-transitions.md](playbook/session-transitions.md) |
| The confirm key's checks with the owner | [playbook/confirm-key.md](playbook/confirm-key.md) |
| What to work on next | [GitHub Issues](https://github.com/NecturaLabs/Jarvis/issues), the only backlog |

## How far to trust each kind

- **Kept true with the code**: `configuration.md`, `models.md`, `architecture.md`, `privacy.md`, the status
  table in `plan.md`, `roadmap.md` and the playbook. A change that makes one wrong fixes it in the same pull
  request. Where one disagrees with the code, the code wins: report it.
- **True when written**: the ADRs, and the rest of `plan.md`. They record why, at a date. A later ADR
  supersedes an earlier one rather than editing it.
- **Measurements** (latency, memory) name the machine and the versions they were taken on.
