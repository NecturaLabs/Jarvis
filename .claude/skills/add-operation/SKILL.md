---
name: add-operation
description: End-to-end recipe for adding or changing something Jarvis can do (a typed operation such as a new Spotify action or a PC operation), from the Operation and its argument schema through the policy class, confirmation wording, audit, tests and docs. Use whenever an issue or request adds a tool the model can call or changes what an existing one does.
---

# Add an operation

Every action is a typed operation checked in code before it runs ([ADR 0002](../../../docs/adr/0002-typed-operation-policy.md)).
Work through the steps in order; each has a done condition. The policy engine, the confirmation flow and any PC
operation are security-sensitive (`AGENTS.md`, Boundaries): they get a security pass and the `jarvis-reviewer`
agent before merging.

## 1. The operation

- Name the operation in a group of `src/jarvis/toolgroups.py` (a test fails for one no group names); the group's
  summary says what its tools are for, so extend it when the new tool widens that.
- Add an `Operation` (`src/jarvis/operations.py`) to the owning module's `operations()` list, for example
  `SpotifyOps.operations()` in `src/jarvis/spotify/ops.py`. Give it:
  - a `name` with its area prefix (`spotify_…`); never a name that delegates to another agent (one agent only);
  - a `description` the model reads: what it does and when to pick it, in one or two sentences;
  - `parameters`: a flat JSON schema object with `"additionalProperties": false`, an `enum` for every closed set
    and bounds for every number; `validate_args` enforces it before the policy sees the call;
  - `default_class`: `SAFE`, `CONFIRM`, `STRONG` or `DENY` (`src/jarvis/policy.py`); anything irreversible or that
    spends money is at least `CONFIRM`;
  - `side_effect=False` only for a pure read (it is then not raised to confirm by outside text);
  - `concurrent=True` on a pure read that waits on the network, when its `run` is safe in several threads at
    once (shared state behind a lock): calls of one model message to such reads then run side by side;
  - `outbound=True` when it sends words the model wrote to a service off this PC (a search, a place name), with
    `outbound_args` naming the arguments that carry them: while file text is in the conversation such a call needs
    a yes;
  - `hashed` for arguments the audit log must keep only as a SHA-256 and a length, and `audit_text=False` when a
    result can hold anything the owner can read (a shell command's output);
  - `lowest_class` when the owner's `boundaries.classes` must not set it below a floor, and `preview` when a
    confirmed call writes text the question should show the start of;
  - `spoken_summary`: the words of the confirmation question, naming the exact target;
  - `prepare` when the call acts on "whatever is current" (the current song, the active device): it binds the exact
    target before the question, so the yes is for that target and not for what is current later.
- `run` returns a `Result`; text from outside (Spotify names, web text) keeps its `origin` so it stays data. Raise
  `UncertainOutcomeError` when the call may or may not have happened: it is not retried.
- No shortcuts: a new operation gets no `shortcuts=` phrasing, keyword list or pattern over the owner's words
  ([ADR 0013](../../../docs/adr/0013-free-speech-no-new-shortcuts.md)). The owner says it in any words, so the
  description must tell the model when to use it; the shortcuts ADR 0004 already has stay as they are.

Every tool's schema is in every request, and published tests find models pick less well as the list grows (#31
measures it for Jarvis). Before adding one, check whether an argument on an existing operation of the same area
does the job.

Done when: the operation is in the registry `src/jarvis/app.py` builds, so the model's tool schemas carry it; the
tests below show it.

## 2. Tests (`tests/`)

Mirror the existing tests of the module (`tests/test_spotify_ops.py`, `tests/test_agent.py`, `tests/test_policy.py`):

- argument validation: a missing, extra and out-of-range argument are refused before `run`;
- a refused case (the `deny` class or an owner override) runs nothing and is audited;
- an unconfirmed case (no answer, or anything but a clear yes) runs nothing;
- for `prepare`: the confirmed call runs on the bound target even if the current one changed;
- the happy path, with fakes for the outside service (no network, no audio; `tests/conftest.py` refuses both).

Each new test is seen failing first (break the operation, watch it fail, restore).

## 3. Docs

- `docs/configuration.md` `[boundaries.classes]` when the class matters to the owner; `README.md` "What it does today";
  `docs/plan.md` build status; `CHANGELOG.md` (Unreleased); a new ADR only for a consequential decision.

Done when: `tools/check.sh` passes and the `jarvis-reviewer` agent has no blocking finding.
