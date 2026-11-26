# ADR: Use only the skills of plugins from Claude Code-format marketplaces

**Status:** proposed
**Date:** 2026-11-26

## Context and problem statement

Jarvis ships a few skills and reads the owner's own ([ADR 0024](0024-owner-skills.md)). Useful third-party
skills already exist as plugins in the Claude Code marketplace format, and the owner wants to use them and
have them kept up to date, not copied into Jarvis and left to age. A plugin can also carry code (hooks,
commands, agents, MCP servers, scripts), and Jarvis runs no plugin code, no MCP server and one agent only. How
does Jarvis take third-party skills and keep them current?

## Considered options

1. Copy chosen skills into the repository at a pinned release (vendoring). Rejected by the owner: it ages, and
   it needs its own update machinery.
2. Read the owner's coding-agent plugin folders. Rejected: Jarvis never uses the owner's agent setup, and those
   plugins carry code for other agents.
3. Jarvis keeps its own copies of marketplaces it is told about, in the Claude Code format, and uses only their
   skills.

## Decision outcome

Chosen option 3.

- **Format.** A marketplace is a Git repository with `.claude-plugin/marketplace.json` (a `name` and a list of
  plugins). A plugin's source is a folder of that repository (`./` or `./path`), a GitHub repository or an
  https Git address; a source that pins a `ref` or `sha` is refused, since Jarvis follows the latest. A plugin's
  `skills/<name>/SKILL.md` files, with their front matter `description`, are its skills, named `plugin:skill`.
- **Skills only.** Hooks, commands, agents, MCP servers, scripts and settings are never read or run
  (`jarvis plugins` names those a plugin holds). Jarvis starts nothing from a downloaded repository: Git is the
  only program involved, with hooks off, https only, no links written, no submodules and none of the owner's own
  Git configuration, started through the CLI's launcher.
- **Config.** `[plugins]` lists `marketplaces` (edited only in `config.toml`, defaulting to NecturaLabs's own),
  the `enabled` plugins, the skills `always_on` in each channel, and `auto_update`. **Nothing is fetched while
  `enabled` is empty**, so a default install contacts nobody; no third-party plugin is enabled by default.
- **Storage.** Jarvis's own folder, `<data_dir>/plugins/`, closed to the file operations like its config and
  state, and read once per process like the owner's skills: only folders and regular files that are the owner's
  alone (not writable by group or others), opened without following a link, within size limits (16,000 bytes a skill file; at most 8,000 bytes for an always-on body and
  12,000 for a channel's). A skill over the limit is not always-on and `jarvis doctor` says why.
- **Use.** An always-on skill's body (front matter removed) is appended to that channel's prompt under a header
  Jarvis writes and followed by a closing line Jarvis writes, so the text cannot fake its own end: it is
  third-party text that changes how the model works and grants nothing. **An always-on skill is in the system
  prompt, a trusted context, by the owner's choice of enabling it always-on**; the owner accepts that its text
  steers the model like their own instructions, bounded by the policy in code. Any other skill is listed as
  third-party with a short description and read with `read_skill`, which gives it as outside text
  (`Origin.PLUGIN_SKILL`, not trusted) under the same heading and closing line, so an action that changes
  something after it needs a yes, as after a web page. A plugin's text never widens an operation: policy,
  confirmation and audit are in code.
- **Updates, not pinned.** Each repository is shallow-cloned once and then, to update, cloned again beside
  the old copy and swapped in whole (the old one stays as `.old` until the next update), after checking that
  the new commit builds on the old one (a rewritten history, or more than 100 commits ahead, is a failure the
  owner resolves). A repository over 50 MB is refused and removed; clones ask the host to skip files over 1 MB.
  Folders under `<data_dir>/plugins/` are named by host, path and a digest of the address.
  `jarvis update --apply`, and its daily timer, do this before Jarvis's own update when `auto_update` is on;
  `jarvis plugins update` always does. Each change is audited with the commit before and after, and
  `jarvis doctor` shows each marketplace's commit and date and any failed update.

### Consequences

- Good: third-party skills are used and tracked without copying them, and a plugin's code never runs.
- Bad: following the latest commit means upstream changes reach the model's instructions without review. The
  owner chose this over pinning. The limits are that it only guides the model, every operation still passes
  the policy in code, the audit shows each change, and only plugins the owner enabled are fetched. A hostile
  update to an enabled plugin could still steer the model within what the policy allows, as the owner's own
  words could.
- Bad: a plugin host that points a source at a name resolving to a private address is not stopped by Jarvis
  (Git does the lookup); a source must be a public host name, never an IP address or a name without a dot.
- Neutral: Git must be installed, and an update needs the network; with the network down a failed update is
  reported and the last downloaded text stays in use.
