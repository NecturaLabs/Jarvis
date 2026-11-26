# ADR: Read the owner's own skills from Jarvis's config folder

**Status:** accepted
**Date:** 2026-11-26

## Context and problem statement

Jarvis ships its own skills (`prompts/skills/`, read with `read_skill`) and reads Omarchy's guides. The owner
asked for a skill of their own: instructions for a task that matter only on their PC, such as what to watch for
in one program. Such a skill cannot ship with Jarvis, which is public and kept generic. The old rule allowed no
skills but the shipped ones and Omarchy's. The owner also ruled that Jarvis never uses the owner's agent setup
for coding agents (their global instructions, plugins and skills): what Jarvis needs is written for Jarvis. How
does Jarvis get the owner's own instructions?

## Considered options

1. Owner-written skill files in Jarvis's own config folder, `~/.config/jarvis/skills/<name>.md`, in the same
   format as the shipped ones.
2. Longer free text in `persona.addendum` in `config.toml`.
3. Read skills from the owner's coding-agent setup.

## Decision outcome

Chosen option 1. A skill is read only when needed, so the prompt holds one line per skill rather than its
whole text, which option 2 would put in every request. Option 3 is what the owner ruled out, and it would load
instructions written for other agents with other powers.

- An owner skill has the trust of the config file: the owner's own words. It guides the model and never widens
  what an operation may do, since every boundary is in code before an operation runs.
- A file counts only when the folder and the file belong to the owner and nobody else can write to them, the
  folder and the file themselves are opened without following a link (a link higher up, such as a managed
  `~/.config`, is followed), its name is lowercase letters, digits and hyphens, and it fits the
  limits: 20 skills, 200 characters for each description and 1,600 for all of them, and 16,000 bytes for each
  file. A name a shipped skill has stays the shipped skill's. `jarvis doctor` says which files were left out and why.
- The prompt marks an owner skill "(yours)". The folder is read once per process, so the prompt and
  `read_skill` always list the same skills, and a new or changed skill takes effect at the next `jarvis restart`
  or chat window. `jarvis doctor` reads it afresh.
- Jarvis's file operations never read or write its config folder, whatever `[files]` allows: the settings and
  skills there steer Jarvis, so no request, however it was led, can rewrite them.

### Consequences

- Good: the owner can teach Jarvis their own routines without changing its code or the public repository.
- Bad: any program running as the owner can write that folder, as it can the config file, and a skill's text is
  trusted like the owner's own words. Such a program can then steer the model on every request: it does not
  count as outside text, so the extra caution after outside text does not apply. Every operation still passes
  the policy (its class, confirmation where the class needs it, the cap on actions and the audit), so a skill
  cannot make Jarvis do what a request of the owner's could not. Jarvis itself cannot write the folder: its
  file operations are refused there, and the opt-in shell command is the owner's own risk as before.
