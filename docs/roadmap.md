# Roadmap

What Jarvis plans, and what it will not do. The detail lives in
[GitHub Issues](https://github.com/NecturaLabs/Jarvis/issues), ordered by priority (`priority:p0-critical` to
`priority:p3-low`); this page names the themes and links them. [docs/plan.md](plan.md) holds the design and
says which parts are built.

## Themes

- **Verification on real hardware**: most of what is built is still open for checks against a real machine:
  the model service (#2), the wake word and transcription (#3), the policy engine (#4), configuration and
  setup (#5), the agent loop (#6) and Spotify control (#7). Whether the local model can drive Jarvis's tools
  is measured in #17, and which speech-to-text model serves Jarvis best in #28.
- **Install and maintenance**: one install command and a doctor that names the fix for each problem (#16);
  update, rollback and uninstall (#21); commands for desktop session changes such as lock and suspend (#22).
- **Operating the PC**: typed PC operations (#9; stats, files, stopping and opening programs, the PC's volume and
  other media players are built; workspaces and Omarchy's commands are not), and an off-by-default shell
  operation behind strong confirmation (#19, built).
- **Research**: built (web search, page reading and Reddit through a local SearXNG); the owner's spoken check
  of a sourced answer and SearXNG in the installer remain (#10, #20).
- **Voice polish**: speaking the first sentence while the rest is generated (#12) and follow-up requests
  without the wake word (#11) are built and wait for live measurements; the bare word "Jarvis" (#15) is built
  as the Azure keyword model and waits for the owner's trained model; the Omarchy bar module (#14) is built
  and waits for the owner to add it ([configuration](configuration.md#the-omarchy-bar-module-14)). Still to
  build: interrupting Jarvis by voice without its own voice waking it (#13).
- **Accessibility** (not filed as issues yet): a text-only way to use Jarvis and a screen-reader check of
  the chat window; see [ACCESSIBILITY.md](../ACCESSIBILITY.md).

Everyone who runs Jarvis creates their own Spotify developer app (the maintainer's was #8, now done).

## Not planned

These are product decisions, not gaps (see [AGENTS.md](../AGENTS.md#product-rules) and
[ADR 0005](adr/0005-local-only.md)):

- a hosted model backend or a paid API, even as an optional fallback;
- sending what the owner says, or what the PC plays, to a service other than the one a feature needs;
- subagents, delegation or a second-agent tool: one agent does all the work;
- loading global agent instructions, skills, plugins or MCP servers at runtime; the Omarchy OS skills are the
  only outside knowledge;
- desktop notifications: Jarvis replies by voice;
- tool calls that skip the policy engine, or boundaries that are enforced only in the prompt;
- support for operating systems or desktops other than Linux with Omarchy, PipeWire, Voxtype and Strata (a
  port would need its own discussion first).
