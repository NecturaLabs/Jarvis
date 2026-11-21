# Roadmap

What Jarvis plans, and what it will not do. The detail lives in
[GitHub Issues](https://github.com/NecturaLabs/Jarvis/issues), ordered by priority (`priority:p0-critical` to
`priority:p3-low`); this page names the themes and links them. [docs/plan.md](plan.md) holds the design and
says which parts are built.

## Themes

- **Verification on real hardware**: most of what is built is still open for checks against a real machine:
  the model service (#2), the wake word and transcription (#3), the policy engine (#4), configuration and
  setup (#5), the agent loop (#6) and Spotify control (#7). Whether the local model can drive Jarvis's tools
  is measured in #17, and spoken replies through Piper in #18.
- **Install and maintenance**: one install command and a doctor that names the fix for each problem (#16);
  update, rollback and uninstall (#21); commands for desktop session changes such as lock and suspend (#22).
- **Operating the PC**: typed PC operations (#9), and an off-by-default shell operation behind strong
  confirmation (#19).
- **Research**: built (web search, page reading and Reddit through a local SearXNG); the owner's spoken check
  of a sourced answer and SearXNG in the installer remain (#10, #20).
- **Voice polish**: speaking the first sentence while the rest is generated (#12), interrupting Jarvis by
  voice with echo cancellation (#13), follow-up requests without the wake word (#11), the bare word "Jarvis"
  (#15) and a state indicator in the Omarchy bar (#14).
- **Accessibility** (not filed as issues yet): a text-only way to use Jarvis and a screen-reader check of
  the chat window; see [ACCESSIBILITY.md](../ACCESSIBILITY.md).

Issue #8 is the maintainer's own Spotify developer app setup; everyone who runs Jarvis creates their own.

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
