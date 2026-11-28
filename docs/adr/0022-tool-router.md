# ADR: Let the model route itself through tool groups, off by default until measured

**Status:** accepted (2026-11-26); the default stays `all` until the benchmark below is run
**Date:** 2026-11-26

## Context and problem statement

The owner asked for a router that lets Jarvis understand which tools and skills it has and use them as the
requests need. Today every model call carries every tool's definition: about 7,600 tokens of the 24,576-token
window for 48 tools by voice, about 7,700 for 50 in the chat (the repository's own token estimate). That is
room the conversation cannot use, and it grows with every tool added. Strata's server pins the start of the
prompt and reuses what it already read ([ADR 0001](0001-own-agent-loop-on-strata.md), `Agent._step`):
changing the tool list changes the prompt after the system text and costs seconds to read again, so a router
that changes tools freely would trade tokens for time.

## Considered options

1. Keep offering every tool (the status quo).
2. A separate classifier, a small model or a rule that picks tools from the request before the main model
   runs. A model there is a second agent, which Jarvis never has; a rule is a keyword list, which
   [ADR 0013](0013-free-speech-no-new-shortcuts.md) rules out.
3. The same model in the same loop chooses: it reads a toolbox (each group of tools and what it is for) and
   asks for groups with one typed tool, `use_tools`; the system offers only the core tools until it does.

## Decision outcome

Chosen option 3, as a setting: `tools = "all" | "routed"` in `[agent]` and `[chat]`, default `all`.

- **Groups** (`src/jarvis/toolgroups.py`). Every operation belongs to exactly one group: `music`, `sound`,
  `screen`, `apps`, `system`, `files`, `everyday`, `web`, `browser`, `schedule`, `guides` and `jarvis`, plus
  the **core**. A group has a one-line summary written from what its tools do. Membership is one list of
  names in one place and a test fails for an operation no group names, so a new tool cannot be left out.
  Nothing matches the owner's words against a group: the model reads the summaries and decides.
- **Core** is what every request may need: `no_action`, `stay_quiet` and `chat_hand_off` (how Jarvis
  answers; the hand-off only while a chat window is open, [ADR 0028](0028-voice-and-chat-as-two-channels.md)),
  `think_harder` and `switch_model` (ways to go on; `switch_model` must be the first and only call of its reply, so a
  detour through `use_tools` would only cost a call) and `use_tools`. Everything else is a group.
- **The toolbox** (`Registry.groups()`, `router.toolbox_section`): in the system prompt of a routed agent, each
  group, its summary and its tool names, then Jarvis's own skills and Omarchy's guides as before (their tools
  are the `guides` group). It is generated from the registry, so a tool switched off or not installed is not
  in it. It costs about 730 tokens by voice and 790 in the chat (estimate), and is left out when every tool is
  offered anyway: there it would only take room from the remembered conversation, whose defaults the
  window tests pin.
- **Routing.** A routed request's first call offers the core tools alone (about 1,000 tokens, against 7,600).
  `use_tools(groups)` is a `safe` operation with no side effect, classified by the policy, audited (`tool_call`,
  then `tools_loaded`) and open to `boundaries.classes` like any other. The agent offers the named groups
  from the next call to the end of the request, always as the core tools, then the loaded groups in the
  registry's order, whatever order they were asked for in, so calls with the same groups share their whole
  start. The call that may only answer (after a repeated call, or on the last step) is offered the tools an
  ordinary call would have then, the core and the groups loaded so far, so the server keeps the prompt it pinned. A request that
  continues the remembered conversation starts with the groups the last one had loaded (amended 2026-11-28, below); a new or
  forgotten conversation, and a request heard from PC audio, start with the core.
- **Enforcement.** A call to a tool of a group that was not offered when the call was written is not run, even
  in the same reply as the `use_tools` that asks for it. The model is told which group to ask for, and the
  refusal is audited (`tool_not_offered`). A plain command matched in code (ADR 0004, the owner's own words, no model call) runs as in `all` mode and
  is never refused for its group, and a refused call is no action tried: the model may still ask for the group,
  call the tool or switch modes. Loading never lowers a check: a request heard from PC audio that
  loads `apps` still gets the same verdict for `stop_process` as before. Without `use_tools` in the registry,
  `routed` offers everything.
- **Argument type.** The tool's `groups` is the first array argument: `validate_args` accepts a list of
  scalars checked against `items` and `maxItems`.

### Why the default stays `all`

Routing saves about 6,600 tokens a call and costs a re-read: the first call after `use_tools` has new tools in
front of the conversation, so the server reads the prompt after the system text again, and `use_tools` is an
extra model call. Whether that is faster, and whether the model asks for the right groups as often as it picks
the right tool from all of them, is a measurement, not an argument. `routed` becomes the default only when a
live run of the same requests as the model benchmark ([models.md](../models.md)) shows, in `all` and `routed`:

- the same accuracy (right tool, right arguments, no wrong action), across the cases that need two groups and
  those that need none; and
- no more seconds per request, prompt re-read included, for speech and for the chat.

Until then `routed` is an opt-in, and the memory defaults are sized for `all`: the room kept for the remembered
conversation counts every tool definition and the toolbox, the request that loads every group, so a routed chat
remembers a little less by default (about 4,500 tokens, with every optional tool on, when it was capped at 5,700). Making room
inside a request counts only what that call is offered.

### Consequences

- A group ordered by its first tool in the registry puts `music` before `web`, so asking for both in any order
  gives the same tools in the same order. If a benchmark shows requests mostly add groups one by one,
  appending in the order asked would share more of the previous call; the cost is a second order for the same
  set.
- A conversation remembered with `memory_detail = "full"` can show a past call to a tool the new request has
  not been offered; calling it is refused with the hint, which costs a call. Since the amendment below this
  happens only after a new or forgotten conversation.
- Owner-written skills and Omarchy's guides stay in the `guides` group and the skills list; the router never
  reads anything outside Jarvis's own files and Omarchy's guides.

## Amendment (2026-11-28): the groups stay for the conversation

In coding mode a follow-up ("keep going", "try again") started with the core tools, though the request before
it had just used the file tools. The model called `write_file` from memory, was refused, and spent a call
(30 to 40 seconds with the coding model) loading the group again, three times in one evening's log; once the
refused call ended the request with no reply. A request that continues the remembered conversation now
starts with the groups the last remembered request had loaded (audited as `tools_kept`). A new or forgotten
conversation (`/clear`, the idle time, a restart) starts with the core again, and so does a request heard
from PC audio, which reads no history; what such a request loads is not kept. Groups decide only what is
offered: every call is still classified as before, so a heard request still gets the same verdicts.

## Measurement (2026-11-28)

Keep `all` as the default for voice and chat. The synthetic benchmark on Qwen3.8-Flash-Next
IQ2_XS had fewer prompt tokens per call with routing and a longer median request in both
channels. See [the measured results](../models.md#router-benchmark) for accuracy, latency and
the scope of the run. Routing remains available for owners who value a smaller prompt.
