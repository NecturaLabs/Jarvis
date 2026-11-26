# 0023. Serve Jarvis to coding apps over the Agent Client Protocol; consent stays Jarvis's own

Status: proposed  
Date: 2026-11-26

## Context

The owner wants to talk to Jarvis from T3 Code, which takes any agent that speaks the Agent Client
Protocol (ACP) over standard input and output (Settings, Providers, Add provider, Local ACP command).
T3 Code starts the program it is given, so a second Jarvis process runs beside the voice service. Jarvis
must still be one agent: no second model, no second agent, nothing spoken for a typed request, and the
boundaries in code ([ADR 0002](0002-typed-operation-policy.md)) unchanged.

T3 Code answers ACP's `session/request_permission` by itself in its default Full access mode, and in
Auto modes too. Its answer says nothing about what the owner wants.

## Decision

Add `jarvis acp`, a front end to the agent `jarvis chat` already has.

- **One agent, as the chat has it.** Each ACP session is `build(via="acp")`: the chat's written replies
  and limits, its own conversation in memory (none kept on disk, and the chat's conversation file is left
  alone), and the one model Strata serves, reached over HTTP as the chat reaches it. Nothing loads a
  second model. No sound is made, and the tools that show a picture in the chat window are not offered,
  because the client has no place for them. Jarvis loads no MCP servers, so a session's `mcpServers` and
  `cwd` are ignored. Requests have the chat's provenance: the owner's own typed words. A picture block is
  decoded under a size limit and checked like any picture, and what it shows counts as outside data.
- **A small hand-written JSON-RPC 2.0 loop** (`acp.py`), one message a line, for `initialize`,
  `session/new`, `session/prompt` and the `session/cancel` notification. It does not add the
  `agent-client-protocol` package: the part of the protocol used is small, a new dependency would need its
  own licence, lock and advisory work, and the loop needs control of threads (a cancel is read while a
  request runs). Standard output carries only protocol JSON: the command keeps private copies of standard
  input and output and points descriptors 0 and 1 elsewhere, so a program a tool starts cannot take or
  forge protocol lines.
- **The reply streams** as `agent_message_chunk` updates, taken from the same draft the chat window shows,
  and the prompt ends with `end_turn`, `cancelled` or, for a prompt Jarvis will not read (a picture too
  big or not a picture, an unsupported block), `refusal` with the reason written first. A chunk cannot be
  taken back, so a later message that does not continue the draft follows it, set apart.
- **Consent stays Jarvis's own.** `session/request_permission` is never sent, and a client's answer is
  never read, so no client can stand in for the owner. A question about one exact action
  (`Hooks.confirm`) is written into the reply and the prompt ends. The owner's next prompt is the answer:
  only a clear yes (`is_yes`) runs the action, anything else cancels it, and that prompt is not a new
  request. A cancel, a closed connection or five minutes without an answer is a no. A `strong` action still
  needs the confirm key, which `jarvis acp` cannot press, so it is refused as in `jarvis ask`.
- **Tool steps are not shown.** The agent's hooks give no step events, so `tool_call` updates are left
  out; the reply is the text.

## Consequences

- T3 Code's own approval buttons never appear for Jarvis, and its permission mode changes nothing about
  what Jarvis does.
- An answer that is not a yes is not also treated as a request: "no, do this instead" cancels the action,
  and the owner asks again.
- A client that sends a second prompt while a request runs gets an error, not a queued request.
- Every ACP session builds an agent (a few small clients), so the number of sessions is capped.
- A client that supports only a newer protocol version than 1 is answered with version 1 and decides
  whether to go on.

## Not decided

Showing tool steps (`tool_call` updates) needs step events from the agent; showing a picture Jarvis
fetched needs the client to take an image content block in a reply.
