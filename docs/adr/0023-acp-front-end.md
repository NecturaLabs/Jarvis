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
- **Announcements reach it too.** The product rule "replies by voice only; no desktop notifications"
  becomes (owner's change, 2026-11-26): Jarvis speaks its replies and its announcements, and the
  announcements (reminders, watch alerts) also go to the open text front ends, the chat window and T3 Code
  through `jarvis acp`. There are still no desktop notifications. `announce.py` holds the announcements the
  voice service makes (`publish(text, kind)`, a ring of the last 50 in memory, a bounded queue per
  subscriber that drops its oldest line when full, so the voice loop never waits for a reader). The
  control socket gains `subscribe`, which keeps the connection and streams each announcement as one JSON
  line until the client closes (the same owner-only socket as every request, at most 8 at once).
  `jarvis acp` follows it, looking again every ten seconds while Jarvis is not running, and keeps each line
  as "Jarvis: <line>" for every open session. T3 Code v0.1.11 drops assistant message chunks that arrive
  while no `session/prompt` is open (its ACP client shows them only between a prompt's start and end), so a
  line is written first in the session's next reply, or after the reply when it came during one, never
  inside a reply being written, and before a waiting question's answer goes on. It is not seen until the
  owner's next message; ACP gives a client no other way to show a line. A session keeps its latest 50.
- **Shown at once in T3 Code.** T3 Code's command line can start a turn: `t3 thread send <thread> -` ("a busy
  thread gets it as its next turn; nothing is interrupted"). With `acp.push_to_t3` (off until the owner has tried it live and turns it on) `jarvis acp`
  sends each announcement that way into the newest open thread, not settled, of the project a session was
  opened in (`session/new`'s `cwd`) whose provider is a Jarvis ACP provider, found from T3 Code's `settings.json`
  (only `providerInstances`' command path and arguments are read; nothing else of `~/.t3`) or named by
  `acp.t3_provider`. The message is the words and a tag: a keyed hash (a private key file in the state
  folder) of the announcement's number, time and words. The turn reaches a `jarvis acp` as a prompt; one whose
  tag is valid is answered with the words alone, with no model and no tool, and is never the answer to a
  question that waits. A prompt that starts with Jarvis's mark but has no valid tag, or carries more than text, is dropped:
  answered with a fixed refusal and never given to the model (also with the push off); one that merely
  resembles an announcement without the mark is the owner's ordinary input. Two Jarvis processes
  send an announcement once. `t3` is started only with an argument list, the words on standard input, and
  only for `thread list` and `thread send`. Without `t3`, a Jarvis provider or such a thread nothing happens
  and the line waits as before. The voice service does not push: T3 Code's projects are known only to the
  sessions that opened in them.
- **More announcements.** The reply to a request that ran on a schedule is published too (kind `schedule`),
  and the chat window shows each announcement as a dim `Jarvis:` line while it is open.
- **Tool steps are not shown.** The agent's hooks give no step events, so `tool_call` updates are left
  out; the reply is the text.

## Trust and limits of the push

- A valid tag is not single-use: a pushed message sent again is accepted again. All it does is show its own
  words (no model, no tool), so a replay can show an old announcement and nothing more.
- Any local program that can send into a Jarvis thread (another agent running `t3 thread send`, say) types as
  the owner there, the same trust as `jarvis ask`: its words are an ordinary request, and an action that
  needs a yes still asks for it. A `strong` action stays unreachable, because `jarvis acp` has no confirm key.
- `acp.t3_provider` only picks among T3 Code providers whose settings run `jarvis ... acp`. A name that is not
  one of them pushes nothing (`jarvis doctor` says so), so no announcement or model text reaches another
  agent's thread.
- A pushed prompt that arrives while a request of the same session is being written is answered as busy, like
  any other prompt. Its announcement stays queued for the end of the current reply or the next reply.
  Cleanup after `t3 thread send` removes only the original pending copy for that announcement. This
  preserves the copy kept on BUSY even if the prompt arrives before the command returns. A pushed prompt
  between requests is written whole under the session lock. A forged prompt is audited as `acp_refused`
  without its words, including when the session is busy.
- Announcements wait for the push on one worker with a short queue (the oldest is dropped when full), so a
  hung `t3` holds one thread and closing does not wait for it.
- At the 32-session limit, a new session evicts the oldest idle one. `session/close` removes a session.
  Cancel and close stop deferred prompts before they can reserve a run. The protocol writer checks
  stop state under its lock for every chunk, including validation refusals.
- A prompt can answer a confirmation only after Jarvis writes the question. A deferred prompt
  remains a request of its own.

## Consequences

- T3 Code's own approval buttons never appear for Jarvis, and its permission mode changes nothing about
  what Jarvis does.
- An answer that is not a yes is not also treated as a request: "no, do this instead" cancels the action,
  and the owner asks again.
- An announcement reaches T3 Code at once only while a `jarvis acp` runs for a session of that project, a
  `t3` is installed and a Jarvis thread exists; otherwise it is seen after the owner's next message there.
  The voice still says it aloud when it is made.
- The push goes to one thread per project (the newest); the other threads see it as the line queued for their
  next reply only where no thread of the project was pushed to.
- Only what the voice service publishes is announced; a program that wants to announce something calls
  `announce.publish` in that process (the reminders do).
- A client that sends a second prompt while a request runs gets an error, not a queued request.
- Every ACP session builds an agent (a few small clients), so the number of sessions is capped.
- A client that supports only a newer protocol version than 1 is answered with version 1 and decides
  whether to go on.

## Not decided

Showing tool steps (`tool_call` updates) needs step events from the agent; showing a picture Jarvis
fetched needs the client to take an image content block in a reply.

## One provider, two models

The modes are offered as the session's models, not as two providers: `session/new` returns a `model`
select (each mode by the name of the model that serves it, with what it is for) and a `thought_level`
select (`none` to `high`), and also `models.availableModels`, whose `_meta.reasoningEffort` is each mode's
own starting effort (T3 Code reads it from there). `session/set_config_option` changes either while no
request runs. Each mode is its own agent in the session, made when first chosen and kept, so going back
keeps its conversation; a new mode starts at its own effort. A client that lists one section per provider
then shows Jarvis once. `--coding` makes coding mode the first one.

## Coding mode

A session in coding mode is built in coding mode
(`build(..., mode="coding")`): the chat's coding prompt part and `[coding.chat]` over `[chat]`, `coding.tools_off` and
`coding.write_roots`, as the chat gets in coding mode. The first request reloads the configuration and compares
the resolved run-config paths. If coding has its own run config, the request uses the same switch as
`/model coding` and reports the result. If the switch fails, Jarvis reports the failure and tries again
on the next request. Cancellation while waiting for either switch lock skips it. With a shared run
config, the provider keeps the shared mode and budgets for the normal Strata context on local setups.

A coding-mode session holds the shared answering lock while handling a request and resets the idle timer
when it finishes in shared coding mode. The voice service can return Strata to normal after the idle interval.
The session's prompt and tools stay in coding mode. The push finds the newest open thread that runs Jarvis
(`jarvis acp`, or `jarvis acp --coding` as set up before the modes became models).

## Kept sessions (2026-11-26)

The first decision kept no conversation on disk. T3 Code reopens a thread with `session/load` after
either side restarts, and failed with "Provider session failed to open" when Jarvis did not offer it.
So `jarvis acp` now keeps each session's conversation in `~/.local/state/jarvis/acp-sessions/` (folder
0700, files 0600, the newest 200), with the same provenance marks the chat archive keeps, and answers
`session/load` by replaying the remembered turns. Loading raises no turn's trust: the marks are read
back as the chat's resume reads them. A session id with nothing kept opens empty under that id, because
T3 Code treats any error as a failed thread. The chat's privacy settings cover these files too:
`chat.archive_keep = 0` keeps none and deletes those kept before, and `/forget all` in the chat deletes
them, also with `chat.archive_keep = 0`, where sessions kept before would otherwise stay until the
next `jarvis acp` starts. `/clear`, the command the protocol lets a client offer, forgets one session
without the model. When it has to wait for a stopped request or a declined question to end, a prompt
sent after it waits for it too (at most `CANCEL_WAIT_S`) and runs on the cleared session: the two are
ordered by the order they were sent in, not by which thread wakes first.

A running `jarvis acp` still holds its sessions' turns in memory when `/forget all` deletes the files,
and its next reply would write them back. So `forget_kept` first writes the time to a `forgotten` file in
the folder (and puts the earlier time back if the deletion fails, since then nothing was forgotten), and
each session remembers when it was opened or loaded. A session opened before that time forgets what it
remembered before its next request begins (`acp_forgotten` in the audit log) and then keeps its new turns
as a new conversation; a reply that was already being answered when the files were deleted is not
written, and files written while the deletion ran are taken back. A session opened or loaded after
the time is never touched, so no conversation the owner did not forget is lost. The time is a file because
`jarvis acp` and the chat window are separate processes; clearing the agent's memory at once would need
a channel to every running `jarvis acp`, and the check at the start of a request needs none.
