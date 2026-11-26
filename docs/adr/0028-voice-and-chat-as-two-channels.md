# ADR: Voice and chat as two cooperating channels, with a typed hand-off between them

**Status:** proposed
**Date:** 2026-11-26

## Context and problem statement

Jarvis has two ways in: the voice service (`jarvis run`) and the chat window (`jarvis chat`). They are two
processes with their own agent, memory, limits and tools, over one Strata server. Until now the chat followed the
whole audit log, so every spoken request showed in an open chat, and nothing let the
owner pass spoken words to the chat's work. Two owner rules apply: a spoken
request such as playing music should just be done, whether or not a chat is open, and not show there; and
whatever the owner asks in the chat is routed there. Neither channel may freeze or cancel the other's work
unless the owner's words call for it (a stop, a pause, a mode switch).

AGENTS.md says "one agent does all the work; never add a subagent, delegation or second-agent tool". How can the
voice pass words to the chat without breaking that?

## Considered options

1. Leave both channels sharing the log, and show every spoken request in the chat.
2. A tool the model calls with text it writes for the chat (`chat_hand_off(text)`).
3. A tool with no arguments: the code forwards the voice request's own transcript to the open chat.
4. Let the chat read the voice agent's conversation, or merge the two conversations.

## Decision outcome

Chosen option 3, with these parts.

- **Spoken requests stay the voice agent's own.** The voice service tags every audit event it writes with
  `via` `voice`; the chat's follower leaves those out. Typed requests, `jarvis ask` and a window's own work show
  as before. Nothing about the voice agent's tools, memory or boundaries changes.
- **`chat_hand_off`** is a typed operation of the voice channel, class `safe`, offered only while a chat window
  answers on its socket (`Operation.offered`; the schema is the voice agent's own tool list, so it changes when a
  window opens or closes). Its description says when to use it: the owner asks for something to be done in or added
  to the chat's ongoing work or conversation. The model decides from that description; Jarvis has no phrase list.
- **It takes no arguments.** The code sends the request's own words, `Request.words`: what the owner said,
  never text the model wrote. It is refused for a request heard in PC audio, one run by a schedule, and one that
  read outside text this request (remembered text from earlier turns does not count), so a page cannot make the
  voice pass words to the chat. It is also refused when the message holds other calls or the request already
  tried an action, since the owner's words would carry that other work into the chat too. It counts as the
  request's action. It is audited like any call; the window audits `chat_handoff_received` before the request
  starts (and the request carries the source "spoken") and `chat_handoff_refused` for each refusal.
- **The voice request ends** after a successful hand-off with one short spoken line (`Result.ends`): the work
  is the chat's now, and no second model call is spent on it. A hand-off that was refused or never reached a window
  ends nothing: the model is told and answers itself. One that was sent and not answered in 5 seconds is an
  uncertain outcome (`UncertainOutcomeError`): the model is told not to retry, so the words are neither sent twice
  nor done by the voice as well. With two windows open the spoken line says the newest took them.
- **Delivery** is a Unix socket per chat window in the runtime folder (folders 0700, socket 0600). The window
  takes words only from the voice service process itself: the peer's PID (`SO_PEERCRED`) must be the one holding
  the control socket's lock, which the kernel lists in `/proc/locks`, so a program the voice service started
  (`pc_shell`, an app) cannot send words. The newest window that answers gets them. Whether a window is open
  is looked at at most every 3 seconds and with a 0.2 s probe, so a window that is stuck never stalls the voice
  loop; a socket nobody answers on is skipped and never removed by the sender (a window removes its own stale
  one when it starts).
- **The chat receives them as the owner's own typed input**, marked "spoken": they join the running request
  before its next model call (steering) or queue behind it when `chat.while_working` is `queue`, or start a
  request when the window is idle. Typing or speaking never pauses the work in progress.
- **No cutting.** Strata has one slot (`parallel` 1), so a typed and a spoken request share it; no Jarvis
  request holds it for its whole run, only each model call occupies it. The order Strata serves waiting calls in
  has not been measured. A cross-process gate was drafted and dropped: nothing measured justifies it. The two
  waits that were not the owner's words are removed. A restart of the voice service that the chat's `/config`
  asks for is made while no voice request can begin, or, with one under way, left as a note
  (`voice-restart-pending`) that the voice service acts on itself once idle, also when the window has closed.
  `jarvis pause` unloads the model only while holding off every request that would begin, once none is answering
  in any process; a chat request, or a question it waits for the owner to answer, therefore keeps the model
  loaded until it ends (Esc or an answer ends it), and the chat's status line says so.

### Against "never add delegation"

The rule exists so that one agent holds the policy, the memory and the audit trail of a request, and no agent
can widen another's powers. The hand-off keeps that. There is still one agent per request: the voice agent ends
its request when it calls the tool and does not wait for, read or steer the chat agent. The chat agent is not
asked to do a subtask and report back; it receives the owner's own transcribed words as if they had been typed,
and handles them with its own tools, its own boundaries (`boundaries.classes`, the action cap, consent) and its
own audit, exactly as for typing. The model cannot put its own words into the chat, and outside text cannot reach
the tool; no program the voice service started can reach the window either. What the hand-off adds is routing the owner's words, which is what the owner asked for.

## Consequences

- Spoken requests are unseen in the chat: the owner reads their audit through `jarvis logs` or the audit file.
- When the chat opens or closes the voice agent's tool list changes, and Strata reads its pinned prompt again
  once (not measured).
- Without a voice service nothing sends a hand-off; without a chat window the tool is not offered.
- The shared settings (`boundaries`, `[spotify]`, `[files]`, `[persona]`, `[model]`, `[strata]`) are shared on
  purpose and stay so ([configuration.md](../configuration.md#voice-and-chat-which-settings-are-each-channels-own)).
