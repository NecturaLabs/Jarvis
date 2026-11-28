# ADR: One conversation across voice and chat, each answered the way it was asked

**Status:** accepted (2026-11-28)
**Date:** 2026-11-28
**Amends:** [ADR 0028](0028-voice-and-chat-as-two-channels.md) (its "spoken requests stay the voice agent's
own" part)

## Context and problem statement

ADR 0028 kept the two channels apart: a spoken request was done and spoken, never shown in an open chat
window, and neither agent knew what the other had been asked. The owner reversed that on 2026-11-28: voice and
chat should be one conversation, spoken requests should show in the chat, and a typed message should get a text
reply, as a spoken one gets a spoken reply; configurable, so the split can come back.

Three ways were weighed with the owner: show spoken requests in the chat only; share the conversation as well;
or one agent behind both, the chat window sending typed text to the voice service's agent. The owner chose the
shared conversation.

## Decision outcome

- **`[channels] shared`**, on by default. Off is ADR 0028 as it was.
- **Two agents still**, each with its own tools, limits, thinking and memory budget (the chat's coding mode,
  browser and pictures stay the chat's). Each answers the way it was asked: the voice service aloud, the chat
  window in text. Nothing is delegated: an agent only reads what the owner asked of the other, and its reply,
  before its own next request.
- **The shared file** (`shared.py`, `~/.local/state/jarvis/shared-turns.jsonl`, 0600 in a 0700 folder, with a
  lock file beside it) holds the newest 40 turns of the two: the request, the reply (4,000 characters each at
  most), what was done, what was looked up, where its outside text came from (`Turn.outside`) and whether it
  read the owner's files (`Turn.private`). Not the tool calls and results a turn kept, and never a request heard
  in PC audio or run by a schedule.
- **Before each request** (not one heard in PC audio, which reads no history, nor a scheduled one, which runs
  with the talk it was asked in; the turns wait for the owner's next request), an agent adds the other channel's
  turns since its last request to its own conversation, marked "(said aloud)" or "(typed in the chat)", with
  their outside and private marks. So a page read in the chat makes an action by voice ask for a yes, exactly as
  a page the voice itself read earlier would (remembered outside text). The audit records `shared_turns_taken`.
- **Only the turns of the last 15 minutes** (the voice conversation's idle time) are read, by a process that
  starts and by one that sat idle, so a window opened later picks up the current talk but not an old one. A chat
  window that reopens its saved conversation starts after the turns that conversation already took (its
  `shared_mark`), and a file that was removed and written again is read from its first turn (each row names
  the file's `epoch`). Older rows share a fixed file identity, which the first write carries forward
  so it preserves existing read positions.
- **The chat window shows spoken requests and their replies** (marked "by voice" and "spoken"), and the calls
  the voice made for them, but never a request heard in PC audio, its work or its reply, as it shows `jarvis ask` and ACP work. Only the voice service and a chat window share;
  `jarvis ask`, ACP sessions and speech tests keep their own conversation.
- **`chat_hand_off` stays**: it still routes spoken words to the chat's own work. The voice does not share the
  hand-off turn; the chat shares the words it was handed, marked "(said aloud)".

## Consequences

- "What did I just ask you?" works across the two; each channel's memory budget and summary hold the other's
  turns too, so the other channel's talk takes room in each.
- Forgetting (`forget_conversation`, `/clear`) clears the channel that was asked and the shared file, so a
  window opened later does not read what was forgotten; turns the other channel already took stay in its
  conversation until it forgets or its idle time runs out.
- Typing in the chat keeps the voice conversation from expiring, since a taken turn counts as activity.
- With several chat windows open, the voice reads the turns of all of them, and each window reads the voice's.
- The voice's tool list, prompt and policy are unchanged; the hand-off's rules (ADR 0028) still hold.
