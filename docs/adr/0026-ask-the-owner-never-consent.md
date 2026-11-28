# ADR: Let the model put a choice to the owner in the chat, with a timer; an answer is never consent

**Status:** accepted (2026-11-26)
**Date:** 2026-11-26

## Context and problem statement

In the chat the model sometimes needs a decision it cannot make for the owner. Until now it stopped and
asked in its reply, and nothing moved on if the owner was away. The owner asked for a selection picker, a todo
list that stays current, work that goes on while a question waits, and a timer that sends Jarvis on with what
it judges best after five minutes without an answer.

A question the model writes must not become a way to act without the owner's yes. Consent in Jarvis is a
yes to one exact action ([ADR 0002](0002-typed-operation-policy.md)), asked by code. A model that can write
the options, and the words around them, could otherwise word a question so that a pick, or a silence, reads
as a yes.

## Considered options

1. Let `ask_owner` stand for a confirmation when its options say so, and refuse it when they do not.
2. Offer the question and its answer to the model, and keep confirmations on their own path, so that nothing
   on the question's path can reach a confirmation.
3. No question tool: the model asks in its reply, as before.

## Decision outcome

Chosen option 2.

- **Two typed operations, chat only**: `todo_write` and `ask_owner`, both `safe` and audited. Neither is
  offered by voice, in `jarvis ask` or in `jarvis acp` (`CHAT_WINDOW_ONLY`). They add schema to every chat request,
  so their descriptions are short, and the chat's remembered conversation still keeps its floor.
- **The recommended option is required.** `recommended` is part of the arguments; the picker lists that option
  first, marked "(Recommended)", and at a timeout the model is told to go on with it unless it judges
  another better.
- **Answers cannot be consent, by structure rather than by wording.** An answer, picked or timed out, reaches the
  model in one of two ways: as the result of a call that waits, or as a `Notice` that joins the running request
  (`Hooks.notices`), or starts one. It is never given to the confirmation flow: that flow is a separate question
  the window asks (the yes-or-no choice), bound to one exact action, and only that choice answers it. The picker
  and the yes-or-no choice are different widgets with different keys; while an action is asked about, the picker
  waits behind it. No test or code reads the words of a question or an option for what it means; the model decides
  when to use the tool from its description.
- **The note's origin is not the owner's.** The timeout's "go on with what you judge best" enters as Jarvis's own
  status text (`Origin.TOOL_STATUS`), never as typed words. A pick is returned as "the owner picked: <label>",
  with the label (model-written) shown as data: angle brackets spelled out, one line, no control characters. Neither
  is added to the owner's own words (`Request.words`), so a web address in an option the owner picked is not one
  they wrote and a page still cannot be opened that way. What the owner types under Other is theirs and is added
  (`Notice.words`), so an address typed there can be opened.
- **A request begun by a note.** When no request runs, a pick, a timeout or Esc starts one (`handle(noted=...)`).
  Its words are the note's, none of them the owner's (but for the typed text). A request begun by a timeout or
  Esc is not the owner's own: the operations that act only on the owner's own request (`pc_shell`, `open_app`,
  window changes, screenshots, stopping a process) refuse it (`Request.owner_asked`), as they refuse PC audio.
  One begun by a pick counts as the owner's, since they chose. The note is not shown as something the owner said,
  in the window, the log or a resumed conversation.
- **The timer waits for a confirmation.** While the window asks yes or no about an action, the question's
  countdown stands still and resumes where it was, so a timeout cannot fire while the owner is deciding something
  else, and cannot be used to run out the clock on a question about an action.
- **Not blocking.** With `wait` false (the default) the call returns at once and the model carries on with what
  does not depend on the answer. With `wait` true the request pauses on the question, and the time waited is not
  counted against the request's limit (`Request.waited`) or its steps.
- **One question at a time**, refused for a request heard in PC audio or run by a schedule (nobody is there to
  answer), and dropped without a word by `/clear`, `/resume` and when the window closes.
- **Esc closes the picker**; the model is told that, as it is told of a timeout, to go on with its pick. A second
  Esc stops the request, and so does Ctrl+C while a request waits on a question (also with no timeout).
- **Written after outside text.** A question asked after the request read outside text (a page, a file) is marked
  in the picker, since a page may have steered its words. It is not refused: choosing between results found on
  the web is a fair use. Its answer is still never a yes.
- The todo list belongs to the conversation: kept by `/compact`, emptied by `/clear` and `/resume`.

## Consequences

- **Good:** the model can ask and go on; an unanswered question cannot stall a request for ever; consent stays one
  exact action's yes.
- **Good:** nothing here depends on matching the owner's or the model's phrasing ([ADR 0013](0013-free-speech-no-new-shortcuts.md)).
- **Bad:** the picker is modal: while a question is open the input is hidden, so a message is typed only after
  an answer or Esc (the owner types under `Other`).
- **Bad:** the two tools add about 380 tokens to every chat request, kept short so the chat memory stays above its floor.
- **Not measured:** how well the model uses the tools unprompted, and how often it asks when it should decide.

## Related

- `src/jarvis/askowner.py`, `src/jarvis/todos.py`, `src/jarvis/agent.py` (`Hooks.notices`),
  `src/jarvis/chat/app.py`
- [ADR 0002](0002-typed-operation-policy.md), [ADR 0013](0013-free-speech-no-new-shortcuts.md),
  [ADR 0028](0028-voice-and-chat-as-two-channels.md)
