# ADR: A coding mode with its own model, switched by the owner or the model

**Status:** accepted
**Date:** 2026-11-26

## Context and problem statement

Jarvis is also used for programming, mostly in the chat. The everyday setup is sized for quick spoken
commands: short replies, no thinking, a small quantised model chosen for fast tool use beside a game. A
request to build a program can run into the reply and step limits. The goal is a coding mode on the best
programming model Strata can run on the PC, set up for programming, that Jarvis moves to by itself when a
request is programming work, with a manual switch as well. Three rules bound
it: one model in memory at a time beside a game, nothing model-specific in the code, and no list of phrases
that mean "programming".

## Considered options

1. Raise the everyday limits and keep one model and one setup.
2. A second model server for the coding model, on its own port.
3. One model server and a mode: the server's profile is rewritten for the mode and the server restarted.
4. For who switches: only the owner; a keyword list; or a typed tool the model calls.

## Decision outcome

Option 3, with the owner's switch and a typed tool.

- **The mode** is `normal` or `coding`, kept in a state file every Jarvis process reads
  (`~/.local/state/jarvis/mode.json`), so the voice service, each chat window and `jarvis model` agree. An
  agent looks at it when a request starts and works with that mode's prompt, tools and limits.
- **What coding mode changes** is laid over the usual settings, never a copy of them: `[coding.chat]` over
  `[chat]` (60 model calls, 8,192 reply tokens, `medium` thinking, 20,000 tokens of conversation) and
  `[coding.voice]` over `[agent]`; a part on programming work added to the prompt; `coding.write_roots` and
  `coding.tools_off`; and for the model server `strata.coding_run_config` (Strata's own run config for the
  coding model) and `coding.context`. Without a coding run config the mode keeps the model and changes only
  the rest.
- **The switch** (`modes.Switcher`) holds a lock, checks that the mode's run config can be served, writes
  the model server's profile and unit, and restarts the server only when either changed. The old engine has
  ended before the new one starts, so two models are never in memory together. It then waits for the server
  and loads the model. If the server does not come back, the previous profile is restored. Every switch is
  audited (`mode_switched`: from, to, by whom, the model, the seconds). A switch is refused while any
  Jarvis process answers a request: requests are held off first (the shared lock below), then the voice
  service is asked whether it is hearing or saying one. Asking for the mode Jarvis is already in changes
  nothing, unless a switch was killed midway. The switch notes the mode it is going to in the state file
  before it touches the files and clears the note when it ends, so a note still there (or files that are not
  the mode's) means the server may run either mode: the next switch then restarts it whatever the files say.
  `jarvis setup` and `jarvis restart` write the same files under the switch's lock; a restart of the
  server by `jarvis restart` is the owner's own, so it is made even over a request and says that it cut
  one; while no request is being answered, none begins until the server is back. A switch into coding mode
  loads its model even while Jarvis is paused, as the work follows at
  once; `jarvis unload` frees it.
- **The owner is told** before a restart in one line, with the seconds the model took to load the last time
  once that is known: spoken for a spoken request, shown in the chat.
- **The model switches itself** with `switch_model`, a typed tool like `think_harder`
  ([ADR 0014](0014-model-raises-its-own-thinking.md)): its description says what programming work is, the
  model decides, and no phrase is matched ([ADR 0013](0013-free-speech-no-new-shortcuts.md)). The agent
  then makes the switch and answers the request again in the new mode, with that mode's prompt, limits and
  time; words typed into it meanwhile go with it. It counts as an action, so a request steered by
  outside text needs the owner's yes for it (with `bypass_consent` on it runs unasked, like every action).
  A request heard in PC audio never switches and is not asked about it, and neither does a scheduled
  one, whatever `coding.voice.auto` says: nobody is there to be told or to wait for the load. What the
  request read before the switch stays read: outside text or the owner's own data seen then still puts the
  actions after the switch behind a yes. A request switches once, only by a
  call that stands alone in the model's reply (beside other calls none of them runs; a reply the
  length limit ended counts as holding more), not after one of its
  actions was tried, whether it succeeded or not (answering again could run it twice), not once it was
  stopped, and not to a mode whose model cannot see a picture it came with. A switch that is refused or
  cannot be made is the tool's result, and the request goes on in the mode it is in. In coding mode the
  same tool goes back when the owner says the work is over.
- **Going back**: `jarvis model normal`, `/model normal`, the model's own call, or after
  `coding.back_after_s` (30 minutes) without a request in coding mode. The voice service looks once a minute
  while idle; a chat, `jarvis ask` and the voice look when a request starts, and tell the owner in a line
  before the model loads. A request in flight anywhere holds the return off, and the idle time is
  read again once the switch holds its locks, so a request that ended a moment before keeps coding mode. A
  running chat keeps the
  `[coding]` values it started with; `/model` reads the config anew.
- **The chat and the voice differ.** In the chat the model may switch by itself (`coding.chat.auto`, on):
  programming is asked for there, the work is long, and a line in the window covers the load. For spoken,
  `jarvis ask` and scheduled requests it may not (`coding.voice.auto`, off): a load in the middle of a spoken
  request is a long silence, a misheard request would cost one, and every spoken request after it would be
  answered by a model that is weaker outside code. Such a request in coding mode stays a quick command with
  its short memory; the overlay only gives it room to write a file.
- **The model** is the owner's choice of run config. The one chosen for the reference machine and why is in
  [models.md](../models.md#the-coding-model).

Option 1 was not taken: thinking and long replies make every spoken request slower, and the stronger coding
model is weaker at everything else. Option 2 would hold two models' RAM and VRAM beside a game. A keyword
list is ruled out by ADR 0013.

### Consequences

- Good: programming work gets the model and the room it needs without the owner setting anything per request.
- Good: one server, one profile, one model in memory; the usual model and its settings are untouched.
- Bad: a switch between two models costs a model load each way, and the system file cache cannot hold both
  models on a 64 GB PC, so a load probably reads the disk (not measured).
- Good: a switch does not cut a request, by design (tested with fakes, not with a real server). Every Jarvis
  process (the voice service, a chat window, `jarvis ask`)
  holds a shared lock on one small file while it answers a request, and a switch is made only while it can
  take that lock alone; a request that begins during a switch waits until the model server is back. A
  request's own switch (`switch_model`) lets its own hold go for it and takes it back before another switch
  can begin, so the request goes on in the mode it switched to. A summary (`/compact`) and the prompt read
  ahead at start hold the lock too; a request's own switch is refused as busy while one of them runs beside
  it in the same process.
- Bad: the model can call a request programming work when it is not; `coding.chat.auto = false` leaves the
  switch to the owner.
- Once the headless browser tool of ADR 0016 is in, it can show the model a page it built, from a project
  folder inside the file write roots.

### Confirmation

`tests/test_modes.py` (the overlay, the state, the switch with a fake unit and server: restart only on a
change, the refusal while busy or while another process answers a request, a config that cannot be served, the
way back when the server stays down,
idle return), `tests/test_coding_mode.py` (the agent answering again in the new mode, the refusals,
following a switch made elsewhere, `jarvis model`, the doctor) and `tests/test_chat.py` (`/model`). Not
measured: the coding model on the reference machine, and how often the model calls the tool when it should.
