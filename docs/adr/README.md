# Decisions

Each record states one consequential decision, the options weighed and why this one won. A record is not
edited once accepted; a later decision supersedes it with a new record that links back.

| Record | Decision | Status |
|---|---|---|
| [0001](0001-own-agent-loop-on-strata.md) | Run Jarvis on its own single-agent loop over Strata's API | accepted |
| [0002](0002-typed-operation-policy.md) | Bound every action with typed operations and default deny | accepted |
| [0003](0003-pc-audio-is-not-the-owner.md) | Treat speech heard in PC audio as someone else's | accepted |
| [0004](0004-model-free-path-for-plain-commands.md) | Answer plain commands without the model | accepted; no new shortcuts since 0013 |
| [0005](0005-local-only.md) | Run entirely on this PC, with no hosted backend | accepted |
| [0006](0006-gpl-licence.md) | Publish Jarvis under GPL-3.0-or-later | accepted |
| [0007](0007-outside-services-for-answers.md) | Reach SearXNG's engines, Open-Meteo and Arctic Shift to answer questions | accepted |
| [0008](0008-speak-while-writing-and-barge-in.md) | Speak a reply while the model writes it, and let the owner speak over Jarvis | accepted |
| [0009](0009-lookups-without-a-yes-and-searxng-idle.md) | Settle ADR 0007's open points: read-only lookups need no yes after web text, and SearXNG keeps running while idle | accepted |
| [0010](0010-azure-keyword-wake-word.md) | Offer an Azure custom keyword model as a second wake word, run offline in its own process | accepted |
| [0011](0011-reminders-and-scheduled-requests.md) | Keep reminders and scheduled requests in Jarvis's own jobs file, said by the voice service; a scheduled request runs as the owner's own | accepted |
| [0012](0012-lower-wake-score-while-speaking.md) | Hold the wake word to a lower score while Jarvis speaks | accepted |
| [0013](0013-free-speech-no-new-shortcuts.md) | Understand free speech with typed tools; add no new shortcuts | accepted |
| [0014](0014-model-raises-its-own-thinking.md) | Let the model raise its own thinking within a request | accepted |
| [0015](0015-listen-to-pc-audio.md) | Listen to PC audio for the owner through the wake-word watcher's frames, kept in memory only | accepted |
| [0016](0016-headless-browser.md) | Give Jarvis a headless browser, fenced by a local proxy and a yes for every click and typing | accepted |
| [0017](0017-coding-mode.md) | A coding mode with its own model on the one model server, switched by the owner or by the model with a typed tool | accepted |
| [0018](0018-sing-a-song-the-model-writes.md) | Sing a song the model writes as notes, with a speaking-voice singer and an optional GPU one | proposed |
| [0019](0019-install-and-update.md) | Install and update Jarvis from a release branch | proposed |
| [0020](0020-model-on-another-machine.md) | Optionally serve Strata on another machine owned by the user, with the Jarvis client on the PC it controls | proposed |
| [0021](0021-screen-watch.md) | Watch a window or the screen and say when something shows up, with one check by the same model that runs no call | proposed |
| [0022](0022-tool-router.md) | Let the model route itself through tool groups, off by default until measured | proposed |
| [0024](0024-owner-skills.md) | Read the owner's own skills from Jarvis's config folder | accepted |
| [0025](0025-workspace-folder.md) | Give Jarvis a workspace folder it can always write in, where a new file needs no yes | proposed |
| [0028](0028-voice-and-chat-as-two-channels.md) | Voice and chat as two cooperating channels: a spoken request is its own, and a typed tool hands the owner's words to the open chat | proposed |
