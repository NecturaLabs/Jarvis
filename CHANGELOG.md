# Changelog

Notable changes to Jarvis, in the [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format.
Jarvis has no tagged releases yet; everything below is unreleased. Once releases start, this project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Install Jarvis with `install.sh` from a terminal or checkout. It checks prerequisites, uses the
  locked environment, fetches the default wake word, VAD and Kokoro files with SHA-256 verification
  (`tools/get_models.py`, which you can also run yourself),
  and starts the user services when requested.
- Preview and apply release updates with `jarvis update`. It fast-forwards a clean checkout, defers
  while the voice service is busy and restores the previous version if validation or startup fails. A daily
  user timer is available with `jarvis update --auto on` ([ADR 0019](docs/adr/0019-install-and-update.md)).
- The headless browser is required: the installer stops without agent-browser and has it download its
  Chrome, and `jarvis doctor` fails a missing agent-browser or Chrome while the browser is switched on.
- Jarvis handles your windows on Hyprland: `pc_windows` lists them, `pc_window` shows one, moves it to
  a workspace, or turns fullscreen or floating on or off, and `pc_window_close` closes one after
  asking. Each action is checked afterwards, because Hyprland can answer "ok" without having done it,
  and acts only on a window list the model has already read. A fullscreen window is never moved,
  resized, toggled or closed, and nothing is moved or opened beside it (`open_app` included, which
  also waits while a fullscreen window is on screen), since a fullscreen game can crash when that
  happens.
- `pc_screenshot` takes a picture of the screen, a workspace or one window. A window or workspace
  that is not on screen is captured off-screen without switching to it. The picture is saved where
  Omarchy keeps screenshots, copied to the clipboard and shown in the chat.
- Jarvis sees its own screenshots: where its model can see pictures, `pc_screenshot` also hands it the
  picture, as outside data (a side effect after it needs a yes, like after a shared picture). The
  conversation keeps the picture's name, not the picture.
- Screen watches: ask Jarvis to tell you when something shows up in a window, a workspace or the screen
  and `pc_watch` checks now and then (`every_s`, default 20, 10 to 300; `for_min`, default 120, up to
  720; `repeat`), said by voice in a fixed line built from what you asked for, never words read off the
  screen. `pc_watches` lists and `pc_watch_stop` stops. Two at most, only in `jarvis run`, only
  between requests (a check under way gives way to the wake word), and each check is one model call that
  uses the GPU
  ([ADR 0021](docs/adr/0021-screen-watch.md), proposed).
- Every `jarvis` command answers a script or an AI agent in one line of JSON with `--agent` (or
  `--agent-format`, or `JARVIS_AGENT_FORMAT=1`): `command`, `ok` and `exit`, then its own keys (`doctor`,
  `status`, `tools`, `speed`, `consent`) or its output lines, even for a wrong argument or a crash. Interactive
  commands refuse it. The `jarvis-cli` skill tells coding agents how to read and keep it.
- Set up Jarvis without Spotify. Tools and background Spotify refresh stay off until a client ID is set.

- A headless browser for pages `web_read` can't read or that must be used (a search box, a button), through
  agent-browser when it is installed: `browser_open` opens a search result, an address you wrote or a link the
  page shows, and reads it, less the text and elements the page hides (best effort); `browser_screenshot`
  shows the chat a picture of it; `browser_act` clicks or types there, and always asks first, even with
  questions turned off and with `jarvis ask --yes`, which asks it on the terminal. It runs headless and muted with a throwaway profile, every connection goes through a
  loopback proxy that reaches only public addresses, and it closes when idle. On in the chat (`chat.browser`), off for speech (`agent.browser`); `[browser] idle_close_s`.
  The chat's `/config` sets all three.
  A preview session for the owner's own builds (one project folder, served read-only on loopback, with a
  proxy that reaches that port alone) is ready for coding mode to use.
- Coding mode, for programming work ([ADR 0017](docs/adr/0017-coding-mode.md)): more model calls, longer
  replies, more thinking and a larger context, a part on programming in the prompt, and a model of its own
  when `strata.coding_run_config` names one (`jarvis setup --coding-run-config`). `jarvis model` shows the mode
  and each mode's model, `jarvis model coding` or `normal` switches, and so does `/model` in the chat. In the
  chat the model moves a request that is programming work to coding mode by itself (`switch_model`,
  `coding.chat.auto`), and back when you say the work is over; Jarvis also goes back after 30 minutes unused
  (`coding.back_after_s`). With two models the model server restarts for a switch, so one model is in
  memory at a time, and Jarvis says first how long the load took the last time. Every switch is audited,
  and `jarvis doctor` checks the coding model's files; a missing model file is named for the mode Jarvis is
  in. Settings: `[coding]`, `[coding.chat]` (laid over `[chat]`) and
  `[coding.voice]` (laid over `[agent]`).
- A mode switch waits for requests: none is made while the voice service, a chat window or `jarvis ask`
  answers one, and a request that begins during a switch waits for the model server. The model's own switch
  is made only by a call that stands alone in its reply, before any action of the request was tried, and
  one that cannot be made leaves the request going on in the mode it is in.
  A scheduled request never switches, and what a request read before its switch (outside text, your own
  data) still puts the actions after it behind a yes. A switch that was killed midway is made whole by the
  next one, and `jarvis setup` and `jarvis restart` no longer write the model server's files in the middle
  of a switch; `jarvis restart` says when its restart of the model server cut a request.
- `/config` in the chat has a "Mode" section: how long coding mode may go unused before Jarvis leaves it
  (`coding.back_after_s`), and whether the model switches the mode by itself in the chat (`coding.chat.auto`)
  and for speech (`coding.voice.auto`, asked about before it is turned on).
- What the chat changed since it started (`/effort`, `/format`, `/config`) is kept when the mode changes, and
  `/format` in coding mode keeps the prompt's part on programming. Coding mode still starts at its own
  `coding.chat` values where it sets them, such as `thinking`.
- The chat keeps every conversation in an archive (`~/.local/state/jarvis/chat-archive/`, 0600, the newest
  `chat.archive_keep` = 50) with its full log of requests, steps, results and replies. `/resume` lists them and
  `/resume 2` continues one, with what came from outside text still treated as data; `/rename` names the
  current one, and `/forget all` deletes the archive and forgets the conversation in the window, so none of it
  is archived again. A conversation is written only once something happened in it, never while a request
  works on it: closing the chat mid-request archives the stopped request once the agent is done, and opening
  and closing the chat writes no copy. Listing and the cap read only a small file per conversation, and the
  log is added to, not rewritten. With `archive_keep = 0` the chat deletes the conversations kept before.
- `chat.response_format` says how the chat's replies are written: `concise` (the default), `normal` or
  `verbose`. `/format verbose` and `/config` change it from the next request on; the voice keeps its own style.
- The chat shows calls of one tool in a row as one step: "Write file "…/game.js" · 10 calls" with the newest
  result, instead of a row for each. Every result is listed with every detail shown (ctrl+o); a failure, or a
  call that asks first, still gets its own row. A step names what it works on (the file, page or search), not
  its first argument.
- `/effort` in the chat shows how hard the model thinks before it answers, and `/effort high` (or `none`,
  `low`, `medium`) sets it from the next request on, saved as `chat.thinking`. Each chat setting it or
  `/config` saves is now audited, and the chat's activity feed shows setting and tool changes.
- The chat's `/config` is a settings screen: sections (Chat, Voice, Listening, Safety, Tools), every value a
  setting takes with the one in force lit, when a change applies, and a line on what it does; ← and → step a
  value. Acting without asking and taking requests from PC audio turn on only after the same step again to
  the same value, once the row says what that means and which value, and are marked while they are on.
  Acting without asking changed there applies to the open chat at once too, on or off. The Tools list shows each tool's setting in
  colour, how it ships when changed, and what it does.
- `memory_detail` in `[agent]` and `[chat]` says what later requests keep of one: `condensed` (the request, the
  reply, what was done and what was looked up) or `full` (also its tool calls and their results). The chat keeps
  `full` by default, so a follow-up has the pages, files and outputs a request read instead of looking them up
  again, and the model server reuses what it already read; speech, `jarvis ask` and scheduled requests stay
  `condensed`. Outside text kept this way is still data and keeps actions behind a yes. When the conversation
  outgrows `memory_tokens`, the oldest work goes first, then the oldest turns are summarized, and the chat says in
  one line what it condensed; the context meter then shows what the next request starts from.
- `pc_audio_listen`: with PC audio on, ask Jarvis what is being said or played there (a call, a video) and it
  listens for up to a minute and tells you. `audio.pc_audio_recall_s` (off by default) keeps the last seconds in
  memory, so it can also tell you what was just said. The audio reaches the disk only for the moment Voxtype
  transcribes it, the words are someone else's data, and when Jarvis asks first (`bypass_consent` off) a search
  or action after them waits for your yes.
- Jarvis sings: ask in any words and the model writes the song as notes (words, pitches and lengths), which
  the new `sing` tool checks (at most 30 seconds and 64 notes) before Jarvis sings it by voice, spoken over
  (but not a stretch of about 3 seconds that sounds like the wake word), hushed and paused like a reply. It
  sings with the Kokoro voice moved to each note (Praat, through the new `praat-parselmouth` dependency), or
  with SoulX-Singer on the GPU once `tools/get_singer.py` has installed it (it skips files already in place);
  `[sing] engine` chooses. A song still being made stops at once when you speak over Jarvis, hush or pause
  it, and making one never takes longer than the request's time. Asked for after a web page while Jarvis
  asks first, a song waits for your yes to a question that reads out all its words. Before each song, and
  in `jarvis doctor`, every file of SoulX-Singer's runner is checked against its pinned release; when one is
  not, Jarvis sings with the speaking voice. `tools/get_singer.py` puts a new runner in place only whole
  ([ADR 0018](docs/adr/0018-sing-a-song-the-model-writes.md)).
  or action after them waits for your yes. Only your own request can listen, never a scheduled one; a stop ends
  a listen at once, and the words heard are not kept in the remembered conversation after that request.
- The chat's `/config` also sets Jarvis's speaking speed, whether it acts without asking first and whether it
  takes requests from PC audio. Changes there restart the voice service once, when the settings close, instead of
  at every Enter, and its Tools list never takes a tool below how it ships.
- `jarvis speed` shows how fast Jarvis speaks and `jarvis speed 1.5` sets it; `audio.speech_speed` now goes up to 3.0.
- In `jarvis chat`, words typed while a request runs steer it: they join the request before the model's next
  call, instead of being refused with "esc stops it". They are your own words like the request's (a web address
  in them can be read) and are remembered with it. `chat.while_working = "queue"` (or `/config`) runs them as
  the next request instead.
- Switch a tool on or off, or make it ask first, without editing the config: `jarvis tools` lists them all,
  `jarvis tools NAME on|ask|key|off|default` sets one, and the chat's `/config` has a Tools list. A tool that is
  off (`deny`) is no longer offered to the model at all, rather than offered and refused.
- `strata.speed_projection`, on by default, keeps a control vector the model's run config loads, such as
  Strata's experimental speed projection, with which the model declines far fewer requests; off, Jarvis leaves it
  out. Jarvis never adds one itself, and `jarvis doctor` notes when the setting is on and none is loaded.
- Jarvis reads a web address you write in the request ("what does https://… say?") without searching for it first.
  Only an address in your own words counts, exactly as written: never one in a long paste, a page, a file, PC audio
  or a scheduled request that outside text steered, and every address is still checked to be public.
- `pc_volume`: the PC's own output volume, read, set, turned up or down or muted, in any words; it is every
  program's sound, games and videos included. A change is bound to its start and end before any question.
- `audio.transcribe_backend` says where the GGUF speech-to-text engine runs (Voxtype's `gguf_backend`): `cpu` keeps it
  off the graphics card on Voxtype's GPU build, for a game that needs all of its memory. Empty, the default, leaves
  it to Voxtype.
- The weather at an hour of today or tomorrow ("will it rain at five?"): `weather` takes an `hour` on the place's
  own clock, said on yours (`clock.twelve_hour`); an hour already past today means tomorrow's unless today is said.
  Only the hours' columns are added to what Open-Meteo is asked; nothing new is sent.
- Spotify: jump within what plays or restart it (`spotify_seek`), move playback to another device by its name
  (`spotify_resume`'s `device`), hear what is queued next and which devices there are (`spotify_now_playing`'s
  `more`), and unlike the song playing or add it to one of your playlists (`spotify_library`, which asks first).
  Adding to a playlist needs two new permissions: run `jarvis spotify-login` once; until then only that part
  says so.
- `strata.effort_position` passes Strata's `effort_position` into its profile: with `end`, a request whose model
  raised its effort to `low` or `medium` reuses the pinned prompt instead of reading it all again. Empty, the default, keeps
  the run config's own, until the model's answers with it are measured.
- `jarvis doctor` checks `wpctl` (the PC's volume) and, with PC audio on, `pw-dump`; it notes a missing
  `playerctl`, a Spotify login without a permission Jarvis now asks for, and a user manager whose limit on
  locked memory keeps the model from being locked in RAM.
- Look-ups the model asks for together run side by side: two pages, a search and the weather, or the time in
  two places take about as long as the slowest one. Each is still decided, asked about and audited one by
  one first; actions always run on their own, in order.
- `pc_media`: play, pause or skip a video in the browser, VLC or another MPRIS player, or say what plays,
  through `playerctl` (`pacman -S playerctl`). Spotify keeps its own tools; `everything` pauses it too. Only a
  player's plain name is said, never a title.

- Reminders and scheduled requests can be set in seconds ("in 30 seconds"), up to an hour, for a one-off job;
  the smallest step was a minute. One due within two minutes is said as "in 30 seconds".

- The model thinks harder only when a request needs it. Each request still starts at `thinking` (`none` by
  default). For one that needs careful thought (several steps, a name it may have misheard, a correction, a
  try that failed), the model raises its own effort for the rest of that request with the new `think_harder`
  tool, up to `thinking_max` (`high` by default) in `[agent]` and `[chat]`. After a raise each call thinks up
  to `thinking_raised_tokens` (768 for speech, 1024 in the chat), and the effort never drops within the
  request. Set `thinking_max` equal to `thinking` to keep one effort
  ([ADR 0014](docs/adr/0014-model-raises-its-own-thinking.md)).

- Parakeet, Cohere Transcribe and Qwen3-ASR (Voxtype's `gguf` engine) can transcribe requests:
  `audio.transcribe_engine` names the engine and `audio.transcribe_model` the full path to its `.gguf` file,
  which Jarvis writes into its own Voxtype config, with the architecture the file names for `gguf`.
  `jarvis speech-test score --engine parakeet --model /path/to/model.gguf` scores one on your recordings,
  with settings of its own, so a running Jarvis keeps its engine.
  The vocabulary hint goes only to Whisper, and `jarvis doctor` checks that the model file is there.
- `jarvis speech-test` measures speech-to-text on your own voice. `record` shows one card at a time (play
  one of your artists, ask the time somewhere, tell Jarvis to be quiet, anything you would ask) and saves
  what you say through Jarvis's own microphone, voice detection and lead-in, with the words you confirm or
  type; `--music` marks a session said with music playing. `score` transcribes the set again with the
  current settings (`--engine`, `--model` and `--no-hint` try others) and reports the word error rate, the
  names heard right and the time taken, by kind of card; `compare` pairs two saved scores request by
  request and says whether they differ beyond chance (an exact McNemar test on the names, a sign test on
  the words). The set stays in its folder (0700, files 0600) and is never sent anywhere; an existing folder
  holding anything but a set is refused and left as it is, and `record` waits until a running Jarvis is
  paused, so a card said aloud is never acted on.
- Each request's sound levels are logged before it is transcribed, as one `levels:` line of numbers: peak,
  speech level, the room's floor, the signal-to-noise ratio, how loud its first 20 ms already were (high
  when the start was cut off) and the share of clipped samples. A misheard request can then be told apart:
  a quiet voice, a loud room, or a first sound lost. The sound itself is not kept.

- Tell Jarvis to be quiet, in any words, and it says nothing: the model calls `stay_quiet` (voice only),
  which ends the request without a reply, stops the sentence playing, drops anything queued and opens no
  follow-up window, even when the model had started a sentence first.
  Before, "shut up" got "Understood, I'll be quiet. Just say the word when you need me."
- The time in a country: `time_now` takes a country as well as a city, and a country with several time zones
  gets the time on each of its clocks, from east to west ("In the United States, it's 4:35 AM in New York,
  ..."), each named by its largest city, with the day when it is not today here. The zones come from the time zone database on the
  PC, so nothing is sent anywhere. The model is told never to search the web for the time: asked the time in
  the United States, it had made seven web searches and page reads, 25.6 s before the first word.
- `audio.barge_in_threshold` (0.5): while Jarvis speaks, a wake word needs at most this score, since its own
  voice in the microphone lowers yours. With its voice as loud as yours, a measured 71 % of wake words got
  through at 0.76 and 94.5 % at 0.5; its own speech never scored above 0.052
  ([ADR 0012](docs/adr/0012-lower-wake-score-while-speaking.md)). Its sentences are checked against the same
  score, and a wake word missed over its speech logs the closest score heard.
- Jarvis gets two more things ready ahead of a request. At start and at each wake word it reads the speech
  model's file into memory while the request is still being said, so Voxtype does not wait for the disk:
  with a 2.2 GB model out of the file cache, transcription took 0.7 s instead of 4.2 s. And once the model is
  loaded at start or on `jarvis resume`, it reads the system prompt and tools once (`prompt_primed` in the
  audit log), so the first request no longer spends 2 to 4 s on that
  ([docs/models.md](docs/models.md#warm-up)).
- `strata.keep_in_ram` (on by default) keeps the model server's memory in RAM: its unit gets
  `MemorySwapMax=0` and `LimitMEMLOCK=infinity`. On a PC short of memory the kernel otherwise swaps part of
  the model out between requests, and the graphics driver pauses the GPU while it comes back: 12 of 119
  model calls were paused, by up to 9.8 s. Swapping it out frees little, since quantized weights barely
  compress (2 % under zstd). `jarvis restart` writes it into the unit and restarts the model server
  ([docs/configuration.md](docs/configuration.md#keeping-the-model-in-ram)).

### Fixed

- Web reads and the browser no longer reach a public address on one of this PC's own networks: its own
  addresses, and a network it reaches without a router, such as the home network's global IPv6 prefix
  (checked at each connection, including for embedded IPv4 addresses). Missing route or local-address
  data now blocks the connection. Site-local IPv6 (`fec0::/10`) is refused too.
- Browser confirmations reject ambiguous links, name the full host and require interactive input under
  `jarvis ask --yes`. A lost browser reply after sending an action is reported as uncertain. Pages keep
  their visible text when Chrome has already omitted hidden text; a page too big to check for hidden text
  shows nothing, and the model is told why.
- The browser proxy caps unauthenticated connections and request headers. Preview startup verifies its
  open folder before serving and closes it if startup fails. Directory URLs redirect to the correct
  index URL.
- The chat archive survives a write that was cut short: moving a conversation kept in the first format is
  made again from that file, a half-written log line no longer swallows the next entry, and files no
  listed conversation owns are deleted once they are an hour old. `/forget all` that cannot delete says so
  and forgets nothing. `/compact` is written to the archive, and `/clear`, `/resume` and `/forget all` read
  the log's last events first, so none lands in the next conversation. A chat closed before its request
  began is still archived.
- A request the model answered while the conversation held outside text now keeps that mark itself: what the
  model wrote then may repeat that text, so actions still need a yes once the turn it came from is forgotten. A
  request stopped while the model was writing keeps the mark too once any of its words were said or shown.
- A voice-service restart that fails after `/config` is now shown in the chat, saying the old settings still
  hold; it went to a hidden error stream before. A setting is audited only once it is saved, and acting without
  asking turned on from the chat is recorded as `consent_changed`, as `jarvis consent` records it.
- A tool call's arguments now count toward the context: a long file being written was counted as almost nothing,
  so the check that clears a long request's oldest results to fit the window, and the remembered conversation's
  size, came out too low.
- A request you stop (Esc in the chat, or speaking over Jarvis) is no longer forgotten. It is remembered with
  what it looked up and a note that it was stopped, so the next request does not search all over again or lose
  the task, and the audit log's `reply` says `stopped`. A stop now takes effect while the model is still
  thinking, not only once it starts to answer. In the chat, a request that ends without words says so instead
  of showing nothing. A reply the model wrote whole is not marked stopped when you speak over its last words.
- A long file is no longer lost to the reply cap. A call the cap cuts off is not run: the model is told how much
  one answer holds and writes the text in parts, adding each to the file, and the cut text no longer fills the
  context. A second cut call ends the request with a reply that says so, instead of retrying until the steps
  run out.
- A wake word heard in PC audio with no request after it is no longer dropped silently: the log now says why
  (no speech, too little speech, the loop busy, paused or still holding another one).
- With PC audio on, Jarvis no longer tells you it cannot hear it: the voice prompt says whose audio it hears and
  that a request from there is that person's, not yours. With it off, the prompt says Jarvis cannot hear it.
- A follow-up on something Jarvis looked up no longer sends it searching all over again. Each remembered
  request names what it looked up (the searches and pages, as data when they came from outside), and a page read
  in the conversation is kept for 30 minutes, so reading it again needs no network. An address the owner
  wrote can be read again in a later request of the conversation, the 30 minutes count time asleep, and of a
  request's look-ups the newest are kept, with their words only and no address cut short.
- `jarvis doctor` checks for `pw-dump` whenever Jarvis needs to know what else plays, which by default it does
  (follow-ups stay shut while other audio plays), not only for PC audio.
- Where the system cannot tell whether the speech model is in memory, a wake word no longer reads all of it
  again within 5 minutes of the last whole read.
- `jarvis doctor` notes a lock limit on the model's memory only when the model is larger than it.
- After writing a file, the model is told its full path and link, so a link it gives the owner points to the
  right home folder instead of a guessed one.
- `pc_media` binds a command to the players it is for before any question, and the question names them
  ("pause firefox and spotify"); a skip not answered in time is reported as uncertain, so it is not sent twice;
  a player that closes while it is listed is left out, and a player's own error words are not passed on.
- Turning the PC's volume up when another program set it above 100 percent no longer lowers it.
- A Spotify seek past the end of the track asks to jump to its end, at the time it really goes to; a playlist
  whose id is not a Spotify id is not offered.
- Look-ups approved in one message no longer run when Jarvis is stopped while a later one of them is asked about.
- A request that used up its tool calls or its time now still says the answer the model gave on its last step,
  instead of "That needed too many steps".
- Clearer words: `pc_media` says "Nothing is playing." when no player plays, the Spotify device list marks
  the active device "(playing now)", and a reminder given two times says to give only one.
- Speech remembers a reply that drew on outside text (a search, a page, a song name) as data, as the chat does,
  so "what was the second one?" has the answer to go on. While it is remembered, an action the model chooses
  asks first; a plain command does not. It was left out of later requests before.
- A scheduled request keeps where its request came from. One first heard in PC audio runs as PC audio when it is due,
  and one added while outside text (a web page, a file, a remembered answer) was in the conversation runs with that
  text counted, so its actions need a yes as they would have then. Requests the owner alone asked for still run
  unasked ([ADR 0011](docs/adr/0011-reminders-and-scheduled-requests.md)).
- `tools/idle_check.py` no longer reports "ok: idle" for the GPU when it could not see the process's GPU use
  (Strata's ROCm engine publishes no per-process engine time): the verdict says it was not measured, and
  each card's whole busy share over the window is shown beside it, also when another process of the tree
  publishes counters.
- A place said with its state or country, "Springfield (Illinois)" or "Lisbon, Portugal", is found for the time
  and the weather: the geocoder takes only a name and found nothing for the whole phrase, so the name is
  looked up alone and the qualifier picks among the matches; with several ("Springfield, Illinois, USA") the
  place most of them name wins.
- Spotify's "Restriction violated" (shuffle, previous or a seek that what plays now does not allow) is said as
  what it means, that Spotify does not allow that command here and nothing changed, instead of a raw 403.

- `audio.transcribe_threads` reaches the GGUF engine too; before, only Whisper got it and the GGUF engine used
  its own default.

- A step that may only answer (after a repeated call, or a request's last step) no longer makes Strata read
  the whole prompt again. It kept the tool list out of the request, which changed the prompt Strata had
  pinned; now the tools stay and the model is made to answer through `no_action`. A call to any other tool
  on that step is still not run ([docs/models.md](docs/models.md#warm-up)).

- The audit log could grow past `max_bytes` when several writers ran at once on a busy PC: a writer that
  had just rotated a full log reopened the new one and wrote without checking its size, though others may
  have filled it meanwhile. The size is now checked again after each rotation. No event was ever lost.
- `jarvis restart` writes Strata's profile and unit from the config, so a `strata` setting changed there
  reaches the model server. Before, only `jarvis setup` wrote them: a `strata.context` of 49,152 in the config
  left Strata running at 24,576 while Jarvis planned for the larger window. The voice service's unit is left
  as `jarvis setup` wrote it, so a restart run from another checkout does not move the service there.
- The time or weather in a US state, a Canadian province or territory, or an Australian state or territory is
  its main city's ("Texas" is Houston's), and its time is worked out on the PC. Before, the place went to the geocoder
  as said: its best "Texas" is a village in Mexico, "California" a town in Colombia and "Ontario" a city in
  California, and asking the model to name a city instead did not change what it passed. A bare "Georgia" is
  the country, and a bare "Washington" or "Quebec" the city; "Georgia state", "Washington state" and "the
  province of Quebec" are the regions. Since the model passes "the state of Georgia" as "Georgia", the time
  for a plain shared name also gives the region's when its clock differs, and the model answers the one asked
  for. A home set in `[weather]` is always the place it names.

### Changed

- The chat's default `memory_tokens` is now 5700 (was 6200), so the default 24,576-token window
  still holds it beside the new window and screenshot tools.
- `jarvis doctor` ends with the names of the checks that failed, not only their count, writes every note
  the same way (`note …`), and colours its status words on a terminal (never in a pipe or with `NO_COLOR`).
- Speech keeps a short memory for quick commands: `agent.memory_idle_s` is now `300.0` (was `1800.0`), so the
  conversation is forgotten after five idle minutes, and `agent.memory_tokens` is now `3000` (was `6000`). A
  scheduled request shares speech's conversation and limits; `jarvis ask` remembers nothing between runs. The
  chat's own memory is unchanged.
- The chat's default `memory_tokens` is now 6200 (was 9100), so the default 24,576-token window keeps room for
  the thinking the model may raise to and for the new tools' and arguments' schemas, the browser's and
  `switch_model`'s among them.
  A larger `strata.context` leaves room for more.
- Spotify names are matched by sound as well as spelling, whichever is closer, using espeak's phonemes. A
  name speech-to-text wrote as words that sound like it ("flocks days" for "Phlox Daze", "7" for "seven")
  now finds your artist or song, and an item of yours of the kind asked for that sounds closer to what was
  heard beats a search hit only spelled like the mishearing; an exact hit still wins. Measured on a private
  set of 67 artist names misheard in speech-to-text test runs against a 60-artist library: the right artist
  scored at least `close_match` (0.8) for 47, against 33 by spelling alone, and at least `loose_match` (0.6)
  for 58, against 51, with no more of 26 names outside the library matched at either. A name in another
  script than Latin is compared by spelling only, since the English voice reads all such letters alike.
  Espeak calls now share Piper's own lock, so a Piper voice, Kokoro and name matching never phonemize at
  once, and characters espeak cannot take (a NUL, a lone surrogate) are left out of what it reads.
- The capture keeps 320 ms from before speech is detected, not 160 ms, so a soft first sound the voice
  detector was not yet sure of reaches speech recognition. The wake word's tail is kept as before: only
  when the request follows it within 160 ms, and none of it is in the longer lead-in otherwise.
- The model can be asked to think before each answer: `thinking` (`none`, `low`, `medium` or `high`,
  Strata's reasoning effort) in `[agent]` and `[chat]`, capped by `thinking_tokens` (192 for speech, 512 in
  the chat), after which Strata closes the thinking and lets the model answer. It is `none` by default:
  measured three times on 27 spoken requests, every effort handled all of them, as `none` did, and made the
  median request 1.1 to 1.3 s slower ([models.md](docs/models.md#thinking)). The thinking is never spoken
  or shown, and it comes on top of the reply's own cap; the context keeps room for it, out of the
  remembered conversation when it is turned on. A summary of older turns is written without thinking.

- `jarvis doctor` recognises the picture readers of Qwen3.8-Flash-Next and Inkling 2 Mini by their pinned
  sha256, and the docs say Qwen3.8-Flash-Next needs `strata.memory_max = "44G"` (at 40G the Strata unit sat
  at its memory mark).

- Jarvis serves whichever model Strata is set up for, from Strata's own run config
  (`strata.run_config`, the `strata-<model>.json` that `strata launch` writes): the model, its engine, its
  flags, tokenizer and picture reader come from that file, and Jarvis adds only its own limits (context,
  VRAM, one request at a time, lazy load). Nothing model-specific is hardcoded any more. The VRAM limit goes
  through whichever flag the model's engine takes. `strata.learn_experts` (on) lets an engine that keeps a
  routing profile learn which experts Jarvis's requests use and start with them on the card. Jarvis no longer
  has a model default of its own; the model set up and measured here is Qwen3.8-Flash-Next (measured beside a game's budget: 7.95 GiB of VRAM, 35.5 GiB of RAM,
  under a second for a two-tool request, and the prompt reused between requests); Inkling 2 Mini is set up
  to compare and is too slow with only 8 GiB of VRAM today ([docs/models.md](docs/models.md)).
  **Migration:** `strata.model_file`, `tokenizer_dir`, `model_id`, `prefill` and `vision_file` are gone,
  and a config that still sets one is refused with what to do: run `strata launch` for the model, then
  `jarvis setup --run-config <its strata-<model>.json>` (it replaces `--model-file` and `--tokenizer-dir`),
  and delete the old keys. `strata.pictures = false` replaces leaving `vision_file` unset. `jarvis setup`
  without a run config saves the config and says what comes next.

- The command line reads better. `jarvis help` lists the commands in groups (everyday, running Jarvis,
  settings, first-time setup, other) with what each does, instead of argparse's list of every name,
  twice. Every help page is plain text within 80 columns. `jarvis <command> --help` says what the command
  does and explains every option, including what `consent bypass`, `ask --yes` and `spotify-login
  --no-browser` do not do, and a mistyped command gets a suggestion when one is close. `jarvis` alone
  names the services as running or stopped and points at `jarvis help`. `jarvis doctor` lines its checks
  up and ends with how many failed.

- The chat remembers a little less: `chat.memory_tokens` defaults to 10,000 (was 10,500), to make room in
  the default context for the list of Omarchy's guides. A config that still sets 10,500 gets what the window
  leaves, about 10,000.

- From a practices audit (#30): at most 30 actions that change something (playback, a file write) run in
  any 10 minutes, the voice service and the chat counted together and even with questions bypassed; one
  more is refused, recorded as `action_limit`, and Jarvis says why (`boundaries.max_actions`,
  `boundaries.actions_window_s`). Only actions that run count: with the window full Jarvis refuses before
  asking, and an action you decline takes no place. A damaged count file starts the count again instead of
  failing every request, a failed write leaves the old count, and a lock held for 2 seconds falls back to
  the process's own count. Before a write Jarvis names the folders it would make and, for a
  replace, the old file's size; the chat also shows the first three lines of the new text. A name with a
  newline or another control character is shown with a mark, on the question's own line. A write is
  refused if a folder it was not asked about would be needed, or the old file is not the one asked about
  (its size, inode, modification or change time differ). A replace keeps the old file's read and write
  permissions instead of 0644. Files are
  opened one folder at a time without following a link, so a folder swapped for a link after the check is
  refused, and a file under home with a second hard link is not read or added to. The audit log keeps
  five older generations instead of one and holds only the length and SHA-256 of what a write puts in a
  file. While Jarvis asks first, chat pastes over 200 characters in all count as outside data, a request
  brought back with ↑ keeping what was pasted into it. Starting a new conversation (or one forgotten
  after the idle time, checked before every request, PC audio included) also forgets the pages and
  threads earlier searches returned.
- A wake word said while your Voxtype dictation runs is ignored: no request, no chime, one `wake_ignored`
  event in the audit log; and a running Voxtype meeting keeps follow-ups off like dictation does. Jarvis
  follows `voxtype status --follow --format json --extended` in the background, so asking never waits for
  Voxtype (#29). A status that cannot be read no longer counts as dictation: the wake word is taken, so a
  broken or older Voxtype does not leave Jarvis deaf, and follow-ups stay off until the status is known.
- The Azure keyword helper writes 1.5 s of silence into the engine's stream before the first sample and after
  each detection, which the engine is documented to want before a keyword (#29).
- Both services are safer and kinder to a game beside them (#33). Both set `NoNewPrivileges`, `LockPersonality`,
  `RestrictSUIDSGID` and `CPUWeight=50`; Strata also sets `UMask=0077`, and the voice service
  `SystemCallArchitectures=native`. A service that fails five times in 15 minutes now stays `failed` instead
  of restarting for ever; `jarvis start`, `jarvis restart`, `jarvis setup --enable` and the commands that
  restart the voice service after a change (`consent`, `pc-audio`, `wake`) clear that first. Strata's
  `MemoryHigh` is `strata.memory_max` and its `MemoryMax` is 10% above it (44G for the default 40G).
  `strata.prefill` also accepts `"auto"`, and the new `strata.vram_reserve_mib` (unset by default) passes
  `--vram-reserve-mib`. Run `jarvis setup` to rewrite the units. Not measured yet: the effect on time to
  first word, VRAM and a running game (playbook).
- `boundaries.bypass_consent` no longer runs `strong` actions unasked: they wait for the confirm key.
  No operation Jarvis ships is `strong` yet; this matters for one you set to `strong` yourself.
- `strata.idle_unload_s` defaults to 86400 (24 hours) instead of 300, so a fresh install keeps the model
  loaded, as the owner decided (#34). Set a lower value to free the RAM sooner. A saved config or a Strata
  profile already written keeps its value until you change it or run `jarvis setup` again.
- `jarvis doctor` asks Voxtype which speech models are installed (`voxtype info models`), which also checks
  each file's size and header, and falls back to looking for the file when Voxtype cannot say.
  `stop_process` is the one shipped `strong` operation, and `boundaries.classes` may set it only to
  `strong` or `deny`; any other class stops Jarvis with an error.
- The chat remembers a little less: `chat.memory_tokens` defaults to 10,500 (was 11,500), to make room in
  the model's context for `find_processes`, `stop_process`, `calculate` and `open_app`.
- Jarvis never opens Spotify, or any program, unasked: with no player it says Spotify is closed and asks
  whether to open it and where (#36). Before, playing or resuming started the Spotify app.
- From a practices audit: a name in `boundaries.classes` that is no operation (a typo, say) stops Jarvis
  with an error instead of being ignored; turning questions on or off with `jarvis consent`, and the mode
  the voice service starts in, are recorded in the audit log and shown in the chat; words heard from the PC
  reach the model as marked outside data, so someone in a call cannot pose as you; a saved chat
  conversation is read only if it is your own file, readable by nobody else; only web links in a chat
  reply open when clicked; a folder with more than 10,000 entries is listed from the first 10,000; an
  argument shorter than its minimum length is refused; and both units start as `Type=exec`, the voice
  service after WirePlumber too.
- Hidden files and folders under your home are no longer read unless `files.hidden_allow` opens them;
  by default it opens the desktop's own settings (`~/.config/hypr`, `omarchy`, `waybar`, `kitty` and the
  like). Shell history and cookie stores are never read anywhere. With questions bypassed, this keeps
  logins and tokens kept in hidden folders from reaching the model, and from leaving in a search.
- Under each chat reply: "✻ Worked for 4m 41s · done 7:37 PM · 50 TPS" (how long it took, when it was done,
  and how fast the model wrote, from Strata's own timings), a blank line below the reply and in the column of
  its dot. Replies are set as Claude Code sets them: headings bold (the first level also italic and
  underlined) with a blank line above and below, lists marked with a dash and numbers at the text's edge,
  code with no side bar, quotes in italics behind a dim bar, table headers in plain weight. Typing `/` opens
  the command menu over the transcript's last lines, right above the input, as in Claude Code, so nothing
  moves while it is open; the letters typed are in bold and the chosen row is lit whole. `/exit` leaves
  (`/quit` still works).
- Jarvis remembers the conversation. Instead of the last 4 turns, forgotten after 5 minutes, it keeps as
  much as fits a token budget (`agent.memory_tokens` 6,000 for speech, `chat.memory_tokens` 10,500 for the
  chat) and has the model summarize the oldest turns when it outgrows it. Speech forgets after 30 idle
  minutes; the chat never does and resumes its conversation when it opens again (`chat.resume`).
- Replies are no longer cut. The model is told its length and capped in tokens (`agent.reply_tokens` 400,
  `chat.reply_tokens` 2,048, was `model.max_tokens` 200); a reply that reaches the cap anyway is kept and
  says so. A request may take more steps (speech 8 model calls and 12 tool calls, the chat 30 and 40 within
  10 minutes; all in `[agent]` and `[chat]`), and a long one clears its oldest tool results to stay within
  the context window.
- The chat gets its own prompt: written replies in Markdown, sources as links. Speech keeps short spoken
  replies.
- `jarvis chat` is rebuilt closer to Claude Code: replies are rendered as Markdown (tables, lists, code,
  links) while they are written, and never cut; by default it shows only requests and replies, with a working
  line that says what Jarvis is doing ("Searching the web… (8s · esc to interrupt)"); Ctrl+O or
  `/verbosity` shows each step and its result, then every detail. New commands: `/clear` (a new
  conversation), `/compact`, `/context`, `/verbosity`, `/config`, `/status`, `/copy` and `/export`; the menu
  narrows as you type and offers a command's arguments. The input takes several lines (Shift+Enter), ↑ and ↓
  bring back requests from earlier sessions, Ctrl+C clears the input and twice leaves, and a toolbar shows
  the PC's CPU, GPU, VRAM, RAM and temperatures and how full the model's context is.

- `jarvis chat` is redesigned after Claude Code's terminal interface: a welcome card, each request grouped
  with its steps (`●`) and their results (`⎿`), a working line with Esc to stop the request, a yes-or-no
  choice for questions (1 or y, 2, n or Esc), a `/` command menu, ↑ and ↓ for earlier requests, `?` for the
  shortcuts, Ctrl+O for every step, and the terminal's own colours, so it follows the Omarchy theme. A typed
  request shows at once.
- `jarvis chat` reads well on every terminal theme. Omarchy sets the terminal's black to its background, so
  the shaded bands behind requests and code were invisible; requests now start with a cyan `❯` and say how
  and when they came in at the right, code blocks and quotes have a quiet rule at their left, and tables,
  headings, bullets and numbers use the theme's text and grey instead of four different colours. The
  command menu, `/config` and the yes-or-no choice point at the highlighted row with `❯`, the working line
  says "Waiting for your answer" while a question is open, the footer and toolbar shorten themselves to
  stay on one line in a narrow window, and the input says when typed requests are off.
- Jarvis says the time on a 12-hour clock ("3:42 PM") by default; `clock.twelve_hour = false` keeps "15:42".
- Jarvis starts speaking sooner. Each sentence of a reply is spoken as soon as the model has written it and
  synthesized while the one before plays. The silence before the first word is cut from 0.23 s to 0.03 s. A
  sentence claiming an action is not spoken before an action succeeded, and a sentence said before a tool
  call plays to its end instead of being cut off mid-word
  ([ADR 0008](docs/adr/0008-speak-while-writing-and-barge-in.md)).
- The service log's timing line now shows when the first words were ready.
- The default Strata context window is 24,576 tokens (was 8,192), about 8.1 GiB of VRAM with the default model, and the model sees up to 3,000 characters
  of each tool result (was 1,500), so a search and two page reads fit in one request. A saved
  `strata.context` keeps its value; to change it, edit it and run `jarvis setup`.

### Added

- Jarvis knows itself: the prompt names its version and the model Strata serves (from Strata's run
  config), `about_jarvis` gives its model, engine, context, voice, wake word, speech-to-text model,
  boundaries, limits and where its files are, and `read_jarvis_file` reads one of its own configuration
  files (the config, Strata's run config and the profile served from it, its own Voxtype settings, the two
  systemd units), whole or one `[section]`. Both only read. The file is picked from that fixed list,
  never by a path; `files.deny` and the never-read lists still apply; a value under a name such as
  `api_key`, `token` or `password` is hidden; and what the file says counts as file text, so an action
  after it needs a yes. Jarvis never changes its own configuration. To make room in the default context,
  `chat.memory_tokens` defaults to 9,100 (was 9,400).
  The README no longer names one model: Jarvis serves whichever Strata's run config names.

- Reminders and scheduled requests ([ADR 0011](docs/adr/0011-reminders-and-scheduled-requests.md)):
  `schedule_add` sets a reminder Jarvis says, or a request it carries out as if you had just asked, in so
  many minutes or at a time of day (today, tomorrow or a weekday), once, every day, on weekdays, at
  weekends, every week, or every 5 to 1,440 minutes; `schedule_list` and `schedule_cancel` list and cancel
  them. They are kept in `~/.local/state/jarvis/schedule.json` (mode 0600) and said by the voice service
  between requests, never while you dictate or a meeting runs. A one-off reminder up to six hours late is
  still said, with how late it is; anything else more than ten minutes late is not done, and the audit log
  and the chat window's activity show it as missed. The chat window and `jarvis ask` say when the voice
  service is not running to say a job. To make room for the three operations in the default context,
  `chat.memory_tokens` defaults to 9,400 (was 9,900); a config that still sets a higher number gets what
  the window leaves.

- `strata.kv_resident` keeps that many context cells in VRAM and the whole context in pinned RAM
  (`--kv-resident`, Strata's KV streaming), so `strata.context` can grow without more VRAM. It is off
  (`0`) by default until a bigger context has been measured on this PC.

- Jarvis answers questions about Omarchy from the guides Omarchy ships for agents
  (`/usr/share/omarchy/default/agents/skills/`, part of #9). The prompt lists each guide by its first
  sentence, read when Jarvis starts, so an Omarchy update is picked up at the next start. The safe
  `os_skill` operation reads one file of one guide, in parts that fit a tool result, and only a file
  listed at start. Only folders and files that root owns and nobody else can write are listed or read,
  each opened from `/` without following a link, and the list keeps to a fixed size so it always fits the
  context. The text counts as outside text, like a web page, and Jarvis is told never to run a command
  from it. Every guide's SKILL.md is listed before any other file, and `boundaries.classes` may name
  `os_skill` on a PC without the guides.

- An opt-in shell command, `pc_shell` (#19), off unless `[shell] enabled` is set. Each command needs the
  confirm key, which no class setting lowers, and runs once in its own user scope with time, memory, CPU
  and task limits. It runs with no new privileges, so `sudo` cannot work, and with only `PATH`, `HOME`,
  `LANG` and `TERM` set. Before the key, the command is read out character by character, so two
  commands that run differently never sound the same. A request heard in PC audio, or one that read
  outside text, runs nothing. The audit log keeps the command but only a digest of its output.
- Wake word training and benchmark tools: `tools/wake_train.sh` trains and scores a phrase with a chosen
  LiveKit project's key, `tools/wake_ungated.py` runs the trainer without the gated test sets (#37), and
  `tools/wake_bench.py` scores models on 55 local voices and about 24 hours of real media, alone and
  together as Jarvis runs them. [docs/playbook/wake-training.md](docs/playbook/wake-training.md) says how.
- `jarvis doctor` now loads each wake word file that is on and fails one that does not load, and notes a
  `wake_thresholds` name that is not in `wake_words`. `jarvis wake` shows each model's own threshold in its
  first line, and the log names every wake word that passed in a frame, not only the highest.
- Each wake word can have its own threshold (`[audio.wake_thresholds]`, such as `jarvis = 0.85`); the
  others keep `audio.wake_threshold`. Models score on different scales: on the trainer's reduced test set
  the trained `jarvis` falsely woke 5 times in about an hour at 0.5. `jarvis wake` shows a model's own
  threshold beside its name. `docs/models.md` records the measured recall and false wakes of the trained
  `hey_jarvis`, `yo_jarvis` and `jarvis` models (#15, #37).

- `tools/first_audio.py` times, on a loaded model, how long the first sentence of a reply takes to reach the
  speaker and how long the whole reply takes to write (#12). Fake-clock tests show the first sentence goes to the
  speaker while the model is still writing, and each sentence goes once, in order.
- A state indicator for the Omarchy bar (#14): `tools/bar_status.py` asks the running Jarvis for its status over
  the control socket and turns it into the bar's text, tooltip and class for idle, listening, thinking,
  speaking, paused, model loading and unavailable, and into a state of its own for a Jarvis that is not
  running, a timeout, a broken connection and an unreadable answer. The module offers only a left click for
  `jarvis pause` and a right click for `jarvis resume`, polls every 5 seconds and never loads the model. The
  script's own cost is 0.015 s of CPU a poll: it does not start `jarvis` (about 0.2 s a poll) and runs without
  Python's `site` import. Each poll also makes Jarvis ask Strata for its model state (no tokens), about 0.003 s more
  across both, 0.018 s in all or 0.4 % of one core. With Jarvis not running it shows `not running` without the
  model's state or the click hint, and `jarvis status` still gives the state. The entry to add to
  `~/.config/omarchy/shell.json` is in
  [configuration](docs/configuration.md#the-omarchy-bar-module-14); nothing is installed for you.
- `audio.wake_words` picks which LiveKit wake word models listen, side by side, each
  `<data_dir>/wake/<name>.onnx`; `hey_jarvis` and `yo_jarvis` are on by default. `jarvis wake --words
  hey_jarvis,yo_jarvis` sets the list and `jarvis wake` shows every model in the folder as on or off. A
  model that is missing or cannot be loaded is left out with a warning and the others still listen.
  `jarvis doctor` fails only when no model has a file, and lists each missing one. The log names the
  phrase each wake came from, and they share `wake_threshold` unless
  `wake_thresholds` names one. The check that stops Jarvis
  hearing itself during barge-in uses the same models.
- `open_app` (#36): ask Jarvis, in any words, to open a program from the new `[apps]` list (Spotify by
  default) on a workspace from 1 to 10 or in the background. Jarvis asks where when you did not say.
  Hyprland starts it there without taking focus, and Jarvis says where it opened only once it saw the
  window there, moving it when it landed elsewhere. A request heard in PC audio opens nothing, and a
  program already open is not opened again ([Opening a program](docs/configuration.md#opening-a-program)).
- `calculate`: Jarvis works out sums in code, exactly, however you ask (products, percentages, powers,
  roots, logarithms, factorials), and says the
  answer; the model never does the arithmetic itself. Numbers in replies and answers come in
  groups of three (1,488); the new `format.thousands_separator` sets the separator, or none.
- `audio.vad_neg_threshold`: once speech has started it goes on while the probability stays above this
  (default: `vad_threshold` less 0.15, Silero's own margin), so a breath inside a sentence no longer ends a
  request early. A request after the wake word, from the mic or PC audio, needs 0.15 s of speech, so a click
  after "Hey Jarvis" is dropped untranscribed (#29). With the Azure wake word, which is reported up to 0.6 s
  after the keyword ends, only the keyword's last quarter second is left out of that count, so a short
  request said before the report still counts.
- `audio.kokoro_phoneme_map` (off by default): rewrite espeak's phonemes into the symbols Kokoro was trained
  on before they are spoken, to compare by ear (#29).
- `jarvis doctor` checks that `audio.source` names a microphone PipeWire lists, and says so when PipeWire
  cannot be read rather than blaming the name (#29).
- Stop a program by voice, asked in any words:
  `find_processes` lists your own programs that match the name you said (at most 8, by program name,
  executable, a Wine game's `.exe`, or window title), Jarvis asks which one when several match, and
  `stop_process` stops only a PID it listed in the same request, after the confirm key. The PID is bound
  to its program and start time, so a PID reused in between is never signalled. A request heard in PC
  audio, or one that also read a web page, a file or a picture, stops nothing; a match by one window's
  title alone says that all the program's windows close. The desktop session, its helpers, the sound
  server, Voxtype, Strata and Jarvis itself are never listed or stopped
  ([Stopping a program](docs/configuration.md#stopping-a-program)).
- `jarvis confirm`, the confirm key: bind it to a key, and a `strong` action runs only after you press it.
  Jarvis says what it is about to do, then "Press the confirm key to allow it", and waits
  `audio.confirm_key_s` (30 s). One press allows that one action, once; a spoken yes never does. A press
  with nothing waiting holds the key off for 30 s, and, as a best effort, a press from a program Jarvis
  started is refused.
  Only spoken requests can use it: the chat window and `jarvis ask` still refuse `strong` actions.

- `jarvis pause` lets go of the microphone and, once the request in flight has ended, frees the model;
  `jarvis resume` listens again and reloads it. `jarvis hush` stops the speech. `jarvis status` says what
  the running Jarvis is doing (idle, listening, thinking, speaking, paused, model loading, unavailable), and
  `--json` prints it for a status bar (#14). These reach `jarvis run` through a socket only your user can
  open, in `$XDG_RUNTIME_DIR/jarvis/`, and each is recorded in the audit log. A pause lasts until `resume`
  or until the voice service restarts. `jarvis stop` still stops the services; `hush` is the speech
  interrupt #22 asked `stop` to be.
- Jarvis carries on across desktop session changes (#22). A microphone whose capture ends (PipeWire
  restarting, the microphone unplugged) or goes silent for 5 s is opened again: after 1 s, then twice as
  long after each try that brings no frame, up to 30 s. `jarvis run` used to end with "the microphone
  stream ended" and wait for systemd to restart it. Meanwhile `jarvis status` says `unavailable`, and a
  request the lost microphone cuts short ends without an apology. After a suspend, Jarvis starts listening
  afresh and checks the model server, loading the model again only if its load failed. A model server that is
  down because its model file is missing (its drive not mounted) is no longer waited for: Jarvis says so at
  once, and `jarvis status` names the model `missing`. A new `audio.listen_when_locked` (on by default)
  can make Jarvis take no request while the Omarchy lock screen is up (the wake word still stops its speech);
  `jarvis doctor` then checks the lock state can be read.

- `boundaries.bypass_consent`, on by default (the owner's choice): Jarvis acts without asking for a yes,
  including after outside text and for requests heard in PC audio; an operation set to `deny` is still
  refused, and each bypassed question is audited. `jarvis consent ask` turns asking back on, `jarvis consent
  bypass` off again.
- `jarvis chat` shows pictures: ask to see something and Jarvis searches SearXNG's image category
  (`image_search`) and draws the best result in the window (`show_image`), with kitty's graphics protocol
  through textual-image 0.14.2 (new dependency, LGPL-3.0-or-later). A read page offers its own picture too.
  Only a picture a search or page in the same conversation offered is fetched, through the same address
  checks as pages, then checked and shrunk; the newest 50 are kept 0600 in `chat-pictures`.
- Pictures in `jarvis chat`, as in Claude Code: Ctrl+V pastes a copied picture as an `[Image #1]` tag (or
  copied text where the cursor is), a picture file dropped on the terminal or `/image <path>` attaches that
  file, and `/image` alone the newest screenshot. Pictures are checked and shrunk with Pillow 13.0.0 (new
  dependency, MIT-CMU) and go only to the local model, which needs `strata.vision_file`. What a picture
  shows counts as outside text, so an action the model chooses in that request needs your yes.
- Pictures for the model: `strata.vision_file` names the model's picture reader (`mmproj-F16.gguf`, pinned
  in `jarvis doctor`), and the Strata profile then reads pictures with Strata's processor reader, started by
  the first picture and stopped after `strata.vision_idle_s` (120 s), so it costs no VRAM and no RAM while
  unused.
- PC operations: `pc_stats` ("how's my PC doing": CPU, graphics card, VRAM, memory, temperatures, disks,
  uptime), `list_dir` and `read_file` within `files.read_roots` (home by default), and `write_file` within
  `files.write_roots` (`~/Documents` by default), which always asks first. Keys, tokens, password stores,
  browser profiles and Jarvis's own state are never read or written; writes never touch hidden files.
  Jarvis's own `files` and `pc` skills say how to use them.
- While text read from a file is in the conversation, operations that send words off the PC (web and Reddit
  searches, a named place for the weather or time) need a yes, so a file's contents cannot leave in a search
  unseen.
- Skills: Jarvis's own instructions for web research and game look-ups, listed in its prompt and read
  through the safe `read_skill` operation before such a task.

- Follow-up requests: the next request within `audio.follow_up_s` (6 s; 0 turns it off) of a spoken reply
  needs no wake word. It never opens while Voxtype dictation is recording or transcribing; a sound too
  short for a request (a cough, a click) is ignored, and unclear speech is dropped without a word.
- Pull requests check every commit's subject against the Conventional Commits types in `CONTRIBUTING.md`,
  and, from a fork, each commit's DCO sign-off (`tools/check_pr_commits.py`).
- CI lints the workflow files with actionlint (and shellcheck over their `run:` steps); `tools/check.sh` runs
  it too when `actionlint` is installed.
- An Azure custom keyword model can be the wake word instead of LiveKit's: `audio.wake_engine = "azure"`
  or `jarvis wake azure --model FILE`, and `jarvis wake livekit` to switch back. Microsoft's Speech SDK runs
  only in a separate MIT-licensed helper with no network; an Azure model that cannot start falls back to
  LiveKit's, and `jarvis doctor` checks the setup
  ([ADR 0010](docs/adr/0010-azure-keyword-wake-word.md)). While an Azure model listens, a sentence that
  says "Jarvis" cannot be spoken over.
- The service log shows the score of each LiveKit wake, to tune `audio.wake_threshold` against false wakes.
- Speaking over Jarvis: say the wake word while it talks and it stops and listens to the new request; a
  question spoken over counts as no. `audio.barge_in = false` turns it off. A sentence that would trigger the
  wake word itself cannot be spoken over.
- `config.example.toml`: every setting at its default, with no personal values, to copy to
  `~/.config/jarvis/config.toml`.
- The time and date, here or in a named place, and the weather now, today, tomorrow or for the week, at home
  (`[weather]`) or wherever you name, from Open-Meteo. "What time is it" and "what's the weather" skip the
  model.
- Research: web search through a local SearXNG (`deploy/searxng/`), reading the pages a search returned, and
  Reddit threads found through SearXNG and read from the Arctic Shift archive. Pages are opened only on public
  addresses, checked at every connection and redirect, and everything they say is kept as outside data
  ([ADR 0007](docs/adr/0007-outside-services-for-answers.md)).
- The first public version of Jarvis, a local, always-on voice agent for Omarchy, open source under
  GPL-3.0-or-later.
- A "Hey Jarvis" wake word and end-of-speech detection, with speech to text through Voxtype and Jarvis's own
  settings file.
- One agent on a local model served by Strata, with plain commands such as "pause" or "volume 40" answered
  without the model.
- Spoken replies through Kokoro, or a Piper voice; no desktop notifications.
- Spotify by voice: play a track, album, artist, playlist or liked songs, pause, resume, skip, volume,
  shuffle, repeat, queue, like and what is playing, with PKCE login.
- Boundaries in code: typed operations with default deny, provenance on every message part, confirmation
  bound to one exact action and an audit log.
- Optional listening to PC audio, treated as someone else's speech that needs the owner's yes.
- `jarvis setup`, `doctor`, `start`, `stop`, `restart`, `status`, `unload`, `logs`, `ask`, `chat`,
  `pc-audio` and `spotify-login`, and a versioned TOML configuration.
- Community files: contributing guide, code of conduct, support, governance, accessibility, security policy
  and roadmap.
- `tools/check.sh`, one command for everything CI runs, and checks of the dependency licences and lock
  sources (`tools/check_licences.py`), the docs (`tools/check_docs.py`) and the workflow files.
- Tests that cannot reach the network, audio, the real programs Jarvis drives or your home folder
  (`tests/conftest.py`), and the product rules checked in code (`tests/test_product_rules.py`, Ruff's
  banned imports).
- Opt-in git hooks (`tools/install-hooks.sh`) and a read-only idle check (`tools/idle_check.py`).
- A docs map, an architecture overview, a statement of what Jarvis keeps and sends (`docs/privacy.md`), and
  recipes and agents for coding agents (`.claude/`, `.codex/`, `.agents/`).

### Fixed

- The calculator works out a factorial of a function's answer (`sqrt(9)!`, `abs(-3)!`) instead of
  reporting a function that does not exist, and refuses `5!!` with a reason rather than guessing between a
  double factorial (15) and two factorials (120!); `(5!)!` still works.

- A reminders file written by a newer version of Jarvis is no longer written over (losing its jobs) when an
  older one adds a reminder: the older one refuses and says why.

- A reminder repeating every few minutes or hours that starts on another day says which day ("every 30
  minutes, next tomorrow at 9:00 AM"), not only the time.

- A reminder or scheduled request the model asked for with a time that does not work (one that has passed
  today, two times, a day that is not one) is no longer "an internal error": the model is told what was
  wrong and can fix it, and so can a cancel by something other than a number. A `day` beside a repeat that
  sets its own days (`daily`, `weekdays`, `weekends`) is ignored. Found by a benchmark with a local model.

- A web page whose server names no charset is now read in the one the page declares (`<meta charset>`)
  rather than always as UTF-8, and a gzip body made of several parts is read whole instead of only up
  to the end of the first.
- The chat window's view of the voice service no longer misses a line written to the audit log just before
  it rotated.
- A Strata key written in quotes in its environment file (`STRATA_API_KEY="..."`) is now read without
  them, as systemd reads it for Strata. Jarvis used to send the quotes too, and Strata refused it.
- After a file was read, "what time is it" and the weather at home no longer ask first. Only a call that
  names a place sends the model's words off the PC; the time and weather here send none. This is what
  the configuration guide already said.
- In the chat window, a request sent while another is still working now stays in the box with its pictures
  and pastes, ready to send again. It used to be cleared, and the attached pictures were lost.
- A streamed model answer that stalls now ends at the call's time limit. The limit was checked only when a
  piece arrived, and the read timeout restarted with each one, so a stalled stream could run to about
  twice its time.
- PC audio from an app that plays through several streams is now heard whole. Only its first stream was
  recorded, which also weakened the check for other people's speech during a confirmation.
- `jarvis ask` takes a typed `y` as yes. It asked `[yes/no]` but read the answer as speech, where `y` is
  no, so a `y` declined the action.
- When the model answers a question with no words after looking something up, Jarvis now asks it once
  more for an answer, then says it found none. It used to say "Done.", which the claim check then turned
  into "I didn't manage to do that". "Done." is kept for an action that succeeded.
- Summarising the remembered conversation no longer happens silently at the start of a request, on that
  request's clock. It now runs right after the reply that made the conversation too long, while the reply
  is spoken and the model is loaded. With a cold model, the summary used to time out and the older turns
  were dropped instead.
- Time the PC spends asleep now counts. The PC-audio permission (`audio.pc_audio_grant_s`) and the
  conversation's idle expiry (`memory_idle_s`) ran on a clock that stops during a suspend, so a yes
  given to PC audio before a night asleep still held in the morning. They now run on the boot clock.
- A Spotify library refresh in which every request failed (Spotify erroring, the network down with the
  token still valid) is now tried again after a minute. It used to count as done, so the artist names that
  help speech to text stayed empty for `spotify.library_refresh_s` (six hours by default).
- A reply with a sentence that sounds like the wake word can be spoken over again once that sentence
  has played (a second later, for the wake model's lag). The guard used to hold until the whole reply
  ended, and it started when the sentence was made, which could be before the ones ahead of it had played.
- `write_file` no longer replaces a file you made read-only: a replace renamed a new file over it, which
  needs only the folder's permission. A read-only file is now refused for a replace and an append alike,
  before the question.
- An audit event is no longer lost when the log rotates while another Jarvis process (the voice service,
  the chat window, `jarvis ask`) writes to it, and a full log is rotated once rather than twice. Every
  writer now holds a lock on the log for the size check, the rotation and the write.
- Emptying or damaging the shared action count (`actions.json`) no longer lifts `boundaries.max_actions`:
  each process's own count must allow an action too. The shared file's answer used to replace it.

- "Play my liked songs" draws from the whole library: 50 songs picked at random from the newest page and
  three more pages at random places in it. It always played the newest 50 shuffled and said "your liked
  songs"; with more than 50 it now says "50 of your liked songs".

- Saving a setting (`jarvis setup`, `consent`, `pc-audio`, `wake`, the chat's `/config`) changes only that
  setting in `config.toml` and keeps the rest of the file, comments included. It wrote every setting with
  its value at the time, about 140 lines, so a default changed in a later version never reached the install
  and comments were lost. A setting set back to its default is taken out of the file. Defaults written by
  an earlier save stay until they are changed or removed by hand. Jarvis now depends on tomlkit (MIT).

- A question in the chat no longer takes words typed ahead as a yes. It opened with "Yes" chosen and took
  Enter, `1` or `y` for yes after 0.4 s, so the owner typing the next request could answer it. "No" is now
  chosen at first (Enter says no), `y` no longer means yes, and a `1` typed among other letters waits until
  the typing stops; `1`, or moving up to "Yes" and Enter, says yes.

- The calculator's `log` with one number is to base 10, as on a calculator: `log(100)` is 2. It was the
  natural logarithm (4.6), which `ln` still gives.

- The calculator bounds a logarithm like every other step: a 47-character `log` of a huge exact base ran for
  more than 30 s while it checked for a whole answer, and now answers at once. A logarithm to a base just
  above 1 (`log(5, 1 + 1/10**30)`) is worked out instead of failing with an internal error, and one too
  close to 1 to work out says so. A number with a leading zero is refused, so `min(1,000, 2,500)` says to
  write 1000 instead of answering 0.

- The calculator keeps a floor division exact: `(7 // 2)!` is 6 and `10**18 // 7` is 142857142857142857. A
  floor division of two exact numbers gave a plain integer, which a factorial or `round`'s digits refused
  and which was shown through a float (142857142857142864).

- A request that finds the model unloaded gives its first model call the 180 s the load may take on top of
  `model.request_timeout_s`. The request was allowed 270 s, but the call was cut at 120 s, leaving about 13 s
  for the answer after a load measured at 107 s while gaming; a slower load ended in "My model isn't
  answering right now."

- No follow-up window opens while another program plays audio (music, a video, a game, a call): anything the
  microphone heard for 6 s after a reply was taken as your next request, so sung or played lyrics started
  requests, and each reply opened the window again. Speech heard in the window is dropped too if such audio
  started meanwhile. `audio.follow_up_over_audio = true` keeps the window open over other audio.

- The chat no longer closes on a picture whose header claims more than about 179 million pixels (a big scan
  or map, or a crafted file): Pillow refused it with an error the picture check did not catch, and the
  window ended. It is now refused as too large to read, from the chat and from `show_image`, and any other
  failure while loading a picture leaves a note instead of ending the window.

- `files.deny` closes a folder however its path is written: `~/Documents/private`, an absolute path, a
  trailing slash or a leading `./`. Only `Documents/private` matched before; the other forms were accepted
  and matched nothing, so the folder stayed readable. An absolute path now also closes a folder under a
  read root outside home, and an entry that could never match is refused when the config loads.

- The calculator reads a percentage followed by a plus or minus as a percentage: `80 * 15% + 80` is 92 and
  `100 - 20% - 5` is 94.8. It read the `%` as a remainder and answered 0 and 100. A `%` with space on both
  sides (`17 % 5`) or between two numbers (`17%5`) is still the remainder, and so is `mod`.

- The wait for your answer no longer counts against a request's time limit. Reading a question out, waiting
  for a spoken yes or for the confirm key came out of `agent.max_seconds`, so an approved action could run
  and the reply still said "That took too long, so I stopped." or, with a few seconds left, "My model isn't
  answering right now." Once an action has run, the model gets at least 15 s to say how it went, and a
  request that runs out of time after acting says what it did instead of that it stopped.

- "Skip two songs" skips two: `spotify_next` and `spotify_previous` take a `count` (1 to 10), since a second
  identical call in one request is refused as a repeat. It skipped one and said two. A skip that fails
  part-way says how many tracks it moved by. To make room for the count in the default context,
  `chat.memory_tokens` defaults to 9,900 (was 10,000); a config that still sets 10,000 gets what the
  window leaves.

- An action that ran but did not end cleanly is no longer reported as not done: a shell command that
  exits non-zero, a program asked to stop that is still running, and a program that started without its
  window showing where asked. A reply may say what happened, instead of being replaced by "Sorry, I didn't
  manage to do that", and the chat's steps show "done, not cleanly" rather than "not done".

- A reply that says something was done is checked against what ran: only an action that succeeded
  counts, so after Jarvis has only looked something up (what is playing, a process list, a folder, a
  search) "Paused it for you" is challenged like any claim with nothing behind it. The check also knows
  the PC verbs (opened, launched, wrote, created, appended, ran, closed, killed, deleted, sent and the
  like), and a player command with nothing playing counts as not done.

- From the review of the run-config change: a model whose drive is not mounted when Jarvis starts is known
  again by its path, so a request says the model file is missing at once instead of waiting for the server,
  and pictures stay on for when the drive is back. `jarvis setup` refuses a run config whose engine would
  fill the card with its expert cache unless `strata.vram_reserve_mib` is set, and `jarvis doctor` fails
  "VRAM left for games" then; it also names a missing model drive as such. The learned expert profile is
  kept per model, a run config that sets its own `--host` or `--port` is refused, a relative routing
  profile is a setup error, and a bare `--vision` without a picture reader is dropped.

- The chat's working line says "Checking the time" while Jarvis reads the clock; it said "Running Time now".

- The calculator works out a negative number that is not exact (`cos(pi)`, `sin(4)`) to a whole power;
  it refused it as a fractional power.

- The calculator rounds a half away from zero, as calculators do: `round(2.5)` is 3 and `round(0.125, 2)`
  is 0.13. It used Python's rounding to even, which gave 2 and 0.12.

- A plain command that needs the model after all (it failed, or its result needs wording) no longer sends
  the model a remembered conversation larger than its budget: the conversation is summarized or trimmed
  first, as for any other request.

- `open_app` builds a valid Hyprland command for a program whose command or window name holds an invisible
  character beyond Latin-1 (a zero-width space, a line separator). Such a character was escaped in a form Lua
  refuses, so the program did not open.

- Reading in parts no longer skips text full of angle brackets (HTML, XML, code). `read_file`, `list_dir`
  and `os_skill` sized each part in plain characters, but each `<` and `>` reaches the model spelled out
  in four, and the result is then cut to its cap: the cut took the line that says where the next part
  starts, so what followed was never read. Parts are now sized as they are sent. A cut `pc_shell` output
  keeps its note that it was cut, and a page that has a picture names the picture before the text, so a cut
  never takes either.

- `read_file` no longer loses most of a line longer than one part (a one-line JSON or minified file). It
  cut the line at about 2,900 characters and reported it as read; now it says which characters it gave,
  and `from_char` reads on inside the line.

- A `[boundaries.classes]` entry that sets `pc_shell` below `strong` is refused with a message that names
  what `pc_shell` takes; it named only `stop_process` before.

- A shell command's output (`pc_shell`) is now outside text of its own kind, `shell`. It was marked as a
  process listing, which `stop_process` lets through. So a remembered output (a file shown with `cat`, a page
  fetched with `curl`) did not block a later shell command in the chat, and within one request it did not
  block `stop_process`. The output also counts as file text now: a search that sends text off the
  PC afterwards needs your yes.

- With the Azure wake word, a pause after "Hey Jarvis" no longer loses the request. The keyword's own last
  quarter second started the capture, so a pause longer than `end_silence_s` ended it before the request
  was said, and it was dropped as too little speech. Now only speech after the keyword starts a request,
  and Jarvis waits up to `no_speech_timeout_s` for it, as with the built-in wake word.

- Reading an Omarchy guide in parts: each part names the guide's version, and a later part asked for
  after an Omarchy update changed the guide is refused with a note to read it again from part 1, instead
  of mixing two versions.

- A model server that answers but lists no model (Strata while its engine is down or restarting) is
  shown as `unloaded`, not as unreachable, and a wake word then loads the model.

- A pause during the start-up load, or right after a wake word, now frees the model once the load
  is over. Both loads skipped the turn-taking with a pause, so the pause's unload met a busy server and
  Jarvis stayed paused with the model loaded.

- Outside text in a tool result (a web page, a file) now fits the size planned for it: its angle
  brackets are spelled out before it is cut, not after, which could make it four times as long. A long
  request also keeps its tool results whole while they fit: the system prompt was counted twice when
  making room.

- `jarvis status`, `status --json` and the status bar say the model is loading while a load Jarvis asked
  for is under way (at start, after a resume, on a wake word). Strata reports only loaded or unloaded, so a
  2-minute load after `jarvis restart` showed `model: unloaded`. Each load request now waits as long as
  Strata may take to answer it (three engine starts of 15 minutes), the start-up load is sent even while
  Strata lists no model, a load a resume asked for stops retrying once you pause again, and a wake word during a load joins it
  instead of queueing a second load.
  A running Jarvis whose model server is down now says `server unreachable` or `file missing`, as it
  does when Jarvis is not running.

- `jarvis start` and `jarvis restart` no longer print "Failed to reset failed state of unit jarvis.service:
  Unit jarvis.service not loaded" when Jarvis was stopped: only a unit in the `failed` state is reset.
  `jarvis help` now lists the commands, as `jarvis --help` does.
- The audit log now says why a confirmation ended as no (#38). Before, the PC sleeping with the question
  open, no answer in time, a lost microphone and a spoken no all wrote the same line. Each no now carries a
  `reason` from a fixed list, and the chat's activity pane shows it. Only a clear yes approves, as before.
- A yes no longer approves a question that was hushed, paused or could not be played before it was heard
  out: that question is a no (`not_heard`), and no answer is listened for.
- A yes spoken after the PC woke from a suspend no longer answers a question asked before it. The question is
  now counted as a no, and Jarvis listens afresh (#22). A sleep also ends the follow-up window, so speech
  after waking is no longer taken as a request without the wake word, and drops a request begun after the
  wake word; speech heard before the sleep is not answered either (#22).
- Your Voxtype dictation starting while Jarvis is capturing a request now drops that request: before, what you
  dictated could be transcribed and answered as a request. Jarvis stops capturing, says nothing, records one
  `wake_ignored` event and never touches Voxtype's recording (#3). A pause that lands after the last
  frame of a reply's follow-up window was read, and before the request heard in it is taken, no longer lets
  that request through (#11).
- `jarvis pause`, `resume` and every other stop of the microphone, and a restart of the Voxtype status
  follower, no longer leak the capture's pipe (#32). A stop waits at most half a second for the reader of a
  capture whose command left a child holding the pipe, and keeps that pipe instead of closing it under the
  reader; a stop that lands while a start is under way no longer closes the new capture's pipe; and a
  status line that cannot be handled ends the Voxtype command instead of waiting for it for ever. The
  tests now fail on any warning, so an unclosed pipe or file shows up in CI.
- The chat no longer shows a reply twice for a moment: the draft is gone before the whole reply
  appears.
- Web research is safer and steadier. Text a browser hides (the `hidden` attribute, `aria-hidden`, and
  inline `display:none`, `visibility:hidden`, `opacity:0` or `font-size:0`) no longer reaches the model, a page
  with more than 20,000 elements is read from its start so it cannot hold up a request, the weather and
  Reddit archive answers must arrive within 15 seconds, the SearXNG container is limited to 200 processes, and
  the documented command now creates its `.env` file readable only by you. After Spotify says "too many
  requests", Jarvis sends nothing to Spotify until the wait it asked for is over.
- Voice and speech to text, from a practices audit: the check that asks whether a sentence would wake Jarvis
  itself now low-pass filters before it drops Kokoro's and Piper's speech to 16 kHz, so sounds above 8 kHz
  no longer fold into the band; a sentence of more than 510 phonemes is split between words and all of it
  is spoken, instead of cut mid-word; Kokoro refuses a model file it cannot run when it loads, not at the
  first sentence, and accepts the older build's `tokens` input; Voxtype runs quietly and keeps no audio of a
  failed transcription from Jarvis's runs; and Jarvis waits as long as Strata's `Retry-After` says while it loads the model at start.
- `jarvis chat` could end with a `NoMatches` traceback when it was closed: a timer that redraws the
  window, reads the log or draws the toolbar could fire after the window's parts were taken down.
- In `jarvis chat`, a request from the log without a time no longer ends in a lone "·", a reply that
  starts with a heading no longer leaves its `●` on a line of its own, the details after a step's name are
  no longer bold, and the transcript fills the window's width instead of stopping six columns short.
- A web page read a second time in a session (or one much like a page read before) came back as "no
  readable text": trafilatura's duplicate check remembers text across calls. It is off now.
- A request that reaches its step limit no longer ends with "That needed too many steps, so I stopped.":
  its last model call gets no tools and answers from what the earlier steps found. Research questions
  asked by voice that kept searching and re-reading pages now get an answer.
- `docs/configuration.md` warns that its echo-cancel setup broke the PC's sound output when it was applied
  on 2026-11-21, and says how to undo it.
- Asked the time, Jarvis says only the time, and asked the date, only the date: the time tool no longer
  hands the model both, and the core prompt tells it to answer only what was asked.
- The check that Jarvis does not claim an action no tool did now also catches "I set the volume to
  thirty", "I've lowered the volume", "I muted it", "The volume is now at 20" and "Sure — I muted it".
  A question that asks ("Shall I put it on shuffle?", "Muted, or just quieter?") is not a claim, but
  "Done, anything else?" still is.

[Unreleased]: https://github.com/NecturaLabs/Jarvis/commits/dev
