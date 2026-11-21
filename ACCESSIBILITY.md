# Accessibility

Jarvis is a voice agent, so it is built for people who can speak and hear, and that leaves out others. This
page says what it offers today, what it does not, and where the gaps are tracked.

## Voice

- You wake Jarvis with the spoken phrase "Hey Jarvis" and speak a request; it answers aloud. Replies are
  spoken only, with no desktop notifications, so without the chat window (below) a reply is audible only.
- Jarvis is built for English: the default speech-to-text model (`audio.transcribe_model`, `small.en`) is
  English-only and the wake word is English.
- You can tune it to your speech in [docs/configuration.md](docs/configuration.md): `audio.wake_threshold`
  (how readily the wake word fires), `audio.end_silence_s` and `audio.no_speech_timeout_s` (how long a pause
  Jarvis waits through, and how long before it gives up), `audio.max_utterance_s` (the longest request),
  `audio.confirm_window_s` (how long Jarvis waits for a yes or no) and `audio.speech_speed` (how fast it
  speaks).
- Confirmations are spoken questions answered by voice. Anything but a clear yes is a no, so an unclear or
  missed answer never approves an action.

## Without speaking or hearing

- `jarvis chat` opens a terminal window where you type requests and read Jarvis's replies, live, whether the
  request was typed or spoken. Replies are formatted (headings, lists, tables, code) and shown as they are
  written. By default only requests, replies, refusals and errors show; Ctrl+O or `/verbosity` adds every
  step and its result, then every detail. A question that needs a yes opens a choice: 1 or y for yes, 2, n or
  Esc for no.
- `jarvis ask <text>` sends one typed request and prints the reply, with no audio device needed.
- Jarvis still speaks its replies aloud in both cases; there is no setting to turn speech off yet.

## Keyboard only

Jarvis has no pointer-only controls. Setup and every command run from the terminal
([README](README.md#everyday-commands)). In the chat window the cursor starts in the input line; Enter sends,
Shift+Enter (or a backslash then Enter, or Ctrl+J) starts a new line, Esc stops a request, ↑ and ↓ bring back
earlier requests (kept between sessions), `/` lists the commands as you type (Tab completes one), `?` on an
empty line shows the shortcuts, Ctrl+O shows more or less of the work, Ctrl+L redraws, Ctrl+C clears the input
(twice on an empty one leaves), and `/quit` or Ctrl+Q leaves. `/config` changes the chat's settings with the
arrow keys and Enter.

## The chat window

- The window is a terminal application (Textual), so it uses your terminal's font, size, colours and
  zoom, and works with a terminal screen reader. Whether a given reader announces new lines as they arrive
  has not been tested.
- The window uses the terminal's own colours (an ANSI theme), so it follows the Omarchy theme.
- Colour only reinforces meaning. A request starts with `>` on a shaded band and says who asked ("typed",
  "you, by voice", "PC audio:"); each step and the reply start with `●`, and a step's result sits under it
  after `⎿`. Words, not colour, say when something was refused, not done or failed. The toolbar under the
  input gives the PC's load and how full the model's context is as numbers; the coloured bars repeat them.
- Text from outside, such as song names and PC speech, is shown as plain characters, with control
  characters removed.

## Known gaps

Tracked in [issues](https://github.com/NecturaLabs/Jarvis/issues): interrupting Jarvis by voice (#13),
following up without repeating the wake word (#11), and a state indicator in the Omarchy bar (#14). Not
tracked yet: a text-only mode that skips speech, and a screen-reader test of the chat window.

To report a barrier, file a [bug](https://github.com/NecturaLabs/Jarvis/issues/new/choose) naming the
assistive technology, terminal, operating system and the step that fails, or start a
[discussion](https://github.com/NecturaLabs/Jarvis/discussions) if you are not sure it is a bug.
