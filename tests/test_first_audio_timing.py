"""The first sentence of a long reply reaches the speaker while the model is still writing it (#12).

A fake model writes a reply one word at a time on a fake clock; the real agent, its stream and the
real sentence splitter decide when each sentence goes to a fake speaker, which notes the fake time.
Nothing here sleeps, plays a sound or touches a server. What it proves is the order and the
overlap; how many real seconds the overlap is worth depends on the model's speed and is measured
on the owner's PC with `tools/first_audio.py`.
"""

import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pytest
from test_agent import Clock, Counter

from jarvis.agent import Agent, Hooks
from jarvis.app import no_action_operation
from jarvis.audit import AuditLog
from jarvis.model import Delta, Thinking
from jarvis.operations import Registry

if TYPE_CHECKING:
    from collections.abc import Callable

START = 1000.0  # the fake clock's reading when the owner's request arrives
TOKEN_S = 0.1  # one word written every tenth of a second

SENTENCES = [
    "Owls turn heads.",  # 3 words: complete at 0.3 s
    "They hear very well in the dark.",  # 7 words: complete at 1.0 s
    "Their feathers make no sound as they fly at night.",  # 10 words: complete at 2.0 s
    "Goodnight, and sleep well, my dear friend, and thank you.",  # 10 words: complete at 3.0 s
]
REPLY = " ".join(SENTENCES)
FINISHED_S = 3.0  # thirty words at a tenth of a second


def _words(text: str) -> list[str]:
    """Return the text as the pieces a model writes: each word with the space before it."""
    return re.findall(r"\s*\S+", text)


def _text_pieces(pieces: list[str], token_s: float) -> list[tuple[float, Delta]]:
    return [((number + 1) * token_s, Delta(text=piece)) for number, piece in enumerate(pieces)]


def _reply_pieces(pieces: list[str], token_s: float) -> list[tuple[float, Delta]]:
    """Return the pieces of a `no_action` call whose `reply` is the text, as its JSON arrives."""
    arguments = [json.dumps(piece)[1:-1] for piece in pieces]  # JSON string escapes, no quotes
    arguments[0] = '{"reply": "' + arguments[0]
    arguments[-1] += '"}'
    opener = [(0.0, Delta(call=0, name="no_action"))]
    return opener + [
        ((number + 1) * token_s, Delta(call=0, arguments=piece))
        for number, piece in enumerate(arguments)
    ]


def _reply_call(text: str) -> dict[str, Any]:
    function = {"name": "no_action", "arguments": json.dumps({"reply": text})}
    call = {"id": "c1", "type": "function", "function": function}
    return {"role": "assistant", "content": "", "tool_calls": [call]}


def _tool_call(name: str) -> dict[str, Any]:
    function = {"name": name, "arguments": "{}"}
    call = {"id": "t1", "type": "function", "function": function}
    return {"role": "assistant", "content": "", "tool_calls": [call]}


@dataclass
class Written:
    """One model call: what it finally returns and the pieces it streams, each at its own time."""

    message: dict[str, Any]
    pieces: list[tuple[float, Delta]]


@dataclass
class TimedModel:
    """A model that writes on a fake clock: each piece arrives at its time after the call began.

    With `streaming` off it behaves like a server that sends the answer whole: it takes the same
    time, then hands the message back without having called `on_delta`.
    """

    clock: Clock
    calls: list[Written]
    streaming: bool = True
    ended: list[float] = field(default_factory=list[float])
    offered: list[list[str]] = field(default_factory=list[list[str]])

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        require_tool: bool = False,
        timeout_s: float | None = None,
        on_delta: Callable[[Delta], None] | None = None,
        max_tokens: int | None = None,
        load_s: float = 0.0,
        thinking: Thinking | None = None,
    ) -> dict[str, Any]:
        del messages, require_tool, timeout_s, max_tokens, load_s, thinking
        self.offered.append([str(tool["function"]["name"]) for tool in tools])
        written = self.calls.pop(0)
        began = self.clock.now
        for at, delta in written.pieces:
            self.clock.now = began + at
            if self.streaming and on_delta is not None:
                on_delta(delta)
        self.ended.append(self.clock.now - START)
        return written.message

    def is_loaded(self) -> bool | None:  # ruff: ignore[no-self-use] - the ChatModel protocol
        return True


@dataclass
class Speaker:
    """What was handed to the speaker, and when (seconds after the request)."""

    clock: Clock
    handed: list[tuple[float, str]] = field(default_factory=list[tuple[float, str]])
    hushes: int = 0

    def speak(self, text: str) -> None:
        self.handed.append((self.clock.now - START, text))

    def hush(self) -> None:
        self.hushes += 1

    @property
    def words(self) -> list[str]:
        return [text for _, text in self.handed]

    @property
    def times(self) -> list[float]:
        return [at for at, _ in self.handed]

    def hooks(self) -> Hooks:
        return Hooks(confirm=lambda _q: False, speak=self.speak, hush=self.hush)


def _run(
    calls: list[Written], *, streaming: bool = True, ops: Counter | None = None
) -> tuple[Speaker, TimedModel, str]:
    clock = Clock()
    clock.now = START
    model = TimedModel(clock, calls, streaming=streaming)
    registry = Registry([*([ops.op("pause")] if ops else []), no_action_operation()])
    agent = Agent(model, registry, "system", AuditLog(None), {}, clock=clock)
    speaker = Speaker(clock)
    reply = agent.handle("tell me about owls", speaker.hooks())
    return speaker, model, reply


def owl_reply() -> Written:
    return Written(_reply_call(REPLY), _reply_pieces(_words(REPLY), TOKEN_S))


def test_the_timeline_is_what_the_comments_say() -> None:
    assert len(_words(REPLY)) * TOKEN_S == pytest.approx(FINISHED_S)
    assert [len(_words(sentence)) for sentence in SENTENCES] == [3, 7, 10, 10]


def test_the_first_sentence_is_handed_over_while_the_model_is_still_writing() -> None:
    speaker, model, reply = _run([owl_reply()])

    assert reply == REPLY
    assert model.ended == [pytest.approx(FINISHED_S)]
    # A sentence is whole once the space after it arrives, which comes with the next word: the
    # first one at 0.4 s, the second at 1.1 s, the third at 2.1 s. The last has no next word, so
    # it goes when the reply ends.
    assert speaker.times == pytest.approx([0.4, 1.1, 2.1, FINISHED_S])
    assert speaker.words == SENTENCES
    assert speaker.times[0] < model.ended[0]  # before generation ended ...
    assert FINISHED_S - speaker.times[0] == pytest.approx(2.6)  # ... by this much
    assert speaker.hushes == 0


def test_every_sentence_but_the_last_is_handed_over_before_generation_ends() -> None:
    speaker, model, _ = _run([owl_reply()])

    assert all(at < model.ended[0] for at in speaker.times[:-1])
    assert speaker.times == sorted(speaker.times)


def test_each_sentence_is_spoken_once_and_in_order() -> None:
    speaker, _, reply = _run([owl_reply()])

    assert speaker.words == SENTENCES
    assert len(set(speaker.words)) == len(speaker.words)
    assert " ".join(speaker.words) == reply


def test_a_server_that_sends_the_answer_whole_speaks_only_when_it_ends() -> None:
    whole, model, reply = _run([owl_reply()], streaming=False)

    assert reply == REPLY
    assert whole.words == [REPLY]
    assert whole.times == pytest.approx([model.ended[0]])

    streamed, _, _ = _run([owl_reply()])
    # The gain from streaming on this timeline: the first sentence 2.6 s sooner.
    assert whole.times[0] - streamed.times[0] == pytest.approx(2.6)


def test_a_reply_that_is_one_sentence_has_nothing_to_overlap() -> None:
    sentence = "Owls turn their heads a long way round."
    calls = [Written(_reply_call(sentence), _reply_pieces(_words(sentence), TOKEN_S))]

    speaker, model, _ = _run(calls)

    assert speaker.words == [sentence]
    assert speaker.times == pytest.approx(model.ended)


@pytest.mark.parametrize("path", ["reply call", "plain text"])
def test_the_gain_does_not_depend_on_how_the_model_writes_the_reply(path: str) -> None:
    pieces = _words(REPLY)
    if path == "reply call":
        written = Written(_reply_call(REPLY), _reply_pieces(pieces, TOKEN_S))
    else:
        written = Written({"role": "assistant", "content": REPLY}, _text_pieces(pieces, TOKEN_S))

    speaker, model, _ = _run([written])

    assert speaker.words == SENTENCES
    assert speaker.times[0] == pytest.approx(0.4)
    assert speaker.times[0] < model.ended[0]


@pytest.mark.parametrize("size", [1, 4, 17])
def test_the_first_sentence_goes_at_the_first_piece_that_completes_it(size: int) -> None:
    """Pieces of any size: the piece that brings the space after the sentence hands it over."""
    text = REPLY
    pieces = [text[i : i + size] for i in range(0, len(text), size)]
    step = FINISHED_S / len(pieces)
    written = Written({"role": "assistant", "content": text}, _text_pieces(pieces, step))
    seen = ""
    expected = FINISHED_S
    for at, delta in written.pieces:
        seen += delta.text
        if len(seen) > len(SENTENCES[0]) and seen[len(SENTENCES[0])].isspace():
            expected = at
            break

    speaker, model, _ = _run([written])

    assert speaker.words == SENTENCES
    assert speaker.times[0] == pytest.approx(expected)
    assert speaker.times[0] < model.ended[0]


def test_an_answer_written_after_a_tool_ran_is_spoken_while_it_is_written() -> None:
    counter = Counter()
    first = Written(_tool_call("pause"), [(0.2, Delta(call=0, name="pause"))])
    answer = Written({"role": "assistant", "content": REPLY}, _text_pieces(_words(REPLY), TOKEN_S))

    speaker, model, _ = _run([first, answer], ops=counter)

    assert counter.runs == [{"op": "pause"}]
    # The second call starts at 0.2 s, when the first ended; its words come 0.1 s apart from there.
    assert model.ended == pytest.approx([0.2, 0.2 + FINISHED_S])
    assert speaker.words == SENTENCES
    assert speaker.times == pytest.approx([0.6, 1.3, 2.3, 0.2 + FINISHED_S])
    assert speaker.times[0] < model.ended[1]
