"""Speaking a reply while the model writes it: only words that will be the reply, never a claim
made before the action, nothing twice, and nothing after the owner speaks over Jarvis."""

import json
from dataclasses import dataclass, field
from typing import Any

import pytest
from test_agent import Clock, Counter, ScriptedModel, stream_pieces

from jarvis import app
from jarvis.agent import (
    MAX_STREAMED_CHARS,
    UNDONE,
    Agent,
    Hooks,
    claims_action,
    partial_reply,
)
from jarvis.app import no_action_operation
from jarvis.audit import AuditLog
from jarvis.operations import Operation, Registry
from jarvis.sentences import sentence_end, split_sentences
from jarvis.shortcuts import Shortcut


@dataclass
class Ear:
    """What the owner hears, and how far the model had written when each part was handed over."""

    model: ScriptedModel
    heard: list[str] = field(default_factory=list[str])
    when: list[int] = field(default_factory=list[int])
    hushes: int = 0
    hushed_at: list[int] = field(default_factory=list[int])
    stop_after: str = ""  # cancel the request once this has been heard
    cancelled: bool = False
    _pieces: int = 0

    def __post_init__(self) -> None:
        self.model.after_piece = self._piece

    def _piece(self, _number: int) -> None:
        self._pieces += 1  # pieces streamed so far, over every model call of the request

    def speak(self, text: str) -> None:
        self.heard.append(text)
        self.when.append(self._pieces)
        if self.stop_after and text == self.stop_after:
            self.cancelled = True

    def hush(self) -> None:
        self.hushes += 1
        self.hushed_at.append(self._pieces)

    def hooks(self) -> Hooks:
        return Hooks(
            confirm=lambda _q: False,
            cancelled=lambda: self.cancelled,
            speak=self.speak,
            hush=self.hush,
        )


def _call(name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
    function = {"name": name, "arguments": json.dumps(args or {})}
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "c1", "type": "function", "function": function}],
    }


def _say(text: str) -> dict[str, Any]:
    return {"role": "assistant", "content": text}


def _agent(
    model: ScriptedModel, ops: list[Operation], overrides: dict[str, str] | None = None
) -> tuple[Agent, AuditLog]:
    audit = AuditLog(None)
    registry = Registry([*ops, no_action_operation()])
    return Agent(model, registry, "system", audit, overrides or {}, clock=Clock()), audit


def _events(audit: AuditLog) -> list[object]:
    return [e["event"] for e in audit.events]


def test_a_spoken_answer_starts_before_the_model_has_finished_writing_it() -> None:
    reply = "Owls can turn their heads a long way. They hear very well. Goodnight."
    model = ScriptedModel([_call("no_action", {"reply": reply})])
    ear = Ear(model)
    agent, _ = _agent(model, [])

    assert agent.handle("tell me about owls", ear.hooks()) == reply

    assert ear.heard == [
        "Owls can turn their heads a long way.",
        "They hear very well.",
        "Goodnight.",
    ]
    assert ear.when[0] < ear.when[-1]  # the first sentence was handed over mid-stream
    assert model.streamed == [True]


def test_an_answer_after_a_tool_streams_and_may_report_what_was_done() -> None:
    counter = Counter()
    model = ScriptedModel([_call("pause"), _say("Paused it for you. Enjoy the quiet.")])
    ear = Ear(model)
    agent, _ = _agent(model, [counter.op("pause")])

    agent.handle("pause", ear.hooks())

    assert ear.heard == ["Paused it for you.", "Enjoy the quiet."]
    assert ear.when[0] < ear.when[-1]


def test_a_claim_before_any_action_is_never_spoken_while_streaming() -> None:
    counter = Counter()
    model = ScriptedModel(
        [
            _call("no_action", {"reply": "Sure thing. Paused it for you. Anything else?"}),
            _call("no_action", {"reply": "Paused it for you. Anything else?"}),
        ]
    )
    ear = Ear(model)
    agent, _ = _agent(model, [counter.op("pause")])

    assert agent.handle("pause", ear.hooks()) == UNDONE

    assert "Paused it for you." not in " ".join(ear.heard)
    assert ear.heard[-1] == UNDONE
    assert counter.runs == []


def test_words_before_a_tool_call_stop_but_what_was_said_plays_to_its_end() -> None:
    counter = Counter()
    first = _call("pause")
    first["content"] = "Let me look. One moment."
    model = ScriptedModel([first, _say("Paused.")])
    ear = Ear(model)
    agent, audit = _agent(model, [counter.op("pause")])

    assert agent.handle("pause", ear.hooks()) == "Paused."

    # "One moment." was not finished when the call began, so it is never said; "Let me look."
    # passed the claim check, so it is heard to its end rather than cut off mid-word.
    assert ear.heard == ["Let me look.", "Paused."]
    assert ear.hushes == 0
    assert counter.runs == [{"op": "pause"}]
    assert "spoken_before_tool_call" in _events(audit)


def test_no_action_after_another_call_is_not_spoken_until_the_answer_is_whole() -> None:
    counter = Counter()
    both = _call("pause")
    both["tool_calls"].append(_call("no_action", {"reply": "Done. Paused."})["tool_calls"][0])
    model = ScriptedModel([both, _say("Paused it.")])
    ear = Ear(model)
    agent, _ = _agent(model, [counter.op("pause")])

    agent.handle("pause", ear.hooks())

    assert ear.heard == ["Paused it."]


def test_a_no_action_the_owner_switched_off_is_never_streamed() -> None:
    model = ScriptedModel([_call("no_action", {"reply": "Hello there. Nice day."}), _say("Hi.")])
    ear = Ear(model)
    agent, _ = _agent(model, [], overrides={"no_action": "deny"})

    agent.handle("hello", ear.hooks())

    assert "Hello there." not in ear.heard


def test_an_actions_note_the_model_copied_is_held_and_never_spoken() -> None:
    counter = Counter()
    model = ScriptedModel([_call("pause"), _say("[actions taken: pause -> ok] Paused. Enjoy.")])
    ear = Ear(model)
    agent, _ = _agent(model, [counter.op("pause")])

    assert agent.handle("pause", ear.hooks()) == "Paused. Enjoy."
    assert ear.heard == ["Paused. Enjoy."]


def test_a_cancelled_request_stops_the_stream_and_keeps_only_what_was_said() -> None:
    reply = "First part. Second part. Third part."
    model = ScriptedModel([_call("no_action", {"reply": reply})])
    ear = Ear(model, stop_after="First part.")
    agent, audit = _agent(model, [])

    assert agent.handle("talk", ear.hooks()) == "First part."
    assert ear.heard == ["First part."]
    assert audit.events[-1]["event"] == "reply"


def test_nothing_past_the_stream_cap_is_spoken_before_the_answer_is_whole() -> None:
    counter = Counter()
    sentence = "This is a sentence of words. "
    reply = (sentence * (MAX_STREAMED_CHARS // len(sentence) + 10)).strip()
    answer = _say(reply)
    model = ScriptedModel([_call("pause"), answer])
    ear = Ear(model)
    agent, _ = _agent(model, [counter.op("pause")])

    agent.handle("pause and talk", ear.hooks())

    whole = len(stream_pieces(_call("pause"), model.piece)) + len(
        stream_pieces(answer, model.piece)
    )
    streamed = [h for h, at in zip(ear.heard, ear.when, strict=True) if at < whole]
    assert sum(len(h) + 1 for h in streamed) <= MAX_STREAMED_CHARS + 1
    assert " ".join(ear.heard) == reply


def test_the_stream_cap_is_the_no_action_reply_cap() -> None:
    assert MAX_STREAMED_CHARS == app.MAX_REPLY_CHARS


def test_without_a_speak_hook_the_model_is_not_streamed() -> None:
    model = ScriptedModel([_call("no_action", {"reply": "Hi."})])
    agent, _ = _agent(model, [])

    assert agent.handle("hi", Hooks(confirm=lambda _q: False)) == "Hi."
    assert model.streamed == [False]


def test_a_shortcut_reply_is_spoken_through_the_hook() -> None:
    counter = Counter()
    model = ScriptedModel([])
    ear = Ear(model)
    pause = counter.op("pause", spoken="Paused.", shortcuts=(Shortcut("pause"),))
    agent, _ = _agent(model, [pause])

    assert agent.handle("pause", ear.hooks()) == "Paused."
    assert ear.heard == ["Paused."]
    assert model.seen == []


@pytest.mark.parametrize(
    ("arguments", "words"),
    [
        ("", ""),
        ('{"rep', ""),
        ('{ "reply" : "Hi th', "Hi th"),
        ('{"reply":"a\\', "a"),
        ('{"reply":"a\\\\', "a\\"),
        ('{"reply":"caf\\u00', "caf"),
        ('{"reply":"caf\\u00e9!', "café!"),
        ('{"reply":"x\\ud83d', "x"),
        ('{"reply":"x\\ud83d\\ude00', "x\U0001f600"),
        ('{"reply":"done."}', "done."),
        ('{"other": 1}', None),
        ("[1, 2]", None),
    ],
)
def test_a_partial_reply_is_read_only_as_far_as_it_is_whole(
    arguments: str, words: str | None
) -> None:
    assert partial_reply(arguments) == words


@pytest.mark.parametrize(
    ("text", "first"),
    [
        ("Paused. Next", "Paused."),
        ("Really? Yes", "Really?"),
        ("Line one\nLine two", "Line one"),
        ("Dr. Who is on. Next", "Dr. Who is on."),
        ("Use e.g. vinegar. Next", "Use e.g. vinegar."),
        ("J. R. R. Tolkien wrote it. Next", "J. R. R. Tolkien wrote it."),
        ("Steps:\n1. Boil. Next", "Steps:"),
        ("It costs 3.5 euros. Next", "It costs 3.5 euros."),
        ("Not finished.", None),
        ("Not finished", None),
    ],
)
def test_a_sentence_ends_only_where_it_really_ends(text: str, first: str | None) -> None:
    end = sentence_end(text)
    assert (None if end is None else text[:end]) == first


def test_split_sentences_keeps_the_unfinished_tail() -> None:
    assert split_sentences("One. Two!  Three") == ["One.", "Two!", "Three"]


@pytest.mark.parametrize(
    "reply",
    [
        "Sure. Paused it for you.",
        "I found it. Playing it now. Enjoy.",
        "Here you go.\nSkipped to the next one.",
        "No problem. I couldn't find that. Nothing was saved.",
        "I set the volume to thirty. Shall I put it on shuffle?",
        "Done, anything else? I muted it.",
        "Sure \u2014 I muted it. Want more?",
        "Set it to what? The volume's up to you.",
    ],
)
def test_the_claim_check_gives_the_same_answer_in_pieces_as_whole(reply: str) -> None:
    pieces = split_sentences(reply)
    assert claims_action(reply) == any(claims_action(piece) for piece in pieces)


def _two_calls(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    message = dict(first)
    other = dict(second["tool_calls"][0], id="c2")
    message["tool_calls"] = [*first["tool_calls"], other]
    return message


def test_an_exact_copy_of_the_reply_call_does_not_cut_the_reply() -> None:
    reply = _call("no_action", {"reply": "Alpha one. Beta two. Gamma three."})
    model = ScriptedModel([_two_calls(reply, reply)])
    ear = Ear(model)
    agent, _ = _agent(model, [])

    agent.handle("talk", ear.hooks())

    assert ear.hushes == 0
    assert ear.heard == ["Alpha one.", "Beta two.", "Gamma three."]


def test_a_call_after_a_held_claim_never_says_the_claim_and_audits_what_was_said() -> None:
    counter = Counter()
    first = _call("pause")
    first["content"] = "Let me look. Paused it. One moment."
    model = ScriptedModel([first, _say("Paused.")])
    ear = Ear(model)
    agent, audit = _agent(model, [counter.op("pause")])

    agent.handle("pause", ear.hooks())

    assert ear.heard[0] == "Let me look."
    assert "Paused it." not in ear.heard
    assert ear.hushes == 0
    assert _events(audit).count("spoken_before_tool_call") == 1


def test_words_of_a_reply_call_that_is_refused_are_dropped_and_audited() -> None:
    bad = _call("no_action")
    bad["tool_calls"][0]["function"]["arguments"] = '{"reply": "Hello there. Nice day.", "x": 1}'
    model = ScriptedModel([bad, _call("no_action", {"reply": "Hi."})])
    ear = Ear(model)
    agent, audit = _agent(model, [])

    assert agent.handle("hello", ear.hooks()) == "Hi."

    assert ear.hushes == 1
    assert "spoken_before_tool_call" in _events(audit)
    assert ear.heard[-1] == "Hi."


def test_another_tools_arguments_are_never_spoken() -> None:
    counter = Counter()
    # Shaped like a reply, and refused later by the operation's schema: never heard either way.
    model = ScriptedModel(
        [_call("pause", {"reply": "Secret words. More."}), _say("I could not pause it.")]
    )
    ear = Ear(model)
    agent, _ = _agent(model, [counter.op("pause")])

    agent.handle("pause", ear.hooks())

    assert ear.heard == ["I could not pause it."]


def test_nothing_is_delivered_after_the_request_was_cancelled() -> None:
    answer = _say("Unfinished")  # text, so the reply is final without running a call
    model = ScriptedModel([answer])
    ear = Ear(model)
    agent, _ = _agent(model, [])
    last = len(stream_pieces(answer, model.piece)) - 1

    def cancel_at_the_end(number: int) -> None:
        if number == last:  # after the last piece: only the final delivery can see it
            ear.cancelled = True

    model.after_piece = cancel_at_the_end
    agent.handle("talk", ear.hooks())

    assert ear.heard == []


def test_a_claim_said_before_its_own_call_is_dropped_where_not_heard() -> None:
    # An earlier action succeeded, so the claim was not held; its own call then fails, so the
    # claim was not true: the speaker is hushed at the call, as for a reply that is not one.
    counter = Counter()
    second = _call("volume")
    second["content"] = "Volume is set to thirty. One moment."
    model = ScriptedModel([_call("pause"), second, _say("I could not change the volume.")])
    ear = Ear(model)
    agent, audit = _agent(
        model, [counter.op("pause"), counter.op("volume", error=RuntimeError("down"))]
    )

    agent.handle("pause and set the volume to thirty", ear.hooks())

    assert claims_action("Volume is set to thirty.")
    assert ear.heard[0] == "Volume is set to thirty."
    assert ear.hushes == 1
    assert _events(audit).count("spoken_before_tool_call") == 1


def test_words_before_a_lone_reply_call_are_heard_and_audited() -> None:
    first = _call("no_action", {"reply": "Hello there."})
    first["content"] = "Let me look. One moment."
    model = ScriptedModel([first])
    ear = Ear(model)
    agent, audit = _agent(model, [])

    assert agent.handle("hello", ear.hooks()) == "Hello there."

    assert ear.heard == ["Let me look.", "Hello there."]
    assert ear.hushes == 0
    assert _events(audit).count("spoken_before_tool_call") == 1


def test_words_before_a_refused_reply_call_stay_heard_and_are_audited_once() -> None:
    bad = _call("no_action")
    bad["content"] = "Let me look. One moment."
    bad["tool_calls"][0]["function"]["arguments"] = '{"reply": "Hello there.", "x": 1}'
    model = ScriptedModel([bad, _call("no_action", {"reply": "Hi."})])
    ear = Ear(model)
    agent, audit = _agent(model, [])

    assert agent.handle("hello", ear.hooks()) == "Hi."

    assert ear.heard == ["Let me look.", "Hi."]
    assert ear.hushes == 0
    assert _events(audit).count("spoken_before_tool_call") == 1


def test_a_claim_before_a_lone_reply_call_is_hushed_and_the_reply_still_heard() -> None:
    counter = Counter()
    second = _call("no_action", {"reply": "All set."})
    second["content"] = "Volume is set to thirty. One moment."
    model = ScriptedModel([_call("pause"), second])
    ear = Ear(model)
    agent, audit = _agent(model, [counter.op("pause")])

    assert agent.handle("pause", ear.hooks()) == "All set."

    assert ear.heard == ["Volume is set to thirty.", "All set."]
    assert ear.hushes == 1
    assert _events(audit).count("spoken_before_tool_call") == 1
