"""The Kokoro voices Jarvis can speak with, each with a short description (`jarvis voice`).

Jarvis reads text with American English sounds for an `a*` voice and British (RP) for a `b*`
one (`kokoro.py`), so only those 28 voices of Kokoro's voice file are listed; the others are
voices for other languages and sound wrong with English sounds. The grade is the voice's overall
grade in Kokoro's own VOICES.md (how much and how good its training audio was); the description
says what it sounds like, from listeners' write-ups.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from jarvis.operations import Operation, OperationError, Result, current_request

if TYPE_CHECKING:
    from collections.abc import Callable

    from jarvis.config import Config

KOKORO = "kokoro:"
SET_VOICE = "set_voice"


@dataclass(frozen=True, slots=True)
class Voice:
    """One Kokoro voice: its id, accent, gender, overall grade and what it sounds like."""

    id: str
    accent: str
    gender: str
    grade: str
    about: str

    @property
    def setting(self) -> str:
        """The voice as `audio.voice` takes it."""
        return KOKORO + self.id


def _v(voice_id: str, grade: str, about: str) -> Voice:
    accent = {"a": "American", "b": "British"}[voice_id[0]]
    gender = {"f": "female", "m": "male"}[voice_id[1]]
    return Voice(voice_id, accent, gender, grade, about)


VOICES: tuple[Voice, ...] = (
    _v("af_heart", "A", "warm and expressive, steady pace; the best-graded voice"),
    _v("af_bella", "A-", "warm and inviting, lively"),
    _v("af_nicole", "B-", "calm and close, slow with long pauses"),
    _v("af_aoede", "C+", "soft and melodic"),
    _v("af_kore", "C+", "bright, youthful and energetic"),
    _v("af_sarah", "C+", "friendly and easygoing"),
    _v("af_alloy", "C", "even and neutral"),
    _v("af_nova", "C", "high-energy, like a broadcaster"),
    _v("af_sky", "C-", "airy, lower than the other women"),
    _v("af_jessica", "D", "crisp and professional, reads fast"),
    _v("af_river", "D", "calm and measured, reads fast"),
    _v("am_michael", "C+", "polished and professional; the best-graded American man"),
    _v("am_fenrir", "C+", "forceful and firm"),
    _v("am_puck", "C+", "playful and light"),
    _v("am_echo", "D", "even and plain, darker"),
    _v("am_eric", "D", "warm storyteller, the highest man's voice"),
    _v("am_liam", "D", "young and laid-back"),
    _v("am_onyx", "D", "very deep and theatrical"),
    _v("am_adam", "F+", "deep and commanding; the lowest-graded voice"),
    _v("am_santa", "D-", "deep and jolly, a novelty"),
    _v("bf_emma", "B-", "warm narrator, bright; the best British voice"),
    _v("bf_isabella", "C", "youthful and animated"),
    _v("bf_alice", "D", "refined classic British, much pitch movement"),
    _v("bf_lily", "D", "gentle and airy"),
    _v("bm_george", "C", "mature and authoritative; Jarvis's default"),
    _v("bm_fable", "C", "storyteller"),
    _v("bm_lewis", "D+", "natural and conversational"),
    _v("bm_daniel", "D", "businesslike and clear, flat pitch"),
)


def find(name: str) -> Voice | None:
    """Return the listed voice called `name` (with or without `kokoro:`), else None."""
    wanted = name.strip().lower().removeprefix(KOKORO)
    return next((v for v in VOICES if v.id == wanted), None)


def with_voice(config: Config, setting: str) -> Config:
    """Return the same configuration with a different speaking voice."""
    return config.model_copy(update={"audio": config.audio.model_copy(update={"voice": setting})})


def _owner_only(args: dict[str, Any]) -> dict[str, Any]:
    request = current_request()
    if (
        request is None
        or not request.owner_asked
        or not request.words.strip()
        or request.scheduled
        or request.outside
    ):
        msg = "Only the owner's own request without outside text can change the speaking voice."
        raise OperationError(msg)
    return args


class VoiceControl:
    """The voice actually in use by the running speaker."""

    def __init__(self, setting: str, switch: Callable[[Voice], None] | None = None) -> None:
        """Start with `setting`. A settings listing has no live speaker to switch."""
        super().__init__()
        self._setting = setting
        self._switch = switch

    def setting(self) -> str:
        """Return the current speaking voice setting (for example, `kokoro:bf_emma`)."""
        return self._setting

    def _run(self, args: dict[str, Any]) -> Result:
        _owner_only(args)
        if self._switch is None:
            msg = "The voice service is not available to switch voices."
            raise OperationError(msg)
        chosen = find(str(args["voice"]))
        if chosen is None:
            msg = "That voice is not in the speaking voice catalogue."
            raise OperationError(msg)
        self._switch(chosen)
        self._setting = chosen.setting
        return Result(f"Speaking voice switched to {chosen.id}.", acted=True)

    def operation(self) -> Operation:
        """Let Jarvis select a fixed catalogue voice for a spoken request."""
        catalogue = "\n".join(f"{v.id}: {v.accent} {v.gender}, {v.about}" for v in VOICES)
        return Operation(
            "set_voice",
            "Change the speaking voice when the owner asks. Choose an ID below that fits "
            "the owner's description:\n" + catalogue,
            {
                "type": "object",
                "properties": {"voice": {"type": "string", "enum": [v.id for v in VOICES]}},
                "required": ["voice"],
                "additionalProperties": False,
            },
            self._run,
            prepare=_owner_only,
            spoken_summary=lambda args: f"switch my speaking voice to {args['voice']}",
        )
