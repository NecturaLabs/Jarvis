"""The Kokoro voices Jarvis can speak with, each with a short description (`jarvis voice`).

Jarvis reads text with American English sounds for an `a*` voice and British (RP) for a `b*`
one (`kokoro.py`), so only those 28 voices of Kokoro's voice file are listed; the others are
voices for other languages and sound wrong with English sounds. The grade is the voice's overall
grade in Kokoro's own VOICES.md (how much and how good its training audio was); the description
says what it sounds like, from listeners' write-ups.

The voices speak at levels up to 8 dB apart, so each has a gain that brings its speech to the
median of all 28 (`tools/voice_loudness.py` measures them: the RMS of the 20 ms frames within
40 dB of the loudest, over six sentences, at speed 1.0, measured 2026-11-28 on voices-v1.0.bin).
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from jarvis.operations import Operation, OperationError, Result, current_request

if TYPE_CHECKING:
    from collections.abc import Callable

    from jarvis.config import Config

KOKORO = "kokoro:"
SET_VOICE = "set_voice"
DEFAULT_FEMALE = "af_heart"  # the voice "female" picks
DEFAULT_MALE = "am_michael"  # the voice "male" picks


@dataclass(frozen=True, slots=True)
class Voice:
    """One Kokoro voice: its id, accent, gender, grade, what it sounds like and its gain in dB."""

    id: str
    accent: str
    gender: str
    grade: str
    about: str
    gain_db: float

    @property
    def setting(self) -> str:
        """The voice as `audio.voice` takes it."""
        return KOKORO + self.id


def _v(voice_id: str, grade: str, gain_db: float, about: str) -> Voice:
    accent = {"a": "American", "b": "British"}[voice_id[0]]
    gender = {"f": "female", "m": "male"}[voice_id[1]]
    return Voice(voice_id, accent, gender, grade, about, gain_db)


VOICES: tuple[Voice, ...] = (
    _v("af_heart", "A", +1.6, "warm and expressive, steady pace; the best-graded voice"),
    _v("af_bella", "A-", +0.4, "warm and inviting, lively"),
    _v("af_nicole", "B-", +0.1, "calm and close, slow with long pauses"),
    _v("af_aoede", "C+", -3.4, "soft and melodic"),
    _v("af_kore", "C+", -2.5, "bright, youthful and energetic"),
    _v("af_sarah", "C+", -0.4, "friendly and easygoing"),
    _v("af_alloy", "C", -0.1, "even and neutral"),
    _v("af_nova", "C", +4.9, "high-energy, like a broadcaster"),
    _v("af_sky", "C-", +1.8, "airy, lower than the other women"),
    _v("af_jessica", "D", +1.0, "crisp and professional, reads fast"),
    _v("af_river", "D", +0.4, "calm and measured, reads fast"),
    _v("am_michael", "C+", +2.5, "polished and professional; the best-graded American man"),
    _v("am_fenrir", "C+", -1.6, "forceful and firm"),
    _v("am_puck", "C+", -2.6, "playful and light"),
    _v("am_echo", "D", +0.2, "even and plain, darker"),
    _v("am_eric", "D", -0.3, "warm storyteller, the highest man's voice"),
    _v("am_liam", "D", +0.4, "young and laid-back"),
    _v("am_onyx", "D", +0.6, "very deep and theatrical"),
    _v("am_adam", "F+", -2.8, "deep and commanding; the lowest-graded voice"),
    _v("am_santa", "D-", +1.5, "deep and jolly, a novelty"),
    _v("bf_emma", "B-", -0.4, "warm narrator, bright; the best British voice"),
    _v("bf_isabella", "C", -3.0, "youthful and animated"),
    _v("bf_alice", "D", -0.6, "refined classic British, much pitch movement"),
    _v("bf_lily", "D", -0.1, "gentle and airy"),
    _v("bm_george", "C", -0.3, "mature and authoritative; Jarvis's default"),
    _v("bm_fable", "C", +2.5, "storyteller"),
    _v("bm_lewis", "D+", +2.2, "natural and conversational"),
    _v("bm_daniel", "D", -0.1, "businesslike and clear, flat pitch"),
)


def find(name: str) -> Voice | None:
    """Return the listed voice called `name` (with or without `kokoro:`), else None.

    "female" and "male" are the default voice of each.
    """
    wanted = name.strip().lower().removeprefix(KOKORO)
    wanted = {"female": DEFAULT_FEMALE, "male": DEFAULT_MALE}.get(wanted, wanted)
    return next((v for v in VOICES if v.id == wanted), None)


def with_voice(config: Config, setting: str) -> Config:
    """Return the same configuration with a different speaking voice."""
    return config.model_copy(update={"audio": config.audio.model_copy(update={"voice": setting})})


def _owner_only(args: dict[str, Any]) -> dict[str, Any]:
    request = current_request()
    if request is None or not request.owner_asked or not request.words.strip() or request.scheduled:
        msg = "Only the owner's own request can change the speaking voice."
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
            # The voice is saved in the config, so outside text in the conversation (a page read
            # just now or remembered) asks for the owner's yes first, like any action.
            side_effect=True,
            spoken_summary=lambda args: f"switch my speaking voice to {args['voice']}",
        )


def gain(voice_id: str) -> float:
    """Return the factor that brings `voice_id`'s speech to the common level (1 if unlisted)."""
    found = find(voice_id)
    return 1.0 if found is None else 10 ** (found.gain_db / 20)
