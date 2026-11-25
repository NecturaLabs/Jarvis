"""`pc_volume`: the PC's own output volume (PipeWire's default output), read, set or muted.

Everything the PC plays goes through it: a game, a video, a call, Spotify and Jarvis's own voice.
The volume is changed through PipeWire's `wpctl`, run by `jarvis.audio.wpctl` with a fixed
argument list; this module only builds the arguments and reads the answer.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from jarvis.operations import Operation, OperationError, Result

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

SINK = "@DEFAULT_AUDIO_SINK@"  # wpctl's name for the default output, whichever device it is
MAX_PERCENT = 100  # never above full: wpctl allows more, and that distorts
MAX_DELTA = 50
_VOLUME = re.compile(r"Volume:\s*([0-9]+(?:\.[0-9]+)?)(\s*\[MUTED\])?")
_WAYS = ("percent", "delta", "mute")


def _clamp(percent: int) -> int:
    return max(0, min(MAX_PERCENT, percent))


class VolumeOps:
    """The `pc_volume` operation over a `wpctl` runner (`jarvis.audio.wpctl`, a fake in tests)."""

    def __init__(self, wpctl: Callable[[Sequence[str]], str]) -> None:
        """Change the volume through `wpctl`, which raises OperationError when it cannot."""
        super().__init__()
        self._wpctl = wpctl

    def _read(self) -> tuple[int, bool]:
        """Return the output's volume in percent and whether it is muted.

        Raises:
            OperationError: wpctl failed or answered something else.
        """
        match = _VOLUME.search(self._wpctl(["get-volume", SINK]))
        if match is None:
            msg = "I couldn't read the PC's volume."
            raise OperationError(msg)
        return round(float(match.group(1)) * 100), bool(match.group(2))

    @staticmethod
    def _way(args: dict[str, Any]) -> str:
        """Return the one way the call changes the volume, or "" for a reading.

        Raises:
            OperationError: more than one way was given.
        """
        given = [way for way in _WAYS if way in args]
        if len(given) > 1:
            msg = "give one of percent, delta or mute, not more"
            raise OperationError(msg)
        return given[0] if given else ""

    def prepare(self, args: dict[str, Any]) -> dict[str, Any]:
        """Bind a change ("up 10") to the volume it starts from and ends at, before any question.

        A yes is then for that exact volume, not for whatever the volume is when it runs.
        """
        if self._way(args) != "delta":
            return args
        current, _ = self._read()
        # Above the usual top (another program set it), turning up keeps it rather than lowering it.
        top = max(MAX_PERCENT, current)
        target = max(0, min(current + int(args["delta"]), top))
        return {**args, "current": current, "target": target}

    def run(self, args: dict[str, Any]) -> Result:
        """Read, set, change or mute the output volume and say the outcome."""
        way = self._way(args)
        if not way:
            percent, muted = self._read()
            said = f"The PC's volume is {percent} percent{', and it is muted' if muted else ''}."
            return Result(said, spoken=said)
        if way == "mute":
            self._wpctl(["set-mute", SINK, "1" if args["mute"] else "0"])
            said = "Muted." if args["mute"] else "Unmuted."
            return Result(f"The PC's output is {said.lower()}", spoken=said, acted=True)
        if way == "delta":
            prepared = args if "target" in args else self.prepare(args)
            target = int(prepared["target"])
        else:
            target = _clamp(int(args["percent"]))
        self._wpctl(["set-volume", SINK, f"{target / 100:.2f}"])
        said = f"The PC's volume is now {target} percent."
        return Result(said, spoken=said, acted=True)

    @staticmethod
    def summary(args: dict[str, Any]) -> str:
        """The words of a question about the call."""
        if "mute" in args:
            return "mute the PC's sound" if args["mute"] else "unmute the PC's sound"
        if "target" in args:
            return f"change the PC's volume from {args['current']} to {args['target']} percent"
        if "percent" in args:
            return f"set the PC's volume to {args['percent']} percent"
        return "read the PC's volume"

    def operations(self) -> list[Operation]:
        """Return `pc_volume`."""
        return [
            Operation(
                "pc_volume",
                "The PC's own output volume: everything it plays (games, videos, calls, Spotify). "
                "No argument reads it; percent sets it; delta turns it up (positive) or down; "
                "mute true or false. For Spotify alone, use spotify_set_volume.",
                {
                    "type": "object",
                    "properties": {
                        "percent": {"type": "integer", "minimum": 0, "maximum": MAX_PERCENT},
                        "delta": {"type": "integer", "minimum": -MAX_DELTA, "maximum": MAX_DELTA},
                        "mute": {"type": "boolean"},
                    },
                    "additionalProperties": False,
                },
                self.run,
                spoken_summary=self.summary,
                prepare=self.prepare,
            )
        ]
