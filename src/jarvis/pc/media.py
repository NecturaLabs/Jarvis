"""`pc_media`: play, pause and skip in the PC's other media players (MPRIS, through playerctl).

A video in the browser, VLC or mpv answers to it. Spotify has its own tools, through its Web API,
so it is left alone unless the owner means everything. Only a player's plain name is ever said
or returned, never a title: a browser's title is a web page's text.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from jarvis.operations import Operation, OperationError, Result

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

ACTIONS = ("status", "play", "pause", "next", "previous")
_OWN_TOOLS = frozenset({"spotify"})  # players Jarvis drives through their own tools
_NO_PLAYERS = "No players found"
_STATUSES = frozenset({"Playing", "Paused", "Stopped"})
_PLAIN = re.compile(r"[a-z][a-z0-9_-]{0,23}")


def _said(name: str) -> str:
    """Return a player's bus name as said: its plain leading word ("firefox"), or "a player"."""
    match = _PLAIN.fullmatch(name.split(".", 1)[0])
    return match.group() if match else "a player"


class MediaOps:
    """The `pc_media` operation over a `playerctl` runner (`jarvis.audio.playerctl`)."""

    def __init__(self, playerctl: Callable[[Sequence[str]], str]) -> None:
        """Send commands through `playerctl`, which raises OperationError when it cannot."""
        super().__init__()
        self._playerctl = playerctl

    def _players(self) -> list[tuple[str, str]]:
        """Return every player's bus name and status ("Playing", "Paused" or "Stopped").

        Raises:
            OperationError: playerctl failed other than by finding no player.
        """
        try:
            names = [line.strip() for line in self._playerctl(["-l"]).splitlines() if line.strip()]
        except OperationError as exc:
            if _NO_PLAYERS in str(exc):
                return []
            raise
        statuses = [self._playerctl(["-p", name, "status"]).strip() for name in names]
        # The status is the player's own word: only MPRIS's three are kept.
        return [
            (name, status if status in _STATUSES else "Unknown")
            for name, status in zip(names, statuses, strict=True)
        ]

    def run(self, args: dict[str, Any]) -> Result:
        """Say what plays, or play, pause or skip, and say the outcome."""
        action = str(args["action"])
        players = self._players()
        if action == "status":
            if not players:
                said = "No media player is open."
            else:
                said = "; ".join(f"{_said(n)} is {s.lower()}" for n, s in players) + "."
            return Result(said, spoken=said)
        everything = bool(args.get("everything"))
        mine = [(n, s) for n, s in players if everything or n.split(".", 1)[0] not in _OWN_TOOLS]
        wanted = "Paused" if action == "play" else "Playing"
        targets = [n for n, s in mine if s == wanted]
        if action != "pause" or not everything:
            targets = targets[:1]
        if not targets:
            said = "Nothing else is paused." if action == "play" else "Nothing else is playing."
            return Result(said, spoken=said)
        for name in targets:
            self._playerctl(["-p", name, action])
        names = " and ".join(dict.fromkeys(_said(name) for name in targets))
        said = {
            "play": f"Playing {names} again.",
            "pause": f"Paused {names}.",
            "next": f"Skipped ahead in {names}.",
            "previous": f"Went back in {names}.",
        }[action]
        return Result(said, spoken=said, acted=True)

    def operations(self) -> list[Operation]:
        """Return `pc_media`."""
        return [
            Operation(
                "pc_media",
                "Play, pause or skip in other media players (a video in the browser, VLC), or "
                "say what plays. Spotify has its own tools; everything true also pauses it.",
                {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "enum": list(ACTIONS)},
                        "everything": {"type": "boolean"},
                    },
                    "required": ["action"],
                    "additionalProperties": False,
                },
                self.run,
                spoken_summary=lambda a: (
                    "say what plays" if a["action"] == "status" else f"{a['action']} the media"
                ),
            )
        ]
