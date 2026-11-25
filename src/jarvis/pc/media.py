"""`pc_media`: play, pause and skip in the PC's other media players (MPRIS, through playerctl).

A video in the browser, VLC or mpv answers to it. Spotify has its own tools, through its Web API,
so it is left alone unless the owner means everything. Only a player's plain name is ever said
or returned, never a title: a browser's title is a web page's text.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from jarvis.operations import Operation, OperationError, Result, UncertainOutcomeError

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
        except UncertainOutcomeError as exc:
            msg = "the media players did not answer in time"
            raise OperationError(msg) from exc
        except OperationError as exc:
            if _NO_PLAYERS in str(exc):
                return []
            raise
        players: list[tuple[str, str]] = []
        for name in names:
            try:
                status = self._playerctl(["-p", name, "status"]).strip()
            except OperationError, UncertainOutcomeError:
                continue  # closed since it was listed, or not answering: left out
            # The status is the player's own word: only MPRIS's three are kept.
            players.append((name, status if status in _STATUSES else "Unknown"))
        return players

    def _targets(self, action: str, *, everything: bool) -> list[str]:
        """Return the players `action` is for: the first that fits, or every playing one."""
        mine = [
            (n, s) for n, s in self._players() if everything or n.split(".", 1)[0] not in _OWN_TOOLS
        ]
        wanted = "Paused" if action == "play" else "Playing"
        targets = [n for n, s in mine if s == wanted]
        return targets if action == "pause" and everything else targets[:1]

    def prepare(self, args: dict[str, Any]) -> dict[str, Any]:
        """Bind a command to the players it is for, before any question.

        A yes is then for those players, named in the question, not for whatever plays when it
        runs.
        """
        if args["action"] == "status":
            return args
        return {
            **args,
            "players": self._targets(args["action"], everything=bool(args.get("everything"))),
        }

    def run(self, args: dict[str, Any]) -> Result:
        """Say what plays, or play, pause or skip, and say the outcome.

        Raises:
            OperationError: a player refused, or did not answer a play or pause in time.
            UncertainOutcomeError: a skip was sent and not answered in time, so it may have run.
        """
        action = str(args["action"])
        if action == "status":
            players = self._players()
            if not players:
                said = "No media player is open."
            else:
                said = "; ".join(f"{_said(n)} is {s.lower()}" for n, s in players) + "."
            return Result(said, spoken=said)
        prepared = args if "players" in args else self.prepare(args)
        targets = [str(name) for name in prepared["players"]]
        if not targets:
            said = "Nothing is paused." if action == "play" else "Nothing is playing."
            return Result(said, spoken=said)
        for name in targets:
            try:
                self._playerctl(["-p", name, action])
            except UncertainOutcomeError:
                if action in {"next", "previous"}:
                    raise  # a skip may have gone through: never sent twice
                msg = f"{_said(name)} did not answer in time"
                raise OperationError(msg) from None
            except OperationError:
                msg = f"{_said(name)} refused that"  # never the player's own words
                raise OperationError(msg) from None
        names = " and ".join(dict.fromkeys(_said(name) for name in targets))
        said = {
            "play": f"Playing {names} again.",
            "pause": f"Paused {names}.",
            "next": f"Skipped ahead in {names}.",
            "previous": f"Went back in {names}.",
        }[action]
        return Result(said, spoken=said, acted=True)

    @staticmethod
    def summary(args: dict[str, Any]) -> str:
        """The words of a question about the call, naming the players it is bound to."""
        action = str(args["action"])
        if action == "status":
            return "say what plays"
        bound = [str(name) for name in args.get("players", [])]
        names = " and ".join(dict.fromkeys(_said(name) for name in bound)) or "no player"
        return {
            "play": f"play {names} again",
            "pause": f"pause {names}",
            "next": f"skip to the next track in {names}",
            "previous": f"go back a track in {names}",
        }[action]

    def operations(self) -> list[Operation]:
        """Return `pc_media`."""
        return [
            Operation(
                "pc_media",
                "Play, pause or skip in other media players (a video in the browser, VLC), or "
                "say what plays. Spotify has its own tools; everything: true also pauses Spotify.",
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
                spoken_summary=self.summary,
                prepare=self.prepare,
            )
        ]
