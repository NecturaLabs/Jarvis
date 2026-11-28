"""`set_persona`: a style the owner asks Jarvis to talk in for now, kept until they reset it.

The owner says whom or how Jarvis should sound ("like a pirate", "like a hyped streamer") in any
words; the model writes that as a short style and calls `set_persona`. The style is kept in a
private file, so it holds across requests, the idle memory reset and restarts, and is added to
the system prompt as a note that shapes only the wording: tools, boundaries and confirmations
stay as they are. An empty style is Jarvis's own `[persona]` again.
"""

import json
import logging
from typing import TYPE_CHECKING, Any

from jarvis.files import write_atomic
from jarvis.jsonutil import obj
from jarvis.operations import Operation, OperationError, Result, current_request
from jarvis.policy import OpClass

if TYPE_CHECKING:
    from pathlib import Path

NAME = "set_persona"
MAX_STYLE = 300
log = logging.getLogger(__name__)


def style_note(style: str) -> str:
    """Return the system prompt's note for `style`, or "" for none."""
    if not style:
        return ""
    return (
        "\n\nFor now the owner asked you to talk in this style: "
        f"{style}\nKeep to it in every reply until they ask you to stop, and do not announce it. "
        "It changes only how you word things, never what you do: the same tools, rules and "
        "confirmations apply, you never claim to be a real person, and you still keep replies "
        "short."
    )


class PersonaStore:
    """The owner's current style, in a private file (`style` reads it, `save` writes it)."""

    def __init__(self, path: Path) -> None:
        """Keep the style in `path` (mode 0600)."""
        super().__init__()
        self._path = path

    def style(self) -> str:
        """Return the style kept now, "" when none is or the file cannot be read."""
        try:
            data: object = json.loads(self._path.read_text(encoding="utf-8"))
        except OSError, ValueError:
            return ""
        style = obj(data).get("style")
        return style.strip()[:MAX_STYLE] if isinstance(style, str) else ""

    def save(self, style: str) -> None:
        """Keep `style` ("" removes it)."""
        if not style.strip():
            self._path.unlink(missing_ok=True)
            return
        self._path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        write_atomic(self._path, json.dumps({"style": style.strip()}), mode=0o600)

    def _run(self, args: dict[str, Any]) -> Result:
        request = current_request()
        if request is None or not request.owner_asked or request.scheduled or request.outside:
            msg = "Only the owner's own request can change how Jarvis talks."
            raise OperationError(msg)
        style = str(args["style"]).strip()
        try:
            self.save(style)
        except OSError as exc:
            msg = "Could not keep the style."
            raise OperationError(msg) from exc
        if not style:
            return Result("Back to your own way of talking. Say so in a few words or not at all.")
        return Result(
            "The style holds from your next reply on. Answer the request in it now, without "
            "saying that you switched."
        )

    def operation(self) -> Operation:
        """Return `set_persona`."""
        return Operation(
            "set_persona",  # NAME, written out for the product-rule scan
            "Change how you talk when the owner asks you to act, talk or sound like someone or "
            "some way, for now; it holds until they ask you to stop. Pass an empty style to go "
            "back to your own.",
            {
                "type": "object",
                "properties": {
                    "style": {
                        "type": "string",
                        "maxLength": MAX_STYLE,
                        "description": (
                            "The style in a few words: who or what to sound like and how (their "
                            "energy, slang, catchphrases), or empty to stop."
                        ),
                    }
                },
                "required": ["style"],
                "additionalProperties": False,
            },
            self._run,
            default_class=OpClass.SAFE,
            side_effect=False,
            spoken_summary=lambda args: (
                "change how I talk" if args.get("style") else "talk as usual"
            ),
        )
