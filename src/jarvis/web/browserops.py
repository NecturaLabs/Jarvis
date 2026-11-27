"""The browser operations: open and read pages that need a real browser, and act on them.

Opening, following a link and reading are reads, like `web_read`, and need no yes: an address is
opened only when a search in the conversation returned it or the owner wrote it (`WebOps.admit`),
and a link only when the page shows it. Everything a page shows is someone else's words
(`Origin.WEB`). A click and typing can act (send a form, post, buy, log in), and typing sends the
owner's words to whoever runs the page, so `browser_act` is `confirm` and cannot be set lower: the
owner hears the exact element, the page and the words before anything is done, even with consent
bypassed (`Operation.always_ask`). Typing is `outbound` as well. The fences around the browser
itself are in `jarvis.web.browser`.
"""

import unicodedata
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from jarvis.operations import Operation, OperationError, Result, UncertainOutcomeError, cut_inert
from jarvis.pictures import SHOWN_SIDE, PictureError, prepare
from jarvis.policy import OpClass, Origin
from jarvis.web.browser import (
    MAX_PARTS,
    TYPABLE,
    UNTYPABLE,
    BrowserError,
    Element,
    UncertainActionError,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from jarvis.pictures import PictureCache
    from jarvis.web.browser import Browser, View

# Every operation's name, known to `boundaries.classes` even where the browser is not offered.
NAMES = frozenset({"browser_open", "browser_preview", "browser_act", "browser_screenshot"})
MAX_PATH = 1000
MAX_URL = 2000
MAX_TEXT = 200  # what may be typed: about a search, like `web_search`'s query
ELEMENT_SHARE = 3  # at most a third of a result lists elements; the rest is the page's text
_SAID_NAME = 60  # characters of an element's name in a question
_SAID_ADDRESS = 80  # characters of where a link goes, in a question
_REF = {"type": "string", "minLength": 2, "maxLength": 12}
_PART = {"type": "integer", "minimum": 1, "maximum": MAX_PARTS}


def _ref(value: object) -> str:
    """Return a reference as agent-browser writes it (`e12`), whatever the model put around it.

    Raises:
        OperationError: it is not a reference.
    """
    ref = str(value).strip().lstrip("@[").rstrip("]").lower()
    if not (ref.startswith("e") and ref[1:].isdigit()):
        msg = "An element is named by its ref from the page, like e12."
        raise OperationError(msg)
    return ref


def _parts(text: str, room: int) -> list[str]:
    """Cut `text` into parts that each fit `room` once made inert, at line ends where possible."""
    parts: list[str] = []
    rest = text.strip()
    while rest and len(parts) < MAX_PARTS:
        piece = cut_inert(rest, room)
        if len(piece) < len(rest):
            cut = piece.rfind("\n")
            piece = piece[:cut] if cut > room // 2 else piece
        parts.append(piece)
        rest = rest[len(piece) :].lstrip()
    return parts or [""]


def _rank(element: Element) -> int:
    """Order elements for listing: fields first, then other controls, then links."""
    if element.role in TYPABLE:
        return 0
    return 2 if element.role == "link" else 1


def _listed(elements: tuple[Element, ...], room: int) -> tuple[str, int]:
    """Return the lines naming the elements that fit `room`, and how many were listed."""
    lines: list[str] = []
    used = 0
    for element in sorted(elements, key=_rank):
        line = f"[{element.ref}] {element.role}" + (f' "{element.name}"' if element.name else "")
        cost = len(line) + 1 + 3 * (line.count("<") + line.count(">"))
        if used + cost > room:
            break
        lines.append(line)
        used += cost
    return "\n".join(lines), len(lines)


def _where(args: dict[str, Any]) -> str:
    """Name a bound element and its site for a question."""
    name = "".join(
        char
        if char.isprintable()
        and char not in "\"'\u201e\u201a\uff02\uff07\uff1f?"
        and unicodedata.category(char) not in {"Pi", "Pf"}
        else " "
        for char in str(args.get("name", ""))
    )
    name = " ".join(name.split())[:_SAID_NAME]
    role = str(args.get("role", "element"))
    host = urlsplit(str(args.get("page", ""))).hostname or "the page"
    label = f' "{name}"' if name else ""
    return f"the {role}{label} on {host}"


def _goes_to(args: dict[str, Any]) -> str:
    """Say where a bound link goes, for its question: its site and path, or that it goes nowhere."""
    if "goes_to" not in args:
        return ""
    parts = urlsplit(str(args["goes_to"]))
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        return ", which goes to no web address"
    path = parts.path if parts.path != "/" else ""
    if parts.query:
        path += f"?{parts.query}"
    shown_path = " ".join(path.split())[: max(0, _SAID_ADDRESS - len(parts.hostname))]
    return f", which goes to {parts.hostname}{shown_path}"


def act_summary(args: dict[str, Any]) -> str:
    """The words of the question about a bound `browser_act`: the element, its site, the words.

    A link's question says where it goes as well: its name alone can say anything.
    """
    words = " ".join(str(args.get("text", "")).split())
    if not words:
        return f"click {_where(args)}{_goes_to(args)}"
    then = " and press Enter" if args.get("submit", True) else ""
    return f'type "{words}" into {_where(args)}{then}'


class BrowserOps:
    """The browser operations over one `Browser`, opening only what `admit` allows."""

    def __init__(
        self,
        browser: Browser,
        admit: Callable[[str], None],
        *,
        page_chars: int,
        shown: PictureCache | None = None,
        preview: Callable[[str], tuple[Browser, str]] | None = None,
    ) -> None:
        """Drive `browser`; `admit` raises OperationError for an address that may not be opened.

        `page_chars` is the room for one result. With `shown`, where the chat keeps the pictures it
        shows, `browser_screenshot` is offered too. With `preview`, `browser_preview` is: it starts
        a preview of the owner's own build at a path (ADR 0016) and returns its browser session
        and first address, raising OperationError for a path that may not be served.
        """
        super().__init__()
        self._web = self._browser = browser
        self._admit = admit
        self._room = page_chars
        self._shown = shown
        self._start_preview = preview
        self._preview: Browser | None = None  # the open preview, while the model works in it

    def reset(self) -> None:
        """Close the browser and any preview: a forgotten conversation leaves no page open."""
        self._end_preview()
        self._web.close()

    def _end_preview(self) -> None:
        """Close the open preview, if any, and drive the web session again."""
        preview, self._preview = self._preview, None
        self._browser = self._web
        if preview is not None:
            preview.close()

    # --- reading ----------------------------------------------------------------------------

    def _page(self, view: View, part: int = 1) -> Result:
        """Word what a page shows: its address, a part of its text and the elements it offers."""
        head = f"Page {view.title} ({view.host}): {view.url}" if view.title else f"Page {view.url}"
        listed, count = _listed(view.elements, self._room // ELEMENT_SHARE)
        room = max(self._room - len(head) - len(listed) - 200, 200)
        parts = _parts(view.text, room)
        number = min(part, len(parts))
        lines = [head]
        if len(parts) > 1:
            lines.append(f"Text, part {number} of {len(parts)} (browser_open with part for more):")
        if view.withheld:
            lines.append(
                "(Nothing of this page is shown: it is too big to check for hidden text. Try"
                " another page or web_read.)"
            )
        else:
            lines.append(parts[number - 1] or "(no text)")
        if view.elements:
            total = len(view.elements)
            more = f", the first {count} of {total}" if count < total else ""
            lines.extend((f"Elements{more}:", listed))
        return Result(cut_inert("\n".join(lines), self._room), Origin.WEB)

    def open(self, args: dict[str, Any]) -> Result:
        """Open an address or follow a link the page shows; with only `part`, read on.

        Raises:
            OperationError: both an address and a link were given, nothing is open to read on,
                the address may not be opened, or the browser failed.
        """
        url = str(args.get("url", "")).strip()
        link = str(args.get("link", "")).strip()
        part = int(args.get("part", 1))
        if url and link:
            msg = "Give a url or a link from the page, not both."
            raise OperationError(msg)
        if not url and not link:
            view = self._browser.view
            if view is None:
                msg = "No page is open in the browser; give a url."
                raise OperationError(msg)
            return self._page(view, part)
        if url:
            self._admit(url)
            self._end_preview()  # a web address is the web session's, never the preview's
        try:
            address = self._browser.link(_ref(link)) if link else url
            return self._page(self._browser.open(address), part)
        except BrowserError as exc:
            raise OperationError(f"Couldn't open that page: {exc}.") from exc

    def preview(self, args: dict[str, Any]) -> Result:
        """Open the owner's own build (a folder, or a page in it) in a preview session.

        The preview reaches only a read-only server over that folder, not the internet (ADR
        0016). Until a web address is opened again, `browser_open` with a link, `browser_act` and
        `browser_screenshot` work on the preview's page.

        Raises:
            OperationError: no preview can be made here, the path may not be served, or the
                page could not be opened.
        """
        if self._start_preview is None:
            msg = "Previews are not offered here."
            raise OperationError(msg)
        session, address = self._start_preview(str(args["path"]))
        self._end_preview()
        self._preview = self._browser = session
        try:
            return self._page(session.open(address), int(args.get("part", 1)))
        except BrowserError as exc:
            raise OperationError(f"Couldn't open the preview: {exc}.") from exc

    def screenshot(self, args: dict[str, Any]) -> Result:
        """Show the owner one screen of the open page in the chat.

        Raises:
            OperationError: no page is open, or the picture could not be taken, read or kept.
        """
        view = self._browser.view
        if view is None or self._shown is None:
            msg = "No page is open in the browser; open one first."
            raise OperationError(msg)
        part = int(args.get("part", 1))
        try:
            data = self._browser.screenshot(part)
            picture = prepare(data, view.title or view.host, max_side=SHOWN_SIDE)
            path = self._shown.keep(picture)
        except BrowserError as exc:
            raise OperationError(f"Couldn't take that picture: {exc}.") from exc
        except PictureError as exc:
            raise OperationError(f"That picture couldn't be shown: {exc}.") from exc
        except OSError as exc:
            msg = f"That picture couldn't be kept to show: {exc.strerror or type(exc).__name__}."
            raise OperationError(msg) from exc
        return Result(
            f"Shown to the owner in the chat: screen {part} of {view.host}. You can't see it.",
            Origin.WEB,
            picture=str(path),
            caption=f"{view.title or view.host} · {view.host}",
        )

    # --- acting -----------------------------------------------------------------------------

    def bind(self, args: dict[str, Any]) -> dict[str, Any]:
        """Bind a click or typing to the element the page shows now, before the owner is asked.

        The question names it and its page, and the call acts only if both are still there.

        Raises:
            OperationError: no page is open, the page shows no such element, or text is to be
                typed into an element that takes none, or into a password, hidden or file field
                (refused before the question).
        """
        view = self._browser.view
        ref = _ref(args.get("element", ""))
        element = view.element(ref) if view is not None else None
        if view is None or element is None:
            msg = "The page shows no such element now; open the page again."
            raise OperationError(msg)
        if len(view.host) > _SAID_ADDRESS:
            msg = "The page's host is too long to name in the confirmation."
            raise OperationError(msg)
        typing = bool(str(args.get("text", "")).strip())
        if typing and element.role not in TYPABLE:
            # Not the element's name: that is the page's words, and this message is Jarvis's own.
            msg = f"Element {ref} takes no typing; pick a text box."
            raise OperationError(msg)
        if typing:
            try:
                kind = self._browser.input_type(ref)
            except BrowserError as exc:
                raise OperationError(f"Couldn't read that field: {exc}.") from exc
            if kind in UNTYPABLE:
                msg = "Jarvis never types into a password, hidden or file field."
                raise OperationError(msg)
        bound = {
            **args,
            "element": ref,
            "role": element.role,
            "name": element.name,
            "page": view.url,
        }
        if element.role == "link" and not typing:
            try:
                destination = self._browser.href(ref)
            except BrowserError as exc:
                raise OperationError(f"Couldn't read that link: {exc}.") from exc
            if len(urlsplit(destination).hostname or "") > _SAID_ADDRESS:
                msg = "The link's destination host is too long to name in the confirmation."
                raise OperationError(msg)
            bound["goes_to"] = destination
        return bound

    def act(self, args: dict[str, Any]) -> Result:
        """Click the element the owner agreed to, or type the words they agreed to into it.

        Raises:
            UncertainOutcomeError: the click or the words were sent, and may have gone through:
                the browser did not answer, or the page could not be read afterwards.
            OperationError: the page changed, or the click or typing failed.
        """
        expect = Element(str(args["element"]), str(args["role"]), str(args["name"]))
        page, words = str(args["page"]), str(args.get("text", "")).strip()
        try:
            if words:
                view = self._browser.fill(
                    expect, page, words, submit=bool(args.get("submit", True))
                )
            else:
                goes_to = args.get("goes_to")
                view = self._browser.click(expect, page, None if goes_to is None else str(goes_to))
        except UncertainActionError as exc:
            raise UncertainOutcomeError(f"Not known whether that happened: {exc}.") from exc
        except BrowserError as exc:
            raise OperationError(f"Couldn't do that on the page: {exc}.") from exc
        return self._page(view)

    # --- the operations ---------------------------------------------------------------------

    def operations(self) -> list[Operation]:
        """Return the browser operations; `browser_screenshot` only where pictures are shown."""
        ops = [
            Operation(
                "browser_open",
                "Open a page in a real browser when web_read can't read it (built by scripts) or "
                "to use it: a url web_search returned or the owner wrote, or a link ref the open "
                "page shows. Returns text and element refs; part reads on.",
                {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "maxLength": MAX_URL},
                        "link": _REF,
                        "part": _PART,
                    },
                    "additionalProperties": False,
                },
                self.open,
                side_effect=False,
                spoken_summary=lambda _a: "open that page",
            ),
            *(
                [
                    Operation(
                        "browser_preview",
                        "Open what you built (an HTML page, or its folder for index.html) in a "
                        "browser that reaches only that folder, to check it works; then use "
                        "browser_act and its link refs on it. Never open your own files with "
                        "file:// in browser_open.",
                        {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string", "minLength": 1, "maxLength": MAX_PATH},
                                "part": _PART,
                            },
                            "required": ["path"],
                            "additionalProperties": False,
                        },
                        self.preview,
                        side_effect=False,
                        spoken_summary=lambda a: f"preview {a.get('path', 'that page')}",
                    )
                ]
                if self._start_preview is not None
                else []
            ),
            Operation(
                "browser_act",
                "Click an element of the open page, or with text type into a text box (then "
                "Enter unless submit is false). The owner is asked first; never a password.",
                {
                    "type": "object",
                    "properties": {
                        "element": _REF,
                        "text": {"type": "string", "maxLength": MAX_TEXT},
                        "submit": {"type": "boolean"},
                    },
                    "required": ["element"],
                    "additionalProperties": False,
                },
                self.act,
                default_class=OpClass.CONFIRM,
                lowest_class=OpClass.CONFIRM,
                always_ask=True,
                outbound=True,
                outbound_args=frozenset({"text"}),
                spoken_summary=act_summary,
                prepare=self.bind,
            ),
        ]
        if self._shown is not None:
            ops.append(
                Operation(
                    "browser_screenshot",
                    "Show the owner a picture of the open page: part 1 is the top, each next "
                    "part a screen lower. You can't see it.",
                    {
                        "type": "object",
                        "properties": {"part": _PART},
                        "required": ["part"],
                        "additionalProperties": False,
                    },
                    self.screenshot,
                    side_effect=False,
                    spoken_summary=lambda _a: "show you the page",
                )
            )
        return ops
