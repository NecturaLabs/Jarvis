"""`jarvis acp`: Jarvis as an Agent Client Protocol agent, over standard input and output.

A coding app that speaks ACP (T3 Code's "Local ACP command") starts `jarvis acp` and talks JSON-RPC
2.0 to it, one message per line. This is a front end to the same agent `jarvis chat` has, not a
second one: each ACP session is one chat conversation (`build(via="acp")`, its own memory, written
replies), and the model is the one Strata already serves. Nothing here is spoken.

Owned here: the transport, `initialize`, `session/new`, `session/prompt` and `session/cancel`.
Anything else is "method not found". Jarvis loads no MCP servers and no sessions are kept across
runs, so `mcpServers` and `cwd` are ignored and `loadSession` is off.

A request is the owner's own typed words, so it has the chat's provenance. A picture block is
decoded under a size limit and checked like any chat picture (`jarvis.pictures`).

Consent stays Jarvis's own (ADR 0023). The client's `session/request_permission` is never used: T3
Code answers it by itself in its default mode, so it could not stand for the owner. A question
about one exact action is written into the reply, the prompt ends, and the owner's next prompt is
the answer: `is_yes` of its words runs the action, anything else cancels it. That prompt is not a
new request.
"""

import base64
import binascii
import json
import logging
import queue
import secrets
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, TextIO

from jarvis.agent import Declined, Hooks, NoReason
from jarvis.jsonutil import obj, objs
from jarvis.pictures import MAX_FILE_BYTES, Picture, PictureError, prepare
from jarvis.policy import is_yes

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

log = logging.getLogger(__name__)

SUPPORTED_VERSIONS = (1,)  # the ACP protocol versions this speaks, oldest first
MAX_LINE = 2 * MAX_FILE_BYTES  # one message: a picture's base64 and the JSON round it
MAX_BASE64 = MAX_FILE_BYTES * 4 // 3 + 4  # characters of one picture
MAX_IMAGES = 4
MAX_TEXT = 100_000  # characters of one prompt
MAX_SESSIONS = 32
QUESTION_S = 300.0  # how long a question waits for the owner's next prompt
CANCEL_WAIT_S = 10.0  # how long a new prompt waits for a cancelled request to end
CLOSE_WAIT_S = 10.0
IMAGE_TYPES = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/gif": ".gif",
    "image/webp": ".webp",
}
NO_REPLY = "(Jarvis had nothing to say.)"
NO_WORDS = "(a picture, no words)"
ASK_HOW = "Reply yes to go ahead. Anything else cancels it."
SEPARATOR = "\n\n"

# JSON-RPC 2.0 error codes.
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
BUSY = -32000


class StartError(Exception):
    """The agent for a new session cannot be made; the message says why and what to do."""


class AgentSession(Protocol):
    """What a session needs of the agent: `Agent.handle`, as the chat calls it."""

    def handle(self, text: str, hooks: Hooks, *, pictures: Sequence[Picture] = ()) -> str:
        """Answer one typed request and return the reply."""
        ...


class _Rpc(Exception):  # ruff: ignore[error-suffix-on-exception-name] - the JSON-RPC error answer, not a failure
    """A request answered with a JSON-RPC error."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class _Refused(Exception):  # ruff: ignore[error-suffix-on-exception-name] - what a prompt is refused for
    """A prompt Jarvis will not answer; the message is what the owner is told."""


@dataclass(slots=True)
class _Question:
    text: str
    answered: threading.Event = field(default_factory=threading.Event)
    answer: bool | Declined = False


@dataclass(slots=True)
class _Run:
    """One request, on its own thread: it may outlive the prompt that began it (a question)."""

    events: queue.Queue[str] = field(default_factory=queue.Queue[str])  # "asked" or "done"
    cancel: threading.Event = field(default_factory=threading.Event)
    finished: threading.Event = field(default_factory=threading.Event)
    question: _Question | None = None  # set while it waits for the owner's next prompt
    attached: bool = True  # a prompt is waiting on it, so its words may be written
    sent: str = ""  # the draft as last written
    total: str = ""  # everything written for this request
    error: str = ""  # the kind of failure, when it failed


@dataclass(slots=True)
class _Session:
    id: str
    agent: AgentSession
    lock: threading.Lock = field(default_factory=threading.Lock)
    run: _Run | None = None


class AcpServer:
    """One ACP connection: reads messages, answers them, writes them under one lock."""

    def __init__(
        self,
        make_agent: Callable[[str], AgentSession],
        out: TextIO,
        *,
        names: Sequence[str] = (),
        version: str = "",
        question_s: float = QUESTION_S,
    ) -> None:
        """Serve sessions whose agents `make_agent` makes (it takes the session's id).

        `names` are what the assistant is called, so "yes, Jarvis" is a yes; `question_s` is how
        long a question waits for its answer.
        """
        super().__init__()
        self._make_agent = make_agent
        self._out = out
        self._names = tuple(names)
        self._version = version
        self._question_s = question_s
        self._write_lock = threading.Lock()
        self._sessions: dict[str, _Session] = {}
        self._threads: list[threading.Thread] = []
        self._threads_lock = threading.Lock()
        self._closing = threading.Event()

    # The transport

    def serve(self, source: TextIO) -> None:
        """Read messages from `source` until it ends, then stop every request in flight."""
        try:
            while line := self._read(source):
                if line.strip():
                    self._message(line)
        finally:
            self.close()

    def close(self) -> None:
        """Stop every request and wait a little for its thread."""
        self._closing.set()
        for session in list(self._sessions.values()):
            self._cancel(session)
        with self._threads_lock:
            threads = list(self._threads)
        for thread in threads:
            thread.join(CLOSE_WAIT_S)

    def _spawn(self, target: Callable[..., None], *args: object, name: str) -> None:
        thread = threading.Thread(target=target, args=args, name=name, daemon=True)
        with self._threads_lock:
            self._threads = [t for t in self._threads if t.is_alive()]
            self._threads.append(thread)
        thread.start()

    def _read(self, source: TextIO) -> str:
        """Return the next line; one too long is answered with an error and skipped whole."""
        line = source.readline(MAX_LINE + 1)
        if len(line) <= MAX_LINE or line.endswith("\n"):
            return line
        while (rest := source.readline(MAX_LINE)) and not rest.endswith("\n"):
            pass
        self._error(None, INVALID_REQUEST, "message too long")
        return "\n"

    def _send(self, message: dict[str, Any]) -> None:
        """Write one message as one line: the only thing this process writes to its output."""
        line = json.dumps({"jsonrpc": "2.0", **message}, separators=(",", ":"))
        with self._write_lock:
            try:
                self._out.write(line + "\n")
                self._out.flush()
            except OSError, ValueError:
                self._closing.set()  # the client has gone

    def _result(self, request_id: object, result: dict[str, Any]) -> None:
        self._send({"id": request_id, "result": result})

    def _error(self, request_id: object, code: int, message: str) -> None:
        self._send({"id": request_id, "error": {"code": code, "message": message}})

    def _update(self, session: _Session, update: dict[str, Any]) -> None:
        self._send(
            {"method": "session/update", "params": {"sessionId": session.id, "update": update}}
        )

    def _say(self, session: _Session, text: str, kind: str = "agent_message_chunk") -> None:
        content = {"type": "text", "text": text}
        self._update(session, {"sessionUpdate": kind, "content": content})

    # The methods

    def _message(self, line: str) -> None:
        try:
            message: object = json.loads(line)
        except ValueError:
            self._error(None, PARSE_ERROR, "not JSON")
            return
        body = obj(message)
        if not isinstance(message, dict):
            self._error(None, INVALID_REQUEST, "one request or notification per line")
            return
        method = body.get("method")
        if not isinstance(method, str):
            return  # a response to nothing this sent: there is no request of ours to answer
        params = obj(body.get("params"))
        if "id" not in body:
            if method == "session/cancel":
                self._cancel_session(params)
            return  # a notification gets no answer, known or not
        request_id = body["id"]
        if not isinstance(request_id, str | int) or isinstance(request_id, bool):
            self._error(None, INVALID_REQUEST, "id is a string or a number")
            return
        try:
            self._dispatch(request_id, method, params)
        except _Rpc as exc:
            self._error(request_id, exc.code, exc.message)

    def _dispatch(self, request_id: str | int, method: str, params: dict[str, Any]) -> None:
        match method:
            case "initialize":
                self._result(request_id, self._initialize(params))
            case "session/new":
                self._result(request_id, self._new_session())
            case "session/prompt":
                self._prompt(request_id, params)
            case _:
                raise _Rpc(METHOD_NOT_FOUND, f"method not found: {method}")

    def _initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        asked = params.get("protocolVersion")
        if not isinstance(asked, int) or isinstance(asked, bool):
            raise _Rpc(INVALID_PARAMS, "protocolVersion is a number")
        version = asked if asked in SUPPORTED_VERSIONS else SUPPORTED_VERSIONS[-1]
        return {
            "protocolVersion": version,
            "agentCapabilities": {
                "loadSession": False,
                "promptCapabilities": {"image": True, "audio": False, "embeddedContext": False},
                "mcpCapabilities": {"http": False, "sse": False},
            },
            "agentInfo": {"name": "jarvis", "title": "Jarvis", "version": self._version},
            "authMethods": [],
        }

    def _new_session(self) -> dict[str, Any]:
        if len(self._sessions) >= MAX_SESSIONS:
            raise _Rpc(INTERNAL_ERROR, f"at most {MAX_SESSIONS} sessions at once")
        session_id = f"jarvis-{secrets.token_hex(6)}"
        try:
            agent = self._make_agent(session_id)
        except StartError as exc:
            raise _Rpc(INTERNAL_ERROR, str(exc)) from exc
        self._sessions[session_id] = _Session(session_id, agent)
        return {"sessionId": session_id}

    def _session(self, params: dict[str, Any]) -> _Session:
        found = self._sessions.get(str(params.get("sessionId")))
        if found is None:
            raise _Rpc(INVALID_PARAMS, "unknown session")
        return found

    # A prompt

    def _prompt(self, request_id: str | int, params: dict[str, Any]) -> None:
        """Begin a prompt: a new request, or the owner's answer to the question being asked.

        Raises:
            _Rpc: no such session, no list of blocks, or a request still being answered.
        """
        session = self._session(params)
        blocks = objs(params.get("prompt"))
        if not isinstance(params.get("prompt"), list):
            raise _Rpc(INVALID_PARAMS, "prompt is a list of content blocks")
        old = session.run
        if old is not None and old.cancel.is_set():
            old.finished.wait(CANCEL_WAIT_S)  # a request just stopped ends at its next check
        with session.lock:
            run = session.run
            if run is None:
                run = session.run = _Run()
                fresh = True
            else:
                question = run.question
                if question is None or question.answered.is_set():
                    raise _Rpc(BUSY, "a request is still being answered")
                # The owner's words answer the question; they are not a request of their own.
                question.answer = is_yes(_words(blocks), self._names)
                run.attached = True
                question.answered.set()
                fresh = False
        if fresh:
            self._spawn(self._begin, request_id, session, run, blocks, name="acp-prompt")
        else:
            self._spawn(self._wait, request_id, run, name="acp-answer")

    def _begin(
        self, request_id: str | int, session: _Session, run: _Run, blocks: list[dict[str, Any]]
    ) -> None:
        """Read the prompt, start the request on its own thread and wait for its end."""
        try:
            text, pictures = _read_prompt(blocks)
        except _Refused as exc:
            self._say(session, str(exc))
            self._release(session, run)
            self._result(request_id, {"stopReason": "refusal"})
            return
        except Exception:
            log.exception("a prompt could not be read")
            self._release(session, run)
            self._error(request_id, INTERNAL_ERROR, "Jarvis could not read that prompt")
            return
        if run.cancel.is_set():
            self._release(session, run)
            self._result(request_id, {"stopReason": "cancelled"})
            return
        self._spawn(self._work, session, run, text, pictures, name="acp-request")
        self._wait(request_id, run)

    def _wait(self, request_id: str | int, run: _Run) -> None:
        """Answer the prompt when its request ends or stops to ask the owner something."""
        run.events.get()  # "asked": the question is written, this prompt is over; or "done"
        if run.finished.is_set() and run.error and not run.cancel.is_set():
            self._error(request_id, INTERNAL_ERROR, f"Jarvis failed: {run.error}")
            return
        stopped = run.finished.is_set() and run.cancel.is_set()
        self._result(request_id, {"stopReason": "cancelled" if stopped else "end_turn"})

    @staticmethod
    def _release(session: _Session, run: _Run) -> None:
        with session.lock:
            if session.run is run:
                session.run = None
        run.finished.set()

    def _work(self, session: _Session, run: _Run, text: str, pictures: list[Picture]) -> None:
        """Run the agent's `handle` for one request, on its own thread."""
        hooks = Hooks(
            confirm=lambda question: self._ask(session, run, question),
            waking=lambda: self._note(
                session, run, "Waking the model; the first answer takes longer"
            ),
            still_waiting=lambda: self._note(session, run, "Still loading the model"),
            cancelled=lambda: run.cancel.is_set() or self._closing.is_set(),
            draft=lambda draft: self._draft(session, run, draft),
            condensed=lambda note: self._note(session, run, note),
        )
        try:
            reply = session.agent.handle(text, hooks, pictures=pictures)
            if not run.cancel.is_set():
                self._finish(session, run, reply)
        except Exception as exc:
            log.exception("a request failed")
            run.error = type(exc).__name__
        finally:
            self._release(session, run)
            run.events.put("done")

    def _note(self, session: _Session, run: _Run, text: str) -> None:
        """Say a line about the work in hand, as the agent's thought rather than its reply."""
        if run.attached and not run.cancel.is_set():
            self._say(session, text, "agent_thought_chunk")

    def _write(self, session: _Session, run: _Run, piece: str) -> None:
        if not run.attached or not piece:
            return
        self._say(session, piece)
        run.total += piece

    def _draft(self, session: _Session, run: _Run, draft: str) -> None:
        """Write what the reply gained: a chunk is never taken back, so only its new end goes."""
        if run.sent and draft.startswith(run.sent):
            piece = draft[len(run.sent) :]
        else:  # the agent started another message: it follows the last, set apart
            piece = (SEPARATOR if run.total else "") + draft
        run.sent = draft
        self._write(session, run, piece)

    def _finish(self, session: _Session, run: _Run, reply: str) -> None:
        """Write what of the final reply was not already written as drafts."""
        reply = reply.strip()
        if not reply:
            if not run.total:
                self._write(session, run, NO_REPLY)
            return
        if run.sent and reply.startswith(run.sent):
            piece = reply[len(run.sent) :]
        elif reply in run.total:
            piece = ""
        else:
            piece = (SEPARATOR if run.total else "") + reply
        run.sent = reply
        self._write(session, run, piece)

    def _ask(self, session: _Session, run: _Run, text: str) -> bool | Declined:
        """Ask the owner in the reply; the next prompt is the answer (`QUESTION_S` to give it)."""
        question = _Question(text)
        with session.lock:
            if run.cancel.is_set() or self._closing.is_set():
                return Declined(NoReason.CLOSED)
            run.question = question
        self._write(
            session, run, f"{SEPARATOR if run.total else ''}{text}{SEPARATOR}{ASK_HOW}{SEPARATOR}"
        )
        run.sent = ""
        with session.lock:
            run.attached = False
        run.events.put("asked")  # the prompt that began this ends; the next one answers
        answered = question.answered.wait(self._question_s)
        with session.lock:
            run.question = None
        if not answered:
            return Declined(NoReason.NO_ANSWER)
        return question.answer

    # Stopping

    def _cancel_session(self, params: dict[str, Any]) -> None:
        session = self._sessions.get(str(params.get("sessionId")))
        if session is not None:
            self._cancel(session)

    @staticmethod
    def _cancel(session: _Session) -> None:
        """Stop the request in flight; a question it waits on is answered no."""
        with session.lock:
            run = session.run
            if run is None:
                return
            run.cancel.set()
            question = run.question
            if question is not None and not question.answered.is_set():
                question.answer = Declined(NoReason.CLOSED)
                question.answered.set()


def _words(blocks: list[dict[str, Any]]) -> str:
    """Return the text of a prompt's text blocks, joined."""
    return "\n".join(str(b.get("text", "")) for b in blocks if b.get("type") == "text")


def _read_prompt(blocks: list[dict[str, Any]]) -> tuple[str, list[Picture]]:
    """Return a prompt's words and its pictures, checked.

    Raises:
        _Refused: nothing to answer, text too long, a block that is not text or a picture, too many
            pictures, or a picture too big, not one of `IMAGE_TYPES` or not one Jarvis reads.
    """
    parts: list[str] = []
    pictures: list[Picture] = []
    for block in blocks:
        kind = block.get("type")
        if kind == "text":
            parts.append(str(block.get("text", "")))
        elif kind == "image":
            if len(pictures) >= MAX_IMAGES:
                msg = f"I take at most {MAX_IMAGES} pictures in one request."
                raise _Refused(msg)
            pictures.append(_picture(block, len(pictures) + 1))
        else:
            msg = f"I read text and pictures here; a {kind} block is not something I can read."
            raise _Refused(msg)
    text = "\n".join(parts).strip()
    if len(text) > MAX_TEXT:
        msg = f"That is too long to read in one go (over {MAX_TEXT:,} characters)."
        raise _Refused(msg)
    if not text and not pictures:
        msg = "There was nothing in that request to answer."
        raise _Refused(msg)
    return text or NO_WORDS, pictures


def _picture(block: dict[str, Any], number: int) -> Picture:
    """Return an image block's picture, decoded under the size limit and checked like the chat's.

    Raises:
        _Refused: not a type in `IMAGE_TYPES`, no data, too big, not base64 or not a picture.
    """
    mime = block.get("mimeType")
    data = block.get("data")
    if not isinstance(mime, str) or mime.casefold() not in IMAGE_TYPES:
        msg = "I read PNG, JPEG, GIF and WebP pictures here."
        raise _Refused(msg)
    if not isinstance(data, str):
        msg = "That picture had no data."
        raise _Refused(msg)
    if len(data) > MAX_BASE64:
        msg = f"That picture is too big to send (over {MAX_FILE_BYTES // (1024 * 1024)} MB)."
        raise _Refused(msg)
    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError) as exc:
        msg = "That picture's data is not valid."
        raise _Refused(msg) from exc
    try:
        return prepare(raw, f"picture-{number}{IMAGE_TYPES[mime.casefold()]}")
    except PictureError as exc:
        raise _Refused(str(exc)) from exc
    except (OSError, ValueError, SyntaxError) as exc:  # a decoder's own failure
        msg = f"picture-{number} could not be read."
        raise _Refused(msg) from exc


def serve(
    source: TextIO,
    out: TextIO,
    make_agent: Callable[[str], AgentSession],
    *,
    names: Sequence[str] = (),
    version: str = "",
) -> int:
    """Serve one ACP connection over `source` and `out`; return the exit code (0 at its end)."""
    AcpServer(make_agent, out, names=names, version=version).serve(source)
    return 0
