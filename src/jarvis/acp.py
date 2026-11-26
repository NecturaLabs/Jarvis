"""`jarvis acp`: Jarvis as an Agent Client Protocol agent, over standard input and output.

A coding app that speaks ACP (T3 Code's "Local ACP command") starts `jarvis acp` and talks JSON-RPC
2.0 to it, one message per line. This is a front end to the same agent `jarvis chat` has, not a
second one: each ACP session is one chat conversation (`build(via="acp")`, its own memory, written
replies), and the model is the one Strata already serves. Nothing here is spoken.

Owned here: the transport, `initialize`, `session/new`, `session/load`, `session/prompt`,
`session/cancel`, `session/close` and `session/set_config_option`. Anything else is "method not
found".

With `models` (Jarvis's modes, `ModelChoice`) a session offers them as one model list, each by its
mode's name, described by the model that serves it and what it is for, and a thinking effort
(`thought_level`) that starts at the chosen mode's own. Each mode is its own agent in the session,
made when it is first chosen and kept, so going back to a mode keeps its conversation. Neither
changes while a request is being answered. Jarvis loads no MCP servers, so `mcpServers` is ignored.

With a `store` folder (0700) each session's conversation is kept there after every reply, one file
per mode the session used (0600, `Conversation.save`) beside a small file naming its mode and
effort, and `loadSession` is on: a later `jarvis acp` (the client starts one per run) reopens it by
its id with `session/load`, sends its remembered turns back as the protocol asks, and goes on from
them. A well-formed id with nothing kept opens an empty session under that id, so a thread whose
files were pruned still works; an id Jarvis could not have made is refused. Only the newest
`MAX_KEPT` sessions are kept.

Announcements (a reminder, a watch alert: `jarvis.announce`) the voice service makes reach every
open session as "Jarvis: <line>" in the reply. T3 Code drops message chunks that arrive outside a
prompt (read from its v0.1.11 ACP client: assistant updates are shown only while a `session/prompt`
is open), so a line waits for the session's next prompt and is written first in its reply, or after
the reply when it arrives during one. It never lands in the middle of a reply. With a `pusher`
(`t3push.T3Push`) the line is also sent into the client's own thread by T3 Code's command line, so
T3 Code shows it at once: it comes back as a prompt carrying a tag only Jarvis can make, which the
session answers with the line itself, without the model or any tool. A prompt that merely looks
like one is the owner's ordinary input.

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
import collections
import json
import logging
import queue
import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, TextIO, runtime_checkable

from jarvis.acpstore import SESSION_ID, forget_session, forgotten_at, prune
from jarvis.agent import Declined, Hooks, NoReason
from jarvis.announce import Announcement
from jarvis.config import EFFORTS
from jarvis.files import write_atomic
from jarvis.jsonutil import obj, objs
from jarvis.pictures import MAX_FILE_BYTES, Picture, PictureError, prepare
from jarvis.policy import is_yes
from jarvis.secretfile import read_private
from jarvis.t3push import Delivery

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Sequence
    from pathlib import Path

log = logging.getLogger(__name__)

SUPPORTED_VERSIONS = (1,)  # the ACP protocol versions this speaks, oldest first
MAX_LINE = 2 * MAX_FILE_BYTES  # one message: a picture's base64 and the JSON round it
MAX_BASE64 = MAX_FILE_BYTES * 4 // 3 + 4  # characters of one picture
MAX_IMAGES = 4
MAX_TEXT = 100_000  # characters of one prompt
MAX_SESSIONS = 32
MAX_PUSHES = 8  # announcements waiting to be pushed; a hung `t3` loses the oldest
MAX_PENDING = 50  # announcements a session keeps for its next prompt
FEED_RETRY_S = 10.0  # how long to wait before looking again for the running Jarvis
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
FORGED = "That looks like one of my announcements but its tag is not valid, so I ignored it."
NO_WORDS = "(a picture, no words)"
ASK_HOW = "Reply yes to go ahead. Anything else cancels it."
SEPARATOR = "\n\n"
MODEL_OPTION = "model"
EFFORT_OPTION = "effort"

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
    """What a session needs of the agent: `Agent.handle`, as the chat calls it, and its effort."""

    def handle(self, text: str, hooks: Hooks, *, pictures: Sequence[Picture] = ()) -> str:
        """Answer one typed request and return the reply."""
        ...

    def set_effort(self, level: str) -> None:
        """Think at `level` (one of `config.EFFORTS`) from the next request on."""
        ...


@runtime_checkable
class Kept(Protocol):
    """An agent whose conversation can be kept across runs (`session/load`)."""

    def save(self, path: Path) -> None:
        """Write the conversation to `path` (0600)."""
        ...

    def load(self, path: Path) -> bool:
        """Read a conversation `save` wrote; return whether one was read."""
        ...

    def history(self) -> list[tuple[str, str]]:
        """Return the remembered turns, oldest first, as (request, reply)."""
        ...


@runtime_checkable
class Forgets(Protocol):
    """An agent that can start its conversation afresh (`/clear`)."""

    def forget(self) -> None:
        """Forget the conversation."""
        ...


CLEAR = "/clear"
COMMANDS = [{"name": "clear", "description": "Forget this conversation and start afresh"}]
CLEARED = "Started afresh: I've forgotten this conversation."


@dataclass(frozen=True, slots=True)
class ModelChoice:
    """One of Jarvis's modes as the app lists it: id, title, model and use, starting effort."""

    id: str
    name: str
    description: str
    effort: str


class Feed(Protocol):
    """The running Jarvis's announcements (`control.subscribe`): iterate it, `close` it to stop."""

    def __iter__(self) -> Iterator[Announcement]:
        """Iterate over the announcements as they come."""
        ...

    def close(self) -> None:
        """Stop; a read waiting for the next announcement ends."""
        ...


class Pusher(Protocol):
    """Sends an announcement into the client's own threads (`t3push.T3Push`), and knows its own."""

    def push(self, item: Announcement, projects: Iterable[str]) -> set[str]:
        """Send `item` for each of `projects`; return those it was sent for."""
        ...

    def marked(self, prompt: str) -> bool:
        """Whether `prompt` starts as a message this makes does, whether its tag is valid or not."""
        ...

    def verified_delivery(self, prompt: str) -> Delivery | None:
        """Return the authenticated announcement when `prompt` is a message this made."""
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
    agent: AgentSession  # the chosen model's
    lock: threading.Lock = field(default_factory=threading.Lock)
    run: _Run | None = None
    pending: dict[tuple[int, int], Announcement | Delivery] = field(
        default_factory=dict[tuple[int, int], Announcement | Delivery]
    )
    cwd: str = ""  # the project folder the client opened it in, when it said
    stops: int = 0  # how many times the client stopped it (`session/cancel`)
    closed: bool = False
    model: str = ""  # the chosen `ModelChoice.id`, "" when the server offers none
    effort: str = ""
    agents: dict[str, AgentSession] = field(default_factory=dict[str, AgentSession])
    synced: float = field(default_factory=time.time)  # when its memory last matched the store
    clearing: threading.Event | None = None  # the latest `/clear` waiting for a stopped request

    def keep(self, note: Announcement | Delivery) -> None:
        """Keep this copy of the announcement for the next reply."""
        self.pending[note.id, int(note.at)] = note
        if len(self.pending) > MAX_PENDING:
            del self.pending[next(iter(self.pending))]


class AcpServer:
    """One ACP connection: reads messages, answers them, writes them under one lock."""

    def __init__(  # ruff: ignore[too-many-arguments] - a connection's collaborators and options
        self,
        make_agent: Callable[[str, str], AgentSession],
        out: TextIO,
        *,
        names: Sequence[str] = (),
        version: str = "",
        question_s: float = QUESTION_S,
        follow: Callable[[], Feed | None] | None = None,
        retry_s: float = FEED_RETRY_S,
        pusher: Pusher | None = None,
        record: Callable[..., None] | None = None,
        models: Sequence[ModelChoice] = (),
        store: Path | None = None,
    ) -> None:
        """Serve sessions whose agents `make_agent` makes (it takes the session's id and model).

        With `store`, sessions are kept in that folder and can be loaded by a later run.

        With `models` each session starts on the first and may choose another; without, the
        model `make_agent` is given is "".

        `names` are what the assistant is called, so "yes, Jarvis" is a yes; `question_s` is how
        long a question waits for its answer. `follow` opens the running Jarvis's announcements
        (None when it is not running, and it is tried again every `retry_s`); each is passed on
        to every session, and with a `pusher` also sent into the client's own threads, which turns
        up as a prompt `pusher.verified_delivery` authenticates. The session answers with the
        announcement itself, without calling the model, and clears its pending copy.
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
        self._follow = follow
        self._pusher = pusher
        self._record = record
        self._models = tuple(models)
        self._store = store
        self._pushes: collections.deque[tuple[Announcement, set[str]]] = collections.deque(
            maxlen=MAX_PUSHES
        )
        self._push_ready = threading.Condition()
        self._push_worker: threading.Thread | None = None
        self._retry_s = retry_s
        self._feed: Feed | None = None
        self._feed_lock = threading.Lock()

    # The transport

    def serve(self, source: TextIO) -> None:
        """Read messages from `source` until it ends, then stop every request in flight."""
        if self._follow is not None:
            self._spawn(self._listen, self._follow, name="acp-announcements")
        try:
            while line := self._read(source):
                if line.strip():
                    self._message(line)
        finally:
            self.close()

    def close(self) -> None:
        """Stop every request and wait a little for its thread."""
        self._closing.set()
        with self._push_ready:
            self._push_ready.notify_all()  # the push worker is a daemon: a hung `t3` is left
        with self._feed_lock:
            feed = self._feed
        if feed is not None:
            feed.close()
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

    def _listen(self, follow: Callable[[], Feed | None]) -> None:
        """Pass on the running Jarvis's announcements, looking for it again when it goes."""
        while not self._closing.is_set():
            try:
                feed = follow()
            except OSError:
                feed = None
            if feed is not None:
                with self._feed_lock:
                    self._feed = feed
                try:
                    if not self._closing.is_set():
                        for item in feed:
                            self._announce(item)
                except OSError, ValueError:
                    log.info("the announcements stopped", exc_info=True)
                finally:
                    with self._feed_lock:
                        self._feed = None
                    feed.close()
            self._closing.wait(self._retry_s)

    def _announce(self, item: Announcement) -> None:
        """Keep a line for every session's next prompt, and push it into the client's threads."""
        sessions = list(self._sessions.values())
        for session in sessions:
            with session.lock:
                session.keep(item)
        projects = {s.cwd for s in sessions if s.cwd}
        if self._pusher is not None and projects:
            with self._push_ready:
                self._pushes.append((item, projects))  # a full queue loses its oldest
                if self._push_worker is None:
                    self._push_worker = threading.Thread(
                        target=self._push_loop, args=(self._pusher,), name="acp-push", daemon=True
                    )
                    self._push_worker.start()
                self._push_ready.notify()

    def _push_loop(self, pusher: Pusher) -> None:
        """Push the queued announcements one at a time, on one thread, until closed."""
        while True:
            with self._push_ready:
                self._push_ready.wait_for(lambda: bool(self._pushes) or self._closing.is_set())
                if self._closing.is_set():
                    return
                item, projects = self._pushes.popleft()
            self._push(pusher, item, projects)

    def _push(self, pusher: Pusher, item: Announcement, projects: set[str]) -> None:
        """Push `item`; a session of a project it was pushed for need not keep the line."""
        try:
            sent = pusher.push(item, projects)
        except Exception:
            log.exception("could not push an announcement")
            return
        key = (item.id, int(item.at))
        for session in list(self._sessions.values()):
            if session.cwd in sent:
                with session.lock:
                    if session.pending.get(key) is item:
                        del session.pending[key]

    def _flush(self, session: _Session, run: _Run) -> None:
        """Write the announcements that came while no reply was being written."""
        with session.lock:
            keys = tuple(session.pending)
        for key in keys:
            with session.lock:
                note = session.pending.get(key)
            if note is None:
                continue  # a push took care of this one
            if not self._write(
                session, run, f"{SEPARATOR if run.total else ''}Jarvis: {note.text}"
            ):
                return
            with session.lock:
                session.pending.pop(key, None)  # the notice with this identity was shown

    def _read(self, source: TextIO) -> str:
        """Return the next line; one too long is answered with an error and skipped whole."""
        line = source.readline(MAX_LINE + 1)
        if len(line) <= MAX_LINE or line.endswith("\n"):
            return line
        while (rest := source.readline(MAX_LINE)) and not rest.endswith("\n"):
            pass
        self._error(None, INVALID_REQUEST, "message too long")
        return "\n"

    def _send(
        self, message: dict[str, Any], *, session: _Session | None = None, run: _Run | None = None
    ) -> bool:
        """Write one message as one line: the only thing this process writes to its output."""
        line = json.dumps({"jsonrpc": "2.0", **message}, separators=(",", ":"))
        with self._write_lock:
            if session is not None and (session.closed or self._closing.is_set()):
                return False
            if run is not None and (run.cancel.is_set() or not run.attached):
                return False
            try:
                self._out.write(line + "\n")
                self._out.flush()
            except OSError, ValueError:
                self._closing.set()  # the client has gone
                return False
            return True

    def _result(self, request_id: object, result: dict[str, Any]) -> None:
        self._send({"id": request_id, "result": result})

    def _error(self, request_id: object, code: int, message: str) -> None:
        self._send({"id": request_id, "error": {"code": code, "message": message}})

    def _say(
        self,
        session: _Session,
        text: str,
        kind: str = "agent_message_chunk",
        *,
        run: _Run | None = None,
    ) -> bool:
        content = {"type": "text", "text": text}
        update = {"sessionUpdate": kind, "content": content}
        return self._send(
            {"method": "session/update", "params": {"sessionId": session.id, "update": update}},
            session=session,
            run=run,
        )

    # The methods

    def _message(self, line: str) -> None:
        try:
            message: object = json.loads(line)
        except ValueError, RecursionError:  # not JSON, or nested past what the parser reads
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
        except Exception:
            log.exception("a request failed: %s", method)
            self._error(request_id, INTERNAL_ERROR, "Jarvis failed on that request")

    def _dispatch(self, request_id: str | int, method: str, params: dict[str, Any]) -> None:
        match method:
            case "initialize":
                self._result(request_id, self._initialize(params))
            case "session/new":
                answer = self._new_session(params)
                self._result(request_id, answer)
                self._offer_commands(str(answer["sessionId"]))
            case "session/load" if self._store is not None:
                self._result(request_id, self._load_session(params))
                self._offer_commands(str(params["sessionId"]))
            case "session/prompt":
                self._prompt(request_id, params)
            case "session/close":
                self._result(request_id, self._close_session(params))
            case "session/set_config_option":
                self._result(request_id, self._set_option(params))
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
                "loadSession": self._store is not None,
                "sessionCapabilities": {"close": {}},
                "promptCapabilities": {"image": True, "audio": False, "embeddedContext": False},
                "mcpCapabilities": {"http": False, "sse": False},
            },
            "agentInfo": {"name": "jarvis", "title": "Jarvis", "version": self._version},
            "authMethods": [],
        }

    def _new_session(self, params: dict[str, Any]) -> dict[str, Any]:
        full = len(self._sessions) >= MAX_SESSIONS
        if full and all(session.run is not None for session in self._sessions.values()):
            raise _Rpc(INTERNAL_ERROR, f"at most {MAX_SESSIONS} sessions at once, all busy")
        session_id = f"jarvis-{secrets.token_hex(6)}"
        opened = time.time()
        first = self._models[0] if self._models else None
        model = first.id if first is not None else ""
        agent = self._agent_for(session_id, model)
        if first is not None:
            agent.set_effort(first.effort)
        if full:
            self._evict_idle()
        cwd = params.get("cwd")
        folder = cwd if isinstance(cwd, str) and cwd.startswith("/") and "\0" not in cwd else ""
        session = _Session(session_id, agent, cwd=folder, model=model, synced=opened)
        if first is not None:
            session.effort = first.effort
            session.agents[model] = agent
        self._sessions[session_id] = session
        if first is None:
            return {"sessionId": session_id}
        return {"sessionId": session_id, **self._offered(session)}

    def _offer_commands(self, session_id: str) -> None:
        """Tell the client the commands a session takes (`/clear`), as the protocol does."""
        update = {"sessionUpdate": "available_commands_update", "availableCommands": COMMANDS}
        self._send(
            {"method": "session/update", "params": {"sessionId": session_id, "update": update}}
        )

    def _clear(self, request_id: str | int, session: _Session) -> None:
        """Forget the session's conversations and their kept files; the model is not asked.

        Refused (`_idle`) while the session is closed or a request is being answered.
        """
        with session.lock:
            self._idle(session)
            self._forget(session)
        self._say(session, CLEARED)
        self._result(request_id, {"stopReason": "end_turn"})

    def _forget(self, session: _Session) -> None:
        self._forget_memory(session)
        if self._store is not None:
            for path in self._store.glob(f"{session.id}.*.memory.json"):
                path.unlink(missing_ok=True)
        self._audit("acp_cleared", session=session.id)

    @staticmethod
    def _forget_memory(session: _Session) -> None:
        """Make every agent of the session forget its conversation; it matches the store again."""
        agents = session.agents or {session.model: session.agent}
        for agent in agents.values():
            if isinstance(agent, Forgets):
                agent.forget()
        session.synced = time.time()

    def _answer(
        self, session: _Session, hooks: Hooks, text: str, pictures: Sequence[Picture]
    ) -> str:
        """Return the agent's reply to a request.

        The agent first forgets what it remembers when the kept sessions were forgotten after the
        session was opened.
        """
        if self._forgotten(session):
            self._forget_memory(session)
            self._audit("acp_forgotten", session=session.id)
        return session.agent.handle(text, hooks, pictures=pictures)

    def _forgotten(self, session: _Session) -> bool:
        """Whether the kept sessions were forgotten after this one's memory was read or cleared."""
        return self._store is not None and forgotten_at(self._store) > session.synced

    def _agent_for(self, session_id: str, model: str) -> AgentSession:
        """Make a session's agent for `model`, with its kept conversation when there is one.

        Raises:
            _Rpc: it cannot be made; the message says why.
        """
        try:
            agent = self._make_agent(session_id, model)
        except StartError as exc:
            raise _Rpc(INTERNAL_ERROR, str(exc)) from exc
        if self._store is not None and isinstance(agent, Kept):
            agent.load(self._kept_file(self._store, session_id, model))
        return agent

    # Sessions kept across runs

    @staticmethod
    def _kept_file(store: Path, session_id: str, model: str) -> Path:
        return store / f"{session_id}.{model or 'chat'}.memory.json"

    def _keep(self, session: _Session) -> None:
        """Write the session's conversations and its mode and effort to the store."""
        store = self._store
        if store is None or self._forgotten(session):
            return  # forgotten since it was read: what it remembers is not written back
        try:
            self._write_kept(store, session)
            if self._forgotten(session):  # forgotten while it wrote: take it back
                forget_session(store, session.id)
                return
        except OSError as exc:
            log.warning("the session could not be kept: %s", exc)
            return
        except Exception:  # a reply already given is never failed by keeping it
            log.exception("the session could not be kept")
            return
        prune(store)

    def _write_kept(self, store: Path, session: _Session) -> None:
        agents = session.agents or {session.model: session.agent}
        meta = {"model": session.model, "effort": session.effort, "at": time.time()}
        store.mkdir(mode=0o700, parents=True, exist_ok=True)
        for model, agent in agents.items():
            if isinstance(agent, Kept):
                agent.save(self._kept_file(store, session.id, model))
        write_atomic(store / f"{session.id}.json", json.dumps(meta), mode=0o600)

    @staticmethod
    def _kept_meta(store: Path, session_id: str) -> dict[str, Any]:
        try:
            text = read_private(store / f"{session_id}.json")
            return obj(json.loads(text)) if text is not None else {}
        except OSError, ValueError, RecursionError, TypeError:
            return {}

    def _load_session(self, params: dict[str, Any]) -> dict[str, Any]:
        """Reopen a session a run before made: its mode, effort and remembered turns.

        Raises:
            _Rpc: the id is not one Jarvis makes, every session is busy, or its agent cannot
                be made.
        """
        session_id = params.get("sessionId")
        if not isinstance(session_id, str) or not SESSION_ID.fullmatch(session_id):
            raise _Rpc(INVALID_PARAMS, "unknown session")
        open_one = self._sessions.get(session_id)
        if open_one is not None:
            return self._offered(open_one) if self._models else {}
        full = len(self._sessions) >= MAX_SESSIONS
        if full and all(session.run is not None for session in self._sessions.values()):
            raise _Rpc(INTERNAL_ERROR, f"at most {MAX_SESSIONS} sessions at once, all busy")
        opened = time.time()  # before the kept files are read: a forget after it is seen
        meta = self._kept_meta(self._store, session_id) if self._store is not None else {}
        choice = next((m for m in self._models if m.id == meta.get("model")), None)
        choice = choice or (self._models[0] if self._models else None)
        model = choice.id if choice is not None else ""
        agent = self._agent_for(session_id, model)
        effort = meta.get("effort")
        effort = effort if effort in EFFORTS else (choice.effort if choice is not None else "")
        if effort:
            agent.set_effort(str(effort))
        if full:
            self._evict_idle()
        cwd = params.get("cwd")
        folder = cwd if isinstance(cwd, str) and cwd.startswith("/") and "\0" not in cwd else ""
        session = _Session(
            session_id, agent, cwd=folder, model=model, effort=str(effort or ""), synced=opened
        )
        if choice is not None:
            session.agents[model] = agent
        self._sessions[session_id] = session
        if isinstance(agent, Kept):
            for request, reply in agent.history():
                self._say(session, request, "user_message_chunk")
                self._say(session, reply)
        return self._offered(session) if choice is not None else {}

    def _offered(self, session: _Session) -> dict[str, Any]:
        """Return the session's options and its model list, as the app shows them."""
        efforts = [{"value": level, "name": level.capitalize()} for level in EFFORTS]
        listed = [
            {"value": choice.id, "name": choice.name, "description": choice.description}
            for choice in self._models
        ]
        return {
            "configOptions": [
                {
                    "id": MODEL_OPTION,
                    "name": "Model",
                    "category": "model",
                    "type": "select",
                    "currentValue": session.model,
                    "options": listed,
                },
                {
                    "id": EFFORT_OPTION,
                    "name": "Effort",
                    "description": "How hard the model thinks before it answers.",
                    "category": "thought_level",
                    "type": "select",
                    "currentValue": session.effort,
                    "options": efforts,
                },
            ],
            "models": {
                "currentModelId": session.model,
                "availableModels": [
                    {
                        "modelId": choice.id,
                        "name": choice.name,
                        "description": choice.description,
                        "_meta": {
                            "reasoningEfforts": [
                                {"value": level, "label": level.capitalize()} for level in EFFORTS
                            ],
                            "reasoningEffort": choice.effort,
                        },
                    }
                    for choice in self._models
                ],
            },
        }

    def _set_option(self, params: dict[str, Any]) -> dict[str, Any]:
        """Choose the session's model or its effort; answer with every option as it now is.

        A new model is its own agent, made the first time it is chosen, at its own effort.

        Raises:
            _Rpc: no such session, option or value; a request is being answered; or the
                chosen model's agent cannot be made.
        """
        session = self._session(params)
        option, value = params.get("configId"), params.get("value")
        choice = next((m for m in self._models if m.id == value), None)
        if option == MODEL_OPTION and choice is not None:
            with session.lock:
                self._idle(session)
                agent = session.agents.get(choice.id)
            if agent is None:
                agent = self._agent_for(session.id, choice.id)  # may take a while: unlocked
            with session.lock:
                self._idle(session)
                session.agents.setdefault(choice.id, agent)
                session.agent = session.agents[choice.id]
                session.model, session.effort = choice.id, choice.effort
                session.agent.set_effort(choice.effort)
                return self._offered(session)
        if option == EFFORT_OPTION and self._models and value in EFFORTS:
            with session.lock:
                self._idle(session)
                session.agent.set_effort(str(value))
                session.effort = str(value)
                return self._offered(session)
        raise _Rpc(INVALID_PARAMS, f"no option {option} with the value {value}")

    @staticmethod
    def _idle(session: _Session) -> None:
        """Refuse a change while the session answers a request (the lock held).

        Raises:
            _Rpc: the session is closed or a request is in flight.
        """
        if session.closed:
            raise _Rpc(INVALID_PARAMS, "unknown session")
        if session.run is not None:
            raise _Rpc(BUSY, "a request is still being answered")

    def _evict_idle(self) -> None:
        """Make room for a new session after its agent has been built.

        Raises:
            _Rpc: every session is busy.
        """
        for session in self._sessions.values():
            with session.lock:
                if session.run is None:
                    with self._write_lock:
                        session.closed = True
                    del self._sessions[session.id]
                    return
        raise _Rpc(INTERNAL_ERROR, f"at most {MAX_SESSIONS} sessions at once, all busy")

    def _close_session(self, params: dict[str, Any]) -> dict[str, Any]:
        session = self._session(params)
        self._cancel(session, close=True)
        self._sessions.pop(session.id, None)
        return {}

    def _session(self, params: dict[str, Any]) -> _Session:
        found = self._sessions.get(str(params.get("sessionId")))
        if found is None:
            raise _Rpc(INVALID_PARAMS, "unknown session")
        return found

    # A prompt

    def _prompt(
        self, request_id: str | int, params: dict[str, Any], *, stops: int | None = None
    ) -> None:
        """Begin a prompt: a new request, or the owner's answer to the question being asked.

        `stops` is set by a deferred prompt to reject any stop that came while it waited.

        Raises:
            _Rpc: no such session, no list of blocks, or a request still being answered.
        """
        session = self._session(params)
        blocks = objs(params.get("prompt"))
        if not isinstance(params.get("prompt"), list):
            raise _Rpc(INVALID_PARAMS, "prompt is a list of content blocks")
        if self._answered_here(request_id, session, blocks):
            return
        answering: _Question | None = None
        with session.lock:
            if session.closed:
                raise _Rpc(INVALID_PARAMS, "unknown session")
            if self._closing.is_set() or (stops is not None and stops != session.stops):
                self._result(request_id, {"stopReason": "cancelled"})
                return
            run = session.run
            clearing = session.clearing
            if stops is None and clearing is not None and not clearing.is_set():
                # A `/clear` sent earlier still waits for a stopped request: it goes first.
                self._spawn(
                    self._prompt_later,
                    request_id,
                    params,
                    run,
                    session.stops,
                    clearing,
                    name="acp-prompt-wait",
                )
                return
            if run is None:
                run = session.run = _Run()
            elif stops is None and run.cancel.is_set():
                # A stopped request ends at its next check; the reader goes on meanwhile.
                self._spawn(
                    self._prompt_later,
                    request_id,
                    params,
                    run,
                    session.stops,
                    None,
                    name="acp-prompt-wait",
                )
                return
            else:
                answering = run.question
                if (
                    stops is not None
                    or run.attached
                    or answering is None
                    or answering.answered.is_set()
                ):
                    raise _Rpc(BUSY, "a request is still being answered")
                # Only a fresh prompt can answer the question.
                answering.answer = is_yes(_words(blocks), self._names)
                run.attached = True
        if answering is None:
            self._spawn(self._begin, request_id, session, run, blocks, name="acp-prompt")
            return
        self._flush(session, run)  # before the request goes on and writes
        answering.answered.set()
        self._spawn(self._wait, request_id, run, name="acp-answer")

    def _answered_here(
        self, request_id: str | int, session: _Session, blocks: list[dict[str, Any]]
    ) -> bool:
        """Answer a prompt the model never sees: Jarvis's own pushed message, or `/clear`.

        Raises:
            _Rpc: `/clear` while a request runs that is neither stopped nor waiting on a question.
        """
        if (note := self._announced(blocks)) is not None:
            # Jarvis's own message, pushed into the thread: shown, never given to the model, and
            # never taken for the owner's answer to a question that waits. One that starts as
            # such a message but has no valid tag is dropped, also never given to the model.
            if note == FORGED:
                self._audit("acp_refused", reason="forged_announcement", session=session.id)
            with session.lock:
                self._spawn(
                    self._show, request_id, session, note, session.stops, name="acp-announcement"
                )
            return True
        if not _is_clear(blocks):
            return False
        done = previous = None
        with session.lock:
            run = session.run
            if run is not None:
                question = run.question
                if question is not None and not question.answered.is_set() and not run.attached:
                    # A question waits: it is declined and its request stopped, then cleared.
                    question.answer = Declined(NoReason.CLEARED)
                    question.answered.set()
                    run.cancel.set()
                elif not run.cancel.is_set():
                    raise _Rpc(BUSY, "a request is still being answered")
                previous, done = session.clearing, threading.Event()
                session.clearing = done  # a prompt that comes now waits for this clear
        if run is None or done is None:
            self._clear(request_id, session)
        else:
            self._spawn(
                self._clear_later, request_id, session, run, done, previous, name="acp-clear-wait"
            )
        return True

    def _clear_later(
        self,
        request_id: str | int,
        session: _Session,
        run: _Run,
        done: threading.Event,
        previous: threading.Event | None,
    ) -> None:
        """Clear once the request that was stopped has ended (or after `CANCEL_WAIT_S`).

        A `/clear` before this one goes first, and a prompt sent after it waits for `done`.
        """
        try:
            run.finished.wait(CANCEL_WAIT_S)
            if previous is not None:
                previous.wait(CANCEL_WAIT_S)
            self._clear(request_id, session)
        except _Rpc as exc:
            self._error(request_id, exc.code, exc.message)
        finally:
            done.set()

    def _prompt_later(
        self,
        request_id: str | int,
        params: dict[str, Any],
        old: _Run | None,
        stops: int,
        clearing: threading.Event | None,
    ) -> None:
        """Begin a prompt once the request that was stopped has ended (or after `CANCEL_WAIT_S`).

        A `/clear` sent before it (`clearing`) is done first, so the prompt runs on the cleared
        session. A stop that came meanwhile (`stops` is how many there were when it waited) stops
        it too.
        """
        if old is not None:
            old.finished.wait(CANCEL_WAIT_S)
        if clearing is not None:
            clearing.wait(CANCEL_WAIT_S)
        try:
            self._prompt(request_id, params, stops=stops)
        except _Rpc as exc:
            self._error(request_id, exc.code, exc.message)
        except Exception:
            log.exception("a prompt failed")
            self._error(request_id, INTERNAL_ERROR, "Jarvis failed on that request")

    def _audit(self, event: str, **fields: object) -> None:
        """Record an event in the audit log, when there is one; a failure only logs."""
        if self._record is None:
            return
        try:
            self._record(event, **fields)
        except Exception:
            log.exception("could not audit %s", event)

    def _announced(self, blocks: list[dict[str, Any]]) -> Delivery | str | None:
        """Return what to answer a prompt that starts as a message Jarvis pushes, else None.

        That is the announcement when it is nothing but such a message with a valid tag,
        and `FORGED` when it starts as one but is not, or carries more than text.
        """
        if self._pusher is None or not self._pusher.marked(_words(blocks)):
            return None
        if not blocks or any(b.get("type") != "text" for b in blocks):
            return FORGED
        return self._pusher.verified_delivery(_words(blocks).strip()) or FORGED

    def _show(
        self, request_id: str | int, session: _Session, note: Delivery | str, stops: int
    ) -> None:
        """Answer a pushed announcement with its words, and drop the copy that was waiting.

        Under the session's lock, so the words are written whole before a prompt that comes
        next can begin its reply, and never into a reply that is being written: then it is
        refused as busy and the announcement waits for the next reply instead.
        """
        with session.lock:
            if session.closed:
                self._error(request_id, INVALID_PARAMS, "unknown session")
                return
            if session.stops != stops or self._closing.is_set():
                self._result(request_id, {"stopReason": "cancelled"})
                return
            running = session.run
            if running is not None and running.attached:
                if isinstance(note, Delivery):
                    waiting = session.pending.get((note.id, note.at))
                    # Keep the full words, but replace the copy the push worker may remove.
                    session.keep(
                        Delivery(note.id, note.at, note.text if waiting is None else waiting.text)
                    )
                self._error(request_id, BUSY, "a request is still being answered")
                return
            if isinstance(note, str):
                self._say(session, note)
                self._result(request_id, {"stopReason": "refusal"})
                return
            if self._say(session, note.text):
                session.pending.pop((note.id, note.at), None)
            self._result(request_id, {"stopReason": "end_turn"})

    def _begin(
        self, request_id: str | int, session: _Session, run: _Run, blocks: list[dict[str, Any]]
    ) -> None:
        """Read the prompt, start the request on its own thread and wait for its end."""
        try:
            text, pictures = _read_prompt(blocks)
        except _Refused as exc:
            wrote = self._write(session, run, str(exc))
            self._release(session, run)
            self._result(request_id, {"stopReason": "refusal" if wrote else "cancelled"})
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
        self._flush(session, run)
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
            reply = self._answer(session, hooks, text, pictures)
            self._keep(session)
            if not run.cancel.is_set():
                self._finish(session, run, reply)
                self._flush(session, run)  # what came while it was answering
        except Exception as exc:
            log.exception("a request failed")
            run.error = type(exc).__name__
        finally:
            self._release(session, run)
            run.events.put("done")

    def _note(self, session: _Session, run: _Run, text: str) -> None:
        """Say a line about the work in hand, as the agent's thought rather than its reply."""
        self._say(session, text, "agent_thought_chunk", run=run)

    def _write(self, session: _Session, run: _Run, piece: str) -> bool:
        if not piece or not self._say(session, piece, run=run):
            return False
        run.total += piece
        return True

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
        self._flush(session, run)  # the prompt ends with the question: nothing is written after
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

    def _cancel(self, session: _Session, *, close: bool = False) -> None:
        """Stop the request in flight; a question it waits on is answered no."""
        with session.lock, self._write_lock:
            if close:
                session.closed = True
            session.stops += 1
            run = session.run
            if run is None:
                return
            run.cancel.set()
            question = run.question
            if question is not None and not question.answered.is_set():
                question.answer = Declined(NoReason.CLOSED)
                question.answered.set()


def _is_clear(blocks: list[dict[str, Any]]) -> bool:
    """Whether the prompt is the `/clear` command alone: one text block, nothing else."""
    return (
        len(blocks) == 1
        and blocks[0].get("type") == "text"
        and str(blocks[0].get("text", "")).strip().casefold() == CLEAR
    )


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


def serve(  # ruff: ignore[too-many-arguments] - what one connection is given
    source: TextIO,
    out: TextIO,
    make_agent: Callable[[str, str], AgentSession],
    *,
    names: Sequence[str] = (),
    version: str = "",
    follow: Callable[[], Feed | None] | None = None,
    pusher: Pusher | None = None,
    record: Callable[..., None] | None = None,
    models: Sequence[ModelChoice] = (),
    store: Path | None = None,
) -> int:
    """Serve one ACP connection over `source` and `out`; return the exit code (0 at its end).

    `follow` opens the running Jarvis's announcements (`control.subscribe`), tried again while
    none runs; `pusher` sends them into the client's own threads as well; `record` writes
    the audit log's events; `models` are the modes a session may choose (`ModelChoice`).
    """
    server = AcpServer(
        make_agent,
        out,
        names=names,
        version=version,
        follow=follow,
        pusher=pusher,
        record=record,
        models=models,
        store=store,
    )
    server.serve(source)
    return 0
