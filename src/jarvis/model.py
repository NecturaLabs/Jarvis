"""OpenAI-compatible chat client for the Strata server, with non-generating health checks."""

import json
import logging
import math
import queue
import re
import ssl
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast
from urllib.parse import urlsplit

import httpx

from jarvis import ramguard
from jarvis.config import STRATA_CONTEXT_MAX, STRATA_CONTEXT_MIN, private_endpoint
from jarvis.jsonutil import obj, objs

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterator
    from pathlib import Path

    from jarvis.config import ModelConfig
    from jarvis.ramguard import Verdict

log = logging.getLogger(__name__)

# A load request waits as long as Strata may take to answer it: up to 3 engine starts of 900 s each
# (`engine_ready_s`), and a margin. A cold load from disk took about 110 s.
LOAD_TIMEOUT_S = 3 * 900.0 + 120.0
WARM_ATTEMPTS = 60
WARM_RETRY_S = 2.0
RETRY_AFTER_RANGE_S = (1.0, 30.0)  # what a server's `Retry-After` is clamped to while warming
SERVICE_UNAVAILABLE = 503
DEFAULT_MAX_TOKENS = 400
REMOTE_CONNECT_S = 2.0
# A remote probe is one attempt of at most REMOTE_CONNECT_S + REMOTE_PROBE_READ_S, and its answer,
# good or bad, is reused for REMOTE_PROBE_CACHE_S, so a backend that is down costs a turn one wait.
REMOTE_PROBE_READ_S = 1.0
REMOTE_PROBE_CACHE_S = 5.0
REMOTE_RETRY_PAUSE_S = 0.5  # before the one retry of a dropped remote connection
_MODEL_ID = re.compile(r"[a-zA-Z0-9_./:@-]{1,160}")
_MODEL_STATES = frozenset({"loaded", "unloaded", "loading", "failed"})


def _tool_choice(*, require_tool: bool | str) -> str | dict[str, Any]:
    """Return the request's `tool_choice`: a named tool, any tool, or the model's own pick."""
    if isinstance(require_tool, str):
        return {"type": "function", "function": {"name": require_tool}}
    return "required" if require_tool else "auto"


class ModelError(Exception):
    """The model server could not answer."""


class ModelMissingError(ModelError):
    """The model server is down and its model file is not there (its drive not mounted, say)."""


class ModelMemoryError(ModelError):
    """The model would not fit in the memory that is free, so loading it was refused."""


class RemoteModelError(ModelError):
    """The remote model server could not be reached or refused a request."""


NOT_PRIVATE = "host resolves outside your own machine or private network"


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelStatus:
    """A non-generating observation; absent metadata is kept distinct from an explicit value."""

    reachable: bool
    latency_ms: int
    state: str | None = None
    name: str | None = None
    context: int | None = None
    pictures: bool | None = None
    issue: str = ""

    def report(self, name: str, label: str = "remote model server") -> str:
        """Describe the probe without exposing the URL, key or a response body."""
        if not self.reachable:
            return f"{label} {self.issue or 'unreachable'} ({self.latency_ms} ms)"
        state = self.state or "state unknown"
        return f"{label} reachable, {self.name or name} ({state}, {self.latency_ms} ms)"


def _model_status(listed: object, latency_ms: int) -> ModelStatus:
    if listed == []:
        return ModelStatus(reachable=True, latency_ms=latency_ms, state="unloaded")
    data = objs(listed)
    if not data:
        return ModelStatus(
            reachable=False, latency_ms=latency_ms, issue="returned an invalid model list"
        )
    entry = data[0]
    state = obj(entry.get("status")).get("value")
    name = entry.get("id")
    context = entry.get("context_length")
    pictures = entry.get("vision")
    return ModelStatus(
        reachable=True,
        latency_ms=latency_ms,
        state=state if isinstance(state, str) and state in _MODEL_STATES else None,
        name=name if isinstance(name, str) and _MODEL_ID.fullmatch(name) else None,
        context=context
        if type(context) is int and STRATA_CONTEXT_MIN <= context <= STRATA_CONTEXT_MAX
        else None,
        pictures=pictures if isinstance(pictures, bool) else None,
    )


def tls_context(ca_file: Path) -> ssl.SSLContext:
    """Trust a specific CA or self-signed server certificate, with hostname verification."""
    return ssl.create_default_context(cafile=str(ca_file))


@dataclass(frozen=True, slots=True)
class Delta:
    """One piece of an answer as it is written: reply text, part of a tool call, or thinking.

    `call` is the tool call's position in the answer (None for reply text); `name` comes with a
    call's first piece and `arguments` is the next stretch of its JSON text. A `thinking` piece
    carries no words: it says the model is still thinking, so a stop need not wait for the answer.
    """

    text: str = ""
    call: int | None = None
    name: str = ""
    arguments: str = ""
    thinking: bool = False


@dataclass(frozen=True, slots=True)
class Thinking:
    """How much the model thinks before it answers: Strata's `reasoning_effort` and a cap.

    Strata closes the thinking once it reaches `tokens` and lets the model answer; the thinking
    is streamed apart from the answer (`reasoning_content`), so it is never spoken or shown.
    """

    effort: Literal["low", "medium", "high"]
    tokens: int


class ChatModel(Protocol):
    """What the agent needs from a model."""

    def chat(  # ruff: ignore[too-many-arguments] - one call's settings, all keyword-only
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        require_tool: bool | str = False,
        timeout_s: float | None = None,
        on_delta: Callable[[Delta], None] | None = None,
        max_tokens: int | None = None,
        load_s: float = 0.0,
        thinking: Thinking | None = None,
    ) -> dict[str, Any]:
        """Return the assistant message; with `require_tool`, the model must call a tool.

        `require_tool` may instead name the one tool the model must call; the tool list is still
        sent whole, so the prompt the server pinned stays the same.

        `timeout_s` caps how long this one call may wait (the time the request has left);
        `load_s` is how much longer than one answer it may take, as the model must load first.
        With `on_delta`, the answer is streamed and each piece is passed to it as it arrives; an
        exception it raises ends the call and is raised from here. `max_tokens` caps the answer.
        With `thinking`, the model thinks first, within its cap; without, it answers at once.
        The message also carries `finish_reason` ("length" when the cap cut it) and `usage`
        (the server's token counts) when the server sent them.
        """
        ...

    def is_loaded(self) -> bool | None:
        """Return whether the model is in memory (None when unknown); never generates."""
        ...


def _message(response: httpx.Response) -> dict[str, Any]:
    """Return the assistant message of a chat completion answer.

    Raises:
        ModelError: the server refused or the answer holds no message object.
    """
    if response.is_error:  # httpx: status 400 to 599
        msg = f"model server answered {response.status_code}: {response.text[:200]}"
        raise ModelError(msg)
    try:
        data = obj(response.json())
        choices = objs(data.get("choices"))
    except ValueError as exc:
        msg = f"model server sent an unexpected answer: {exc}"
        raise ModelError(msg) from exc
    message = choices[0].get("message") if choices else None
    if not isinstance(message, dict):
        msg = "model server sent an answer without a message"
        raise ModelError(msg)
    answer = dict(cast("dict[str, Any]", message))
    finish = choices[0].get("finish_reason")
    if isinstance(finish, str):
        answer["finish_reason"] = finish
    usage = data.get("usage")
    if isinstance(usage, dict):
        answer["usage"] = usage
    timings = data.get("timings")
    if isinstance(timings, dict):
        answer["timings"] = timings
    return answer


def _events(lines: Iterator[str]) -> Iterator[dict[str, Any]]:
    """Yield the JSON events of a server-sent event stream until `[DONE]`.

    Raises:
        ModelError: an event is not a JSON object.
    """
    for line in lines:
        if not line.startswith("data:"):
            continue
        data = line.removeprefix("data:").strip()
        if data == "[DONE]":
            return
        try:
            yield obj(json.loads(data))
        except ValueError as exc:
            msg = f"model server sent an unexpected stream event: {exc}"
            raise ModelError(msg) from exc


@dataclass(slots=True)
class _Assembly:
    """An answer put together from its streamed pieces, in the shape of a whole answer."""

    content: str = ""
    calls: dict[int, dict[str, Any]] = field(default_factory=dict[int, dict[str, Any]])
    finish_reason: str = ""
    usage: dict[str, Any] = field(default_factory=dict[str, Any])
    timings: dict[str, Any] = field(default_factory=dict[str, Any])

    def add(self, delta: dict[str, Any]) -> list[Delta]:
        pieces: list[Delta] = []
        thought = delta.get("reasoning_content")
        if isinstance(thought, str) and thought:
            pieces.append(Delta(thinking=True))
        text = delta.get("content")
        if isinstance(text, str) and text:
            self.content += text
            pieces.append(Delta(text=text))
        for part in objs(delta.get("tool_calls")):
            index = part.get("index")
            if not isinstance(index, int) or isinstance(index, bool):
                msg = "model server streamed a tool call without its position"
                raise ModelError(msg)
            call = self.calls.setdefault(
                index, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
            )
            if isinstance(part.get("id"), str):
                call["id"] = part["id"]
            function = obj(part.get("function"))
            name = function.get("name")
            arguments = function.get("arguments")
            name = name if isinstance(name, str) else ""
            arguments = arguments if isinstance(arguments, str) else ""
            call["function"]["name"] += name
            call["function"]["arguments"] += arguments
            pieces.append(Delta(call=index, name=name, arguments=arguments))
        return pieces

    def message(self) -> dict[str, Any]:
        message: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.calls:
            message["tool_calls"] = [self.calls[i] for i in sorted(self.calls)]
        if self.finish_reason:
            message["finish_reason"] = self.finish_reason
        if self.usage:
            message["usage"] = self.usage
        if self.timings:
            message["timings"] = self.timings
        return message


def _retry_after(response: httpx.Response) -> float | None:
    """Return the seconds a `Retry-After` header asks for, clamped; None without one we read."""
    try:
        asked = float(response.headers["Retry-After"])
    except KeyError, ValueError:
        return None  # absent, or an HTTP date (Strata sends seconds)
    if math.isnan(asked):
        return None
    low, high = RETRY_AFTER_RANGE_S
    return min(max(asked, low), high)


@dataclass(slots=True)
class _Loads:
    """The model loads Jarvis asked for whose answer has not come yet.

    Strata reports a model only as `loaded` or `unloaded`, also while it loads one, and answers a
    load request once the load has ended, so a load counts while its request waits.
    """

    _lock: threading.Lock = field(default_factory=threading.Lock)
    _asking: int = 0

    @contextmanager
    def asking(self) -> Generator[None]:
        """Count a load request while it waits for its answer."""
        with self._lock:
            self._asking += 1
        try:
            yield
        finally:
            with self._lock:
                self._asking -= 1

    @contextmanager
    def asking_alone(self) -> Generator[bool]:
        """Count a load request, unless one is waiting already: then yield False and count none."""
        with self._lock:
            alone = self._asking == 0
            self._asking += alone
        try:
            yield alone
        finally:
            with self._lock:
                self._asking -= alone

    def pending(self) -> bool:
        """Return whether a load request is waiting for its answer."""
        with self._lock:
            return self._asking > 0


@dataclass(slots=True)
class _Probes:
    """The last remote probe and when it was made, shared by every thread that asks."""

    _lock: threading.Lock = field(default_factory=threading.Lock)
    _status: ModelStatus | None = None
    _at: float = 0.0

    def recent(self, now: float) -> ModelStatus | None:
        """Return the last probe when it is younger than the cache time."""
        with self._lock:
            fresh = self._status is not None and 0.0 <= now - self._at < REMOTE_PROBE_CACHE_S
            return self._status if fresh else None

    def keep(self, status: ModelStatus, now: float) -> None:
        """Remember a probe made at `now`."""
        with self._lock:
            self._status, self._at = status, now


@dataclass(frozen=True, slots=True)
class StrataModel:
    """Talks to `strata serve` over its OpenAI API."""

    config: ModelConfig
    api_key: str = field(repr=False)
    transport: httpx.BaseTransport | None = None  # tests answer in place of the server
    sleep: Callable[[float], None] = time.sleep  # tests wait without waiting
    clock: Callable[[], float] = time.monotonic  # tests move time with their sleep
    model_file: Path | None = None  # what the server loads (the run config's `--native`)
    # ... as it is now, where what the server loads changes while this client lives (a mode)
    file_of: Callable[[], Path | None] | None = None
    # Says whether the model fits in the memory free now, for the file the server loads; asked
    # before any load, and by a local server's request that must load it or finds it down.
    ram_guard: Callable[[Path | None], Verdict] | None = None
    remote: bool = False
    tls_ca: Path | None = None
    resolve: Callable[[str], list[str]] | None = None  # tests answer name lookups in place of DNS
    _loads: _Loads = field(default_factory=_Loads, init=False, repr=False, compare=False)
    _probes: _Probes = field(default_factory=_Probes, init=False, repr=False, compare=False)

    def file_missing(self) -> bool:
        """Return whether the server's model file is known and not there."""
        file = self.served_file()
        return file is not None and not file.is_file()

    def served_file(self) -> Path | None:
        """Return the model file the server loads now, when known."""
        return self.model_file if self.file_of is None else self.file_of()

    def _no_room(self) -> str:
        """Return why the model must not load now, or "" (also for a remote server's).

        Nothing is refused while a load of this client waits for its answer, or while the server
        has the model: its memory is no longer in what is available, so it would look too big.
        """
        if self.remote or self.ram_guard is None or self._loads.pending():
            return ""
        verdict = self.ram_guard(self.served_file())
        if not verdict.short or self._listing()[1] in {"loaded", "loading"}:
            return ""
        return ramguard.refusal("model load", verdict)

    def _load_refused(self) -> bool:
        """Return whether a load must not be asked for now, saying why in the log."""
        why = self._no_room()
        if why:
            log.warning("the model was not loaded: %s", why)
        return bool(why)

    def _client(self, timeout: float, *, connect: float | None = None) -> httpx.Client:
        """Return a client for one request; a remote one first checks where its host leads.

        Raises:
            RemoteModelError: the remote host's name does not resolve to private addresses only.
        """
        if self.remote:
            host = urlsplit(self.config.base_url).hostname or ""
            if not private_endpoint(host, self.resolve):
                msg = f"remote model server {NOT_PRIVATE}"
                raise RemoteModelError(msg)
        return httpx.Client(
            base_url=self.config.base_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=httpx.Timeout(
                timeout, connect=min(timeout, REMOTE_CONNECT_S) if connect is None else connect
            )
            if self.remote
            else timeout,
            verify=tls_context(self.tls_ca) if self.tls_ca is not None else True,
            trust_env=False,
            transport=self.transport,
        )

    def probe(self) -> ModelStatus:
        """Read /models with a short timeout; a remote server gets one attempt and a cached answer.

        A remote answer, also a failure, is kept for `REMOTE_PROBE_CACHE_S`, so a turn, a status
        and the voice loop do not each wait out a backend that is down.
        """
        if not self.remote:
            return self._probe_once()
        if (recent := self._probes.recent(self.clock())) is not None:
            return recent
        status = self._probe_once()
        self._probes.keep(status, self.clock())
        return status

    def _probe_once(self) -> ModelStatus:
        began = self.clock()
        issue = "unreachable"
        try:
            with (
                self._client(REMOTE_PROBE_READ_S, connect=REMOTE_CONNECT_S)
                if self.remote
                else self._client(REMOTE_CONNECT_S)
            ) as client:
                response = client.get("/models")
                response.raise_for_status()
                return _model_status(obj(response.json()).get("data"), self._latency(began))
        except httpx.HTTPStatusError as exc:
            issue = f"returned HTTP {exc.response.status_code}"
        except httpx.TransportError:
            pass
        except ValueError:
            issue = "returned an invalid model list"
        except RemoteModelError:
            issue = NOT_PRIVATE
        except OSError:
            issue = "could not read the TLS CA"
        return ModelStatus(reachable=False, latency_ms=self._latency(began), issue=issue)

    def _latency(self, began: float) -> int:
        return round((self.clock() - began) * 1000)

    def _listing(self) -> tuple[bool, str | None]:
        """Return whether the server answered, and the state of the model it lists, if any.

        Strata lists no model while its engine is down or restarting: the model is not in memory.
        """
        if self.remote:
            probe = self.probe()
            return probe.reachable, probe.state
        try:
            with self._client(5.0) as client:
                response = client.get("/models")
                response.raise_for_status()
                listed = obj(response.json()).get("data")
        except httpx.HTTPError, ValueError:
            return False, None
        if listed == []:
            return True, "unloaded"
        data = objs(listed)
        value = obj(data[0].get("status")).get("value") if data else None
        return True, (str(value) if value is not None else None)

    def status(self) -> str | None:
        """Return the model's state (`loaded`, `unloaded`...) without generating.

        While a load Jarvis asked for is under way it is `loading`, whatever the server says
        (Strata says only `loaded` or `unloaded`); a server that cannot be reached is None.
        """
        was_loading = self._loads.pending()  # the server's answer may be from before the load ended
        answered, state = self._listing()
        if not answered:
            return None
        loading = was_loading or self._loads.pending()
        return "loading" if state != "loaded" and loading else state

    def is_loaded(self) -> bool | None:
        """Return whether the model is in memory (None when the server is unreachable)."""
        state = self.status()
        return None if state is None else state == "loaded"

    def load(self) -> bool:
        """Ask the server to load the model now (ahead of a request); return whether it accepted.

        The request waits for the load, as long as Strata lets an engine start; a load still
        going after that carries on in the server. While a load Jarvis asked for is under way,
        this one joins it instead of queueing a second behind it.
        """
        if self.remote or self._load_refused():
            return False
        try:
            with self._loads.asking_alone() as alone:
                if not alone:
                    return True
                with self._client(LOAD_TIMEOUT_S) as client:
                    response = client.post("/load", json={})
        except httpx.ReadTimeout:
            return True
        except httpx.HTTPError:
            return False
        return response.is_success

    def warm(
        self, attempts: int = WARM_ATTEMPTS, *, keep_going: Callable[[], bool] = lambda: True
    ) -> bool:
        """Load the model once the server is up (at service start); return whether it loaded.

        The server may still be starting (unreachable, or answering 409 while busy), so this
        waits and tries again, up to `attempts` times; a 503 that says `Retry-After` is waited
        out for that long (1 to 30 seconds) instead of the usual two. A load reads the whole model
        and can take minutes when it is not in the page cache, so the request waits for it. Each
        attempt first asks `keep_going`, so a load nobody wants any more (a pause came) stops.
        """
        if self.remote or self._load_refused():
            return False
        for _ in range(attempts):
            if not keep_going():
                return False
            wait = WARM_RETRY_S
            # Answering, even with no model listed while it restarts; still wanted after the check.
            if self._listing()[0] and keep_going():
                try:
                    with self._loads.asking(), self._client(LOAD_TIMEOUT_S) as client:
                        response = client.post("/load", json={})
                except httpx.ReadTimeout:
                    return True  # still loading on the server's side
                except httpx.HTTPError:
                    pass
                else:
                    if response.is_success:
                        return True
                    if response.status_code == SERVICE_UNAVAILABLE:
                        wait = _retry_after(response) or wait
            self.sleep(wait)
        return False

    def unload(self) -> bool:
        """Ask the server to free the model now; return whether it did."""
        if self.remote:
            return False
        try:
            with self._client(30.0) as client:
                response = client.post("/unload", json={})
        except httpx.HTTPError:
            return False
        return response.is_success

    def chat(  # ruff: ignore[too-many-arguments] - one call's settings, all keyword-only
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        require_tool: bool | str = False,
        timeout_s: float | None = None,
        on_delta: Callable[[Delta], None] | None = None,
        max_tokens: int | None = None,
        load_s: float = 0.0,
        thinking: Thinking | None = None,
    ) -> dict[str, Any]:
        """Send one chat completion; retry once after a dropped connection.

        The call waits at most `timeout_s` (when given) and never longer than the configured
        request timeout, plus `load_s` when the model must load first. With `on_delta` the
        answer is streamed (see `ChatModel.chat`); a connection that drops after a piece was
        passed on is not retried, since that piece may already have been spoken.
        Without `thinking`, thinking is switched off. With it, the server's `max_tokens` (which
        counts the thinking too) is the answer's cap plus the thinking's, so the answer keeps all of
        its room.

        A failed request raises ModelError. A local server that is down with its model file
        missing raises ModelMissingError.

        Raises:
            ModelMemoryError: the request must load the model (`load_s`) and the memory that is
                free would not hold it: the server would load it all the same, and the kernel
                would take memory from a running program.
        """
        if load_s > 0 and (why := self._no_room()):
            raise ModelMemoryError(why)
        body: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": max_tokens or DEFAULT_MAX_TOKENS,
        }
        if self.config.temperature is not None:
            body["temperature"] = self.config.temperature
        if thinking is None:
            body["reasoning_effort"] = "none"
            body["chat_template_kwargs"] = {"enable_thinking": False}
        else:
            body["reasoning_effort"] = thinking.effort
            body["reasoning_budget_tokens"] = thinking.tokens
            body["max_tokens"] += thinking.tokens
        if tools:
            body["tools"] = tools
            body["tool_choice"] = _tool_choice(require_tool=require_tool)
            if len(messages) > 1 and messages[0].get("role") == "system":
                # Strata keeps the system prompt and tools read once, pinned, so a request reads
                # only its own words (an engine without pin support ignores the mark).
                body["strata_prefix"] = {"messages": 1}
        limit = self.config.request_timeout_s + load_s
        if timeout_s is not None:
            limit = min(limit, timeout_s)
        return self._complete(body, self.clock() + limit, on_delta)

    def _complete(
        self, body: dict[str, Any], deadline: float, on_delta: Callable[[Delta], None] | None
    ) -> dict[str, Any]:
        passed_on: list[Delta] = []
        for attempt in (1, 2):
            try:
                with self._client(max(1.0, deadline - self.clock())) as client:
                    if on_delta is None:
                        return self._answer(client.post("/chat/completions", json=body))
                    body["stream"] = True
                    body["stream_options"] = {"include_usage": True}
                    with client.stream("POST", "/chat/completions", json=body) as response:
                        return self._streamed(response, deadline, on_delta, passed_on)
            except httpx.TransportError as exc:
                retry = attempt == 1 and not passed_on and not isinstance(exc, httpx.ReadTimeout)
                self._recover_connection(exc, deadline, retry=retry)
            except OSError as exc:
                if not self.remote:
                    raise
                msg = "remote model server TLS CA could not be read"
                raise RemoteModelError(msg) from exc
        msg = "unreachable"
        raise ModelError(msg)

    def _recover_connection(
        self, error: httpx.TransportError, deadline: float, *, retry: bool
    ) -> None:
        if self.remote:
            # A probe that just failed already waited out this server: no second wait for it.
            down = self._probes.recent(self.clock())
            if retry and self.clock() < deadline and (down is None or down.reachable):
                self.sleep(min(REMOTE_RETRY_PAUSE_S, max(0.0, deadline - self.clock())))
                return
            msg = "remote model server unreachable or timed out"
            raise RemoteModelError(msg) from error
        if self.file_missing():  # systemd skips Strata then: nothing to wait for
            msg = f"model server down and its model file is missing: {self.served_file()}"
            raise ModelMissingError(msg) from error
        if why := self._no_room():  # its unit's own check refused the start
            raise ModelMemoryError(why) from error
        if not retry:
            msg = f"model server unreachable: {error}"
            raise ModelError(msg) from error
        self._wait_ready(min(30.0, max(0.0, deadline - self.clock())))

    def _answer(self, response: httpx.Response) -> dict[str, Any]:
        if self.remote and not response.is_success:
            msg = f"remote model server returned HTTP {response.status_code}"
            raise RemoteModelError(msg)
        return _message(response)

    def _streamed(
        self,
        response: httpx.Response,
        deadline: float,
        on_delta: Callable[[Delta], None],
        passed_on: list[Delta],
    ) -> dict[str, Any]:
        """Read a streamed answer, passing on each piece; return the whole message.

        Raises:
            ModelError: the server refused, sent something unreadable or ran out of time.
        """
        if self.remote and not response.is_success:
            return self._answer(response)
        if response.is_error:  # httpx: status 400 to 599
            response.read()
            return _message(response)
        answer = _Assembly()
        finished = False
        for event in _events(self._lines_until(response, deadline)):
            usage = event.get("usage")
            if isinstance(usage, dict):
                answer.usage = cast("dict[str, Any]", usage)
            timings = event.get("timings")
            if isinstance(timings, dict):
                answer.timings = cast("dict[str, Any]", timings)
            choices = objs(event.get("choices"))
            if not choices:
                continue
            for piece in answer.add(obj(choices[0].get("delta"))):
                passed_on.append(piece)
                on_delta(piece)
            reason = choices[0].get("finish_reason")
            if isinstance(reason, str):
                answer.finish_reason = reason
                finished = True
        if not finished:  # cut off: what arrived is not the whole answer
            msg = "model server's answer ended before it was finished"
            raise ModelError(msg)
        return answer.message()

    def _lines_until(self, response: httpx.Response, deadline: float) -> Iterator[str]:
        """Yield the response's lines, read on a thread of their own, until `deadline` on `clock`.

        The read timeout restarts with every byte, so a stream that stalls, or trickles, would
        otherwise run on past its time; here it ends at it, whether a line comes or not. Leaving
        `client.stream` closes the response, which ends the reader.

        Raises:
            ModelError: the time ran out.
        """
        lines: queue.Queue[str | Exception | None] = queue.Queue()

        def read() -> None:
            try:
                for line in response.iter_lines():
                    lines.put(line)
            except Exception as exc:  # ruff: ignore[blind-except] - raised on the caller's thread
                lines.put(exc)
            else:
                lines.put(None)

        threading.Thread(target=read, name="jarvis-model-stream", daemon=True).start()
        late = "model server took too long to answer"
        while True:
            try:
                line = lines.get(timeout=max(0.0, deadline - self.clock()))
            except queue.Empty:
                raise ModelError(late) from None
            if self.clock() > deadline:
                raise ModelError(late)
            if line is None:
                return
            if isinstance(line, Exception):
                raise line
            yield line

    def _wait_ready(self, seconds: float) -> None:
        deadline = self.clock() + seconds
        while self.clock() < deadline:
            if self.status() is not None:
                return
            self.sleep(1.0)
