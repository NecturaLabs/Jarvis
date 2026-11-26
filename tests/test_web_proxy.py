"""The browser's egress proxy: only public addresses on ports 80 and 443, at every connection.

Names resolve through a fake resolver and "public" addresses are carried to a server on loopback
that the test started, so nothing leaves the PC.
"""

from __future__ import annotations

import contextlib
import socket
import threading
import time
from typing import TYPE_CHECKING, Any

import pytest

from jarvis.web import proxy as proxy_module
from jarvis.web.proxy import EgressProxy

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

PUBLIC = "93.184.215.14"
NAMES = {
    "site.example": [PUBLIC],
    "inside.example": ["192.168.1.20"],
    "mixed.example": [PUBLIC, "10.0.0.5"],
    "localhost": ["127.0.0.1"],
}


class Upstream:
    """A loopback server standing in for every public address: it records what reaches it."""

    def __init__(self) -> None:
        super().__init__()
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind(("127.0.0.1", 0))
        self.server.listen(8)
        self.port = int(self.server.getsockname()[1])
        self.received: list[bytes] = []
        self.reply = b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok"
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        while True:
            try:
                conn, _ = self.server.accept()
            except OSError:
                return
            with conn, contextlib.suppress(OSError):
                self._answer(conn)

    def _answer(self, conn: socket.socket) -> None:
        conn.settimeout(5)
        data = b""
        while b"\r\n\r\n" not in data:
            chunk = conn.recv(4096)
            if not chunk:
                break
            data += chunk
        self.received.append(data)
        conn.sendall(self.reply)

    def close(self) -> None:
        self.server.close()


def _resolve(host: str, port: int, **_kw: object) -> list[tuple[Any, ...]]:
    try:
        addresses = NAMES[host]
    except KeyError:
        addresses = [host]  # an address literal resolves to itself
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (a, port)) for a in addresses]


@pytest.fixture
def upstream() -> Iterator[Upstream]:
    server = Upstream()
    yield server
    server.close()


@pytest.fixture
def proxy(upstream: Upstream) -> Iterator[EgressProxy]:
    connected: list[tuple[str, int]] = []

    def connect(address: str, port: int, timeout: float) -> socket.socket:
        connected.append((address, port))
        assert address == PUBLIC, f"the proxy connected to {address}"
        return socket.create_connection(("127.0.0.1", upstream.port), timeout=timeout)

    egress = EgressProxy(resolve=_resolve, connect=connect)
    egress.start()
    yield egress
    egress.stop()


def _signed(proxy: EgressProxy, request: bytes) -> bytes:
    """Return the request with the session's credential, as the browser sends it."""
    line, _, rest = request.partition(b"\r\n")
    return line + b"\r\nProxy-Authorization: " + proxy.authorization.encode() + b"\r\n" + rest


def _ask(proxy: EgressProxy, request: bytes, *, signed: bool = True) -> bytes:
    with socket.create_connection(("127.0.0.1", proxy.port), timeout=5) as client:
        client.sendall(_signed(proxy, request) if signed else request)
        data = b""
        while True:
            chunk = client.recv(4096)
            if not chunk:
                return data
            data += chunk


@pytest.mark.parametrize(
    "target",
    [
        "127.0.0.1:443",
        "localhost:443",
        "[::1]:443",
        "192.168.1.1:443",
        "169.254.169.254:80",
        "inside.example:443",
        "mixed.example:443",  # one private answer is enough to refuse
        "site.example:22",
        "site.example:8080",
    ],
)
def test_a_tunnel_to_anything_but_a_public_web_port_is_refused(
    proxy: EgressProxy, upstream: Upstream, target: str
) -> None:
    answer = _ask(proxy, f"CONNECT {target} HTTP/1.1\r\nHost: {target}\r\n\r\n".encode())

    assert answer.startswith(b"HTTP/1.1 403")
    assert upstream.received == []
    assert proxy.refused == 1


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:8080/v1/models",
        "http://127.0.0.1:8888/search?q=x",
        "http://localhost/",
        "http://10.0.0.1/",
        "http://inside.example/",
    ],
)
def test_a_plain_request_to_this_pc_or_the_network_is_refused(
    proxy: EgressProxy, upstream: Upstream, url: str
) -> None:
    answer = _ask(proxy, f"GET {url} HTTP/1.1\r\nHost: x\r\n\r\n".encode())

    assert answer.startswith(b"HTTP/1.1 403")
    assert upstream.received == []


def test_a_refused_host_is_remembered_for_the_browser_to_tell_apart(proxy: EgressProxy) -> None:
    _ask(proxy, b"CONNECT inside.example:443 HTTP/1.1\r\n\r\n")

    assert proxy.was_refused("inside.example", 443)
    assert proxy.was_refused("INSIDE.example", 443)
    assert not proxy.was_refused("inside.example", 80)
    assert not proxy.was_refused("site.example", 443)


@pytest.mark.parametrize(
    "request_line",
    [
        b"GET / HTTP/1.1",  # not an absolute address
        b"GET ftp://site.example/ HTTP/1.1",
        b"GET https://site.example/ HTTP/1.1",  # https comes as CONNECT
        b"CONNECT user:pw@site.example:443 HTTP/1.1",
        b"CONNECT site.example HTTP/1.1",  # no port
        b"CONNECT site.example:0 HTTP/1.1",
        b"CONNECT site.example:443/path HTTP/1.1",
        b"CONNECT site.example:443?q=1 HTTP/1.1",
        b"CONNECT site.example:443#x HTTP/1.1",
        b"GET http://site.example:0/ HTTP/1.1",
        b"GARBAGE",
        b"GET http://site.example/ SPDY/3",
    ],
)
def test_a_malformed_or_other_request_is_refused(
    proxy: EgressProxy, upstream: Upstream, request_line: bytes
) -> None:
    answer = _ask(proxy, request_line + b"\r\nHost: site.example\r\n\r\n")

    assert answer.startswith(b"HTTP/1.1 403")
    assert upstream.received == []


@pytest.mark.parametrize(
    "request_line",
    [
        b"GET http://[::1 HTTP/1.1",
        b"GET http://[::1/x HTTP/1.1",
        b"GET http://site.example:99999/ HTTP/1.1",
        b"CONNECT [::1 HTTP/1.1",
        b"CONNECT site.example:x HTTP/1.1",
    ],
)
def test_an_address_that_does_not_parse_is_answered_403(
    proxy: EgressProxy, upstream: Upstream, request_line: bytes
) -> None:
    # Not dropped without an answer: the browser is told, and the refusal is counted.
    answer = _ask(proxy, request_line + b"\r\nHost: x\r\n\r\n")

    assert answer.startswith(b"HTTP/1.1 403")
    assert upstream.received == []
    assert proxy.refused == 1


def test_a_tunnel_to_a_public_address_carries_bytes_both_ways(
    proxy: EgressProxy, upstream: Upstream
) -> None:
    with socket.create_connection(("127.0.0.1", proxy.port), timeout=5) as client:
        client.sendall(
            _signed(proxy, b"CONNECT site.example:443 HTTP/1.1\r\nHost: site.example:443\r\n\r\n")
        )
        head = b""
        while b"\r\n\r\n" not in head:
            head += client.recv(4096)
        assert head.startswith(b"HTTP/1.1 200")
        client.sendall(b"HELLO / HTTP/1.1\r\n\r\n")
        reply = b""
        while not reply.endswith(b"ok"):
            chunk = client.recv(4096)
            if not chunk:
                break
            reply += chunk

    assert upstream.received == [b"HELLO / HTTP/1.1\r\n\r\n"]
    assert reply.endswith(b"ok")


def test_stopping_ends_the_tunnels_still_open(upstream: Upstream) -> None:
    # A browser that outlives its session must not keep a way out through an open tunnel.
    def connect(_address: str, _port: int, timeout: float) -> socket.socket:
        return socket.create_connection(("127.0.0.1", upstream.port), timeout=timeout)

    egress = EgressProxy(resolve=_resolve, connect=connect)
    egress.start()
    with socket.create_connection(("127.0.0.1", egress.port), timeout=5) as client:
        client.sendall(_signed(egress, b"CONNECT site.example:443 HTTP/1.1\r\n\r\n"))
        head = b""
        while b"\r\n\r\n" not in head:
            head += client.recv(4096)
        assert head.startswith(b"HTTP/1.1 200")

        egress.stop()

        client.settimeout(3)
        assert client.recv(4096) == b""  # closed, not left waiting for its idle timeout


def test_a_plain_request_goes_on_in_origin_form_without_proxy_headers(
    proxy: EgressProxy, upstream: Upstream
) -> None:
    answer = _ask(
        proxy,
        b"GET http://site.example/a/b?q=1 HTTP/1.1\r\nHost: site.example\r\n"
        b"Proxy-Connection: keep-alive\r\nProxy-Authorization: Basic eA==\r\n"
        b"Connection: keep-alive\r\nAccept: */*\r\n\r\n",
    )

    (sent,) = upstream.received
    lines = sent.split(b"\r\n")
    assert lines[0] == b"GET /a/b?q=1 HTTP/1.1"
    assert b"Accept: */*" in lines
    assert b"Connection: close" in lines
    assert not any(line.lower().startswith(b"proxy-") for line in lines)
    assert b"Connection: keep-alive" not in lines
    assert answer.endswith(b"ok")


def test_the_session_stops_moving_bytes_past_its_budget(upstream: Upstream) -> None:
    upstream.reply = b"HTTP/1.1 200 OK\r\nContent-Length: 5000\r\n\r\n" + b"x" * 5000

    def connect(_address: str, _port: int, timeout: float) -> socket.socket:
        return socket.create_connection(("127.0.0.1", upstream.port), timeout=timeout)

    egress = EgressProxy(resolve=_resolve, connect=connect, max_bytes=1000)
    egress.start()
    try:
        answer = _ask(egress, b"GET http://site.example/ HTTP/1.1\r\nHost: site.example\r\n\r\n")
    finally:
        egress.stop()

    assert len(answer) < 1000


def test_connections_past_the_limit_are_turned_away(upstream: Upstream) -> None:
    egress = EgressProxy(resolve=_resolve, max_connections=1)
    egress.start()
    answer = b""
    try:
        with socket.create_connection(("127.0.0.1", egress.port), timeout=5):
            # The first holds the only slot while it waits for its request head.
            for _ in range(50):
                answer = _ask(egress, b"CONNECT site.example:443 HTTP/1.1\r\n\r\n")
                if answer.startswith(b"HTTP/1.1 503"):
                    break
    finally:
        egress.stop()

    assert answer.startswith(b"HTTP/1.1 503")
    assert upstream.received == []


@pytest.mark.parametrize(
    "authorization",
    [None, "Basic eA==", "Bearer x", "Basic amFydmlzOndyb25n"],
    ids=["none", "garbage", "other-scheme", "wrong"],
)
def test_a_request_without_the_sessions_credential_is_refused(
    proxy: EgressProxy, upstream: Upstream, authorization: str | None
) -> None:
    # Another program on this PC may not use the browser's way out, or spend its budget.
    head = b"CONNECT site.example:443 HTTP/1.1\r\nHost: site.example:443\r\n"
    if authorization is not None:
        head += f"Proxy-Authorization: {authorization}\r\n".encode()

    answer = _ask(proxy, head + b"\r\n", signed=False)

    assert answer.startswith(b"HTTP/1.1 407")
    assert b'Proxy-Authenticate: Basic realm="jarvis"' in answer
    assert upstream.received == []


def test_each_proxy_has_a_credential_of_its_own() -> None:
    one, two = EgressProxy(resolve=_resolve), EgressProxy(resolve=_resolve)

    assert one.authorization != two.authorization
    assert one.authorization.startswith("Basic ")
    assert one.url(40123).startswith("http://jarvis:")
    assert one.url(40123).endswith("@127.0.0.1:40123")


def test_a_slot_is_taken_before_a_thread_is_made(
    upstream: Upstream, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A burst of connections past the cap costs no thread each.
    made: list[str] = []
    real = threading.Thread

    class Counted(real):
        def __init__(
            self, *, target: Callable[..., object] | None = None, **kwargs: object
        ) -> None:
            made.append(getattr(target, "__name__", ""))
            super().__init__(target=target, **kwargs)  # pyright: ignore[reportArgumentType] - passed through

    egress = EgressProxy(resolve=_resolve, max_connections=1)
    egress.start()
    try:
        with socket.create_connection(("127.0.0.1", egress.port), timeout=5):
            time.sleep(0.2)  # the first takes the only slot, waiting for its request head
            monkeypatch.setattr(threading, "Thread", Counted)
            answers = [
                _ask(egress, b"CONNECT site.example:443 HTTP/1.1\r\n\r\n") for _ in range(30)
            ]
            monkeypatch.undo()
    finally:
        egress.stop()

    assert all(answer.startswith(b"HTTP/1.1 503") for answer in answers)
    assert made.count("process_request_thread") == 0
    assert upstream.received == []


def test_a_request_head_sent_slowly_is_cut_at_one_deadline(
    upstream: Upstream, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Sending a byte before each read's timeout no longer keeps the slot.
    monkeypatch.setattr(proxy_module, "HEAD_TIMEOUT_S", 0.6)
    egress = EgressProxy(resolve=_resolve)
    egress.start()
    started = time.monotonic()
    try:
        with socket.create_connection(("127.0.0.1", egress.port), timeout=5) as client:
            answer = b""
            for byte in _signed(egress, b"CONNECT site.example:443 HTTP/1.1\r\n\r\n"):
                try:
                    client.sendall(bytes([byte]))
                except OSError:
                    break
                if time.monotonic() - started > 3:
                    break
                time.sleep(0.1)
            client.settimeout(3)
            with contextlib.suppress(OSError):
                answer = client.recv(4096)
    finally:
        egress.stop()

    assert time.monotonic() - started < 3
    assert not answer.startswith(b"HTTP/1.1 200")
    assert upstream.received == []


def test_lookups_at_once_are_bounded(upstream: Upstream, monkeypatch: pytest.MonkeyPatch) -> None:
    # A lookup the system never answers outlives its caller; past the bound, more are refused.
    monkeypatch.setattr(proxy_module, "MAX_LOOKUPS", 2)
    release = threading.Event()
    asked: list[str] = []

    def stuck(host: str, port: int, **_kw: object) -> list[tuple[Any, ...]]:
        asked.append(host)
        release.wait(10)
        return _resolve(host, port)

    def connect(_address: str, _port: int, timeout: float) -> socket.socket:
        return socket.create_connection(("127.0.0.1", upstream.port), timeout=timeout)

    egress = EgressProxy(resolve=stuck, connect=connect)
    egress.start()
    held: list[socket.socket] = []
    try:
        held.extend(socket.create_connection(("127.0.0.1", egress.port), timeout=5) for _ in "ab")
        for client in held:
            client.sendall(_signed(egress, b"CONNECT site.example:443 HTTP/1.1\r\n\r\n"))
        while len(asked) < 2:
            time.sleep(0.01)
        third = _ask(egress, b"CONNECT site.example:443 HTTP/1.1\r\n\r\n")
    finally:
        release.set()
        for client in held:
            client.close()
        egress.stop()

    assert third.startswith(b"HTTP/1.1 403")
    assert len(asked) == 2


def test_a_lookup_that_ends_after_stop_dials_nothing(upstream: Upstream) -> None:
    release = threading.Event()
    looked_up = threading.Event()
    dialled: list[str] = []

    def slow(host: str, port: int, **_kw: object) -> list[tuple[Any, ...]]:
        looked_up.set()
        release.wait(10)
        return _resolve(host, port)

    def connect(address: str, _port: int, timeout: float) -> socket.socket:
        dialled.append(address)
        return socket.create_connection(("127.0.0.1", upstream.port), timeout=timeout)

    egress = EgressProxy(resolve=slow, connect=connect)
    egress.start()
    with socket.create_connection(("127.0.0.1", egress.port), timeout=5) as client:
        client.sendall(_signed(egress, b"CONNECT site.example:443 HTTP/1.1\r\n\r\n"))
        assert looked_up.wait(5)
        egress.stop()
        release.set()
        client.settimeout(3)
        with contextlib.suppress(OSError):
            client.recv(4096)

    time.sleep(0.3)
    assert dialled == []
