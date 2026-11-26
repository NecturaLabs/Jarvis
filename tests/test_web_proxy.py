"""The browser's egress proxy: only public addresses on ports 80 and 443, at every connection.

Names resolve through a fake resolver and "public" addresses are carried to a server on loopback
that the test started, so nothing leaves the PC.
"""

from __future__ import annotations

import contextlib
import socket
import threading
from typing import TYPE_CHECKING, Any

import pytest

from jarvis.web.proxy import EgressProxy

if TYPE_CHECKING:
    from collections.abc import Iterator

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


def _ask(proxy: EgressProxy, request: bytes) -> bytes:
    with socket.create_connection(("127.0.0.1", proxy.port), timeout=5) as client:
        client.sendall(request)
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
        client.sendall(b"CONNECT site.example:443 HTTP/1.1\r\nHost: site.example:443\r\n\r\n")
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
        client.sendall(b"CONNECT site.example:443 HTTP/1.1\r\n\r\n")
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
