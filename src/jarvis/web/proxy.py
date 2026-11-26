"""A local proxy the headless browser must send every connection through: public addresses only.

Which addresses a page reaches is decided by the page (its scripts, frames, redirects and links),
so the browser cannot be trusted to stay off this PC, the home network or a cloud metadata
address. Chrome is started with this proxy as its only way out, loopback included
(`<-loopback>`), and with every name unresolvable on its own (`--host-resolver-rules`), so each
connection a page makes arrives here first. Here, as in `jarvis.web.guard`, the host is resolved,
every address it resolves to must be public, only ports 80 and 443 are reached, and the
connection goes to the address that was checked: a name that resolves to a public address for the
check cannot then connect to a private one.

Only the session's own browser may use it: each proxy has a random credential, which the browser
sends (`Proxy-Authorization`, after a `407`) and which no other program on this PC knows.

HTTPS and secure WebSockets come as `CONNECT host:port` and are tunnelled unread (TLS runs between
the browser and the site). Plain HTTP comes as an absolute address and is sent on with
`Connection: close`, one request per connection. Anything else is refused. All the traffic of one
browser session together is capped (`MAX_BYTES`), so a page cannot fill the disk with a download.
"""

import base64
import contextlib
import hmac
import re
import secrets
import selectors
import socket
import socketserver
import threading
import time
from collections import deque
from typing import TYPE_CHECKING, Any, override
from urllib.parse import urlsplit

from jarvis.web.guard import BlockedDestinationError, public_addresses

if TYPE_CHECKING:
    from collections.abc import Callable

    from jarvis.web.guard import Resolver

LOOPBACK = "127.0.0.1"
USER = "jarvis"  # the credential's user name; its password is each proxy's own
MAX_HEAD = 16 * 1024  # a request line and its headers
MAX_BYTES = 200_000_000  # every byte of one browser session, both ways
REFUSED_KEPT = (
    64  # hosts and ports last refused, so a page refused at its own address is told apart
)
MAX_CONNECTIONS = 48  # at once; Chrome opens about six per site
# Name lookups at once, those the system never answered (each outlives its caller) included.
MAX_LOOKUPS = 16
HEAD_TIMEOUT_S = 10.0  # for a whole request head, however it trickles in
CONNECT_TIMEOUT_S = 5.0
IDLE_TIMEOUT_S = 60.0  # a tunnel with no byte either way for this long is closed
_CHUNK = 64 * 1024
_REQUEST_LINE_WORDS = 3  # method, target, version
_HOP_HEADERS = frozenset(
    {b"connection", b"keep-alive", b"proxy-connection", b"proxy-authorization", b"te", b"upgrade"}
)

type Connector = Callable[[str, int, float], socket.socket]
# A CONNECT target: a host name or an IP literal, and a port, nothing else (no path or query).
_AUTHORITY = re.compile(r"(\[[0-9A-Fa-f:.]+\]|[^\s/?#@\[\]:]+):([0-9]{1,5})")
_MAX_PORT = 65535


class RefusedError(Exception):
    """A request the proxy does not pass on; the message is the reason, for the refusal line."""


def _connect(address: str, port: int, timeout: float) -> socket.socket:
    """Open a TCP connection to a checked address."""
    return socket.create_connection((address, port), timeout=timeout)


def _split_target(target: str, default_port: int) -> tuple[str, int]:
    """Return the host and port of an absolute address's `host[:port]` or `[v6][:port]`.

    Raises:
        RefusedError: not a host and port, or port 0.
    """
    try:
        parts = urlsplit(f"//{target}")
        port = parts.port if parts.port is not None else default_port
    except ValueError as exc:
        msg = "not a host and port"
        raise RefusedError(msg) from exc
    if not parts.hostname or parts.username is not None or parts.password is not None:
        msg = "not a host and port"
        raise RefusedError(msg)
    if port == 0 or parts.path or parts.query or parts.fragment:
        msg = "not a host and port"
        raise RefusedError(msg)
    return parts.hostname, port


def _authority(target: str) -> tuple[str, int]:
    """Return the host and port of a CONNECT target, which must name both and nothing else.

    Raises:
        RefusedError: no port, port 0 or past 65535, or anything but a host and port.
    """
    found = _AUTHORITY.fullmatch(target)
    if found is None or not 0 < int(found.group(2)) <= _MAX_PORT:
        msg = "not a host and port"
        raise RefusedError(msg)
    return _split_target(target, 0)


def _head(sock: socket.socket) -> tuple[bytes, bytes]:
    """Read a request's head (up to its blank line) and whatever arrived after it.

    The whole head must arrive within `HEAD_TIMEOUT_S`, however slowly it trickles in.

    Raises:
        RefusedError: the head is too long, took too long, or the connection closed first.
    """
    deadline = time.monotonic() + HEAD_TIMEOUT_S
    data = b""
    while b"\r\n\r\n" not in data:
        if len(data) > MAX_HEAD:
            msg = "the request head is too long"
            raise RefusedError(msg)
        left = deadline - time.monotonic()
        if left <= 0:
            msg = "the request head took too long"
            raise RefusedError(msg)
        sock.settimeout(left)
        try:
            chunk = sock.recv(_CHUNK)
        except TimeoutError as exc:
            msg = "the request head took too long"
            raise RefusedError(msg) from exc
        if not chunk:
            msg = "the connection closed before the request"
            raise RefusedError(msg)
        data += chunk
    head, _, rest = data.partition(b"\r\n\r\n")
    return head, rest


def _forwarded_head(lines: list[bytes], method: bytes, path: str, version: bytes) -> bytes:
    """Return a plain-HTTP request head for the site: origin-form, no proxy or hop headers."""
    out = [b" ".join((method, path.encode("ascii", "strict"), version))]
    for line in lines[1:]:
        name = line.split(b":", 1)[0].strip().lower()
        if name not in _HOP_HEADERS:
            out.append(line)
    out.append(b"Connection: close")
    return b"\r\n".join(out) + b"\r\n\r\n"


class _Budget:
    """The bytes the session may still move, shared by every connection."""

    def __init__(self, total: int) -> None:
        super().__init__()
        self._left = total
        self._lock = threading.Lock()

    def take(self, count: int) -> bool:
        """Spend `count` bytes; False once the budget is used up."""
        with self._lock:
            self._left -= count
            return self._left >= 0


class _Server(socketserver.ThreadingTCPServer):
    """Takes a connection slot before a thread is made for a connection; past the cap, none."""

    daemon_threads = True

    def __init__(
        self,
        handler: type[socketserver.BaseRequestHandler],
        slots: threading.BoundedSemaphore,
        on_full: Callable[[socket.socket], None],
    ) -> None:
        self.slots = slots
        self._on_full = on_full
        super().__init__((LOOPBACK, 0), handler)

    @override
    def process_request(
        self, request: socket.socket | tuple[bytes, socket.socket], client_address: object
    ) -> None:
        if not self.slots.acquire(blocking=False):
            if isinstance(request, socket.socket):
                self._on_full(request)
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self.slots.release()
            raise

    @override
    def process_request_thread(
        self, request: socket.socket | tuple[bytes, socket.socket], client_address: object
    ) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.slots.release()


class EgressProxy:
    """An HTTP proxy on 127.0.0.1 that reaches only public addresses on ports 80 and 443.

    With `local_port` it is a preview's proxy instead: it reaches `127.0.0.1` on that one port (a
    server Jarvis started over the owner's own project) and nothing else, not the internet either.
    """

    def __init__(
        self,
        *,
        resolve: Resolver | None = None,
        connect: Connector = _connect,
        max_bytes: int = MAX_BYTES,
        max_connections: int = MAX_CONNECTIONS,
        local_port: int | None = None,
    ) -> None:
        """Resolve with `resolve` and connect with `connect` (both real by default; for tests)."""
        super().__init__()
        self._local_port = local_port
        self._resolve = resolve
        self._connect = connect
        self._budget = _Budget(max_bytes)
        self._slots = threading.BoundedSemaphore(max_connections)
        self._max_connections = max_connections
        self._lookups = threading.BoundedSemaphore(MAX_LOOKUPS)
        self._server: socketserver.ThreadingTCPServer | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._refused = 0
        self._refused_hosts: deque[tuple[str, int]] = deque(maxlen=REFUSED_KEPT)
        self._stopped = False
        self._open_sockets: set[socket.socket] = set()  # both ends of every connection served
        self._clients: set[socket.socket] = set()  # the browser's ends of them
        self._password = secrets.token_urlsafe(24)
        token = base64.b64encode(f"{USER}:{self._password}".encode()).decode("ascii")
        self.authorization = f"Basic {token}"  # the browser's `Proxy-Authorization`

    def url(self, port: int) -> str:
        """Return the proxy's address with its credential, for the browser (never put in argv)."""
        return f"http://{USER}:{self._password}@{LOOPBACK}:{port}"

    @property
    def connections(self) -> int:
        """How many connections are being served now."""
        with self._lock:
            return len(self._open_sockets & self._clients)

    @property
    def refused(self) -> int:
        """How many connections were refused (a private address, a port, a malformed request)."""
        with self._lock:
            return self._refused

    def was_refused(self, host: str, port: int) -> bool:
        """Return whether a connection to `host` on `port` was refused lately."""
        with self._lock:
            return (host.lower(), port) in self._refused_hosts

    @property
    def port(self) -> int:
        """The port the proxy listens on, on 127.0.0.1.

        Raises:
            RuntimeError: it is not started.
        """
        if self._server is None:
            msg = "the proxy is not started"
            raise RuntimeError(msg)
        return int(self._server.server_address[1])

    def start(self) -> int:
        """Listen on a free port of 127.0.0.1 and return it."""
        proxy = self

        class Handler(socketserver.BaseRequestHandler):
            @override
            def handle(self) -> None:
                proxy.serve(self.request)

        server = _Server(Handler, self._slots, _turn_away)
        thread = threading.Thread(
            target=server.serve_forever, args=(0.1,), name="jarvis-proxy", daemon=True
        )
        with self._lock:
            self._stopped = False
        self._server, self._thread = server, thread
        thread.start()
        return self.port

    def stop(self) -> None:
        """Stop listening and end every connection still open, tunnels included.

        A browser whose own close failed is then left with no way out, rather than keeping its
        tunnels until it goes.
        """
        server, thread = self._server, self._thread
        self._server = self._thread = None
        if server is not None:
            server.shutdown()
            server.server_close()
        with self._lock:
            self._stopped = True
            still_open = list(self._open_sockets)
        for sock in still_open:
            # Its serving thread is woken by the end of the stream, and closes it itself.
            with contextlib.suppress(OSError):
                sock.shutdown(socket.SHUT_RDWR)
        if thread is not None:
            thread.join(timeout=5.0)

    def _track(self, sock: socket.socket) -> bool:
        """Note a connection as open, so `stop` can end it; False once the proxy is stopped."""
        with self._lock:
            if self._stopped:
                return False
            self._open_sockets.add(sock)
            return True

    def _untrack(self, sock: socket.socket) -> None:
        with self._lock:
            self._open_sockets.discard(sock)
            self._clients.discard(sock)

    def serve(self, client: socket.socket) -> None:
        """Handle one connection from the browser (its slot already taken): check, then pass on."""
        try:
            if self._track(client):
                with self._lock:
                    self._clients.add(client)
                self._serve(client)
        except OSError:
            pass  # either side went away; nothing is left to tell
        finally:
            self._untrack(client)
            with contextlib.suppress(OSError):
                client.close()

    def _serve(self, client: socket.socket) -> None:
        """Read and check the request, then pass it on; a refused one is answered `403`.

        An `OSError` from either side comes through.
        """
        client.settimeout(HEAD_TIMEOUT_S)
        try:
            head, rest = _head(client)
        except RefusedError as exc:
            _reply(client, b"403 Forbidden", str(exc))
            return
        if not self._signed(head):
            _reply(client, b"407 Proxy Authentication Required", headers=_CHALLENGE)
            return
        try:
            upstream, first = self._open(head, rest)
        except (RefusedError, BlockedDestinationError) as exc:
            with self._lock:
                self._refused += 1
            _reply(client, b"403 Forbidden", str(exc))
            return
        except OSError:
            _reply(client, b"502 Bad Gateway")
            return
        with upstream:
            if not self._track(upstream):
                return
            try:
                self._pass_on(client, upstream, first, rest)
            finally:
                self._untrack(upstream)

    def _signed(self, head: bytes) -> bool:
        """Return whether a request head carries this proxy's credential (in constant time)."""
        expected = self.authorization.encode("ascii")
        signed = False
        for line in head.split(b"\r\n")[1:]:
            name, _, value = line.partition(b":")
            if name.strip().lower() == b"proxy-authorization":
                signed |= hmac.compare_digest(value.strip(), expected)
        return signed

    def _pass_on(
        self, client: socket.socket, upstream: socket.socket, first: bytes | None, rest: bytes
    ) -> None:
        """Answer a tunnel `200` (or send a plain request on), then relay both ways."""
        if first is None:
            client.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            first = rest
        if first and self._budget.take(len(first)):
            upstream.sendall(first)
        self._relay(client, upstream)

    def _open(self, head: bytes, rest: bytes) -> tuple[socket.socket, bytes | None]:
        """Connect to the checked destination of a request head.

        Returns the connection and, for plain HTTP, the bytes to send first (the rewritten head
        and any body that came with it); None for a tunnel, which is answered `200` first.

        A `BlockedDestinationError` says the destination is not public or not on port 80 or 443.

        Raises:
            RefusedError: a malformed request, another method or scheme, or no address answered.
        """
        lines = head.split(b"\r\n")
        words = lines[0].split(b" ")
        if len(words) != _REQUEST_LINE_WORDS or not words[2].startswith(b"HTTP/1."):
            msg = "not an HTTP/1 request"
            raise RefusedError(msg)
        method, target, version = words
        try:
            text = target.decode("ascii", "strict")
        except UnicodeDecodeError as exc:
            msg = "the address is not ASCII"
            raise RefusedError(msg) from exc
        if method == b"CONNECT":
            host, port = _authority(text)
            return self._reach(host, port), None
        try:
            parts = urlsplit(text)
        except ValueError as exc:  # an unclosed `[`, as in `http://[::1`
            msg = "not an address"
            raise RefusedError(msg) from exc
        if parts.scheme != "http":
            msg = "only http addresses and CONNECT tunnels are passed on"
            raise RefusedError(msg)
        host, port = _split_target(parts.netloc, 80)
        path = parts.path or "/"
        if parts.query:
            path += f"?{parts.query}"
        upstream = self._reach(host, port)
        return upstream, _forwarded_head(lines, method, path, version) + rest

    def _reach(self, host: str, port: int) -> socket.socket:
        """Connect to the first checked public address of `host` that answers.

        Raises:
            BlockedDestinationError: the host or port is not allowed.
            RefusedError: no address answered.
        """
        try:
            addresses = self._allowed(host, port)
        except BlockedDestinationError:
            with self._lock:
                self._refused_hosts.append((host.lower(), port))
            raise
        for address in addresses:
            if self._is_stopped():  # a lookup that outlived `stop` dials nothing
                break
            try:
                return self._connect(address, port, CONNECT_TIMEOUT_S)
            except OSError:
                continue
        msg = f"{host!r} did not answer"
        raise RefusedError(msg)

    def _is_stopped(self) -> bool:
        with self._lock:
            return self._stopped

    def _lookup(self, host: str, port: int, **options: Any) -> list[tuple[Any, ...]]:  # ruff: ignore[any-type] - getaddrinfo's own options
        """Resolve as `resolve` (or the system) does, holding one of `MAX_LOOKUPS` until it ends.

        Raises:
            OSError: as many lookups are running already (it then counts as not resolving).
        """
        if not self._lookups.acquire(blocking=False):
            msg = "too many name lookups at once"
            raise OSError(msg)
        try:
            return (self._resolve or socket.getaddrinfo)(host, port, **options)
        finally:
            self._lookups.release()

    def _allowed(self, host: str, port: int) -> list[str]:
        """Return the addresses `host` may be reached at: public ones, or a preview's one port.

        Raises:
            BlockedDestinationError: anything else.
        """
        if self._local_port is None:
            return public_addresses(host, port, self._lookup)
        if host == LOOPBACK and port == self._local_port:
            return [LOOPBACK]
        msg = f"a preview reaches only {LOOPBACK}:{self._local_port}"
        raise BlockedDestinationError(msg)

    def _relay(self, client: socket.socket, upstream: socket.socket) -> None:
        """Move bytes both ways until either side closes, goes quiet or the budget runs out."""
        client.setblocking(False)  # ruff: ignore[boolean-positional-value-in-call] - socket's own API
        upstream.setblocking(False)  # ruff: ignore[boolean-positional-value-in-call] - socket's own API
        peer = {client: upstream, upstream: client}
        with selectors.DefaultSelector() as selector:
            selector.register(client, selectors.EVENT_READ)
            selector.register(upstream, selectors.EVENT_READ)
            while True:
                ready = selector.select(IDLE_TIMEOUT_S)
                if not ready:
                    return
                for key, _events in ready:
                    source = key.fileobj
                    if not isinstance(source, socket.socket):
                        return
                    try:
                        data = source.recv(_CHUNK)
                    except BlockingIOError:
                        continue
                    if not data or not self._budget.take(len(data)):
                        return
                    _send_all(peer[source], data)


def _send_all(sock: socket.socket, data: bytes) -> None:
    """Send all of `data` on a non-blocking socket, waiting for room up to the idle timeout.

    Raises:
        TimeoutError: the other side took nothing for `IDLE_TIMEOUT_S`.
    """
    view = memoryview(data)
    with selectors.DefaultSelector() as selector:
        selector.register(sock, selectors.EVENT_WRITE)
        while view:
            try:
                sent = sock.send(view)
            except BlockingIOError:
                if not selector.select(IDLE_TIMEOUT_S):
                    msg = "the other side stopped reading"
                    raise TimeoutError(msg) from None
                continue
            view = view[sent:]


_CHALLENGE = (b'Proxy-Authenticate: Basic realm="jarvis"',)


def _turn_away(client: socket.socket) -> None:
    """Answer a connection past the cap, on the listening thread: never wait on it."""
    with contextlib.suppress(OSError):
        client.settimeout(0.5)
    _reply(client, b"503 Service Unavailable")


def _reply(
    client: socket.socket, status: bytes, reason: str = "", headers: tuple[bytes, ...] = ()
) -> None:
    """Answer the browser with a status and close; the reason is for anyone reading the log."""
    body = reason.encode("utf-8", "replace")[:200]
    head = b"HTTP/1.1 " + status + b"\r\nConnection: close\r\nContent-Type: text/plain\r\n"
    head += b"".join(header + b"\r\n" for header in headers)
    with contextlib.suppress(OSError):
        client.sendall(head + b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
