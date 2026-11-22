"""An HTTP client that reaches only public internet addresses, checked at every connection.

Which page Jarvis opens is decided by text from outside (a search result), so a URL must never
reach this PC, the home network or a cloud metadata address. Before a connection is opened the
host name is resolved here, every address it resolves to must be a public one, and the
connection goes to the address that was checked: a name cannot pass the check with a public
address and then connect to a private one. Only ports 80 and 443 are reached. Each redirect is a
new request (the client never follows them itself), checked the same way. TLS still verifies the
certificate against the host name in the URL.
"""

import ipaddress
import socket
import threading
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, cast, override

import httpcore
import httpx

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable, Iterator

ALLOWED_PORTS = frozenset({80, 443})
DNS_TIMEOUT_S = 5.0
MAX_ANSWER_BYTES = 1_000_000
# The whole of one answer from a fixed service, however slowly it arrives (`TIMEOUT` is per read).
ANSWER_LIMIT_S = 15.0
_NAT64 = ipaddress.ip_network("64:ff9b::/96")
_IPV4_COMPATIBLE = ipaddress.ip_network("::/96")
_LOW_32_BITS = 0xFFFFFFFF
# Generic on purpose: no request names the owner, this machine or the project.
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) voice-assistant"
TIMEOUT = httpx.Timeout(10.0, connect=5.0)

type Resolver = Callable[..., list[tuple[Any, ...]]]

# httpcore's errors as the httpx errors of the same meaning, most specific first.
_ERRORS: tuple[tuple[type[Exception], type[httpx.TransportError]], ...] = (
    (httpcore.ConnectTimeout, httpx.ConnectTimeout),
    (httpcore.ReadTimeout, httpx.ReadTimeout),
    (httpcore.WriteTimeout, httpx.WriteTimeout),
    (httpcore.PoolTimeout, httpx.PoolTimeout),
    (httpcore.ConnectError, httpx.ConnectError),
    (httpcore.ReadError, httpx.ReadError),
    (httpcore.WriteError, httpx.WriteError),
    (httpcore.RemoteProtocolError, httpx.RemoteProtocolError),
    (httpcore.LocalProtocolError, httpx.LocalProtocolError),
    (httpcore.UnsupportedProtocol, httpx.UnsupportedProtocol),
    (httpcore.ProtocolError, httpx.ProtocolError),
    (httpcore.NetworkError, httpx.NetworkError),
    (httpcore.TimeoutException, httpx.TimeoutException),
)


class BlockedDestinationError(httpcore.ConnectError):
    """The destination is not a public internet address on an allowed port."""


def is_public(address: str) -> bool:
    """Return whether an IP address is a public (global unicast) one.

    An IPv4 address carried in IPv6 is judged as that IPv4 address: mapped (`::ffff:10.0.0.1`,
    which Python does itself) and NAT64 (`64:ff9b::a00:1`). The deprecated IPv4-compatible form
    (`::10.0.0.1`) is refused outright.
    """
    try:
        ip = ipaddress.ip_address(address.split("%", 1)[0])
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address):
        if ip in _IPV4_COMPATIBLE:
            return False
        if ip in _NAT64:
            return is_public(str(ipaddress.IPv4Address(int(ip) & _LOW_32_BITS)))
    return ip.is_global and not ip.is_multicast


def _lookup_within(lookup: Resolver, host: str, port: int) -> list[tuple[Any, ...]]:
    """Resolve on a helper thread and give up after `DNS_TIMEOUT_S`: the system call has no timeout.

    Raises:
        BlockedDestinationError: the name took too long to resolve.
    """
    answer: list[list[tuple[Any, ...]]] = []
    failure: list[BaseException] = []

    def work() -> None:
        try:
            answer.append(lookup(host, port, type=socket.SOCK_STREAM))
        except BaseException as exc:  # ruff: ignore[blind-except] - re-raised on the caller's thread
            failure.append(exc)

    worker = threading.Thread(target=work, name="jarvis-dns", daemon=True)
    worker.start()
    worker.join(DNS_TIMEOUT_S)
    if worker.is_alive():
        msg = f"{host!r} took too long to resolve"
        raise BlockedDestinationError(msg)
    if failure:
        raise failure[0]
    return answer[0]


def public_addresses(host: str, port: int, resolve: Resolver | None = None) -> list[str]:
    """Return the addresses `host` resolves to, when every one of them is public.

    Raises:
        BlockedDestinationError: the port is not 80 or 443, the name does not resolve, or any
            address it resolves to is not public (one private answer is enough to refuse).
    """
    if port not in ALLOWED_PORTS:
        msg = f"port {port} is not allowed (only 80 and 443)"
        raise BlockedDestinationError(msg)
    lookup = resolve or socket.getaddrinfo
    try:
        infos = _lookup_within(lookup, host, port)
    except (OSError, UnicodeError) as exc:
        msg = f"{host!r} does not resolve"
        raise BlockedDestinationError(msg) from exc
    addresses = list(dict.fromkeys(str(info[4][0]) for info in infos))
    if not addresses or not all(is_public(a) for a in addresses):
        msg = f"{host!r} is not a public internet address"
        raise BlockedDestinationError(msg)
    return addresses


class PublicOnlyBackend(httpcore.NetworkBackend):
    """Opens TCP connections only to checked public addresses; never a Unix socket."""

    def __init__(
        self, inner: httpcore.NetworkBackend | None = None, resolve: Resolver | None = None
    ) -> None:
        """Connect through `inner` (the real sockets by default); `resolve` is for tests."""
        super().__init__()
        self._inner = inner or httpcore.SyncBackend()
        self._resolve = resolve

    @override
    def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Iterable[Any] | None = None,
    ) -> httpcore.NetworkStream:
        """Resolve and check `host`, then connect to the first checked address that answers."""
        failure: httpcore.ConnectError | None = None
        for address in public_addresses(host, port, self._resolve):
            try:
                return self._inner.connect_tcp(
                    address,
                    port,
                    timeout=timeout,
                    local_address=local_address,
                    socket_options=socket_options,
                )
            except httpcore.ConnectError as exc:
                failure = exc
        raise failure or httpcore.ConnectError(f"could not connect to {host!r}")

    @override
    def connect_unix_socket(
        self,
        path: str,
        timeout: float | None = None,
        socket_options: Iterable[Any] | None = None,
    ) -> httpcore.NetworkStream:
        """Refuse: a web request never reaches a local socket.

        Raises:
            BlockedDestinationError: always.
        """
        msg = "a Unix socket is not a public internet address"
        raise BlockedDestinationError(msg)

    @override
    def sleep(self, seconds: float) -> None:
        """Sleep as the inner backend does."""
        self._inner.sleep(seconds)


@contextmanager
def _as_httpx_errors(request: httpx.Request) -> Generator[None]:
    try:
        yield
    except BlockedDestinationError as exc:
        raise httpx.ConnectError(str(exc), request=request) from exc
    except Exception as exc:
        for theirs, ours in _ERRORS:
            if isinstance(exc, theirs):
                raise ours(str(exc), request=request) from exc
        raise


class _Body(httpx.SyncByteStream):
    def __init__(self, stream: Iterable[bytes], request: httpx.Request) -> None:
        super().__init__()
        self._stream = stream
        self._request = request

    @override
    def __iter__(self) -> Iterator[bytes]:
        with _as_httpx_errors(self._request):
            yield from self._stream

    @override
    def close(self) -> None:
        close = getattr(self._stream, "close", None)
        if callable(close):
            close()


class PublicOnlyTransport(httpx.BaseTransport):
    """An httpx transport over a connection pool whose every connection is checked."""

    def __init__(self, backend: httpcore.NetworkBackend | None = None) -> None:
        """Use `backend` for connections (a `PublicOnlyBackend` over real sockets by default)."""
        super().__init__()
        self._pool = httpcore.ConnectionPool(
            ssl_context=httpx.create_ssl_context(trust_env=False),
            max_connections=8,
            max_keepalive_connections=4,
            keepalive_expiry=30.0,
            network_backend=backend or PublicOnlyBackend(),
        )

    @override
    def handle_request(self, request: httpx.Request) -> httpx.Response:
        """Send one request (no redirects are followed here).

        Raises:
            UnsupportedProtocol: the URL is not http or https.
        """
        if request.url.scheme not in {"http", "https"}:
            msg = f"the {request.url.scheme!r} scheme is not allowed"
            raise httpx.UnsupportedProtocol(msg, request=request)
        core = httpcore.Request(
            method=request.method,
            url=httpcore.URL(
                scheme=request.url.raw_scheme,
                host=request.url.raw_host,
                port=request.url.port,
                target=request.url.raw_path,
            ),
            headers=request.headers.raw,
            content=request.stream,
            extensions=request.extensions,
        )
        with _as_httpx_errors(request):
            response = self._pool.handle_request(core)
        return httpx.Response(
            status_code=response.status,
            headers=response.headers,
            stream=_Body(cast("Iterable[bytes]", response.stream), request),
        )

    @override
    def close(self) -> None:
        """Close every pooled connection."""
        self._pool.close()


def read_capped(
    response: httpx.Response,
    limit: int = MAX_ANSWER_BYTES,
    *,
    clock: Callable[[], float] = time.monotonic,
) -> bytes:
    """Return an answer's body, reading no more than `limit` decoded bytes, in `ANSWER_LIMIT_S`.

    Raises:
        ValueError: the body is larger than `limit`.
        httpx.ReadTimeout: the whole body did not arrive in time.
    """
    deadline = clock() + ANSWER_LIMIT_S
    chunks: list[bytes] = []
    size = 0
    for chunk in response.iter_bytes():
        size += len(chunk)
        if size > limit:
            msg = "the answer is too large"
            raise ValueError(msg)
        chunks.append(chunk)
        if clock() > deadline:
            msg = "the answer took too long to arrive"
            raise httpx.ReadTimeout(msg)
    return b"".join(chunks)


def public_client(backend: httpcore.NetworkBackend | None = None) -> httpx.Client:
    """Return a client that reaches only public addresses and never follows a redirect itself."""
    return httpx.Client(
        transport=PublicOnlyTransport(backend),
        timeout=TIMEOUT,
        follow_redirects=False,
        trust_env=False,
        headers={"User-Agent": USER_AGENT},
    )
