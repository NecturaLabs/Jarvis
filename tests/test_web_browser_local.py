"""The real headless browser cannot reach this PC: run with `uv run pytest -m local` on a PC with
agent-browser. It starts Chrome headless and muted, out of sight, and closes it.

The page is served by the test on loopback and reached through the egress proxy under a made-up
public name, as a public site would be. Its scripts, frames, images and a redirect then try to
reach the same server on IPv4 and IPv6 loopback, by name, a cloud metadata address and the home
network directly. Each try must reach the proxy and be refused there (so it did run), and the
servers must see none of it; one fetch to the made-up public name must arrive (so the page's
scripts ran at all).
"""

from __future__ import annotations

import contextlib
import http.server
import os
import shutil
import socket
import tempfile
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast, override
from urllib.parse import urlsplit

import pytest

from jarvis.config import FilesConfig
from jarvis.pc.files import Boundary
from jarvis.web.browser import Browser, BrowserError
from jarvis.web.preview import FolderServer
from jarvis.web.proxy import EgressProxy

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = [
    pytest.mark.local,
    pytest.mark.skipif(shutil.which("agent-browser") is None, reason="agent-browser is missing"),
]

PUBLIC = "93.184.215.14"
NAME = "probe.example"
PAYLOAD = b"jarvis download probe " * 64
# Unique to this test, so a file of that name anywhere can only be its download.
SAVED_NAME = "jarvis-download-probe-5b1f0c.bin"
DOWNLOAD_PAGE = (
    b"<html><body><h1>Download</h1>"
    b'<a id=d href="/probe.bin" download="' + SAVED_NAME.encode() + b'">get</a>'
    b"<script>document.getElementById('d').click();</script></body></html>"
)


METADATA = "169.254.169.254"
HOME_NETWORK = "192.168.77.1"


def _page(port: int, port6: int) -> bytes:
    here = f"127.0.0.1:{port}"
    return f"""<html><body><h1>Probe</h1><p id=r>waiting</p>
<iframe src="http://{here}/leak-iframe"></iframe>
<img src="http://{here}/leak-img"><img src="http://[::1]:{port6}/leak-img6">
<script>
const out = [];
function note(x) {{ out.push(x); document.getElementById('r').textContent = out.join(' | '); }}
fetch('/ok-fetch').then(() => note('own site reached')).catch(() => note('own site blocked'));
fetch('http://{here}/leak-fetch').then(() => note('fetch reached'))
  .catch(() => note('fetch blocked'));
fetch('http://[::1]:{port6}/leak-fetch6').then(() => note('v6 reached'))
  .catch(() => note('v6 blocked'));
fetch('http://localhost:{port}/leak-name').then(() => note('name reached'))
  .catch(() => note('name blocked'));
fetch('http://{METADATA}/latest/meta-data/').then(() => note('metadata reached'))
  .catch(() => note('metadata blocked'));
fetch('http://{HOME_NETWORK}/').then(() => note('home reached'))
  .catch(() => note('home blocked'));
try {{
  const ws = new WebSocket('ws://{here}/leak-ws');
  ws.onopen = () => note('ws reached'); ws.onerror = () => note('ws blocked');
}} catch (e) {{ note('ws threw'); }}
try {{
  const pc = new RTCPeerConnection({{iceServers: []}});
  pc.createDataChannel('x');
  pc.onicecandidate = e => {{ if (e.candidate) note('candidate ' + e.candidate.candidate); }};
  pc.createOffer().then(o => pc.setLocalDescription(o));
}} catch (e) {{ note('rtc threw'); }}
</script></body></html>""".encode()


class _Server6(http.server.ThreadingHTTPServer):
    address_family = socket.AF_INET6


class Server:
    """A loopback web server, on IPv4 and IPv6, that records every path that reaches it."""

    def __init__(self) -> None:
        super().__init__()
        hits: list[str] = []
        self.hits = hits
        page = _page
        ports: list[int] = []

        class Handler(http.server.BaseHTTPRequestHandler):
            @override
            def log_message(self, format: str, *args: Any) -> None:
                return

            def do_GET(self) -> None:
                hits.append(self.path)
                port = int(cast("tuple[str, int]", self.server.server_address)[1])
                if self.path == "/":
                    body = page(port, ports[1])
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                elif self.path == "/download-page":
                    body = DOWNLOAD_PAGE
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                elif self.path in {"/probe.bin", "/attachment"}:
                    body = PAYLOAD
                    self.send_response(200)
                    self.send_header("Content-Type", "application/octet-stream")
                    self.send_header("Content-Disposition", f'attachment; filename="{SAVED_NAME}"')
                elif self.path == "/redirect":
                    self.send_response(302)
                    self.send_header("Location", f"http://127.0.0.1:{port}/leak-redirect")
                    body = b""
                else:
                    self.send_response(200)
                    body = b"leaked"
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.http = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.http6 = _Server6(("::1", 0), Handler)
        self.port = int(self.http.server_address[1])
        self.port6 = int(self.http6.server_address[1])
        ports.extend((self.port, self.port6))
        for made in (self.http, self.http6):
            made.daemon_threads = True
            threading.Thread(target=made.serve_forever, args=(0.1,), daemon=True).start()

    def close(self) -> None:
        for made in (self.http, self.http6):
            made.shutdown()
            made.server_close()


@pytest.fixture
def root() -> Iterator[Path]:
    """A short folder for the sessions: a socket's path may be at most 103 bytes, and pytest's
    own folders are longer than Jarvis's runtime folder."""
    with tempfile.TemporaryDirectory(prefix="jb-") as made:
        fd = os.open(made, os.O_RDONLY | os.O_DIRECTORY)
        try:
            # The scratch directory can have a long path. Use a short name for the same
            # directory so agent-browser's session socket fits in a sockaddr_un.
            yield Path(f"/proc/{os.getpid()}/fd/{fd}") / "browser"
        finally:
            os.close(fd)


@pytest.fixture
def server() -> Iterator[Server]:
    made = Server()
    yield made
    made.close()


def _proxy(server: Server, made: list[EgressProxy] | None = None) -> EgressProxy:
    def resolve(host: str, port: int, **_kw: object) -> list[tuple[Any, ...]]:
        if host == NAME:
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (PUBLIC, port))]
        return socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)

    def connect(address: str, port: int, timeout: float) -> socket.socket:
        if address == PUBLIC:  # the made-up public site is the test's server
            return socket.create_connection(("127.0.0.1", server.port), timeout=timeout)
        return socket.create_connection((address, port), timeout=timeout)

    proxy = EgressProxy(resolve=resolve, connect=connect)
    if made is not None:
        made.append(proxy)
    return proxy


def _session_commands(folder: Path) -> list[bytes]:
    """Return argv for this session's daemon, Chrome, and their descendants."""
    roots = {int(path.read_text(encoding="ascii")) for path in (folder / "s").glob("*.pid")}
    assert roots, "agent-browser did not record a daemon for this test"
    children: dict[int, list[int]] = {}
    for path in Path("/proc").glob("[0-9]*/stat"):
        with contextlib.suppress(FileNotFoundError, ProcessLookupError, PermissionError):
            parent = int(path.read_text(encoding="utf-8").rsplit(")", 1)[1].split()[1])
            children.setdefault(parent, []).append(int(path.parent.name))
    pids = list(roots)
    commands: list[bytes] = []
    while pids:
        pid = pids.pop()
        pids.extend(children.get(pid, ()))
        with contextlib.suppress(FileNotFoundError, ProcessLookupError):
            commands.append(Path(f"/proc/{pid}/cmdline").read_bytes())
    return commands


def test_the_proxy_password_stays_out_of_the_session_process_arguments(
    root: Path,
    server: Server,
) -> None:
    program = shutil.which("agent-browser")
    assert program is not None
    proxy = _proxy(server)
    browser = Browser(program, root, idle_close_s=30, proxy=lambda: proxy)
    try:
        view = browser.open(f"http://{NAME}/download-page")
        (link,) = (element for element in view.elements if element.role == "link")
        assert view.goes_to(link.ref) == f"http://{NAME}/probe.bin"
        password = urlsplit(proxy.url(proxy.port)).password
        assert password
        (folder,) = root.iterdir()
        commands = _session_commands(folder)
        assert any(b"agent-browser" in command for command in commands)
        assert any(b"--user-data-dir=" in command for command in commands), "Chrome was not scanned"
        if any(password.encode() in command for command in commands):
            pytest.fail("the session's proxy password was present in process arguments")
    finally:
        browser.close()


def test_page_scripts_frames_and_redirects_reach_nothing_on_this_pc(
    root: Path, server: Server
) -> None:
    program = shutil.which("agent-browser")
    assert program is not None
    proxies: list[EgressProxy] = []
    browser = Browser(program, root, idle_close_s=30, proxy=lambda: _proxy(server, proxies))
    try:
        browser.open(f"http://{NAME}/")
        time.sleep(4)  # the page's scripts try their connections
        view = browser.open(f"http://{NAME}/")
        text = view.text
        with pytest.raises(BrowserError):
            browser.open(f"http://{NAME}/redirect")
        with pytest.raises(BrowserError):
            browser.open("http://127.0.0.1/")
    finally:
        browser.close()

    (proxy,) = proxies
    assert "/ok-fetch" in server.hits  # the page's scripts ran, through the proxy
    assert "own site reached" in text
    for host, port in [
        ("127.0.0.1", server.port),
        ("::1", server.port6),
        ("localhost", server.port),
        (METADATA, 80),
        (HOME_NETWORK, 80),
    ]:
        assert proxy.was_refused(host, port), f"{host}:{port} was never tried, or not refused"
    assert [hit for hit in server.hits if hit.startswith("/leak")] == []
    for leak in ("fetch", "v6", "name", "metadata", "home", "ws"):
        assert f"{leak} reached" not in text
    assert "candidate" not in text
    assert list(root.iterdir()) == []


def test_a_preview_reaches_its_own_project_and_nothing_else(
    tmp_path: Path, root: Path, server: Server
) -> None:
    program = shutil.which("agent-browser")
    assert program is not None
    site = tmp_path / "me" / "Documents" / "site"
    site.mkdir(parents=True)
    leak = f"http://127.0.0.1:{server.port}/leak-preview"
    (site / "index.html").write_text(
        f"""<html><body><h1>Built</h1><p id=r>waiting</p><script>
const r = document.getElementById('r');
fetch('{leak}').then(() => r.textContent = 'reached').catch(() => r.textContent = 'blocked');
fetch('https://example.com/').then(() => r.textContent += ' internet')
  .catch(() => r.textContent += ' offline');
</script></body></html>""",
        encoding="utf-8",
    )
    folder = FolderServer(
        site, Boundary.of(FilesConfig(write_roots=["~/Documents"]), tmp_path / "me")
    )
    origin = folder.start()
    proxies: list[EgressProxy] = []

    def preview_proxy() -> EgressProxy:
        made = EgressProxy(local_port=folder.port, local_header=(folder.header, folder.token))
        proxies.append(made)
        return made

    browser = Browser(
        program,
        root,
        idle_close_s=30,
        proxy=preview_proxy,
        origin=origin,
        on_close=folder.stop,
    )
    try:
        browser.open(f"{origin}/")
        time.sleep(4)  # the page's scripts try their connections
        view = browser.open(f"{origin}/")
    finally:
        browser.close()

    (proxy,) = proxies
    assert "Built" in view.text
    # Refused at the proxy, not merely stopped by the page's own CORS rules.
    assert proxy.was_refused("127.0.0.1", server.port)
    assert proxy.was_refused("example.com", 443)
    assert [hit for hit in server.hits if hit.startswith("/leak")] == []
    assert "reached" not in view.text
    assert "internet" not in view.text
    assert list(root.iterdir()) == []


def test_a_download_a_page_starts_saves_nothing(root: Path, server: Server) -> None:
    program = shutil.which("agent-browser")
    assert program is not None
    browser = Browser(program, root, idle_close_s=30, proxy=lambda: _proxy(server))
    try:
        browser.open(f"http://{NAME}/download-page")
        with contextlib.suppress(BrowserError):  # a download opens no page
            browser.open(f"http://{NAME}/attachment")
        time.sleep(3)  # the downloads run, or fail
        saved = [
            path.name
            for path in root.rglob("*")
            if SAVED_NAME in path.name or path.suffix == ".crdownload"
        ]
        downloads = [path for folder in root.glob("*/downloads") for path in folder.iterdir()]
    finally:
        browser.close()

    assert {"/probe.bin", "/attachment"} <= set(server.hits)  # both downloads were tried
    assert saved == []
    assert downloads == []
    assert not (Path.home() / "Downloads" / SAVED_NAME).exists()
    assert list(root.iterdir()) == []
