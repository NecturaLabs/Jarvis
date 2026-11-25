"""The real headless browser cannot reach this PC: run with `uv run pytest -m local` on a PC with
agent-browser. It starts Chrome headless and muted, out of sight, and closes it.

The page is served by the test on loopback and reached through the egress proxy under a made-up
public name, as a public site would be. Its scripts, frames, images and a redirect then try to
reach the same loopback server, other loopback ports and the home network directly: the server
must see none of it.
"""

from __future__ import annotations

import http.server
import shutil
import socket
import threading
import time
from typing import TYPE_CHECKING, Any, cast, override

import pytest

from jarvis.config import FilesConfig
from jarvis.pc.files import Boundary
from jarvis.web.browser import Browser, BrowserError
from jarvis.web.preview import FolderServer, preview_browser
from jarvis.web.proxy import EgressProxy

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

pytestmark = [
    pytest.mark.local,
    pytest.mark.skipif(shutil.which("agent-browser") is None, reason="agent-browser is missing"),
]

PUBLIC = "93.184.215.14"
NAME = "probe.example"


def _page(port: int) -> bytes:
    here = f"127.0.0.1:{port}"
    return f"""<html><body><h1>Probe</h1><p id=r>waiting</p>
<iframe src="http://{here}/leak-iframe"></iframe>
<img src="http://{here}/leak-img"><img src="http://[::1]:{port}/leak-img6">
<script>
const out = [];
function note(x) {{ out.push(x); document.getElementById('r').textContent = out.join(' | '); }}
fetch('http://{here}/leak-fetch').then(() => note('fetch reached'))
  .catch(() => note('fetch blocked'));
fetch('http://localhost:{port}/leak-name').then(() => note('name reached'))
  .catch(() => note('name blocked'));
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


class Server:
    """A loopback web server that records every path that reaches it."""

    def __init__(self) -> None:
        super().__init__()
        hits: list[str] = []
        self.hits = hits
        page = _page

        class Handler(http.server.BaseHTTPRequestHandler):
            @override
            def log_message(self, format: str, *args: Any) -> None:
                return

            def do_GET(self) -> None:
                hits.append(self.path)
                port = int(cast("tuple[str, int]", self.server.server_address)[1])
                if self.path == "/":
                    body = page(port)
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
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
        self.http.daemon_threads = True
        self.port = int(self.http.server_address[1])
        threading.Thread(target=self.http.serve_forever, args=(0.1,), daemon=True).start()

    def close(self) -> None:
        self.http.shutdown()
        self.http.server_close()


@pytest.fixture
def server() -> Iterator[Server]:
    made = Server()
    yield made
    made.close()


def _proxy(server: Server) -> EgressProxy:
    def resolve(host: str, port: int, **_kw: object) -> list[tuple[Any, ...]]:
        if host == NAME:
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (PUBLIC, port))]
        return socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)

    def connect(address: str, port: int, timeout: float) -> socket.socket:
        if address == PUBLIC:  # the made-up public site is the test's server
            return socket.create_connection(("127.0.0.1", server.port), timeout=timeout)
        return socket.create_connection((address, port), timeout=timeout)

    return EgressProxy(resolve=resolve, connect=connect)


def test_page_scripts_frames_and_redirects_reach_nothing_on_this_pc(
    tmp_path: Path, server: Server
) -> None:
    program = shutil.which("agent-browser")
    assert program is not None
    browser = Browser(program, tmp_path / "browser", idle_close_s=30, proxy=lambda: _proxy(server))
    try:
        view = browser.open(f"http://{NAME}/")
        time.sleep(3)  # the page's scripts try their connections
        view = browser.open(f"http://{NAME}/") if "waiting" in view.text else view
        with pytest.raises(BrowserError):
            browser.open(f"http://{NAME}/redirect")
        with pytest.raises(BrowserError):
            browser.open("http://127.0.0.1/")
    finally:
        browser.close()

    assert [hit for hit in server.hits if hit.startswith("/leak")] == []
    assert "reached" not in view.text
    assert "candidate" not in view.text
    assert list((tmp_path / "browser").iterdir()) == []


def test_a_preview_reaches_its_own_project_and_nothing_else(tmp_path: Path, server: Server) -> None:
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
    browser = preview_browser(folder, program, tmp_path / "browser", idle_close_s=30)
    try:
        browser.open(f"{origin}/")
        time.sleep(3)  # the page's scripts try their connections
        view = browser.open(f"{origin}/")
    finally:
        browser.close()
        folder.stop()

    assert "Built" in view.text
    assert [hit for hit in server.hits if hit.startswith("/leak")] == []
    assert "reached" not in view.text
    assert "internet" not in view.text
    assert list((tmp_path / "browser").iterdir()) == []
