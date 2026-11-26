"""Previews of the owner's own builds: one folder, read-only, reached by its own browser alone."""

from __future__ import annotations

import socket
from typing import TYPE_CHECKING

import pytest
from test_web_browser import FakeAgentBrowser, Page

from jarvis.config import FilesConfig
from jarvis.pc.files import Boundary, FileRefusalError
from jarvis.web.browser import Browser, BrowserError
from jarvis.web.preview import FolderServer, preview_browser
from jarvis.web.proxy import EgressProxy

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


@pytest.fixture
def home(tmp_path: Path) -> Path:
    made = tmp_path / "me"
    (made / "Documents" / "site" / "css").mkdir(parents=True)
    site = made / "Documents" / "site"
    (site / "index.html").write_text("<h1>Built</h1>", encoding="utf-8")
    (site / "css" / "a.css").write_text("h1{}", encoding="utf-8")
    (site / ".env").write_text("TOKEN=x", encoding="utf-8")
    (site / "id_rsa").write_text("key", encoding="utf-8")
    (made / "secret.txt").write_text("outside", encoding="utf-8")
    (site / "out.txt").symlink_to(made / "secret.txt")
    return made


def _boundary(home: Path) -> Boundary:
    return Boundary.of(FilesConfig(write_roots=["~/Documents"]), home)


@pytest.fixture
def served(home: Path) -> Iterator[FolderServer]:
    server = FolderServer(home / "Documents" / "site", _boundary(home))
    server.start()
    yield server
    server.stop()


@pytest.mark.parametrize("where", ["secret.txt", ".", "Documents"], ids=["file", "home", "root"])
def test_only_a_folder_jarvis_may_write_is_served(home: Path, where: str) -> None:
    with pytest.raises((FileRefusalError, NotADirectoryError)):
        FolderServer(home / where, _boundary(home))


def test_its_files_are_served(served: FolderServer) -> None:
    assert served.read("/") == b"<h1>Built</h1>"
    assert served.read("/css/a.css") == b"h1{}"
    assert served.read("/css/a.css?v=2") == b"h1{}"


@pytest.mark.parametrize(
    "target",
    [
        "/../secret.txt",
        "/%2e%2e/secret.txt",
        "/out.txt",
        "/.env",
        "/id_rsa",
        "/missing.html",
        "/css",
    ],
)
def test_nothing_outside_hidden_private_or_missing_is_served(
    served: FolderServer, target: str
) -> None:
    assert served.read(target) is None


def _get(port: int, request: bytes) -> bytes:
    with socket.create_connection(("127.0.0.1", port), timeout=5) as client:
        client.sendall(request)
        data = b""
        while chunk := client.recv(4096):
            data += chunk
        return data


def test_it_listens_on_loopback_and_answers_only_reads(served: FolderServer) -> None:
    assert served.origin.startswith("http://127.0.0.1:")
    page = _get(served.port, b"GET / HTTP/1.0\r\n\r\n")
    posted = _get(served.port, b"POST / HTTP/1.0\r\nContent-Length: 1\r\n\r\nx")

    assert page.endswith(b"<h1>Built</h1>")
    assert b"Cache-Control: no-store" in page
    assert posted.startswith(b"HTTP/1.0 501")


def _ask(proxy: EgressProxy, request: bytes) -> bytes:
    line, _, rest = request.partition(b"\r\n")
    signed = line + b"\r\nProxy-Authorization: " + proxy.authorization.encode() + b"\r\n" + rest
    with socket.create_connection(("127.0.0.1", proxy.port), timeout=5) as client:
        client.sendall(signed)
        data = b""
        while chunk := client.recv(4096):
            data += chunk
        return data


def test_the_preview_proxy_reaches_its_one_port_and_nothing_else(served: FolderServer) -> None:
    proxy = EgressProxy(local_port=served.port)
    proxy.start()
    try:
        ok = _ask(proxy, f"GET {served.origin}/ HTTP/1.1\r\nHost: x\r\n\r\n".encode())
        refused = [
            _ask(proxy, f"GET {url} HTTP/1.1\r\nHost: x\r\n\r\n".encode())
            for url in (
                "http://127.0.0.1:8080/v1/models",
                "http://127.0.0.1:8888/",
                f"http://localhost:{served.port}/",
                "http://93.184.215.14/",
            )
        ]
        tunnel = _ask(proxy, b"CONNECT example.com:443 HTTP/1.1\r\n\r\n")
    finally:
        proxy.stop()

    assert ok.startswith((b"HTTP/1.0 200", b"HTTP/1.1 200"))
    assert all(answer.startswith(b"HTTP/1.1 403") for answer in refused)
    assert tunnel.startswith(b"HTTP/1.1 403")


def test_a_preview_browser_opens_only_its_project(tmp_path: Path, served: FolderServer) -> None:
    fake = FakeAgentBrowser(Page(url=f"{served.origin}/"))
    browser = preview_browser(
        served, "/usr/bin/agent-browser", tmp_path / "b", idle_close_s=60, run=fake
    )

    origin = served.origin
    view = browser.open(f"{origin}/")
    for elsewhere in ("https://example.com/", "http://127.0.0.1:8080/", "file:///etc/passwd"):
        with pytest.raises(BrowserError):
            browser.open(elsewhere)
    browser.close()

    assert view.url == f"{origin}/"


def test_the_web_browser_still_refuses_the_preview_origin(
    tmp_path: Path, served: FolderServer
) -> None:
    browser = Browser("/usr/bin/agent-browser", tmp_path / "b", idle_close_s=60)

    with pytest.raises(BrowserError):
        browser.open(f"{served.origin}/")


def test_closing_a_preview_browser_stops_its_server(tmp_path: Path, home: Path) -> None:
    server = FolderServer(home / "Documents" / "site", _boundary(home))
    server.start()
    fake = FakeAgentBrowser(Page(url=f"{server.origin}/"))
    browser = preview_browser(
        server, "/usr/bin/agent-browser", tmp_path / "b", idle_close_s=60, run=fake
    )
    browser.open(f"{server.origin}/")
    port = server.port

    browser.close()

    assert server.port == 0
    with pytest.raises(ConnectionRefusedError):
        socket.create_connection(("127.0.0.1", port), timeout=1).close()
