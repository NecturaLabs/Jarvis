"""Previews of the owner's own builds: one project folder, served read-only to its own browser.

A page Jarvis built is opened neither as `file://` (its scripts could then read other files) nor in
the web session. `FolderServer` serves one folder on a free port of 127.0.0.1: only regular files
inside it with no second name on disk, opened one folder at a time from the folder held open since
the start and never through a link (so a file or folder swapped for one meanwhile is not followed),
never a hidden or private one, and only GET and HEAD. It answers only its preview proxy: a request
must carry the token the proxy adds and name the server itself as its host, so neither another
program on this PC nor a page in the owner's own browser renamed to 127.0.0.1 (DNS rebinding)
reads the project. `preview_browser` gives it a browser session of its own whose proxy reaches
that one port and nothing else, not the internet either (ADR 0016). The folder must be one Jarvis
may write (`Boundary.writable`), the same rule as `write_file`.
"""

import hmac
import http.server
import mimetypes
import os
import secrets
import stat
import threading
from contextlib import ExitStack
from http import HTTPStatus
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, cast, override
from urllib.parse import unquote, urlsplit

from jarvis.pc.files import FileRefusalError
from jarvis.web.browser import Browser, run_program
from jarvis.web.proxy import LOOPBACK, EgressProxy

if TYPE_CHECKING:
    from jarvis.pc.files import Boundary
    from jarvis.web.browser import Runner

MAX_FILE_BYTES = 20_000_000
INDEX = "index.html"
HEADER = "X-Jarvis-Preview"  # carries the server's token, added by its proxy alone
_UNKNOWN_TYPE = "application/octet-stream"  # never sniffed into a page
_DIRECTORY = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_FILE = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK | os.O_NOCTTY


class FolderServer:
    """Serves the regular files of one folder on 127.0.0.1, read-only, until stopped."""

    def __init__(self, folder: Path, boundary: Boundary) -> None:
        """Serve `folder`, which `boundary` must let Jarvis write; a refusal raises its error.

        Raises:
            NotADirectoryError: it is not a folder.
        """
        super().__init__()
        self.folder = boundary.writable(str(folder))
        if not self.folder.is_dir():
            msg = f"{boundary.show(self.folder)} is not a folder"
            raise NotADirectoryError(msg)
        self._boundary = boundary
        self._server: http.server.ThreadingHTTPServer | None = None
        self._root: int | None = None  # the folder, held open while it is served
        self._lock = threading.Lock()
        self.header = HEADER
        self.token = secrets.token_urlsafe(24)

    @property
    def origin(self) -> str:
        """Where the folder is served: `http://127.0.0.1:<port>`.

        Raises:
            RuntimeError: it is not started.
        """
        if self._server is None:
            msg = "the preview server is not started"
            raise RuntimeError(msg)
        return f"http://{LOOPBACK}:{self.port}"

    @property
    def port(self) -> int:
        """The port it serves on (0 before it starts)."""
        made = self._server
        if made is None:
            return 0
        return int(cast("tuple[str, int]", made.server_address)[1])

    def _relative(self, target: str) -> tuple[str, ...] | None:
        """Return the parts of the path a request names, or None when it may not be served."""
        path = unquote(urlsplit(target).path)
        if "\0" in path:
            return None
        parts = Path(path.lstrip("/")).parts
        if any(part in {"..", ""} or part.startswith(".") for part in parts):
            return None
        if self._boundary.secret(self.folder.joinpath(*parts)):
            return None
        return parts

    def read(self, target: str) -> bytes | None:
        """Return the bytes of the file a request names, or None when it may not be served.

        It is opened one folder at a time from the served folder without following a link, and
        must be a regular file with one name, no bigger than `MAX_FILE_BYTES`; a folder serves
        its `index.html`.
        """
        found = self.read_file(target)
        return found[0] if found is not None else None

    def read_file(self, target: str) -> tuple[bytes, bool] | None:
        """Return the file contents and whether they are a directory index."""
        parts = self._relative(target)
        if parts is None:
            return None
        with self._lock:
            if self._root is None:
                return None
            try:
                root = os.dup(self._root)
            except OSError:
                return None
        opened = [root]
        try:
            fd, directory = _open_inside(root, parts, opened)
            data = _bounded(fd)
        except OSError:
            return None
        else:
            return (data, directory) if data is not None else None
        finally:
            for fd in opened:
                os.close(fd)

    def allowed(self, host: str | None, token: str | None) -> bool:
        """Return whether a request came through the preview's proxy, for this server."""
        own_host = host == f"{LOOPBACK}:{self.port}"
        own_token = hmac.compare_digest((token or "").encode(), self.token.encode())
        return own_host and own_token

    def start(self) -> str:
        """Listen on a free port of 127.0.0.1 or raise if the folder is no longer allowed."""
        server = self

        class Handler(_PreviewHandler):
            folder = server

        with self._lock:
            if self._server is None:
                self._listen(Handler)
            return self.origin

    def _listen(self, handler: type[http.server.BaseHTTPRequestHandler]) -> None:
        """Open the verified folder and start the listener, releasing both on failure.

        Raises:
            FileRefusalError: the opened folder is no longer the allowed one.
        """
        with ExitStack() as startup:
            root = os.open(self.folder, _DIRECTORY)
            startup.callback(os.close, root)
            opened = Path(f"/proc/self/fd/{root}").readlink()
            if opened != self.folder or self._boundary.writable(str(opened)) != opened:
                msg = "the preview folder changed before it could be opened"
                raise FileRefusalError(msg)
            made = http.server.ThreadingHTTPServer((LOOPBACK, 0), handler)
            startup.callback(made.server_close)
            made.daemon_threads = True
            self._server = made
            self._root = root
            try:
                threading.Thread(
                    target=made.serve_forever, args=(0.1,), name="jarvis-preview", daemon=True
                ).start()
            except BaseException:
                self._server = None
                self._root = None
                raise
            startup.pop_all()

    def stop(self) -> None:
        """Stop serving."""
        with self._lock:
            made, self._server = self._server, None
            root, self._root = self._root, None
            if made is not None:
                made.shutdown()
                made.server_close()
            if root is not None:
                os.close(root)


class _PreviewHandler(http.server.BaseHTTPRequestHandler):
    folder: ClassVar[FolderServer]

    @override
    def log_message(self, format: str, *args: object) -> None:
        return

    def do_GET(self) -> None:
        self._send(body=True)

    def do_HEAD(self) -> None:
        self._send(body=False)

    def _send(self, *, body: bool) -> None:
        if not self.folder.allowed(self.headers.get("Host"), self.headers.get(HEADER)):
            self.send_error(403)
            return
        found = self.folder.read_file(self.path)
        if found is None:
            self.send_error(404)
            return
        data, directory = found
        parts = urlsplit(self.path)
        if directory and not parts.path.endswith("/"):
            self._redirect(parts.path, parts.query)
            return
        kind = "text/html" if directory else mimetypes.guess_type(parts.path)[0] or _UNKNOWN_TYPE
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if body:
            self.wfile.write(data)

    def _redirect(self, path: str, query: str) -> None:
        target = f"/{path.lstrip('/')}/"
        if query:
            target += f"?{query}"
        self.send_response(HTTPStatus.PERMANENT_REDIRECT)
        self.send_header("Location", target)
        self.send_header("Content-Length", "0")
        self.end_headers()


def preview_browser(
    server: FolderServer,
    program: str,
    root: Path,
    *,
    idle_close_s: float,
    run: Runner = run_program,
) -> Browser:
    """Return a browser session that opens only `server`'s origin, reaching nothing else.

    `server` must be started; closing the session stops it.
    """
    return Browser(
        program,
        root,
        idle_close_s=idle_close_s,
        run=run,
        proxy=lambda: EgressProxy(
            local_port=server.port, local_header=(server.header, server.token)
        ),
        origin=server.origin,
        on_close=server.stop,
    )


def _open_inside(root: int, parts: tuple[str, ...], opened: list[int]) -> tuple[int, bool]:
    """Open `parts` below the folder `root`, one at a time and following no link.

    Every descriptor opened is put in `opened`, for the caller to close. A folder opens its
    `index.html` and is marked as a directory. An `OSError` says a part is missing, a link,
    or not a folder where one must be.
    """
    folder = root
    for part in parts[:-1]:
        folder = os.open(part, _DIRECTORY, dir_fd=folder)
        opened.append(folder)
    fd = root
    if parts:
        fd = os.open(parts[-1], _FILE, dir_fd=folder)
        opened.append(fd)
    directory = stat.S_ISDIR(os.fstat(fd).st_mode)
    if directory:
        fd = os.open(INDEX, _FILE, dir_fd=fd)
        opened.append(fd)
    return fd, directory


def _bounded(fd: int) -> bytes | None:
    """Return an open file's bytes: a regular file with one name, of `MAX_FILE_BYTES` at most."""
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > MAX_FILE_BYTES:
        return None
    chunks: list[bytes] = []
    size = 0
    while chunk := os.read(fd, 1 << 20):
        size += len(chunk)
        if size > MAX_FILE_BYTES:
            return None
        chunks.append(chunk)
    return b"".join(chunks)
