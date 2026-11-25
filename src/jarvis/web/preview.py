"""Previews of the owner's own builds: one project folder, served read-only to its own browser.

A page Jarvis built is opened neither as `file://` (its scripts could then read other files) nor in
the web session. `FolderServer` serves one folder on a free port of 127.0.0.1: only regular files
inside it, never through a link that leads out, never a hidden or private one, and only GET and
HEAD. `preview_browser` gives it a browser session of its own whose proxy reaches that one port
and nothing else, not the internet either (ADR 0016). The folder must be one Jarvis may write
(`Boundary.writable`), the same rule as `write_file`.
"""

import http.server
import mimetypes
import threading
from pathlib import Path
from typing import TYPE_CHECKING, cast, override
from urllib.parse import unquote, urlsplit

from jarvis.web.browser import Browser, run_program
from jarvis.web.proxy import LOOPBACK, EgressProxy

if TYPE_CHECKING:
    from jarvis.pc.files import Boundary
    from jarvis.web.browser import Runner

MAX_FILE_BYTES = 20_000_000
INDEX = "index.html"


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
        if self._server is None:
            return 0
        return int(cast("tuple[str, int]", self._server.server_address)[1])

    def file(self, target: str) -> Path | None:
        """Return the file a request's path names, or None when it may not be served."""
        path = unquote(urlsplit(target).path)
        relative = Path(path.lstrip("/"))
        if any(part in {"..", ""} or part.startswith(".") for part in relative.parts):
            return None
        try:
            found = (self.folder / relative).resolve(strict=True)
            if found.is_dir():
                found = (found / INDEX).resolve(strict=True)
        except OSError:
            return None
        if not found.is_relative_to(self.folder) or not found.is_file():
            return None
        if self._boundary.secret(found):
            return None
        below = found.relative_to(self.folder).parts
        if any(part.startswith(".") for part in below):
            return None
        return found

    def read(self, target: str) -> bytes | None:
        """Return the bytes of the file a request names, or None when it may not be served."""
        found = self.file(target)
        if found is None:
            return None
        try:
            if found.stat().st_size > MAX_FILE_BYTES:
                return None
            return found.read_bytes()
        except OSError:
            return None

    def start(self) -> str:
        """Listen on a free port of 127.0.0.1 and return the origin."""
        server = self

        class Handler(http.server.BaseHTTPRequestHandler):
            @override
            def log_message(self, format: str, *args: object) -> None:
                return

            def do_GET(self) -> None:
                self._send(body=True)

            def do_HEAD(self) -> None:
                self._send(body=False)

            def _send(self, *, body: bool) -> None:
                data = server.read(self.path)
                if data is None:
                    self.send_error(404)
                    return
                kind = mimetypes.guess_type(urlsplit(self.path).path)[0] or "text/html"
                self.send_response(200)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                if body:
                    self.wfile.write(data)

        made = http.server.ThreadingHTTPServer((LOOPBACK, 0), Handler)
        made.daemon_threads = True
        self._server = made
        threading.Thread(
            target=made.serve_forever, args=(0.1,), name="jarvis-preview", daemon=True
        ).start()
        return self.origin

    def stop(self) -> None:
        """Stop serving."""
        made, self._server = self._server, None
        if made is not None:
            made.shutdown()
            made.server_close()


def preview_browser(
    server: FolderServer,
    program: str,
    root: Path,
    *,
    idle_close_s: float,
    run: Runner = run_program,
) -> Browser:
    """Return a browser session that opens only `server`'s origin, reaching nothing else.

    `server` must be started; stop it when the session is closed.
    """
    return Browser(
        program,
        root,
        idle_close_s=idle_close_s,
        run=run,
        proxy=lambda: EgressProxy(local_port=server.port),
        origin=server.origin,
    )
