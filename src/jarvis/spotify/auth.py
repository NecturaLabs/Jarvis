"""Spotify login with PKCE (S256), a loopback-only callback and a validated `state`.

Tokens live in a 0600 file under the state directory and are refreshed atomically, under a file
lock so two processes never spend one refresh token twice. No client secret is used or stored.
Spotify's refresh token stops working six months after the owner granted access and refreshing does
not extend that, so the grant time is kept beside the tokens.
"""

import base64
import contextlib
import errno
import fcntl
import hashlib
import json
import math
import os
import re
import secrets
import threading
import time
import urllib.parse
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING, override

import httpx

from jarvis.jsonutil import obj
from jarvis.policy import Origin
from jarvis.secretfile import read_private, write_private_json
from jarvis.spotify.errors import SpotifyAuthError, SpotifyLoginExpired

if TYPE_CHECKING:
    from collections.abc import Callable, Generator
    from pathlib import Path

AUTHORIZE_URL = "https://accounts.spotify.com/authorize"
TOKEN_URL = "https://accounts.spotify.com/api/token"  # ruff: ignore[hardcoded-password-string] - an endpoint, not a secret
SCOPES = (
    "user-read-playback-state",
    "user-modify-playback-state",
    "user-read-currently-playing",
    "user-library-read",
    "user-library-modify",
    "playlist-read-private",
    "playlist-read-collaborative",
    "playlist-modify-private",  # add the song playing to a playlist (2026-11-26)
    "playlist-modify-public",
    "user-top-read",
    "user-follow-read",
    "user-read-recently-played",
)
_EXPIRY_MARGIN_S = 60
GRANT_LIFETIME_DAYS = 183
_DAY_S = 86400.0
_CALLBACK_IDLE_S = 5
_SERVER_ERROR = 500
_ERROR_CODE = re.compile(r"[a-z_]{1,40}")
_LOGIN_AGAIN = "run `jarvis spotify-login`"


def redirect_uri(port: int) -> str:
    """Return the loopback redirect URI that must be registered in the developer app."""
    return f"http://127.0.0.1:{port}/callback"


def make_pkce() -> tuple[str, str]:
    """Return a (verifier, S256 challenge) pair."""
    verifier = secrets.token_urlsafe(64)[:128]
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


def authorize_url(client_id: str, port: int, challenge: str, state: str) -> str:
    """Return the URL the owner opens to grant access."""
    query = {
        "client_id": client_id,
        "response_type": "code",
        "redirect_uri": redirect_uri(port),
        "code_challenge_method": "S256",
        "code_challenge": challenge,
        "state": state,
        "scope": " ".join(SCOPES),
    }
    return f"{AUTHORIZE_URL}?{urllib.parse.urlencode(query)}"


@dataclass(slots=True)
class _Callback:
    state: str
    code: str | None = None
    error: str | None = None


def _error_code(value: object) -> str | None:
    """Return an OAuth error code if `value` is one (short lowercase words), else None."""
    return value if isinstance(value, str) and _ERROR_CODE.fullmatch(value) else None


def _handler(result: _Callback, done: threading.Event) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        timeout = _CALLBACK_IDLE_S

        def _reply(self, status: HTTPStatus, text: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(text.encode())

        def do_GET(self) -> None:
            """Take the one answer that carries our `state`; ignore everything else."""
            parsed = urllib.parse.urlsplit(self.path)
            if parsed.path != "/callback":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            params = urllib.parse.parse_qs(parsed.query)
            heard = params.get("state", [""])[0]
            if not secrets.compare_digest(heard.encode(), result.state.encode()) or done.is_set():
                self._reply(HTTPStatus.BAD_REQUEST, "Login failed.")
                return
            if "error" in params:
                code = _error_code(params["error"][0]) or "unknown"
                result.error = f"Spotify refused: {code}"
            elif "code" in params and params["code"][0]:
                result.code = params["code"][0]
            else:
                self._reply(HTTPStatus.BAD_REQUEST, "Login failed.")
                return
            if result.code is not None:
                self._reply(
                    HTTPStatus.OK, "Jarvis is connected to Spotify. You can close this tab."
                )
            else:
                self._reply(HTTPStatus.BAD_REQUEST, "Login failed.")
            done.set()

        @override
        def log_message(self, format: str, *args: object) -> None:
            return

    return Handler


class _Server(ThreadingHTTPServer):
    """The callback listener: a stalled connection must never hold up the login or its end."""

    daemon_threads = True
    block_on_close = False


def login(  # ruff: ignore[too-many-arguments] - `http` and `clock` are test seams
    client_id: str,
    port: int,
    token_path: Path,
    show_url: Callable[[str], None],
    timeout_s: float = 900.0,
    *,
    http: httpx.Client | None = None,
    clock: Callable[[], float] = time.time,
) -> None:
    """Run the PKCE flow: show the URL, wait for the loopback callback, store the tokens.

    Requests that do not carry the login's own `state` (or are for another path) are answered
    and ignored; the login keeps waiting for the real answer.

    Raises:
        SpotifyAuthError: the callback port is in use, the owner did not finish in time, Spotify
            refused or the token answer was unusable.
    """
    verifier, challenge = make_pkce()
    state = secrets.token_urlsafe(24)
    result = _Callback(state=state)
    done = threading.Event()
    try:
        server = _Server(("127.0.0.1", port), _handler(result, done))
    except OSError as exc:
        why = "is already in use" if exc.errno == errno.EADDRINUSE else f"can't be used ({exc})"
        msg = f"the login's callback port {port} {why}; free it or change `spotify.redirect_port`"
        raise SpotifyAuthError(msg) from exc
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
    )
    thread.start()
    try:
        show_url(authorize_url(client_id, port, challenge, state))
        if not done.wait(timeout_s):
            msg = "the login was not finished in time"
            raise SpotifyAuthError(msg)
    finally:
        server.shutdown()
        server.server_close()
    if result.code is None:
        raise SpotifyAuthError(
            result.error or "no authorization code came back",
            origin=Origin.SPOTIFY if result.error else Origin.TOOL_STATUS,
        )
    grant = _token_request(
        http,
        {
            "grant_type": "authorization_code",
            "code": result.code,
            "redirect_uri": redirect_uri(port),
            "client_id": client_id,
            "code_verifier": verifier,
        },
    )
    if grant.refresh_token is None:
        msg = "Spotify sent no refresh token"
        raise SpotifyAuthError(msg)
    now = clock()
    TokenStore(client_id, token_path, http, clock=clock).store_login(
        _Record(grant.access_token, grant.refresh_token, now + grant.expires_in, grant.scope, now)
    )


@dataclass(frozen=True, slots=True)
class _Grant:
    """A validated token answer."""

    access_token: str
    refresh_token: str | None
    expires_in: float
    scope: str


@dataclass(frozen=True, slots=True)
class _Record:
    """What the token file holds."""

    access_token: str
    refresh_token: str
    expires_at: float
    scope: str
    authorized_at: float | None

    def as_json(self) -> dict[str, object]:
        """Return the file's content; the grant time is left out while it is unknown."""
        record: dict[str, object] = {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "expires_at": self.expires_at,
            "scope": self.scope,
        }
        if self.authorized_at is not None:
            record["authorized_at"] = self.authorized_at
        return record


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def _token_request(http: httpx.Client | None, form: dict[str, str]) -> _Grant:
    refreshing = form["grant_type"] == "refresh_token"
    try:
        response = (
            http.post(TOKEN_URL, data=form, timeout=20.0)
            if http
            else httpx.post(TOKEN_URL, data=form, timeout=20.0, trust_env=False)
        )
    except httpx.HTTPError as exc:
        msg = "Spotify's token service could not be reached; try again later"
        raise SpotifyAuthError(msg) from exc
    status = response.status_code
    if status >= _SERVER_ERROR:
        msg = f"Spotify's token service could not be reached (it answered {status}); try later"
        raise SpotifyAuthError(msg)
    if status != HTTPStatus.OK:
        code = _error_code(_json(response).get("error"))
        if refreshing and status == HTTPStatus.BAD_REQUEST and code == "invalid_grant":
            msg = f"Spotify login has expired; {_LOGIN_AGAIN}"
            raise SpotifyLoginExpired(msg)
        if code is None:
            msg = f"Spotify's token service refused the request (HTTP {status})"
            raise SpotifyAuthError(msg)
        msg = f"Spotify's token service refused the request ({code})"
        raise SpotifyAuthError(msg, origin=Origin.SPOTIFY)
    body = _json(response)
    access, expires_in = body.get("access_token"), _number(body.get("expires_in"))
    if not isinstance(access, str) or not access or expires_in is None:
        msg = "Spotify's token answer was not usable"
        raise SpotifyAuthError(msg)
    refresh, scope = body.get("refresh_token"), body.get("scope")
    return _Grant(
        access,
        refresh if isinstance(refresh, str) and refresh else None,
        expires_in,
        scope if isinstance(scope, str) else "",
    )


def _json(response: httpx.Response) -> dict[str, object]:
    try:
        return obj(response.json())
    except ValueError:
        return {}


def _parse_record(text: str) -> _Record:
    try:
        data = obj(json.loads(text))
    except ValueError:
        data = {}
    access, refresh = data.get("access_token"), data.get("refresh_token")
    expires_at, scope = _number(data.get("expires_at")), data.get("scope")
    if not (isinstance(access, str) and access and isinstance(refresh, str) and refresh):
        expires_at = None
    if expires_at is None:
        msg = f"Spotify's saved login is damaged; {_LOGIN_AGAIN}"
        raise SpotifyAuthError(msg)
    return _Record(
        str(access),
        str(refresh),
        expires_at,
        scope if isinstance(scope, str) else "",
        _number(data.get("authorized_at")),
    )


class TokenStore:
    """Hands out a valid access token, refreshing it when it is about to expire."""

    def __init__(
        self,
        client_id: str,
        path: Path,
        http: httpx.Client | None = None,
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        """Read tokens from `path` for the given developer app; `http` and `clock` are for tests."""
        super().__init__()
        self._client_id = client_id
        self._path = path
        self._http = http
        self._clock = clock
        self._lock = threading.Lock()
        self._last: str | None = None

    def has_login(self) -> bool:
        """Return whether a stored login file exists, even a damaged or unreadable one.

        `access_token` then fails with the instruction to log in again.
        """
        try:
            return read_private(self._path) is not None
        except OSError, ValueError:
            return True

    def store_login(self, record: _Record) -> None:
        """Replace the stored login with a fresh grant, under the locks a refresh holds.

        A refresh that was running on the old grant has finished before this writes, so it cannot
        overwrite the new login afterwards.
        """
        with self._lock, self._file_lock():
            self._save(record)

    def granted(self) -> frozenset[str] | None:
        """Return the scopes the stored login was granted, or None when that is not known."""
        try:
            scope = self._load().scope
        except SpotifyAuthError:
            return None
        return frozenset(scope.split()) or None

    @property
    def authorized_at(self) -> float | None:
        """When the owner granted access (epoch seconds), or None when it is not known."""
        try:
            return self._load().authorized_at
        except SpotifyAuthError:
            return None

    def days_left(self, now: float | None = None) -> int | None:
        """Return the whole days until Spotify stops honouring this login, or None if unknown."""
        granted = self.authorized_at
        if granted is None:
            return None
        end = granted + GRANT_LIFETIME_DAYS * _DAY_S
        return math.floor((end - (self._clock() if now is None else now)) / _DAY_S)

    def access_token(self, *, force_refresh: bool = False) -> str:
        """Return a current access token.

        A forced refresh is skipped when another store or process already replaced the token
        this one handed out: the refresh token is not spent twice. It fails with
        `SpotifyAuthError` when there is no usable login or the refresh fails, and with its
        subclass `SpotifyLoginExpired` when Spotify no longer accepts the stored login.
        """
        with self._lock:
            if not force_refresh:
                record = self._load()
                if self._fresh(record):
                    return self._hand_out(record)
            with self._file_lock():
                record = self._load()
                replaced = self._last is not None and record.access_token != self._last
                if (force_refresh and replaced) or (not force_refresh and self._fresh(record)):
                    return self._hand_out(record)
                grant = _token_request(
                    self._http,
                    {
                        "grant_type": "refresh_token",
                        "refresh_token": record.refresh_token,
                        "client_id": self._client_id,
                    },
                )
                renewed = _Record(
                    grant.access_token,
                    grant.refresh_token or record.refresh_token,
                    self._clock() + grant.expires_in,
                    grant.scope or record.scope,
                    record.authorized_at,
                )
                self._save(renewed)
                return self._hand_out(renewed)

    def _fresh(self, record: _Record) -> bool:
        return record.expires_at - _EXPIRY_MARGIN_S > self._clock()

    def _hand_out(self, record: _Record) -> str:
        self._last = record.access_token
        return record.access_token

    def _load(self) -> _Record:
        try:
            text = read_private(self._path)
        except OSError as exc:
            msg = f"Spotify's saved login can't be used ({exc}); {_LOGIN_AGAIN}"
            raise SpotifyAuthError(msg) from exc
        except ValueError:  # not UTF-8; the decoder's message would quote file bytes
            msg = f"Spotify's saved login is damaged; {_LOGIN_AGAIN}"
            raise SpotifyAuthError(msg) from None
        if text is None:
            msg = f"Spotify is not connected; {_LOGIN_AGAIN}"
            raise SpotifyAuthError(msg)
        return _parse_record(text)

    def _save(self, record: _Record) -> None:
        write_private_json(self._path, record.as_json())

    @contextlib.contextmanager
    def _file_lock(self) -> Generator[None]:
        """Hold an exclusive lock on `<token file>.lock` (mode 0600) across read, refresh, write."""
        lock_path = self._path.with_name(f"{self._path.name}.lock")
        lock_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
        try:
            os.fchmod(fd, 0o600)
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            os.close(fd)
