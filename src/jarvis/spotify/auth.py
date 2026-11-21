"""Spotify login with PKCE (S256), a loopback-only callback and a validated `state`.

Tokens live in a 0600 file under the state directory and are refreshed atomically. No client
secret is used or stored.
"""

import base64
import hashlib
import json
import secrets
import threading
import time
import urllib.parse
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import TYPE_CHECKING, override

import httpx

from jarvis.secretfile import read_private, write_private_json

if TYPE_CHECKING:
    from collections.abc import Callable
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
    "user-top-read",
    "user-follow-read",
    "user-read-recently-played",
)
_EXPIRY_MARGIN_S = 60


class SpotifyAuthError(Exception):
    """Login or token refresh failed."""


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


def _handler(result: _Callback, done: threading.Event) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urllib.parse.urlsplit(self.path)
            if parsed.path != "/callback":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            params = urllib.parse.parse_qs(parsed.query)
            if params.get("state", [""])[0] != result.state:
                result.error = (
                    "the login answer's state did not match (possible forgery); try again"
                )
            elif "error" in params:
                result.error = f"Spotify refused: {params['error'][0]}"
            else:
                result.code = params.get("code", [None])[0]
            ok = result.code is not None
            self.send_response(HTTPStatus.OK if ok else HTTPStatus.BAD_REQUEST)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(
                b"Jarvis is connected to Spotify. You can close this tab."
                if ok
                else b"Login failed."
            )
            done.set()

        @override
        def log_message(self, format: str, *args: object) -> None:
            return

    return Handler


def login(
    client_id: str,
    port: int,
    token_path: Path,
    show_url: Callable[[str], None],
    timeout_s: float = 900.0,
) -> None:
    """Run the PKCE flow: show the URL, wait for the loopback callback, store the tokens.

    Raises:
        SpotifyAuthError: the owner did not finish in time, the state did not match or
            Spotify refused.
    """
    verifier, challenge = make_pkce()
    state = secrets.token_urlsafe(24)
    result = _Callback(state=state)
    done = threading.Event()
    server = HTTPServer(("127.0.0.1", port), _handler(result, done))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
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
        raise SpotifyAuthError(result.error or "no authorization code came back")
    tokens = _token_request(
        {
            "grant_type": "authorization_code",
            "code": result.code,
            "redirect_uri": redirect_uri(port),
            "client_id": client_id,
            "code_verifier": verifier,
        }
    )
    _store(token_path, tokens, previous_refresh=None)


def _token_request(form: dict[str, str]) -> dict[str, object]:
    try:
        response = httpx.post(TOKEN_URL, data=form, timeout=20.0, trust_env=False)
    except httpx.HTTPError as exc:
        msg = f"could not reach Spotify's token service: {exc}"
        raise SpotifyAuthError(msg) from exc
    if response.status_code != HTTPStatus.OK:
        msg = f"Spotify's token service answered {response.status_code}: {response.text[:200]}"
        raise SpotifyAuthError(msg)
    return response.json()


def _store(
    path: Path, tokens: dict[str, object], previous_refresh: str | None
) -> dict[str, object]:
    refresh = tokens.get("refresh_token") or previous_refresh
    if not isinstance(refresh, str):
        msg = "Spotify sent no refresh token"
        raise SpotifyAuthError(msg)
    expires_in = tokens.get("expires_in", 3600)
    record: dict[str, object] = {
        "access_token": tokens["access_token"],
        "refresh_token": refresh,
        "expires_at": time.time()
        + (float(expires_in) if isinstance(expires_in, (int, float)) else 3600.0),
        "scope": tokens.get("scope", ""),
    }
    write_private_json(path, record)
    return record


class TokenStore:
    """Hands out a valid access token, refreshing it when it is about to expire."""

    def __init__(self, client_id: str, path: Path) -> None:
        """Read tokens from `path` for the given developer app."""
        super().__init__()
        self._client_id = client_id
        self._path = path
        self._lock = threading.Lock()

    def has_login(self) -> bool:
        """Return whether a stored login exists."""
        return read_private(self._path) is not None

    def access_token(self, *, force_refresh: bool = False) -> str:
        """Return a current access token.

        Raises:
            SpotifyAuthError: there is no login, or the refresh failed.
        """
        with self._lock:
            text = read_private(self._path)
            if text is None:
                msg = "Spotify is not connected; run `jarvis spotify-login`"
                raise SpotifyAuthError(msg)
            record = json.loads(text)
            if not force_refresh and float(record["expires_at"]) - _EXPIRY_MARGIN_S > time.time():
                return str(record["access_token"])
            tokens = _token_request(
                {
                    "grant_type": "refresh_token",
                    "refresh_token": str(record["refresh_token"]),
                    "client_id": self._client_id,
                }
            )
            stored = _store(self._path, tokens, previous_refresh=str(record["refresh_token"]))
            return str(stored["access_token"])
