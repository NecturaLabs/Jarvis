"""Spotify login and token upkeep: the PKCE flow, validated tokens, refresh and its failures."""

import base64
import fcntl
import hashlib
import http.client
import json
import os
import socket
import stat
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest

from jarvis.operations import OperationError
from jarvis.policy import Origin
from jarvis.secretfile import read_private, write_private, write_private_json
from jarvis.spotify.auth import (
    SpotifyAuthError,
    SpotifyLoginExpired,
    TokenStore,
    login,
    make_pkce,
)
from jarvis.spotify.errors import NoDeviceError, SpotifyError

if TYPE_CHECKING:
    from collections.abc import Callable

DAY = 86400.0
START = 1_800_000_000.0
GOOD = {"access_token": "acc-1", "refresh_token": "ref-1", "expires_in": 3600, "scope": "s"}


class Accounts:
    """A scripted accounts.spotify.com token endpoint."""

    def __init__(self, *answers: httpx.Response | Exception) -> None:
        super().__init__()
        self.answers = list(answers)
        self.forms: list[dict[str, str]] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/token"
        self.forms.append({k: v[0] for k, v in parse_qs(request.content.decode()).items()})
        answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        if isinstance(answer, Exception):
            raise answer
        return answer

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self.handle))


def _ok(**changes: object) -> httpx.Response:
    return httpx.Response(200, json={**GOOD, **changes})


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _write(path: Path, **changes: object) -> None:
    record: dict[str, object] = {
        "access_token": "acc-0",
        "refresh_token": "ref-0",
        "expires_at": START + 3600,
        "scope": "s",
        **changes,
    }
    write_private_json(path, {k: v for k, v in record.items() if v is not None})


def _store(path: Path, accounts: Accounts, now: Callable[[], float] = lambda: START) -> TokenStore:
    return TokenStore("client", path, accounts.client(), clock=now)


def _saved(path: Path) -> dict[str, Any]:
    text = read_private(path)
    assert text is not None
    return json.loads(text)


def _get(port: int, target: str) -> int:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        conn.request("GET", target)
        return conn.getresponse().status
    finally:
        conn.close()


def _query(url: str) -> dict[str, str]:
    return {k: v[0] for k, v in parse_qs(urlsplit(url).query).items()}


def test_pkce_challenge_is_the_s256_of_the_verifier() -> None:
    verifier, challenge = make_pkce()

    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    assert challenge == base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    assert 43 <= len(verifier) <= 128


def test_login_ignores_other_requests_and_waits_for_the_right_one(tmp_path: Path) -> None:
    accounts = Accounts(_ok())
    port = _free_port()
    seen: list[int] = []

    def show(url: str) -> None:
        query = _query(url)
        assert query["code_challenge_method"] == "S256"
        seen.append(_get(port, "/callback?code=stolen&state=wrong"))
        seen.append(_get(port, "/elsewhere?code=x&state=" + query["state"]))
        seen.append(_get(port, "/callback?state=" + query["state"]))
        seen.append(_get(port, "/callback?code=the-code&state=" + query["state"]))

    login(
        "client", port, tmp_path / "token.json", show, http=accounts.client(), clock=lambda: START
    )

    assert seen == [400, 404, 400, 200]
    assert [form["code"] for form in accounts.forms] == ["the-code"]
    assert _saved(tmp_path / "token.json")["access_token"] == "acc-1"


def test_login_sends_the_verifier_that_matches_the_challenge(tmp_path: Path) -> None:
    accounts = Accounts(_ok())
    port = _free_port()
    challenges: list[str] = []

    def show(url: str) -> None:
        query = _query(url)
        challenges.append(query["code_challenge"])
        _get(port, f"/callback?code=c&state={query['state']}")

    login(
        "client", port, tmp_path / "token.json", show, http=accounts.client(), clock=lambda: START
    )

    form = accounts.forms[0]
    digest = hashlib.sha256(form["code_verifier"].encode("ascii")).digest()
    assert base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii") == challenges[0]
    assert form["redirect_uri"] == f"http://127.0.0.1:{port}/callback"


def test_login_stores_a_private_file_with_the_authorization_time(tmp_path: Path) -> None:
    accounts = Accounts(_ok())
    port = _free_port()
    path = tmp_path / "state" / "token.json"

    def show(url: str) -> None:
        _get(port, f"/callback?code=c&state={_query(url)['state']}")

    login("client", port, path, show, http=accounts.client(), clock=lambda: START)

    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert _saved(path)["authorized_at"] == START


def test_an_idle_connection_does_not_stop_the_login_timeout(tmp_path: Path) -> None:
    port = _free_port()
    idle: list[socket.socket] = []

    def show(_url: str) -> None:
        sock = socket.create_connection(("127.0.0.1", port), timeout=5)
        idle.append(sock)

    with pytest.raises(SpotifyAuthError, match="not finished in time"):
        login(
            "client", port, tmp_path / "token.json", show,
            timeout_s=0.2, http=Accounts(_ok()).client(),
        )  # fmt: skip
    idle[0].close()


def test_an_idle_connection_does_not_block_the_right_answer(tmp_path: Path) -> None:
    accounts = Accounts(_ok())
    port = _free_port()
    idle: list[socket.socket] = []

    def show(url: str) -> None:
        idle.append(socket.create_connection(("127.0.0.1", port), timeout=5))
        _get(port, f"/callback?code=c&state={_query(url)['state']}")

    login("client", port, tmp_path / "token.json", show, http=accounts.client())

    idle[0].close()
    assert accounts.forms


def test_a_refusal_with_the_right_state_ends_the_login(tmp_path: Path) -> None:
    port = _free_port()

    def show(url: str) -> None:
        _get(port, f"/callback?error=access_denied&state={_query(url)['state']}")

    with pytest.raises(SpotifyAuthError, match="access_denied") as caught:
        login("client", port, tmp_path / "token.json", show, http=Accounts(_ok()).client())
    assert caught.value.origin is Origin.SPOTIFY


BAD_TOKEN_ANSWERS = [
    pytest.param(httpx.Response(200, json=["x"]), id="not-an-object"),
    pytest.param(httpx.Response(200, text="<html>"), id="not-json"),
    pytest.param(_ok(access_token=""), id="empty-access-token"),
    pytest.param(_ok(access_token=7), id="numeric-access-token"),
    pytest.param(httpx.Response(200, json={"refresh_token": "r", "expires_in": 5}), id="no-token"),
    pytest.param(_ok(expires_in="3600"), id="text-expiry"),
    pytest.param(_ok(expires_in=True), id="boolean-expiry"),
]


@pytest.mark.parametrize("answer", BAD_TOKEN_ANSWERS)
def test_a_malformed_token_answer_is_an_auth_error_on_refresh(
    tmp_path: Path, answer: httpx.Response
) -> None:
    _write(tmp_path / "token.json", expires_at=START)

    with pytest.raises(SpotifyAuthError):
        _store(tmp_path / "token.json", Accounts(answer)).access_token()


@pytest.mark.parametrize("answer", BAD_TOKEN_ANSWERS)
def test_a_malformed_token_answer_is_an_auth_error_on_login(
    tmp_path: Path, answer: httpx.Response
) -> None:
    port = _free_port()

    def show(url: str) -> None:
        _get(port, f"/callback?code=c&state={_query(url)['state']}")

    with pytest.raises(SpotifyAuthError):
        login("client", port, tmp_path / "token.json", show, http=Accounts(answer).client())
    assert read_private(tmp_path / "token.json") is None


@pytest.mark.parametrize(
    "content",
    [
        b"\xff\xfe\x00garbage",
        "",
        "{",
        "[]",
        '"text"',
        "{}",
        '{"access_token": 1, "refresh_token": "r", "expires_at": 5}',
        '{"access_token": "a", "refresh_token": "", "expires_at": 5}',
        '{"access_token": "a", "refresh_token": "r"}',
        '{"access_token": "a", "refresh_token": "r", "expires_at": "soon"}',
    ],
)
def test_a_damaged_token_file_says_to_log_in_again(tmp_path: Path, content: str | bytes) -> None:
    path = tmp_path / "token.json"
    if isinstance(content, bytes):
        path.write_bytes(content)
        path.chmod(0o600)
    else:
        write_private(path, content)

    with pytest.raises(SpotifyAuthError, match="jarvis spotify-login"):
        _store(path, Accounts(_ok())).access_token()


def test_no_login_says_to_log_in(tmp_path: Path) -> None:
    store = _store(tmp_path / "token.json", Accounts(_ok()))

    assert not store.has_login()
    with pytest.raises(SpotifyAuthError, match="jarvis spotify-login"):
        store.access_token()


def test_a_valid_token_is_returned_without_a_request(tmp_path: Path) -> None:
    _write(tmp_path / "token.json")
    accounts = Accounts(_ok())

    assert _store(tmp_path / "token.json", accounts).access_token() == "acc-0"
    assert accounts.forms == []


@pytest.mark.parametrize(("left", "refreshed"), [(61, False), (59, True), (-5, True)])
def test_a_token_is_refreshed_inside_the_expiry_margin(
    tmp_path: Path, left: int, refreshed: bool
) -> None:
    _write(tmp_path / "token.json", expires_at=START + left)
    accounts = Accounts(_ok())

    token = _store(tmp_path / "token.json", accounts).access_token()

    assert (token == "acc-1") is refreshed
    assert bool(accounts.forms) is refreshed


def test_refresh_sends_the_refresh_grant_for_the_client_only(tmp_path: Path) -> None:
    _write(tmp_path / "token.json", expires_at=START)
    accounts = Accounts(_ok())

    _store(tmp_path / "token.json", accounts).access_token()

    assert accounts.forms == [
        {"grant_type": "refresh_token", "refresh_token": "ref-0", "client_id": "client"}
    ]


def test_refresh_stores_the_new_expiry_and_keeps_the_file_private(tmp_path: Path) -> None:
    path = tmp_path / "token.json"
    _write(path, expires_at=START)

    _store(path, Accounts(_ok(expires_in=1800))).access_token()

    assert _saved(path)["expires_at"] == START + 1800
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_a_rotated_refresh_token_is_stored(tmp_path: Path) -> None:
    path = tmp_path / "token.json"
    _write(path, expires_at=START)

    _store(path, Accounts(_ok(refresh_token="ref-2"))).access_token()

    assert _saved(path)["refresh_token"] == "ref-2"


def test_an_answer_without_a_refresh_token_keeps_the_old_one(tmp_path: Path) -> None:
    path = tmp_path / "token.json"
    _write(path, expires_at=START)
    answer = {"access_token": "acc-1", "expires_in": 3600}

    _store(path, Accounts(httpx.Response(200, json=answer))).access_token()

    assert _saved(path)["refresh_token"] == "ref-0"


def test_refresh_keeps_the_authorization_time(tmp_path: Path) -> None:
    path = tmp_path / "token.json"
    _write(path, expires_at=START, authorized_at=START - 10 * DAY)

    _store(path, Accounts(_ok())).access_token()

    assert _saved(path)["authorized_at"] == START - 10 * DAY


def test_an_old_file_without_an_authorization_time_has_none(tmp_path: Path) -> None:
    path = tmp_path / "token.json"
    _write(path, expires_at=START)
    store = _store(path, Accounts(_ok()))

    assert store.authorized_at is None
    assert store.days_left() is None
    store.access_token()
    assert store.authorized_at is None


def test_days_left_counts_whole_days_to_the_six_month_limit(tmp_path: Path) -> None:
    path = tmp_path / "token.json"
    _write(path, authorized_at=START - 10 * DAY)
    store = _store(path, Accounts(_ok()))

    assert store.authorized_at == START - 10 * DAY
    assert store.days_left() == 173
    assert store.days_left(START + 173 * DAY - 1) == 0
    assert (store.days_left(START + 183 * DAY) or 0) < 0


def test_invalid_grant_means_log_in_again_and_keeps_the_file(tmp_path: Path) -> None:
    path = tmp_path / "token.json"
    _write(path, expires_at=START)
    gone = httpx.Response(
        400, json={"error": "invalid_grant", "error_description": "Refresh token revoked"}
    )

    with pytest.raises(SpotifyLoginExpired, match="jarvis spotify-login"):
        _store(path, Accounts(gone)).access_token()

    assert _saved(path)["refresh_token"] == "ref-0"


@pytest.mark.parametrize(
    "answer",
    [
        pytest.param(httpx.Response(503, text="down"), id="503"),
        pytest.param(httpx.Response(500, text="oops"), id="500"),
        pytest.param(httpx.ConnectError("no route"), id="connect-error"),
        pytest.param(httpx.ReadTimeout("slow"), id="timeout"),
    ],
)
def test_an_unreachable_service_is_not_a_lost_login(tmp_path: Path, answer: Exception) -> None:
    path = tmp_path / "token.json"
    _write(path, expires_at=START)

    with pytest.raises(SpotifyAuthError, match="could not be reached") as caught:
        _store(path, Accounts(answer)).access_token()

    assert not isinstance(caught.value, SpotifyLoginExpired)
    assert "spotify-login" not in str(caught.value)
    assert _saved(path)["refresh_token"] == "ref-0"


def test_another_refusal_is_reported_with_spotifys_code_as_untrusted(tmp_path: Path) -> None:
    path = tmp_path / "token.json"
    _write(path, expires_at=START)
    refused = httpx.Response(400, json={"error": "invalid_client"})

    with pytest.raises(SpotifyAuthError, match="invalid_client") as caught:
        _store(path, Accounts(refused)).access_token()

    assert not isinstance(caught.value, SpotifyLoginExpired)
    assert caught.value.origin is Origin.SPOTIFY


def test_errors_never_carry_a_token(tmp_path: Path) -> None:
    path = tmp_path / "token.json"
    _write(path, expires_at=START)
    refused = httpx.Response(400, text="bad ref-0 acc-0 request")

    with pytest.raises(SpotifyAuthError) as caught:
        _store(path, Accounts(refused)).access_token()

    assert "ref-0" not in str(caught.value)
    assert "acc-0" not in str(caught.value)


def test_a_forced_refresh_refreshes_even_when_the_token_looks_valid(tmp_path: Path) -> None:
    path = tmp_path / "token.json"
    _write(path)
    accounts = Accounts(_ok())

    assert _store(path, accounts).access_token(force_refresh=True) == "acc-1"
    assert len(accounts.forms) == 1


def test_the_second_store_sees_the_first_ones_refresh(tmp_path: Path) -> None:
    path = tmp_path / "token.json"
    _write(path, expires_at=START)
    first, second = Accounts(_ok()), Accounts(_ok(access_token="acc-9"))

    assert _store(path, first).access_token() == "acc-1"
    assert _store(path, second).access_token() == "acc-1"

    assert second.forms == []


def test_a_forced_refresh_after_another_store_refreshed_does_not_refresh_again(
    tmp_path: Path,
) -> None:
    # Both stores handed out acc-0; the first one's request was refused and it refreshed. The
    # second one's retry must use that new token, not spend the refresh token a second time.
    path = tmp_path / "token.json"
    _write(path)
    first, second = Accounts(_ok()), Accounts(_ok(access_token="acc-9"))
    one, two = _store(path, first), _store(path, second)
    assert one.access_token() == two.access_token() == "acc-0"

    assert one.access_token(force_refresh=True) == "acc-1"
    assert two.access_token(force_refresh=True) == "acc-1"

    assert second.forms == []


def test_concurrent_stores_refresh_once(tmp_path: Path) -> None:
    path = tmp_path / "token.json"
    _write(path, expires_at=START)
    accounts = Accounts(_ok())
    tokens: list[str] = []

    def fetch() -> None:
        tokens.append(_store(path, accounts).access_token())

    threads = [threading.Thread(target=fetch) for _ in range(4)]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert tokens == ["acc-1"] * 4
    assert len(accounts.forms) == 1


def test_the_lock_file_is_private(tmp_path: Path) -> None:
    path = tmp_path / "token.json"
    _write(path, expires_at=START)

    _store(path, Accounts(_ok())).access_token()

    assert stat.S_IMODE((tmp_path / "token.json.lock").stat().st_mode) == 0o600


def test_has_login_follows_the_file(tmp_path: Path) -> None:
    path = tmp_path / "token.json"
    store = _store(path, Accounts(_ok()))

    assert not store.has_login()
    _write(path)
    assert store.has_login()


def test_has_login_is_true_for_a_file_that_is_not_text(tmp_path: Path) -> None:
    path = tmp_path / "token.json"
    path.write_bytes(b"\xff\xfe\x00garbage")
    path.chmod(0o600)
    store = _store(path, Accounts(_ok()))

    assert store.has_login()
    with pytest.raises(SpotifyAuthError, match="jarvis spotify-login"):
        store.access_token()


def test_a_busy_callback_port_is_an_auth_error_naming_the_config_key(tmp_path: Path) -> None:
    with socket.socket() as holder:
        holder.bind(("127.0.0.1", 0))
        holder.listen()
        port = int(holder.getsockname()[1])

        with pytest.raises(SpotifyAuthError, match=r"in use.*spotify\.redirect_port"):
            login("client", port, tmp_path / "token.json", lambda _url: None)


def test_a_state_with_non_ascii_characters_is_refused_not_a_crash(tmp_path: Path) -> None:
    accounts = Accounts(_ok())
    port = _free_port()
    seen: list[int] = []

    def show(url: str) -> None:
        seen.append(_get(port, "/callback?code=x&state=%C3%A9%C3%A9"))
        _get(port, f"/callback?code=c&state={_query(url)['state']}")

    login("client", port, tmp_path / "token.json", show, http=accounts.client())

    assert seen == [400]
    assert [form["code"] for form in accounts.forms] == ["c"]


def _eventually(check: Callable[[], bool]) -> bool:
    """Poll until `check` holds, for at most five seconds."""
    end = time.monotonic() + 5.0
    while time.monotonic() < end:
        if check():
            return True
        threading.Event().wait(0.005)
    return False


def _blocked_on(inode: int) -> bool:
    """Whether a process waits for a lock on the file with this inode, as the kernel lists it.

    `/proc/locks` shows a waiter as `N: -> FLOCK ADVISORY WRITE <pid> <major>:<minor>:<inode> ...`.
    """
    for line in Path("/proc/locks").read_text(encoding="ascii").splitlines():
        fields = line.split()
        if "->" in fields and fields[-3].rpartition(":")[2] == str(inode):
            return True
    return False


def test_a_login_waits_for_the_token_file_lock_before_it_writes(tmp_path: Path) -> None:
    # A running service refreshing the old grant holds this lock; the fresh login must not be
    # overwritten by it, so the login writes only once the lock is free.
    path = tmp_path / "token.json"
    _write(path, access_token="old-access", refresh_token="old-refresh", authorized_at=START - DAY)
    port = _free_port()
    posted = threading.Event()
    accounts = Accounts(_ok())

    def handle(request: httpx.Request) -> httpx.Response:
        answer = accounts.handle(request)
        posted.set()
        return answer

    def show(url: str) -> None:
        _get(port, f"/callback?code=c&state={_query(url)['state']}")

    client = httpx.Client(transport=httpx.MockTransport(handle))
    worker = threading.Thread(
        target=login,
        args=("client", port, path, show),
        kwargs={"http": client, "clock": lambda: START},
    )
    lock_fd = os.open(tmp_path / "token.json.lock", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        worker.start()
        assert posted.wait(5)
        assert _eventually(lambda: _blocked_on(os.fstat(lock_fd).st_ino))
        assert worker.is_alive()
        assert _saved(path)["access_token"] == "old-access"
    finally:
        os.close(lock_fd)
    worker.join(5)

    assert not worker.is_alive()
    assert _saved(path)["access_token"] == "acc-1"
    assert _saved(path)["authorized_at"] == START


def test_the_error_family_is_an_operation_failure() -> None:
    assert issubclass(SpotifyLoginExpired, SpotifyAuthError)
    assert issubclass(SpotifyAuthError, SpotifyError)
    assert issubclass(NoDeviceError, SpotifyError)
    assert issubclass(SpotifyError, OperationError)
    assert SpotifyError("x").origin is Origin.TOOL_STATUS
    assert SpotifyError("x", origin=Origin.SPOTIFY).origin is Origin.SPOTIFY
