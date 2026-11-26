"""`tools/get_models.py`: pinned releases fetched over a fake transport and checked before kept."""

import hashlib
from typing import TYPE_CHECKING

import get_models as download
import httpx
import pytest

from jarvis import cli, config
from jarvis.models import KnownModel

if TYPE_CHECKING:
    from pathlib import Path

    from jarvis.config import Config

BODY = b"model bytes " * 1000
ASSET = "https://github.com/example/release/download/v1/model.onnx"
CDN = "https://release-assets.githubusercontent.com/asset/1?sig=x"


def _model(
    body: bytes = BODY, *, url: str = ASSET, file: str = "vad/silero_vad.onnx"
) -> KnownModel:
    return KnownModel(
        role="vad",
        project="Example VAD",
        version="1",
        sha256=hashlib.sha256(body).hexdigest(),
        licence="MIT",
        url=url,
        size=len(body),
        file=file,
    )


class Server:
    """A fake release host: a redirect from the asset address to its CDN, which honours Range."""

    def __init__(self, body: bytes = BODY, *, ranges: bool = True) -> None:
        super().__init__()
        self.body = body
        self.ranges = ranges
        self.requests: list[httpx.Request] = []
        self.redirect_to = CDN

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if str(request.url) == ASSET:
            return httpx.Response(302, headers={"location": self.redirect_to, "set-cookie": "s=1"})
        asked = request.headers.get("range")
        if asked and self.ranges:
            start = int(asked.removeprefix("bytes=").removesuffix("-"))
            rest = self.body[start:]
            end = len(self.body) - 1
            return httpx.Response(
                206,
                content=rest,
                headers={"content-range": f"bytes {start}-{end}/{len(self.body)}"},
            )
        return httpx.Response(200, content=self.body)


def _client(server: Server) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(server), follow_redirects=False)


def test_a_file_is_fetched_through_a_redirect_checked_and_put_in_place(tmp_path: Path) -> None:
    server = Server()
    model = _model()
    path = tmp_path / model.file

    with _client(server) as http:
        done = download.fetch(model, path, http, lambda _line: None)

    assert done.state == "fetched"
    assert path.read_bytes() == BODY
    assert not path.with_name(path.name + ".part").exists()
    assert [str(r.url) for r in server.requests] == [ASSET, CDN]
    # Nothing the first host set is sent on to the next, and nothing names the user.
    assert "cookie" not in server.requests[1].headers
    assert "authorization" not in server.requests[1].headers


def test_a_redirect_to_a_host_outside_the_list_is_refused(tmp_path: Path) -> None:
    server = Server()
    server.redirect_to = "https://downloads.example.com/model.onnx"
    model = _model()

    with _client(server) as http, pytest.raises(download.DownloadError, match="not a host"):
        download.fetch(model, tmp_path / model.file, http, lambda _line: None)

    assert len(server.requests) == 1


def test_a_redirect_to_plain_http_is_refused(tmp_path: Path) -> None:
    server = Server()
    server.redirect_to = "http://release-assets.githubusercontent.com/asset/1"
    model = _model()

    with _client(server) as http, pytest.raises(download.DownloadError, match="plain https"):
        download.fetch(model, tmp_path / model.file, http, lambda _line: None)


def test_a_file_whose_digest_differs_is_not_kept(tmp_path: Path) -> None:
    model = _model(b"what the release should hold " * 400)
    server = Server(b"x" * model.size)
    path = tmp_path / model.file

    with _client(server) as http, pytest.raises(download.DownloadError, match="sha256"):
        download.fetch(model, path, http, lambda _line: None)

    assert not path.exists()
    assert not path.with_name(path.name + ".part").exists()


def test_a_server_that_sends_more_than_the_release_is_cut_off(tmp_path: Path) -> None:
    model = _model()
    server = Server(BODY + b"more")
    path = tmp_path / model.file

    with _client(server) as http, pytest.raises(download.DownloadError, match="more than"):
        download.fetch(model, path, http, lambda _line: None)

    assert not path.exists()


def test_an_interrupted_download_carries_on_from_where_it_stopped(tmp_path: Path) -> None:
    model = _model()
    path = tmp_path / model.file
    path.parent.mkdir(parents=True)
    path.with_name(path.name + ".part").write_bytes(BODY[:5000])
    server = Server()

    with _client(server) as http:
        done = download.fetch(model, path, http, lambda _line: None)

    assert done.state == "fetched"
    assert path.read_bytes() == BODY
    assert server.requests[-1].headers["range"] == "bytes=5000-"


def test_a_server_that_ignores_the_range_sends_the_whole_file_again(tmp_path: Path) -> None:
    model = _model()
    path = tmp_path / model.file
    path.parent.mkdir(parents=True)
    path.with_name(path.name + ".part").write_bytes(BODY[:5000])

    with _client(Server(ranges=False)) as http:
        download.fetch(model, path, http, lambda _line: None)

    assert path.read_bytes() == BODY


def test_a_file_already_in_place_is_not_fetched_again(tmp_path: Path) -> None:
    model = _model()
    path = tmp_path / model.file
    path.parent.mkdir(parents=True)
    path.write_bytes(BODY)
    server = Server()

    with _client(server) as http:
        done = download.fetch(model, path, http, lambda _line: None)

    assert done.state == "present"
    assert server.requests == []


def test_a_different_file_in_place_is_never_overwritten(tmp_path: Path) -> None:
    model = _model()
    path = tmp_path / model.file
    path.parent.mkdir(parents=True)
    path.write_bytes(b"my own build")
    server = Server()

    with _client(server) as http:
        done = download.fetch(model, path, http, lambda _line: None)

    assert done.state == "different"
    assert path.read_bytes() == b"my own build"
    assert server.requests == []


def test_only_https_on_the_listed_hosts_is_fetched() -> None:
    for url in (
        "https://github.com/a/b",
        "https://raw.githubusercontent.com/a/b",
        "https://release-assets.githubusercontent.com/a",
    ):
        download.check_url(url)
    for url in (
        "http://github.com/a",
        "https://github.com:8443/a",
        "https://user:pw@example.com/a",
        "https://github.com.example.net/a",
        "file:///etc/passwd",
    ):
        with pytest.raises(download.DownloadError):
            download.check_url(url)


def test_every_pinned_download_is_https_on_a_listed_host_with_a_size_and_a_place() -> None:
    pinned = download.downloads()
    assert {m.role for m in pinned} == {"wake", "vad", "kokoro-model", "kokoro-voices"}
    for model in pinned:
        download.check_url(model.url)
        assert model.size > 0
        assert not model.file.startswith("/")
        assert ".." not in model.file


def test_the_files_needed_follow_the_wake_words_and_the_voice(tmp_path: Path) -> None:
    cfg = config.Config(data_dir=tmp_path)
    assert {m.file for m, _p in download.needed(cfg)} == {
        "wake/hey_jarvis.onnx",
        "vad/silero_vad.onnx",
        "voices/kokoro/kokoro-v1.0.onnx",
        "voices/kokoro/voices-v1.0.bin",
    }
    piper = config.Config(
        data_dir=tmp_path,
        audio=config.AudioConfig(voice="en_GB-alan-medium", wake_words=["yo_jarvis"]),
    )
    assert [m.file for m, _p in download.needed(piper)] == ["vad/silero_vad.onnx"]


def test_the_files_go_where_the_doctor_and_the_voice_look_for_them(tmp_path: Path) -> None:
    from jarvis.audio import voice_files  # ruff: ignore[import-outside-top-level] - loads ONNX Runtime

    cfg = config.Config(data_dir=tmp_path)
    places = {p for _m, p in download.needed(cfg)}
    assert {
        *cli.wake_files(cfg)[:1],
        tmp_path / cli.VAD_MODEL,
        *voice_files(tmp_path, cfg.audio.voice),
    } == places


def test_the_command_reports_each_file_and_fails_when_one_failed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    good = _model()
    bad = _model(b"other", url=ASSET, file="wake/hey_jarvis.onnx")
    config.save(
        config.Config(data_dir=tmp_path, audio=config.AudioConfig(wake_words=["hey_jarvis"]))
    )

    def needed(cfg: Config) -> list[tuple[KnownModel, Path]]:
        return [(good, cfg.data_dir / good.file), (bad, cfg.data_dir / bad.file)]

    monkeypatch.setattr(download, "needed", needed)
    monkeypatch.setattr(download, "client", lambda: _client(Server()))

    assert download.main([]) == 1

    out = capsys.readouterr().out
    assert f"ok   {tmp_path / good.file}: Example VAD 1 (MIT), sha256 verified (downloaded)" in out
    assert f"FAIL {tmp_path / bad.file}" in out


def test_a_model_placed_at_the_destination_during_download_is_preserved(tmp_path: Path) -> None:
    model = _model()
    path = tmp_path / model.file
    local = b"user's own model"

    def serve(_request: httpx.Request) -> httpx.Response:
        path.write_bytes(local)
        return httpx.Response(200, content=BODY)

    with httpx.Client(transport=httpx.MockTransport(serve)) as http:
        outcome = download.fetch(model, path, http, lambda _line: None)

    assert outcome.state == "different"
    assert path.read_bytes() == local
