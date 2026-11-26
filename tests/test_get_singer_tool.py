"""`tools/get_singer.py`: each download is checked before it is kept, and unpacked safely."""

import hashlib
import io
import tarfile
from typing import TYPE_CHECKING

import get_singer
import httpx
import pytest
from get_singer import Download, DownloadError, fetch, unpack

from jarvis import models
from jarvis.sing import soulx

if TYPE_CHECKING:
    from pathlib import Path

BODY = b"the pinned release"
REAL_CLIENT = httpx.Client  # a test replaces `httpx.Client`; this stays the real one


def _client(body: bytes = BODY, status: int = 200) -> httpx.Client:
    return httpx.Client(
        transport=httpx.MockTransport(lambda _r: httpx.Response(status, content=body))
    )


def _item(body: bytes = BODY) -> Download:
    return Download(
        "https://example.invalid/file", len(body), hashlib.sha256(body).hexdigest(), "file"
    )


def test_a_file_that_matches_its_pin_is_kept(tmp_path: Path) -> None:
    with _client() as client:
        path = fetch(_item(), tmp_path, client)

    assert path.read_bytes() == BODY
    assert [p.name for p in tmp_path.iterdir()] == ["file"]


@pytest.mark.parametrize(
    "served",
    [b"something else", BODY + b" and more", BODY[:-1]],
    ids=["digest", "longer", "shorter"],
)
def test_a_file_that_is_not_the_pinned_release_is_refused_and_not_kept(
    tmp_path: Path, served: bytes
) -> None:
    with _client(served) as client, pytest.raises(DownloadError):
        fetch(_item(), tmp_path, client)

    assert not list(tmp_path.iterdir())


def test_a_server_error_is_a_download_error_and_nothing_is_kept(tmp_path: Path) -> None:
    with _client(status=404) as client, pytest.raises(DownloadError, match="could not download"):
        fetch(_item(), tmp_path, client)

    assert not list(tmp_path.iterdir())


def _archive(tmp_path: Path, members: dict[str, bytes], links: tuple[str, ...] = ()) -> Path:
    path = tmp_path / "audiocpp.tar.gz"
    with tarfile.open(path, "w:gz") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        for name in links:
            info = tarfile.TarInfo(name)
            info.type = tarfile.SYMTYPE
            info.linkname = "/etc/passwd"
            tar.addfile(info)
    return path


def test_only_the_runner_its_libraries_and_licence_are_unpacked(tmp_path: Path) -> None:
    archive = _archive(
        tmp_path,
        {
            f"./{soulx.BINARY}": b"runner",
            "./libggml-vulkan.so": b"lib",
            "./LICENSE": b"licence",
            "./tools/convert.py": b"not needed",
            "./model_specs/soulx_singer.json": b"{}",
            "../escape": b"outside",
        },
        links=("./libggml.so.0",),
    )
    folder = tmp_path / "out"

    taken = unpack(archive, folder)

    assert taken == ["LICENSE", soulx.BINARY, "libggml-vulkan.so"]
    assert sorted(p.name for p in folder.iterdir()) == sorted(taken)
    assert (folder / soulx.BINARY).stat().st_mode & 0o111
    assert not (tmp_path / "escape").exists()


def test_an_archive_without_the_runner_is_refused(tmp_path: Path) -> None:
    with pytest.raises(DownloadError, match="holds no"):
        unpack(_archive(tmp_path, {"./LICENSE": b"licence"}), tmp_path / "out")


def _refusing() -> httpx.Client:
    """A client that fails the test when anything is fetched."""

    def refuse(request: httpx.Request) -> httpx.Response:
        pytest.fail(f"fetched {request.url}")

    return REAL_CLIENT(transport=httpx.MockTransport(refuse))


def test_a_file_already_in_place_with_the_right_sha256_is_not_downloaded_again(
    tmp_path: Path,
) -> None:
    (tmp_path / "file").write_bytes(BODY)

    with _refusing() as client:
        path = fetch(_item(), tmp_path, client)

    assert path == tmp_path / "file"
    assert [p.name for p in tmp_path.iterdir()] == ["file"]


@pytest.mark.parametrize("held", [b"something else", BODY + b" and more"], ids=["digest", "size"])
def test_a_file_in_place_that_is_not_the_pinned_release_is_replaced(
    tmp_path: Path, held: bytes
) -> None:
    (tmp_path / "file").write_bytes(held)

    with _client() as client:
        path = fetch(_item(), tmp_path, client)

    assert path.read_bytes() == BODY


def test_an_interrupted_download_leaves_no_part_file(tmp_path: Path) -> None:
    def interrupt(_request: httpx.Request) -> httpx.Response:
        raise KeyboardInterrupt

    with (
        httpx.Client(transport=httpx.MockTransport(interrupt)) as client,
        pytest.raises(KeyboardInterrupt),
    ):
        fetch(_item(), tmp_path, client)

    assert not list(tmp_path.iterdir())


def _runner(
    folder: Path, binary: bytes = b"runner", libs: tuple[str, ...] = ("libggml.so",)
) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for name in libs:
        (folder / name).write_bytes(b"lib")
    (folder / soulx.BINARY).write_bytes(binary)
    return folder / soulx.BINARY


def _pin(monkeypatch: pytest.MonkeyPatch, binary: bytes) -> None:
    known = models.KnownModel(
        "soulx-binary", "test", "v0", hashlib.sha256(binary).hexdigest(), "Apache-2.0"
    )
    monkeypatch.setattr(models, "KNOWN_MODELS", (known,))


def test_a_runner_in_place_with_its_libraries_and_the_pinned_hash_is_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _pin(monkeypatch, b"runner")

    assert get_singer.runner_in_place(_runner(tmp_path))


@pytest.mark.parametrize(
    ("binary", "libs"),
    [(b"another", ("libggml.so",)), (b"runner", ())],
    ids=["other-build", "no-libraries"],
)
def test_a_runner_that_is_not_the_pinned_one_or_has_no_libraries_is_fetched_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, binary: bytes, libs: tuple[str, ...]
) -> None:
    _pin(monkeypatch, b"runner")

    assert not get_singer.runner_in_place(_runner(tmp_path, binary, libs))


def test_no_runner_is_not_in_place(tmp_path: Path) -> None:
    assert not get_singer.runner_in_place(tmp_path / soulx.BINARY)


def test_an_install_with_everything_in_place_downloads_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = soulx.files(tmp_path).model.parent
    folder.mkdir(parents=True)
    _runner(folder / "audiocpp")
    _pin(monkeypatch, b"runner")
    files = tuple(
        Download(f"https://example.invalid/{n}", len(BODY), hashlib.sha256(BODY).hexdigest(), n)
        for n in ("a", "b")
    )
    for item in files:
        (folder / item.name).write_bytes(BODY)
    monkeypatch.setattr(get_singer, "MODEL_FILES", files)

    def refusing_client(**_options: object) -> httpx.Client:
        return _refusing()

    monkeypatch.setattr(get_singer.httpx, "Client", refusing_client)

    done = get_singer.install(tmp_path)

    assert "already" in done
