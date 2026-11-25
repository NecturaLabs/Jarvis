"""`tools/get_singer.py`: each download is checked before it is kept, and unpacked safely."""

import hashlib
import io
import tarfile
from typing import TYPE_CHECKING

import httpx
import pytest
from get_singer import Download, DownloadError, fetch, unpack

from jarvis.sing import soulx

if TYPE_CHECKING:
    from pathlib import Path

BODY = b"the pinned release"


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
