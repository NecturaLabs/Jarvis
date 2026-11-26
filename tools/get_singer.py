"""Download SoulX-Singer and audio.cpp's runner for it into Jarvis's data folder, checked.

Run it yourself, once: `uv run tools/get_singer.py` (or `--data-dir PATH`). Jarvis never
downloads anything when it sings. Each file is fetched from its pinned upstream release over
HTTPS, checked against its size and sha256 before it is kept, and only the runner, its
libraries and its licence are taken out of the audio.cpp archive. About 1.3 GB in all.
"""

import argparse
import hashlib
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import IO

import httpx

from jarvis import config, models
from jarvis.sing import soulx

CHUNK = 1 << 20
TIMEOUT_S = 60.0
GGUF = (
    "https://huggingface.co/audio-cpp/SoulX-Singer-GGUF/resolve/"
    "ded36295f789661e590e4bf6ad49a25470cf830d/SoulX-Singer-GGUF/"
)
ARCHIVE = (
    "https://github.com/0xShug0/audio.cpp/releases/download/v0.9.6/"
    "audio-v0.9.6-bin-ubuntu-x64-vulkan-portable.tar.gz"
)


@dataclass(frozen=True, slots=True)
class Download:
    """One file of an upstream release: where it is, its size and its sha256."""

    url: str
    size: int
    sha256: str
    name: str  # what it is called in the singing folder


MODEL_FILES = (
    Download(
        GGUF + soulx.MODEL,
        1_173_681_632,
        "c146c09427f88202db37a71869cdd498e6c0e92fa2cd08065bd5c2f735eb82fd",
        soulx.MODEL,
    ),
    Download(
        GGUF + "LICENSE",
        11_357,
        "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
        "LICENSE",
    ),
    Download(
        GGUF + "LICENSE.amphion",
        1_064,
        "3086a4ff95a809b61b74db96318232f4fa8e9fcf0f222a58f3727d7315edeb9a",
        "LICENSE.amphion",
    ),
    Download(
        GGUF + "NOTICE",
        839,
        "d03bde0c699926e7a0bd6d45d923e25d8683861419e365d8d8f12f539d660818",
        "NOTICE",
    ),
)
RUNNER = Download(
    ARCHIVE,
    93_520_711,
    "9effac5d30a24b86a5de9a9ec37a5b2db77695590cb96ae5e6a580f5588be361",
    "audiocpp.tar.gz",
)


class DownloadError(Exception):
    """A file could not be fetched, or is not the release it should be."""


def _copy(item: Download, response: httpx.Response, out: IO[bytes]) -> str:
    """Write the body to `out`, never more than the release's size; return its sha256.

    Raises:
        DownloadError: the body is larger than the release.
    """
    digest = hashlib.sha256()
    size = 0
    for chunk in response.iter_bytes(CHUNK):
        size += len(chunk)
        if size > item.size:
            msg = f"{item.url} is larger than the release ({item.size} bytes)"
            raise DownloadError(msg)
        digest.update(chunk)
        out.write(chunk)
    if size != item.size:
        msg = f"{item.url} is {size} bytes, not the release's {item.size}"
        raise DownloadError(msg)
    return digest.hexdigest()


def _in_place(item: Download, target: Path) -> bool:
    """Whether `target` is already the pinned release: the right size and sha256."""
    try:
        return target.stat().st_size == item.size and models.sha256_file(target) == item.sha256
    except OSError:
        return False


def fetch(item: Download, folder: Path, client: httpx.Client) -> Path:
    """Put `item` in `folder`, checked, and return its path; a bad file is not kept.

    A file already there that is the pinned release is not downloaded again. Nothing but the
    finished file is left in `folder`, whatever stops the download (a Ctrl-C included).

    Raises:
        DownloadError: the server failed, or the size or sha256 is not the release's.
    """
    target = folder / item.name
    if _in_place(item, target):
        print(f"{item.name} is already in place")
        return target
    print(f"downloading {item.name} ({item.size:,} bytes)")
    with tempfile.NamedTemporaryFile(dir=folder, prefix=".part-", delete=False) as part:
        partial = Path(part.name)
        try:
            with client.stream("GET", item.url) as response:
                response.raise_for_status()
                digest = _copy(item, response, part)
        except httpx.HTTPError as exc:
            partial.unlink(missing_ok=True)
            msg = f"could not download {item.url}: {exc}"
            raise DownloadError(msg) from exc
        except BaseException:
            partial.unlink(missing_ok=True)
            raise
    if digest != item.sha256:
        partial.unlink(missing_ok=True)
        msg = f"{item.url} is not the pinned release (sha256 {digest})"
        raise DownloadError(msg)
    partial.replace(target)
    return target


def runner_in_place(binary: Path) -> bool:
    """Whether the runner is there as the pinned build, beside at least one of its libraries."""
    try:
        known = models.verify("soulx-binary", binary).known
    except OSError:
        return False
    return known is not None and any(binary.parent.glob("libggml*"))


def _wanted(name: str) -> bool:
    """Whether an archive member is the runner, one of its libraries or its licence."""
    base = name.removeprefix("./")
    return "/" not in base and (base in {soulx.BINARY, "LICENSE"} or base.startswith("libggml"))


def unpack(archive: Path, folder: Path) -> list[str]:
    """Take the runner, its libraries and licence out of `archive` into `folder`; return them.

    Only regular files at the archive's top level are taken (no links, no paths).

    Raises:
        DownloadError: the archive holds no runner.
    """
    taken: list[str] = []
    folder.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile() or not _wanted(member.name):
                continue
            source = tar.extractfile(member)
            if source is None:
                continue
            name = member.name.removeprefix("./")
            target = folder / name
            target.write_bytes(source.read())
            target.chmod(0o755 if name == soulx.BINARY else 0o644)
            taken.append(name)
    if soulx.BINARY not in taken:
        msg = f"{archive.name} holds no {soulx.BINARY}"
        raise DownloadError(msg)
    return sorted(taken)


def install(data_dir: Path) -> str:
    """Download, check and put every file in place under `data_dir`; return what was done.

    A file that could not be fetched or is not its release raises `DownloadError`, one that
    could not be written an OSError, and an archive that cannot be read a `tarfile.TarError`.
    """
    found = soulx.files(data_dir)
    folder = found.model.parent
    folder.mkdir(parents=True, exist_ok=True)
    with httpx.Client(follow_redirects=True, timeout=TIMEOUT_S) as client:
        for item in MODEL_FILES:
            fetch(item, folder, client)
        if runner_in_place(found.binary):
            print("the audio.cpp runner is already in place")
            return f"installed {soulx.MODEL}; the runner was already in place in {folder}"
        archive = fetch(RUNNER, folder, client)
    try:
        taken = unpack(archive, found.binary.parent)
    finally:
        archive.unlink(missing_ok=True)
    return f"installed {soulx.MODEL} and {len(taken)} runner files in {folder}"


def main(argv: list[str] | None = None) -> int:
    """Download and check the files; 0 when they are all in place."""
    parser = argparse.ArgumentParser(description="Download SoulX-Singer for Jarvis, checked.")
    parser.add_argument("--data-dir", type=Path, default=None, help="Jarvis's data folder")
    args = parser.parse_args(argv)
    try:
        done = install(args.data_dir or config.load().data_dir)
    except (DownloadError, OSError, tarfile.TarError) as exc:
        print(f"get_singer: {exc}", file=sys.stderr)
        return 1
    print(done)
    print("`jarvis doctor` checks them; `jarvis restart` makes Jarvis sing with them")
    return 0


if __name__ == "__main__":
    sys.exit(main())
