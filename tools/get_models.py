"""Download the wake word, VAD and Kokoro files Jarvis's config uses into its data folder, checked.

Run it yourself, once, from the checkout: `uv run tools/get_models.py` (the installer runs it).
Jarvis itself never downloads a model. Each file comes from its pinned upstream release (`URLS`,
with its size and sha256 in `jarvis.models.KNOWN_MODELS`) over HTTPS, following a
redirect only to a host in `HOSTS`. It is written to `<name>.part` beside its place in `data_dir`,
never past the release's size, carried on from where an earlier run stopped, and renamed into place
only once its size and sha256 match the pin. A file already in place that matches is left alone;
one that does not is never overwritten (it may be your own build): it is named, and left to you.
Only the file's address is sent: no account, token, cookie or proxy, and httpx's own user agent.
"""

import argparse
import contextlib
import dataclasses
import fcntl
import os
import re
import shutil
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal
from urllib.parse import urlsplit

import httpx

from jarvis import cli, config, models

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Sequence
    from pathlib import Path

    from jarvis.models import KnownModel

# Where each release Jarvis can fetch publishes its file, by `KnownModel.role`.
KOKORO_RELEASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/"
URLS = {
    "wake": "https://github.com/livekit/livekit-wakeword/releases/download/v2.1.3/hey_jarvis.onnx",
    "vad": (
        "https://raw.githubusercontent.com/snakers4/silero-vad/v6.2.3/"
        "src/silero_vad/data/silero_vad.onnx"
    ),
    "kokoro-model": KOKORO_RELEASE + "kokoro-v1.0.onnx",
    "kokoro-voices": KOKORO_RELEASE + "voices-v1.0.bin",
}

# GitHub serves a release asset from github.com through a redirect to its asset host, and a file in
# a repository from raw.githubusercontent.com.
HOSTS = frozenset(
    {
        "github.com",
        "raw.githubusercontent.com",
        "release-assets.githubusercontent.com",
        "objects.githubusercontent.com",
    }
)
MAX_REDIRECTS = 5
CHUNK = 1 << 20
TIMEOUT_S = 30.0
_RANGE = re.compile(r"bytes (\d+)-\d+/(\d+|\*)")

type State = Literal["present", "fetched", "different", "failed"]


class DownloadError(Exception):
    """A file could not be fetched, or what came is not the release it should be."""


@dataclass(frozen=True, slots=True)
class Outcome:
    """What became of one file."""

    model: KnownModel
    path: Path
    state: State
    detail: str = ""


def downloads(table: Sequence[KnownModel] | None = None) -> list[KnownModel]:
    """Return the releases this tool can fetch: those with a place in `data_dir` and a URL."""
    rows = models.KNOWN_MODELS if table is None else table
    return [
        m if m.url else dataclasses.replace(m, url=URLS[m.role])
        for m in rows
        if m.file and (m.url or m.role in URLS)
    ]


def needed(
    cfg: config.Config, table: Sequence[KnownModel] | None = None
) -> list[tuple[KnownModel, Path]]:
    """Return each release Jarvis can fetch that the config uses, with where it goes.

    A wake word model only when `audio.wake_words` turns it on, the Kokoro files only for a Kokoro
    voice, the VAD always.
    """
    kokoro = cfg.audio.voice.startswith("kokoro:")
    wanted: list[tuple[KnownModel, Path]] = []
    for model in downloads(table):
        name = model.file.rsplit("/", 1)[-1].removesuffix(".onnx")
        if model.role == "wake" and name not in cfg.audio.wake_words:
            continue
        if model.role.startswith("kokoro-") and not kokoro:
            continue
        wanted.append((model, cfg.data_dir / model.file))
    return wanted


def client() -> httpx.Client:
    """Return the client the downloads use: it follows no redirect itself and takes no proxy."""
    return httpx.Client(timeout=TIMEOUT_S, follow_redirects=False, trust_env=False)


def check_url(url: str) -> None:
    """Refuse an address that is not HTTPS on the default port of a host in `HOSTS`.

    Raises:
        DownloadError: naming what is wrong with it.
    """
    parts = urlsplit(url)
    if parts.scheme != "https" or parts.username is not None or parts.password is not None:
        msg = f"refused {url}: only plain https addresses are fetched"
        raise DownloadError(msg)
    if parts.hostname not in HOSTS or parts.port not in {None, 443}:
        msg = f"refused {url}: {parts.hostname} is not a host the releases are published on"
        raise DownloadError(msg)


def fetch_all(
    wanted: Sequence[tuple[KnownModel, Path]], http: httpx.Client, show: Callable[[str], None]
) -> list[Outcome]:
    """Fetch each file in `wanted` that is not in place yet; return what became of each."""
    outcomes: list[Outcome] = []
    for model, path in wanted:
        try:
            outcomes.append(fetch(model, path, http, show))
        except DownloadError as exc:
            outcomes.append(Outcome(model, path, "failed", str(exc)))
        except OSError as exc:
            outcomes.append(Outcome(model, path, "failed", f"{path}: {exc.strerror or exc}"))
    return outcomes


def fetch(
    model: KnownModel, path: Path, http: httpx.Client, show: Callable[[str], None]
) -> Outcome:
    """Put the release's file at `path`, unless it is there already.

    `OSError` comes from a file that cannot be written. A failed or mismatched transfer raises
    `DownloadError` from the transfer or verification step.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_name(f".{path.name}.download.lock")
    fd = os.open(lock, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        return _fetch_locked(model, path, http, show)
    finally:
        os.close(fd)


def _existing(model: KnownModel, path: Path) -> Outcome | None:
    if path.is_file():
        if models.sha256_file(path) == model.sha256:
            return Outcome(model, path, "present")
    elif not path.exists() and not path.is_symlink():
        return None
    return Outcome(
        model,
        path,
        "different",
        "a file is there that is not this release; it is left as it is (move it away to fetch it)",
    )


def _fetch_locked(
    model: KnownModel, path: Path, http: httpx.Client, show: Callable[[str], None]
) -> Outcome:
    if (existing := _existing(model, path)) is not None:
        return existing
    part = path.with_name(path.name + ".part")
    if part.is_symlink() or (part.exists() and not part.is_file()):
        msg = f"{part} is not a regular partial download; move it out of the way"
        raise DownloadError(msg)
    have = part.stat().st_size if part.is_file() else 0
    if have > model.size:
        part.unlink()
        have = 0
    if have < model.size:
        free = shutil.disk_usage(path.parent).free
        if free < model.size - have:
            need = (model.size - have) // 2**20
            msg = f"{path.parent} has {free // 2**20} MiB free; {model.file} needs {need} MiB"
            raise DownloadError(msg)
        host = urlsplit(model.url).hostname
        resumed = f", from {have // 2**20} MiB" if have else ""
        show(f"fetching {model.file} ({model.size / 2**20:.0f} MiB) from {host}{resumed}")
        _get(model, part, have, http)
    try:
        _check(model, part)
    except DownloadError:
        part.unlink(missing_ok=True)
        raise
    try:
        os.link(part, path, follow_symlinks=False)
    except FileExistsError:
        # A user can place their own model while a transfer is in progress.
        return _existing(model, path) or Outcome(
            model, path, "failed", "destination changed; retry"
        )
    part.unlink()
    return Outcome(model, path, "fetched")


def _check(model: KnownModel, part: Path) -> None:
    """Return when `part` has the release's size and sha256.

    Raises:
        DownloadError: it has not.
    """
    size = part.stat().st_size
    if size != model.size:
        msg = f"{model.file}: got {size} bytes, the release has {model.size}; run it again"
        raise DownloadError(msg)
    digest = models.sha256_file(part)
    if digest != model.sha256:
        msg = f"{model.file}: sha256 {digest} is not the release's {model.sha256}; nothing was kept"
        raise DownloadError(msg)


def _get(model: KnownModel, part: Path, have: int, http: httpx.Client) -> None:
    """Write the release's bytes after the first `have` into `part`, following checked redirects.

    Raises:
        DownloadError: an address is refused, the server fails, or redirects go on too long.
    """
    url = model.url
    for _hop in range(MAX_REDIRECTS + 1):
        check_url(url)
        http.cookies.clear()  # nothing one host set goes to the next
        headers = {"Range": f"bytes={have}-"} if have else {}
        with _stream(http, url, headers) as response:
            if response.is_redirect:
                url = str(response.url.join(response.headers.get("location", "")))
                continue
            offset = _offset(response, have)
            _write(response, part, offset, model.size)
            return
    msg = f"{model.file}: more than {MAX_REDIRECTS} redirects from {urlsplit(model.url).hostname}"
    raise DownloadError(msg)


@contextlib.contextmanager
def _stream(http: httpx.Client, url: str, headers: dict[str, str]) -> Generator[httpx.Response]:
    """Open a GET of `url`; a transport failure is a `DownloadError`.

    Raises:
        DownloadError: the request failed.
    """
    try:
        with http.stream("GET", url, headers=headers) as response:
            yield response
    except httpx.HTTPError as exc:
        msg = f"{urlsplit(url).hostname}: {exc}"
        raise DownloadError(msg) from exc


def _offset(response: httpx.Response, have: int) -> int:
    """Return where the answer's bytes go in the file: 0 for the whole file, else `have`.

    Raises:
        DownloadError: the server answered with anything but the file or the rest of it.
    """
    status = response.status_code
    if status == httpx.codes.OK.value:
        return 0
    if status == httpx.codes.PARTIAL_CONTENT.value and have:
        found = _RANGE.fullmatch(response.headers.get("content-range", ""))
        if found is not None and int(found[1]) == have:
            return have
        msg = f"{response.url.host} sent another part of the file than the one asked for"
        raise DownloadError(msg)
    msg = f"{response.url.host} answered {status} for {response.url.path}"
    raise DownloadError(msg)


def _write(response: httpx.Response, part: Path, offset: int, size: int) -> None:
    """Write the body into `part` from `offset`, never past `size`.

    Raises:
        DownloadError: the body is longer than the release (the part is deleted), or it broke off
            (the part is kept, to carry on from).
    """
    flags = os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC
    fd = os.open(part, flags | (os.O_APPEND if offset else os.O_TRUNC), 0o644)
    with os.fdopen(fd, "wb") as out:
        written = offset
        for chunk in _body(response):
            written += len(chunk)
            if written > size:
                part.unlink()
                msg = f"{part.name}: the server sent more than the release's {size} bytes"
                raise DownloadError(msg)
            out.write(chunk)


def _body(response: httpx.Response) -> Generator[bytes]:
    """Yield the body in chunks; a transfer that breaks off is a `DownloadError`.

    Raises:
        DownloadError: the transfer broke off.
    """
    try:
        yield from response.iter_bytes(CHUNK)
    except httpx.HTTPError as exc:
        msg = f"{response.url.host}: {exc}; run it again to carry on"
        raise DownloadError(msg) from exc


SAID = {
    "present": "sha256 verified (already there)",
    "fetched": "sha256 verified (downloaded)",
}


def main(argv: Sequence[str] | None = None) -> int:
    """Fetch the model files the config uses; return 1 if one failed."""
    parser = argparse.ArgumentParser(description="Fetch the model files the config uses, checked.")
    parser.parse_args(argv)
    try:
        cfg = config.load()
    except config.ConfigError as exc:
        print(f"get_models: {exc}", file=sys.stderr)
        return 1
    wanted = needed(cfg)
    with client() as http:
        outcomes = fetch_all(wanted, http, print)
    for done in outcomes:
        release = f"{done.model.project} {done.model.version} ({done.model.licence})"
        if done.state in SAID:
            print(f"ok   {done.path}: {release}, {SAID[done.state]}")
        else:
            print(f"FAIL {done.path}: {done.detail}")
    fetched = {model.file for model, _path in wanted}
    for path in cli.wake_files(cfg):
        if f"{cli.WAKE_DIR}/{path.name}" not in fetched and not path.is_file():
            print(f"note wake word {path.stem} has no release to fetch: train it (docs/models.md)")
    if not cfg.audio.voice.startswith("kokoro:"):
        print("note a Piper voice is not fetched: put it in place as docs/models.md says")
    return 0 if all(done.state in SAID for done in outcomes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
