"""Pictures for the model: read, checked, shrunk and re-encoded before they are sent.

A picture comes from the owner (a file, the clipboard, a screenshot) or from the web. Either
way its bytes are someone else's, so only Pillow's decoders for a few common formats see them,
the pixel count is checked before anything is decoded, and what the model gets is a fresh PNG
(or JPEG, for a photo that stays large) at most `MAX_SIDE` pixels on its longest side. Strata's
picture reader turns each picture into at most `PICTURE_TOKENS` tokens of the model's context.
"""

import base64
import io
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import unquote, urlsplit

from PIL import Image, ImageOps, UnidentifiedImageError

if TYPE_CHECKING:
    from collections.abc import Iterable

PICTURE_TOKENS = 300  # what Strata's picture reader makes of one picture, at most
MAX_SIDE = 1568  # pixels on the longest side: enough to read a screenshot's text
MAX_PIXELS = 40_000_000  # refused before decoding: a small file can claim a huge picture
MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_PNG_BYTES = 3 * 1024 * 1024  # a picture still larger as PNG is a photo: sent as JPEG
JPEG_QUALITY = 85
FORMATS = ("PNG", "JPEG", "WEBP", "GIF", "BMP")
SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"})
FORMATS_SAID = "PNG, JPEG, WebP, GIF or BMP"


class PictureError(Exception):
    """The picture cannot be sent; the message says why in plain words."""


@dataclass(frozen=True, slots=True)
class Picture:
    """A checked picture, ready for the model: its name, media type, bytes and size."""

    name: str
    mime: str
    data: bytes
    width: int
    height: int

    def data_url(self) -> str:
        """Return the picture as a `data:` address, the way the model server takes it."""
        return f"data:{self.mime};base64,{base64.b64encode(self.data).decode('ascii')}"


def _decoded(data: bytes, name: str, max_pixels: int) -> Image.Image:
    """Return the picture's first frame, decoded, after its size was checked.

    Raises:
        PictureError: not a picture in `FORMATS`, too many pixels, or broken.
    """
    try:
        image = Image.open(io.BytesIO(data), formats=FORMATS)
    except (UnidentifiedImageError, ValueError) as exc:
        msg = f"{name} isn't a picture Jarvis reads ({FORMATS_SAID})"
        raise PictureError(msg) from exc
    width, height = image.size
    if width * height > max_pixels:
        msg = f"{name} is too large to read (over {max_pixels // 1_000_000} million pixels)"
        raise PictureError(msg)
    try:
        image.seek(0)
        image.load()
        upright = ImageOps.exif_transpose(image)
    except (OSError, SyntaxError, ValueError, Image.DecompressionBombError) as exc:
        msg = f"{name} is broken or cut short"
        raise PictureError(msg) from exc
    return upright


def _encode(image: Image.Image, fmt: str) -> bytes:
    buffer = io.BytesIO()
    if fmt == "JPEG":
        image.convert("RGB").save(buffer, "JPEG", quality=JPEG_QUALITY)
    else:
        image.save(buffer, "PNG")
    return buffer.getvalue()


def prepare(
    data: bytes,
    name: str,
    *,
    max_side: int = MAX_SIDE,
    max_pixels: int = MAX_PIXELS,
    max_png_bytes: int = MAX_PNG_BYTES,
) -> Picture:
    """Return `data` as a picture for the model: upright, shrunk to `max_side`, re-encoded.

    Bytes that are not a picture Jarvis reads, too large a picture or a broken one raise
    `PictureError`.
    """
    image = _decoded(data, name, max_pixels)
    transparent = image.mode in {"RGBA", "LA", "PA"} or "transparency" in image.info
    image = image.convert("RGBA" if transparent else "RGB")
    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    encoded, mime = _encode(image, "PNG"), "image/png"
    if len(encoded) > max_png_bytes and not transparent:
        encoded, mime = _encode(image, "JPEG"), "image/jpeg"
    return Picture(name, mime, encoded, image.width, image.height)


def read_file(path: Path, *, max_bytes: int = MAX_FILE_BYTES) -> Picture:
    """Return the picture in the file at `path`, named after the file.

    Raises:
        PictureError: no such file, not a regular file, over `max_bytes`, or not a picture.
    """
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC)
    except FileNotFoundError as exc:
        msg = f"There's no file at {path}"
        raise PictureError(msg) from exc
    except OSError as exc:
        msg = f"{path} can't be read: {exc.strerror or type(exc).__name__}"
        raise PictureError(msg) from exc
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or info.st_size > max_bytes:
        os.close(fd)
        msg = (
            f"{path} is not a file"
            if not stat.S_ISREG(info.st_mode)
            else f"{path.name} is too big to send (over {max_bytes // (1024 * 1024)} MB)"
        )
        raise PictureError(msg)
    with os.fdopen(fd, "rb") as handle:
        data = handle.read(max_bytes + 1)
    if len(data) > max_bytes:
        msg = f"{path.name} is too big to send (over {max_bytes // (1024 * 1024)} MB)"
        raise PictureError(msg)
    return prepare(data, path.name)


_QUOTES = ("'", '"')


def pasted_path(text: str, home: Path) -> Path | None:
    """Return the picture file that pasted or dropped `text` names, or None when it is text.

    A terminal pastes a dropped file as its path, quoted, with its spaces escaped, or as a
    `file://` address; `~/` means `home`. Only one absolute path to an existing picture file is
    a picture: anything else stays text.
    """
    value = text.strip()
    if not value or "\n" in value or "\r" in value:
        return None
    if len(value) > 1 and value[0] in _QUOTES and value[-1] == value[0]:
        value = value[1:-1]
    if value.startswith("file://"):
        parts = urlsplit(value)
        if parts.netloc not in {"", "localhost"}:
            return None
        value = unquote(parts.path)
    else:
        value = value.replace("\\ ", " ")
    if value.startswith("~/"):
        value = str(home / value[2:])
    path = Path(value)
    if not path.is_absolute() or path.suffix.lower() not in SUFFIXES:
        return None
    return path if path.is_file() else None


def newest(folders: Iterable[Path]) -> Path | None:
    """Return the most recently changed picture file directly in `folders`, or None."""
    found: list[tuple[float, Path]] = []
    for folder in folders:
        try:
            entries = list(os.scandir(folder))
        except OSError:
            continue
        for entry in entries:
            if Path(entry.name).suffix.lower() not in SUFFIXES:
                continue
            try:
                if entry.is_file():
                    found.append((entry.stat().st_mtime, Path(entry.path)))
            except OSError:
                continue
    return max(found)[1] if found else None
