"""Pictures for the model: only real pictures in a few formats, shrunk, re-encoded and named."""

import base64
import io
import os
from typing import TYPE_CHECKING

import pytest
from PIL import Image

from jarvis.pictures import (
    MAX_SIDE,
    Picture,
    PictureCache,
    PictureError,
    newest,
    pasted_path,
    prepare,
    read_file,
)

if TYPE_CHECKING:
    from pathlib import Path


def _encoded(image: Image.Image, fmt: str = "PNG", **options: object) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, fmt, **options)
    return buffer.getvalue()


def _png(width: int = 40, height: int = 20, colour: str = "red") -> bytes:
    return _encoded(Image.new("RGB", (width, height), colour))


def _opened(picture: Picture) -> Image.Image:
    return Image.open(io.BytesIO(picture.data))


def test_a_small_picture_is_kept_at_its_size_as_png() -> None:
    picture = prepare(_png(40, 20), "shot.png")

    assert (picture.name, picture.mime, picture.width, picture.height) == (
        "shot.png",
        "image/png",
        40,
        20,
    )
    assert _opened(picture).format == "PNG"
    assert _opened(picture).getpixel((0, 0)) == (255, 0, 0)


def test_the_data_url_is_the_picture_in_base64() -> None:
    picture = prepare(_png(), "shot.png")

    head, _, body = picture.data_url().partition(",")
    assert head == "data:image/png;base64"
    assert base64.b64decode(body) == picture.data


def test_a_large_picture_is_shrunk_to_the_longest_side_keeping_its_shape() -> None:
    picture = prepare(_png(MAX_SIDE * 2, MAX_SIDE), "wide.png")

    assert (picture.width, picture.height) == (MAX_SIDE, MAX_SIDE // 2)
    assert _opened(picture).size == (MAX_SIDE, MAX_SIDE // 2)


def test_bytes_that_are_not_a_picture_are_refused() -> None:
    with pytest.raises(PictureError, match="isn't a picture"):
        prepare(b"%PDF-1.7 not a picture at all", "file.pdf")


def test_a_broken_picture_is_refused() -> None:
    with pytest.raises(PictureError, match="broken"):
        prepare(_png(200, 200)[:80], "cut.png")


def test_a_format_outside_the_list_is_refused() -> None:
    with pytest.raises(PictureError, match="isn't a picture"):
        prepare(_encoded(Image.new("RGB", (8, 8)), "TIFF"), "scan.tiff")


def test_a_picture_with_too_many_pixels_is_refused_before_it_is_decoded() -> None:
    with pytest.raises(PictureError, match="too large"):
        prepare(_png(100, 100), "huge.png", max_pixels=9_999)


def test_an_animated_picture_becomes_its_first_frame() -> None:
    frames = [Image.new("RGB", (16, 16), colour) for colour in ("blue", "green")]
    data = _encoded(frames[0], "GIF", save_all=True, append_images=frames[1:])

    picture = prepare(data, "wave.gif")

    assert picture.mime == "image/png"
    assert _opened(picture).convert("RGB").getpixel((0, 0)) == (0, 0, 255)


def test_a_photo_turned_by_its_camera_is_shown_upright() -> None:
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 degrees clockwise to view
    data = _encoded(Image.new("RGB", (40, 20), "white"), "JPEG", exif=exif)

    picture = prepare(data, "phone.jpg")

    assert (picture.width, picture.height) == (20, 40)


def test_a_picture_that_stays_large_as_png_is_sent_as_jpeg() -> None:
    noise = Image.frombytes("RGB", (300, 300), os.urandom(300 * 300 * 3))

    picture = prepare(_encoded(noise), "noise.png", max_png_bytes=10_000)

    assert picture.mime == "image/jpeg"
    assert _opened(picture).format == "JPEG"


def test_transparency_is_kept_in_a_png() -> None:
    data = _encoded(Image.new("RGBA", (10, 10), (0, 0, 255, 0)))

    picture = prepare(data, "icon.png")

    assert _opened(picture).mode == "RGBA"


def test_a_picture_file_is_read_and_named_after_the_file(tmp_path: Path) -> None:
    path = tmp_path / "Screenshot 1.png"
    path.write_bytes(_png())

    picture = read_file(path)

    assert picture.name == "Screenshot 1.png"
    assert picture.width == 40


def test_a_missing_file_says_so(tmp_path: Path) -> None:
    with pytest.raises(PictureError, match="no file"):
        read_file(tmp_path / "gone.png")


def test_a_folder_is_not_a_picture(tmp_path: Path) -> None:
    with pytest.raises(PictureError, match="not a file"):
        read_file(tmp_path)


def test_a_file_over_the_size_cap_is_refused_unread(tmp_path: Path) -> None:
    path = tmp_path / "big.png"
    path.write_bytes(_png(400, 400))

    with pytest.raises(PictureError, match="too big"):
        read_file(path, max_bytes=100)


def test_a_pasted_path_is_found_quoted_as_a_file_address_or_from_home(tmp_path: Path) -> None:
    (tmp_path / "Pictures").mkdir()
    shot = tmp_path / "Pictures" / "a shot.png"
    shot.write_bytes(_png())

    assert pasted_path(str(shot), tmp_path) == shot
    assert pasted_path(f"'{shot}'  ", tmp_path) == shot
    assert pasted_path(f'"{shot}"', tmp_path) == shot
    assert pasted_path(str(shot).replace(" ", "\\ "), tmp_path) == shot
    assert pasted_path(f"file://{str(shot).replace(' ', '%20')}\r\n", tmp_path) == shot
    assert pasted_path("~/Pictures/a shot.png", tmp_path) == shot


def test_pasted_text_that_is_not_one_existing_picture_is_left_as_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)  # so a relative name would find the file if it were read
    notes = tmp_path / "notes.txt"
    notes.write_text("hi", encoding="utf-8")
    shot = tmp_path / "shot.png"
    shot.write_bytes(_png())

    assert pasted_path("what's in this picture?", tmp_path) is None
    assert pasted_path(str(notes), tmp_path) is None
    assert pasted_path(str(tmp_path / "gone.png"), tmp_path) is None
    assert pasted_path(f"{shot}\n{shot}", tmp_path) is None
    assert pasted_path("shot.png", tmp_path) is None  # relative: no folder to read it from
    assert pasted_path("https://example.org/shot.png", tmp_path) is None


def test_the_newest_picture_in_the_folders_is_found(tmp_path: Path) -> None:
    first, second = tmp_path / "a", tmp_path / "b"
    first.mkdir()
    second.mkdir()
    old = first / "old.png"
    new = second / "new.jpg"
    later_text = first / "later.txt"
    for number, path in enumerate((old, new, later_text)):
        path.write_bytes(b"x")
        os.utime(path, (1000 + number, 1000 + number))

    assert newest([first, second, tmp_path / "missing"]) == new


def test_no_picture_in_the_folders_is_none(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")

    assert newest([tmp_path, tmp_path / "missing"]) is None


def test_the_cache_keeps_the_newest_pictures_privately_once_each(tmp_path: Path) -> None:
    cache = PictureCache(tmp_path / "shown", keep=2)
    pictures = [prepare(_png(10 + n, 10), f"p{n}") for n in range(3)]

    paths: list[Path] = []
    for number, picture in enumerate(pictures):
        path = cache.keep(picture)
        os.utime(path, (1000 + number, 1000 + number))
        paths.append(path)
    again = cache.keep(pictures[2])

    assert again == paths[2]
    assert sorted(p.name for p in (tmp_path / "shown").iterdir()) == sorted(
        p.name for p in paths[1:]
    )
    assert (tmp_path / "shown").stat().st_mode & 0o777 == 0o700
    assert paths[2].stat().st_mode & 0o777 == 0o600
    assert cache.holds(paths[2])
    assert not cache.holds(paths[0])
    assert not cache.holds(tmp_path / "shown" / ".." / "shown" / paths[2].name)
