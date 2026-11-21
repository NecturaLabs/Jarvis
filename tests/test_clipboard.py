"""Reading the clipboard for Ctrl+V: a picture before text, fixed arguments, a cap and a timeout.

A fake `wl-paste` stands in: a script that answers from files the test writes and logs how it
was called.
"""

import shlex
from typing import TYPE_CHECKING

import pytest

from jarvis.chat.clipboard import Clip, ClipboardError, read

if TYPE_CHECKING:
    from pathlib import Path


def _fake(tmp_path: Path, types: list[str], contents: dict[str, bytes], *, body: str = "") -> str:
    """Write a fake wl-paste that lists `types` and prints `contents[type]`; return its path."""
    (tmp_path / "types").write_text("".join(f"{t}\n" for t in types), encoding="utf-8")
    for number, (kind, data) in enumerate(contents.items()):
        (tmp_path / f"content{number}").write_bytes(data)
        (tmp_path / f"type{number}").write_text(kind, encoding="utf-8")
    log = shlex.quote(str(tmp_path / "calls"))
    here = shlex.quote(str(tmp_path))
    script = tmp_path / "wl-paste"
    script.write_text(
        f"""#!/bin/sh
echo "$*" >> {log}
{body}
if [ "$1" = "--list-types" ]; then
  if [ ! -s {here}/types ]; then echo "Nothing is copied" >&2; exit 1; fi
  cat {here}/types; exit 0
fi
want="$3"
for t in {here}/type*; do
  if [ "$(cat "$t")" = "$want" ]; then cat "{here}/content${{t##*type}}"; exit 0; fi
done
echo "No suitable type of content copied" >&2; exit 1
""",
        encoding="utf-8",
    )
    script.chmod(0o755)
    return str(script)


def _calls(tmp_path: Path) -> list[str]:
    return (tmp_path / "calls").read_text(encoding="utf-8").splitlines()


def test_a_copied_picture_comes_before_its_text(tmp_path: Path) -> None:
    program = _fake(
        tmp_path,
        ["text/plain", "image/png", "image/jpeg"],
        {"image/png": b"\x89PNG data", "text/plain": b"caption"},
    )

    clip = read(program)

    assert clip == Clip(image=b"\x89PNG data", mime="image/png")
    assert _calls(tmp_path) == ["--list-types", "--no-newline --type image/png"]


def test_copied_text_is_read_as_utf8(tmp_path: Path) -> None:
    program = _fake(
        tmp_path,
        ["TEXT", "text/plain;charset=utf-8", "text/plain"],
        {"text/plain;charset=utf-8": "héllo\nworld".encode()},
    )

    assert read(program) == Clip(text="héllo\nworld")


def test_a_copied_file_is_read_as_its_address(tmp_path: Path) -> None:
    program = _fake(tmp_path, ["text/uri-list"], {"text/uri-list": b"file:///home/a/shot.png\r\n"})

    assert read(program) == Clip(text="file:///home/a/shot.png\r\n")


def test_an_empty_clipboard_is_an_empty_clip(tmp_path: Path) -> None:
    program = _fake(tmp_path, [], {})

    assert read(program) == Clip()
    assert not read(program)


def test_a_clipboard_of_only_other_kinds_is_an_empty_clip(tmp_path: Path) -> None:
    program = _fake(tmp_path, ["application/x-blender"], {"application/x-blender": b"x"})

    assert read(program) == Clip()
    assert _calls(tmp_path) == ["--list-types"]


def test_more_than_the_cap_is_refused(tmp_path: Path) -> None:
    program = _fake(tmp_path, ["image/png"], {"image/png": b"x" * 5000})

    with pytest.raises(ClipboardError, match="too big"):
        read(program, max_bytes=1000)


def test_a_clipboard_that_does_not_answer_in_time_is_given_up(tmp_path: Path) -> None:
    program = _fake(tmp_path, ["text/plain"], {"text/plain": b"x"}, body="sleep 5")

    with pytest.raises(ClipboardError, match="didn't answer"):
        read(program, timeout_s=0.3)


def test_a_missing_wl_paste_says_what_to_install(tmp_path: Path) -> None:
    with pytest.raises(ClipboardError, match="wl-clipboard"):
        read(str(tmp_path / "nowhere" / "wl-paste"))


def test_a_failing_wl_paste_says_why(tmp_path: Path) -> None:
    program = _fake(
        tmp_path, [], {}, body='echo "Failed to connect to a Wayland server" >&2; exit 1'
    )

    with pytest.raises(ClipboardError, match="Wayland server"):
        read(program)
