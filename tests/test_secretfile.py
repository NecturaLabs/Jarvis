"""Private files: mode 0600, atomic writes, and a refusal to read anything less private."""

import os
import stat
from typing import TYPE_CHECKING

import pytest

from jarvis.secretfile import (
    ensure_api_key,
    read_api_key,
    read_private,
    write_private,
    write_private_json,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_a_written_secret_is_private_in_a_private_folder(tmp_path: Path) -> None:
    path = tmp_path / "state" / "token.json"

    write_private_json(path, {"a": 1})

    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert read_private(path) == '{\n "a": 1\n}'


def test_a_rewrite_replaces_the_file_and_leaves_no_temporary_file(tmp_path: Path) -> None:
    path = tmp_path / "key"
    write_private(path, "one")

    write_private(path, "two")

    assert read_private(path) == "two"
    assert [p.name for p in tmp_path.iterdir()] == ["key"]


def test_a_missing_secret_reads_as_none(tmp_path: Path) -> None:
    assert read_private(tmp_path / "nothing") is None


@pytest.mark.parametrize("mode", [0o640, 0o604, 0o660])
def test_a_secret_others_can_read_is_refused(tmp_path: Path, mode: int) -> None:
    path = tmp_path / "token"
    write_private(path, "secret")
    path.chmod(mode)

    with pytest.raises(PermissionError):
        read_private(path)


def test_a_symlinked_secret_is_refused(tmp_path: Path) -> None:
    real = tmp_path / "real"
    write_private(real, "secret")
    link = tmp_path / "link"
    link.symlink_to(real)

    with pytest.raises(PermissionError):
        read_private(link)


def test_a_secret_owned_by_someone_else_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "token"
    write_private(path, "secret")
    monkeypatch.setattr(os, "getuid", lambda: path.stat().st_uid + 1)

    with pytest.raises(PermissionError):
        read_private(path)


def test_the_api_key_is_created_once_and_read_back(tmp_path: Path) -> None:
    path = tmp_path / "strata.env"

    key = ensure_api_key(path)

    assert key
    assert ensure_api_key(path) == key
    assert read_api_key(path) == key


def test_adding_the_api_key_keeps_the_files_other_settings(tmp_path: Path) -> None:
    path = tmp_path / "strata.env"
    write_private(path, "# owner's notes\nOTHER=1\n")

    key = ensure_api_key(path)

    text = read_private(path) or ""
    assert "OTHER=1" in text.splitlines()
    assert "# owner's notes" in text.splitlines()
    assert read_api_key(path) == key
