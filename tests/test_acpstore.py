"""The files `jarvis acp` keeps its sessions in, and the window that deletes them."""

import os
import subprocess
import sys
import time
from typing import TYPE_CHECKING

import pytest

from jarvis import acpstore

if TYPE_CHECKING:
    from pathlib import Path


def test_the_chat_window_does_not_load_the_acp_server_to_delete_kept_sessions() -> None:
    program = (
        "import sys, jarvis.chat.app\n"
        "assert 'jarvis.acp' not in sys.modules\n"
        "assert 'jarvis.acpstore' in sys.modules\n"
    )
    done = subprocess.run(
        [sys.executable, "-I", "-c", program], capture_output=True, text=True, check=False
    )
    assert done.returncode == 0, done.stderr


def test_forget_kept_deletes_only_the_sessions_files(tmp_path: Path) -> None:
    store = tmp_path / "acp"
    store.mkdir()
    for name in (
        "jarvis-0123456789ab.json",
        "jarvis-0123456789ab.chat.memory.json",
        "jarvis-ba9876543210.json",
        "notes.txt",
    ):
        (store / name).write_text("{}", encoding="utf-8")

    assert acpstore.forget_kept(store) == 2
    assert sorted(p.name for p in store.iterdir()) == [acpstore.FORGOTTEN, "notes.txt"]
    assert acpstore.forget_kept(tmp_path / "none") == 0


def _temp(store: Path, name: str, age_s: float = 0.0) -> Path:
    path = store / name
    path.write_text("{}", encoding="utf-8")
    then = time.time() - age_s
    os.utime(path, (then, then))
    return path


STORE_TEMPS = (
    ".forgotten.k3j_x9a2.tmp",
    ".jarvis-0123456789ab.json.k3j_x9a2.tmp",
    ".jarvis-0123456789ab.chat.memory.json.8dkf0q1z.tmp",
)
OTHER_TEMPS = (  # not what `write_atomic` makes for a session's file
    ".notes.txt.k3j_x9a2.tmp",
    ".jarvis-0123456789ab.json.tmp",
    ".jarvis-0123456789ab.json.k3j_x9a2",
    ".jarvis-not-a-session.json.k3j_x9a2.tmp",
)


def test_forget_kept_removes_the_temp_files_a_crash_left_in_the_store(tmp_path: Path) -> None:
    store = tmp_path / "acp"
    store.mkdir()
    for name in (*STORE_TEMPS, *OTHER_TEMPS):
        _temp(store, name)

    acpstore.forget_kept(store)

    assert sorted(p.name for p in store.iterdir()) == sorted([*OTHER_TEMPS, acpstore.FORGOTTEN])


def test_prune_removes_only_temp_files_old_enough_not_to_be_a_writers(tmp_path: Path) -> None:
    store = tmp_path / "acp"
    store.mkdir()
    stale = [_temp(store, name, acpstore.STALE_TEMP_S + 60) for name in STORE_TEMPS]
    fresh = _temp(store, ".jarvis-ba9876543210.json.k3j_x9a2.tmp", acpstore.STALE_TEMP_S - 60)
    other = _temp(store, OTHER_TEMPS[0], acpstore.STALE_TEMP_S + 60)

    acpstore.prune(store)

    assert not any(path.exists() for path in stale)
    assert fresh.exists()
    assert other.exists()


def test_forget_kept_stamps_when_it_forgot_and_leaves_the_stamp_for_the_next_time(
    tmp_path: Path,
) -> None:
    store = tmp_path / "acp"
    assert not acpstore.forgotten_at(store)
    before = time.time()

    acpstore.forget_kept(store)  # nothing was kept, yet a running session must still know

    stamp = acpstore.forgotten_at(store)
    assert before <= stamp <= time.time()
    assert (store.stat().st_mode & 0o777) == 0o700
    assert ((store / acpstore.FORGOTTEN).stat().st_mode & 0o777) == 0o600
    acpstore.prune(store)
    assert acpstore.forgotten_at(store) == stamp


def test_a_forget_that_failed_leaves_the_earlier_stamp(tmp_path: Path) -> None:
    store = tmp_path / "acp"
    acpstore.forget_kept(store)
    earlier = acpstore.forgotten_at(store)
    (store / "jarvis-0123456789ab.json").mkdir()  # a folder: it cannot be unlinked

    with pytest.raises(IsADirectoryError):
        acpstore.forget_kept(store)

    assert acpstore.forgotten_at(store) == earlier


def test_an_unreadable_stamp_is_no_forgetting(tmp_path: Path) -> None:
    store = tmp_path / "acp"
    store.mkdir()
    (store / acpstore.FORGOTTEN).write_text("not a time", encoding="utf-8")

    assert not acpstore.forgotten_at(store)


def test_a_first_forget_that_failed_leaves_no_stamp(tmp_path: Path) -> None:
    store = tmp_path / "acp"
    (store / "jarvis-0123456789ab.json").mkdir(parents=True)

    with pytest.raises(IsADirectoryError):
        acpstore.forget_kept(store)

    assert not acpstore.forgotten_at(store)
    assert not (store / acpstore.FORGOTTEN).exists()
