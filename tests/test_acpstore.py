"""The files `jarvis acp` keeps its sessions in, and the window that deletes them."""

import os
import subprocess
import sys
import time
from typing import TYPE_CHECKING

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
    assert [p.name for p in store.iterdir()] == ["notes.txt"]
    assert acpstore.forget_kept(tmp_path / "none") == 0


def _temp(store: Path, name: str, age_s: float = 0.0) -> Path:
    path = store / name
    path.write_text("{}", encoding="utf-8")
    then = time.time() - age_s
    os.utime(path, (then, then))
    return path


STORE_TEMPS = (
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

    assert sorted(p.name for p in store.iterdir()) == sorted(OTHER_TEMPS)


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
