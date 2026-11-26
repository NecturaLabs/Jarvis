"""The files `jarvis acp` keeps its sessions in, and the window that deletes them."""

import subprocess
import sys
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
