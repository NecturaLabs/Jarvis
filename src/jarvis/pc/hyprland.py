"""Hyprland's own socket: what the window and program tools ask the compositor.

Jarvis talks to `$XDG_RUNTIME_DIR/hypr/$HYPRLAND_INSTANCE_SIGNATURE/.socket.sock` directly, the
socket `hyprctl` uses, so reading windows or placing one starts no program of Jarvis's own. Every
answer is read whole, up to `MAX_HYPR_BYTES`, within `HYPR_TIMEOUT_S`.
"""

import json
import os
import socket
from pathlib import Path

from jarvis.jsonutil import objs

HYPR_TIMEOUT_S = 0.5
MAX_HYPR_BYTES = 4 * 1024 * 1024


def socket_path() -> Path | None:
    """Return this session's Hyprland socket, None when Jarvis runs outside Hyprland."""
    runtime = os.environ.get("XDG_RUNTIME_DIR", "")
    signature = os.environ.get("HYPRLAND_INSTANCE_SIGNATURE", "")
    if not runtime or not signature or "/" in signature or signature in {".", ".."}:
        return None
    return Path(runtime) / "hypr" / signature / ".socket.sock"


def ask(path: Path, request: bytes, timeout_s: float = HYPR_TIMEOUT_S) -> bytes:
    """Return Hyprland's whole answer to one socket request.

    A socket that cannot be reached or does not answer in time raises `OSError`.

    Raises:
        ValueError: the answer is longer than `MAX_HYPR_BYTES`.
    """
    chunks: list[bytes] = []
    size = 0
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
        conn.settimeout(timeout_s)
        conn.connect(str(path))
        conn.sendall(request)
        while chunk := conn.recv(65536):
            size += len(chunk)
            if size > MAX_HYPR_BYTES:
                msg = "Hyprland's answer is too long to read"
                raise ValueError(msg)
            chunks.append(chunk)
    return b"".join(chunks)


def _listing(request: bytes, timeout_s: float) -> list[dict[str, object]]:
    """Return the objects Hyprland lists for one `j/…` request; [] when it cannot answer."""
    path = socket_path()
    if path is None:
        return []
    try:
        found: object = json.loads(ask(path, request, timeout_s))
    except OSError, ValueError:
        return []
    return objs(found)


def clients(timeout_s: float = HYPR_TIMEOUT_S) -> list[dict[str, object]]:
    """Return Hyprland's windows as it describes them (`j/clients`); [] when it cannot answer."""
    return _listing(b"j/clients", timeout_s)


def monitors(timeout_s: float = HYPR_TIMEOUT_S) -> list[dict[str, object]]:
    """Return Hyprland's monitors as it describes them (`j/monitors`); [] when it cannot answer."""
    return _listing(b"j/monitors", timeout_s)


def lua_string(text: str) -> str:
    """Return `text` as a Lua string literal, every byte that could end or bend it escaped.

    Lua reads a decimal escape as one byte, so a character escaped is escaped byte by byte in UTF-8.
    """
    escaped = "".join(
        ch
        if ch.isprintable() and ch not in "\\\"'"
        else "".join(f"\\{byte:03d}" for byte in ch.encode("utf-8", "surrogatepass"))
        for ch in text
    )
    return f'"{escaped}"'


def run_lua(lua: str, timeout_s: float = HYPR_TIMEOUT_S) -> bool:
    """Ask Hyprland to run one Lua statement (`hyprctl eval`); True when it answered `ok`."""
    path = socket_path()
    if path is None:
        return False
    try:
        answer = ask(path, b"eval " + lua.encode("utf-8"), timeout_s)
    except OSError, ValueError:
        return False
    return answer.strip() == b"ok"
