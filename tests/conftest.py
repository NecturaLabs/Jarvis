"""Hermetic tests: no test reads the owner's environment, reaches the network or runs a real tool.

Every test not marked `local` runs with its own home and XDG directories, silent audio, no outside
network and no real `systemctl`, `voxtype` or other program Jarvis drives. A guard raises
`HermeticError`, which no production code catches by accident (it is not an `OSError`), and the
test also fails at teardown if a guard tripped, so code that swallows the error cannot hide it.
Tests of the real PC are marked `local` and are run with the owner.
"""

import ipaddress
import os
import shlex
import shutil
import socket
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

# Programs Jarvis drives; a unit test fakes them, it never starts the real one.
FORBIDDEN_PROGRAMS = frozenset(
    {
        "systemctl",
        "journalctl",
        "voxtype",
        "pw-record",
        "pw-play",
        "pw-dump",
        "pw-cat",
        "wpctl",
        "pactl",
        "paplay",
        "hyprctl",
    }
)
_LOOPBACK_NAMES = frozenset({"localhost", "ip6-localhost"})


class HermeticError(Exception):
    """A test tried to leave its sandbox: the message says what was refused and how to fix it."""


def _is_loopback(host: str) -> bool:
    if host in _LOOPBACK_NAMES:
        return True
    try:
        return ipaddress.ip_address(host.split("%", 1)[0]).is_loopback
    except ValueError:
        return False


def refusal(family: int, address: object, allowed_dir: Path) -> str | None:
    """Return why a socket connection is refused, or None when a test may make it."""
    if family in {socket.AF_INET, socket.AF_INET6}:
        host = address[0] if isinstance(address, tuple) and address else address  # pyright: ignore[reportUnknownVariableType]
        if isinstance(host, str) and _is_loopback(host):
            return None
        return (
            f"a test tried to connect to {host!r} over the network; unit tests may only reach "
            "loopback (127.0.0.1, ::1, localhost). Fake the service, or mark the test `local`."
        )
    if family == socket.AF_UNIX and isinstance(address, (str, bytes)):
        path = Path(os.fsdecode(address))
        if path.is_absolute() and path.resolve().is_relative_to(allowed_dir.resolve()):
            return None
        return (
            f"a test tried to connect to the Unix socket {path}; unit tests may only use a "
            f"socket inside pytest's temporary directory ({allowed_dir}). Mark the test "
            "`local` if it needs a real service."
        )
    return None


def forbidden_program(args: object, *, shell: bool = False, env: object = None) -> str | None:
    """Return the name of the forbidden program `args` would really start, or None.

    A program that is not installed (or whose path does not exist) cannot start, so a test of
    "the tool is missing" may name one, and so may a test of an argument the operating system
    rejects (a NUL byte); only a program that would actually run is refused.
    """
    if isinstance(args, (str, bytes, Path)):
        first = os.fsdecode(args)
        if shell:
            words = shlex.split(first)
            first = words[0] if words else ""
    elif isinstance(args, (list, tuple)) and args:
        first = os.fsdecode(args[0])  # pyright: ignore[reportUnknownArgumentType]
    else:
        return None
    name = Path(first).name
    if name not in FORBIDDEN_PROGRAMS:
        return None
    if isinstance(args, (list, tuple)) and any("\0" in os.fsdecode(a) for a in args):  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
        return None
    search = os.pathsep.join(os.get_exec_path(env if isinstance(env, dict) else None))  # pyright: ignore[reportUnknownArgumentType]
    return name if shutil.which(first, path=search) is not None else None


@pytest.fixture(autouse=True)  # ruff: ignore[pytest-fixture-autouse] - every test is isolated
def hermetic(
    request: pytest.FixtureRequest,
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[list[str]]:
    """Isolate a test from the owner's environment; yield the refusals recorded so far."""
    refused: list[str] = []
    if request.node.get_closest_marker("local") is not None:
        yield refused
        return

    # A sibling of the test's `tmp_path`, not inside it: tests list their own directory.
    root = tmp_path_factory.mktemp("hermetic")
    for variable, name in (
        ("HOME", "home"),
        ("XDG_CONFIG_HOME", "config"),
        ("XDG_STATE_HOME", "state"),
        ("XDG_DATA_HOME", "data"),
        ("XDG_CACHE_HOME", "cache"),
        ("XDG_RUNTIME_DIR", "runtime"),
    ):
        (root / name).mkdir(mode=0o700)
        monkeypatch.setenv(variable, str(root / name))
    monkeypatch.setenv("PIPEWIRE_REMOTE", "/nonexistent")
    monkeypatch.setenv("PULSE_SERVER", "unix:/nonexistent")

    allowed_dir = tmp_path_factory.getbasetemp()

    def guard_connect(real: Callable[..., object]) -> Callable[..., object]:
        def connect(self: socket.socket, address: object, *args: object) -> object:
            why = refusal(self.family, address, allowed_dir)
            if why is not None:
                refused.append(why)
                raise HermeticError(why)
            return real(self, address, *args)

        return connect

    monkeypatch.setattr(socket.socket, "connect", guard_connect(socket.socket.connect))
    monkeypatch.setattr(socket.socket, "connect_ex", guard_connect(socket.socket.connect_ex))

    real_init: Callable[..., None] = subprocess.Popen.__init__

    def guarded_init(
        self: subprocess.Popen[bytes], args: object, *rest: object, **kwargs: object
    ) -> None:
        name = forbidden_program(args, shell=bool(kwargs.get("shell")), env=kwargs.get("env"))
        if name is not None:
            why = (
                f"a test tried to start the real {name!r}; unit tests fake the programs Jarvis "
                "drives (monkeypatch the call), or mark the test `local`."
            )
            refused.append(why)
            raise HermeticError(why)
        real_init(self, args, *rest, **kwargs)

    monkeypatch.setattr(subprocess.Popen, "__init__", guarded_init)

    yield refused
    if refused:
        pytest.fail(
            "hermetic guard tripped (even if the code under test hid the error):\n"
            + "\n".join(refused)
        )
