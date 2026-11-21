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
from typing import TYPE_CHECKING, cast

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Iterator, Sequence

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
        "xdg-open",
    }
)
_LOOPBACK_NAMES = frozenset({"localhost", "ip6-localhost"})
# Variables a test must not inherit from the owner's shell: the audio debug dump directory (every
# `GIT_*` setting is cleared too: it would point git at the owner's repository or identity).
_DEBUG_VARIABLES = ("JARVIS_DEBUG_AUDIO_DIR",)


class HermeticError(Exception):
    """A test tried to leave its sandbox: the message says what was refused and how to fix it."""


def _is_loopback(host: str) -> bool:
    if host in _LOOPBACK_NAMES:
        return True
    try:
        return ipaddress.ip_address(host.split("%", 1)[0]).is_loopback
    except ValueError:
        return False


def refusal(
    family: int, address: object, allowed_dir: Path, bound_ports: Collection[int] = ()
) -> str | None:
    """Return why a socket connection is refused, or None when a test may make it.

    Loopback is reachable only on a port a socket in this test process bound: another loopback
    port may be the owner's real Strata server or the Spotify login callback.
    """
    if family in {socket.AF_INET, socket.AF_INET6}:
        fields: tuple[object, ...] = (
            cast("tuple[object, ...]", address) if isinstance(address, tuple) else ()
        )
        host = cast("object", fields[0] if fields else address)
        if isinstance(host, str) and _is_loopback(host):
            port = fields[1] if len(fields) > 1 else None
            if isinstance(port, int) and port in bound_ports:
                return None
            return (
                f"a test tried to connect to {host}:{port} but no socket in this test process "
                "bound that port; it may be a real service of the owner's. Start a server in the "
                "test (bind port 0) or fake the transport, or mark the test `local`."
            )
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


def resolve_refusal(host: object) -> str | None:
    """Return why resolving `host` is refused, or None for loopback (or no host at all)."""
    if host is None:
        return None
    name = os.fsdecode(host) if isinstance(host, (str, bytes)) else repr(host)
    if _is_loopback(name):
        return None
    return (
        f"a test tried to resolve {name!r}, which would send the name to a DNS server; unit tests "
        "may only resolve loopback (127.0.0.1, ::1, localhost). Fake the service, or mark the "
        "test `local`."
    )


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
        argv = cast("Sequence[object]", args)
        first = os.fsdecode(cast("str | bytes | Path", argv[0]))
    else:
        return None
    name = Path(first).name
    if name not in FORBIDDEN_PROGRAMS:
        return None
    if isinstance(args, (list, tuple)) and any(
        "\0" in os.fsdecode(cast("str | bytes | Path", a)) for a in cast("Sequence[object]", args)
    ):
        return None
    environment = cast("dict[str, str]", env) if isinstance(env, dict) else None
    search = os.pathsep.join(os.get_exec_path(environment))
    return name if shutil.which(first, path=search) is not None else None


def _isolate_environment(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the process at fresh directories, silent audio and an empty git configuration."""
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
    for variable in (*_DEBUG_VARIABLES, *(name for name in os.environ if name.startswith("GIT_"))):
        monkeypatch.delenv(variable, raising=False)
    # Outside `home`, which tests expect to be empty: an empty global git config and no system one.
    gitconfig = root / "gitconfig"
    gitconfig.write_text("", encoding="utf-8")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(gitconfig))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")


def _guard_lookup(
    real: Callable[..., object], refuse: Callable[[str | None], None]
) -> Callable[..., object]:
    """Wrap one of the older name lookups so it refuses what `getaddrinfo` refuses."""

    def lookup(host: object, *args: object, **kwargs: object) -> object:
        refuse(resolve_refusal(host))
        return real(host, *args, **kwargs)

    return lookup


def _guard_network(
    monkeypatch: pytest.MonkeyPatch, refuse: Callable[[str | None], None], allowed_dir: Path
) -> None:
    """Refuse connections, datagrams and name lookups that leave the test's own sockets."""
    bound_ports: set[int] = set()
    real_bind = cast("Callable[..., object]", socket.socket.bind)

    def bind(self: socket.socket, address: object, *args: object) -> object:
        result = real_bind(self, address, *args)
        if self.family in {socket.AF_INET, socket.AF_INET6}:
            # After the real bind, so port 0 (any free port) is recorded as the port it became.
            bound_ports.add(cast("int", self.getsockname()[1]))
        return result

    def guard_connect(real: Callable[..., object]) -> Callable[..., object]:
        def connect(self: socket.socket, address: object, *args: object) -> object:
            refuse(refusal(self.family, address, allowed_dir, bound_ports))
            return real(self, address, *args)

        return connect

    real_sendto = cast("Callable[..., object]", socket.socket.sendto)

    def sendto(self: socket.socket, data: object, *args: object) -> object:
        # `sendto(data, address)` or `sendto(data, flags, address)`: the address is last.
        if args:
            refuse(refusal(self.family, args[-1], allowed_dir, bound_ports))
        return real_sendto(self, data, *args)

    real_getaddrinfo = cast("Callable[..., object]", socket.getaddrinfo)

    def getaddrinfo(host: object, *args: object, **kwargs: object) -> object:
        refuse(resolve_refusal(host))
        return real_getaddrinfo(host, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "bind", bind)
    monkeypatch.setattr(socket.socket, "connect", guard_connect(socket.socket.connect))
    monkeypatch.setattr(socket.socket, "connect_ex", guard_connect(socket.socket.connect_ex))
    monkeypatch.setattr(socket.socket, "sendto", sendto)
    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    for name in ("gethostbyname", "gethostbyname_ex", "gethostbyaddr"):
        monkeypatch.setattr(
            socket,
            name,
            _guard_lookup(cast("Callable[..., object]", getattr(socket, name)), refuse),
        )


def _guard_programs(monkeypatch: pytest.MonkeyPatch, refuse: Callable[[str | None], None]) -> None:
    """Refuse to start any of the programs Jarvis drives."""
    real_init: Callable[..., None] = subprocess.Popen.__init__

    def guarded_init(
        self: subprocess.Popen[bytes], args: object, *rest: object, **kwargs: object
    ) -> None:
        name = forbidden_program(args, shell=bool(kwargs.get("shell")), env=kwargs.get("env"))
        if name is not None:
            refuse(
                f"a test tried to start the real {name!r}; unit tests fake the programs Jarvis "
                "drives (monkeypatch the call), or mark the test `local`."
            )
        real_init(self, args, *rest, **kwargs)

    monkeypatch.setattr(subprocess.Popen, "__init__", guarded_init)


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

    def refuse(why: str | None) -> None:
        if why is not None:
            refused.append(why)
            raise HermeticError(why)

    # A sibling of the test's `tmp_path`, not inside it: tests list their own directory.
    _isolate_environment(tmp_path_factory.mktemp("hermetic"), monkeypatch)
    _guard_network(monkeypatch, refuse, tmp_path_factory.getbasetemp())
    _guard_programs(monkeypatch, refuse)

    yield refused
    if refused:
        pytest.fail(
            "hermetic guard tripped (even if the code under test hid the error):\n"
            + "\n".join(refused)
        )
