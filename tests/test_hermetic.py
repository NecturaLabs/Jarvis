"""The hermetic guards in tests/conftest.py: each trips, and what must stay allowed stays."""

import os
import socket
import stat
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

pytest_plugins = ["pytester"]

_PROGRAMS = [
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
]


def _fake_programs(directory: Path) -> Path:
    """Put a harmless executable of each forbidden name in `directory`; each leaves a marker."""
    directory.mkdir()
    for name in _PROGRAMS:
        script = directory / name
        script.write_text(f"#!/bin/sh\ntouch '{directory}/ran-{name}'\n", encoding="utf-8")
        script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return directory


# --- the environment -------------------------------------------------------------------------


def test_the_environment_points_at_fresh_directories(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    base = tmp_path_factory.getbasetemp().resolve()
    seen: set[Path] = set()
    for variable in (
        "HOME",
        "XDG_CONFIG_HOME",
        "XDG_STATE_HOME",
        "XDG_DATA_HOME",
        "XDG_CACHE_HOME",
        "XDG_RUNTIME_DIR",
    ):
        directory = Path(os.environ[variable])
        assert directory.is_dir(), variable
        assert directory.resolve().is_relative_to(base), variable
        assert list(directory.iterdir()) == [], variable
        seen.add(directory)
    assert len(seen) == 6
    assert Path.home() == Path(os.environ["HOME"])


def test_audio_is_silent() -> None:
    assert os.environ["PIPEWIRE_REMOTE"] == "/nonexistent"
    assert os.environ["PULSE_SERVER"] == "unix:/nonexistent"


@pytest.fixture
def own_config_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    path = tmp_path / "mine"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(path))
    return path


def test_a_test_that_sets_its_own_variable_wins(own_config_home: Path) -> None:
    assert os.environ["XDG_CONFIG_HOME"] == str(own_config_home)


# --- the network -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("family", "address"),
    [
        (socket.AF_INET, ("93.184.216.34", 80)),
        (socket.AF_INET, ("192.168.1.10", 8080)),
        (socket.AF_INET, ("example.com", 443)),
        (socket.AF_INET6, ("2606:2800:220:1:248:1893:25c8:1946", 80, 0, 0)),
    ],
)
@pytest.mark.parametrize("method", ["connect", "connect_ex"])
def test_a_connection_beyond_loopback_is_refused(
    hermetic: list[str], family: int, address: tuple[object, ...], method: str
) -> None:
    with (
        socket.socket(family, socket.SOCK_STREAM) as sock,
        pytest.raises(Exception, match="over the network") as caught,
    ):
        getattr(sock, method)(address)
    assert type(caught.value).__name__ == "HermeticError"
    assert "loopback" in str(caught.value)
    assert len(hermetic) == 1
    hermetic.clear()


@pytest.mark.parametrize("port", [1, 9])
@pytest.mark.parametrize("method", ["connect", "connect_ex"])
def test_a_loopback_port_no_test_bound_is_refused(
    hermetic: list[str], method: str, port: int
) -> None:
    # The owner's real Strata (config default port 8080) and Spotify's callback port live here.
    with (
        socket.socket() as sock,
        pytest.raises(Exception, match="no socket in this test") as caught,
    ):
        getattr(sock, method)(("127.0.0.1", port))
    assert type(caught.value).__name__ == "HermeticError"
    assert len(hermetic) == 1
    hermetic.clear()


def test_a_loopback_port_bound_with_port_zero_connects() -> None:
    with socket.socket(socket.AF_INET6) as server:
        server.bind(("::1", 0))
        server.listen()
        with socket.socket(socket.AF_INET6) as client:
            client.connect(server.getsockname())


def test_a_datagram_to_a_bound_loopback_port_is_sent() -> None:
    with (
        socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as server,
        socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as client,
    ):
        server.bind(("127.0.0.1", 0))
        client.sendto(b"x", server.getsockname())
        client.sendto(b"y", 0, server.getsockname())
        assert server.recv(8) == b"x"


@pytest.mark.parametrize(
    "address", [("127.0.0.1", 1), ("192.0.2.1", 53)], ids=["unbound-loopback", "beyond-loopback"]
)
def test_a_datagram_to_an_unbound_or_outside_address_is_refused(
    hermetic: list[str], address: tuple[str, int]
) -> None:
    with (
        socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock,
        pytest.raises(Exception, match="a test tried to") as caught,
    ):
        sock.sendto(b"x", address)
    assert type(caught.value).__name__ == "HermeticError"
    assert len(hermetic) == 1
    hermetic.clear()


@pytest.mark.parametrize("host", ["leak-check.invalid", "192.0.2.1", "example.com"])
def test_resolving_a_name_beyond_loopback_is_refused(hermetic: list[str], host: str) -> None:
    with pytest.raises(Exception, match="resolve") as caught:
        socket.getaddrinfo(host, 80)
    assert type(caught.value).__name__ == "HermeticError"
    assert len(hermetic) == 1
    hermetic.clear()


@pytest.mark.parametrize("host", ["localhost", "127.0.0.1", "::1", None])
def test_resolving_loopback_still_works(host: str | None) -> None:
    assert socket.getaddrinfo(host, 80)


@pytest.mark.parametrize("method", ["connect", "connect_ex"])
def test_loopback_still_connects(method: str) -> None:
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen()
        with socket.socket() as client:
            result = getattr(client, method)(server.getsockname())
            assert result in {None, 0}


def test_a_unix_socket_inside_the_temporary_directory_still_connects(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    # A short path: a Unix socket address is limited to about 100 bytes.
    path = tmp_path_factory.mktemp("u") / "s"
    with socket.socket(socket.AF_UNIX) as server:
        server.bind(str(path))
        server.listen()
        with socket.socket(socket.AF_UNIX) as client:
            client.connect(str(path))


@pytest.mark.parametrize("address", ["/run/user/1000/pipewire-0", "relative.sock", b"\0abstract"])
def test_a_unix_socket_elsewhere_is_refused(hermetic: list[str], address: str | bytes) -> None:
    with (
        socket.socket(socket.AF_UNIX) as sock,
        pytest.raises(Exception, match="Unix socket") as caught,
    ):
        sock.connect(address)
    assert type(caught.value).__name__ == "HermeticError"
    assert "temporary directory" in str(caught.value)
    hermetic.clear()


# --- the programs ----------------------------------------------------------------------------


@pytest.mark.parametrize("program", _PROGRAMS)
def test_starting_a_real_program_is_refused(
    hermetic: list[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path, program: str
) -> None:
    bin_dir = _fake_programs(tmp_path / "bin")
    monkeypatch.setenv("PATH", str(bin_dir))

    with pytest.raises(Exception, match=f"real '{program}'") as caught:
        subprocess.Popen([program])

    assert type(caught.value).__name__ == "HermeticError"
    assert not (bin_dir / f"ran-{program}").exists()
    assert len(hermetic) == 1
    hermetic.clear()


def _run(_bin: Path) -> object:
    return subprocess.run(["systemctl", "--user", "status"], check=False)


def _check_output(_bin: Path) -> object:
    return subprocess.check_output(["systemctl"])


def _absolute_path(bin_dir: Path) -> object:
    return subprocess.Popen([str(bin_dir / "systemctl")])


def _path_object(bin_dir: Path) -> object:
    return subprocess.Popen(bin_dir / "systemctl")


def _shell_string(_bin: Path) -> object:
    # The shell form is exactly what must be refused too.
    return subprocess.Popen("systemctl --user status", shell=True)  # ruff: ignore[subprocess-popen-with-shell-equals-true] - see above


@pytest.mark.parametrize(
    "launch", [_run, _check_output, _absolute_path, _path_object, _shell_string]
)
def test_every_way_of_starting_one_is_refused(
    hermetic: list[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    launch: Callable[[Path], object],
) -> None:
    bin_dir = _fake_programs(tmp_path / "bin")
    monkeypatch.setenv("PATH", str(bin_dir))

    with pytest.raises(Exception, match="real 'systemctl'"):
        launch(bin_dir)

    assert not (bin_dir / "ran-systemctl").exists()
    hermetic.clear()


def test_an_ordinary_program_still_starts(tmp_path: Path) -> None:
    marker = tmp_path / "marker"
    done = subprocess.run(["touch", str(marker)], check=False)
    assert done.returncode == 0
    assert marker.exists()


def test_a_missing_program_is_not_refused_it_just_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PATH", "/nonexistent")
    with pytest.raises(FileNotFoundError):
        subprocess.Popen(["pw-dump"])


# --- the guard itself ----------------------------------------------------------------------------

_CONFTEST = Path(__file__).with_name("conftest.py")

_INNER = """
import os
import socket
import subprocess

import pytest

HOME = os.environ["HOME"]
CONNECT = socket.socket.connect
INIT = subprocess.Popen.__init__


@pytest.mark.local
def test_local_is_not_guarded():
    assert os.environ["HOME"] == HOME
    assert socket.socket.connect is CONNECT
    assert subprocess.Popen.__init__ is INIT


def test_other_is_guarded():
    assert os.environ["HOME"] != HOME
    assert socket.socket.connect is not CONNECT
    assert subprocess.Popen.__init__ is not INIT


def test_swallowing_the_refusal_does_not_hide_it():
    try:
        socket.socket().connect(("93.184.216.34", 80))
    except Exception:
        pass
"""


def test_a_local_test_is_not_guarded_and_a_swallowed_refusal_still_fails(
    pytester: pytest.Pytester,
) -> None:
    pytester.makeini("[pytest]\nmarkers =\n    local: needs the real PC\n")
    pytester.makeconftest(_CONFTEST.read_text(encoding="utf-8"))
    pytester.makepyfile(_INNER)

    result = pytester.runpytest_inprocess("-p", "no:cacheprovider")

    result.assert_outcomes(passed=3, errors=1)
    result.stdout.fnmatch_lines(["*hermetic guard tripped*"])
    result.stdout.fnmatch_lines(
        ["*ERROR at teardown of test_swallowing_the_refusal_does_not_hide_it*"]
    )


_INNER_ENVIRONMENT = """
import os
from pathlib import Path


def test_inherited_variables_are_gone():
    for name in ("JARVIS_DEBUG_AUDIO_DIR", "GIT_DIR", "GIT_AUTHOR_NAME", "GIT_SSH_COMMAND"):
        assert name not in os.environ, name
    assert os.environ["GIT_CONFIG_NOSYSTEM"] == "1"
    config = Path(os.environ["GIT_CONFIG_GLOBAL"])
    assert config.is_file()
    assert config.read_text() == ""
    assert not config.resolve().is_relative_to(Path(os.environ["HOME"]).resolve())
"""


def test_the_owners_debug_and_git_variables_are_not_inherited(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("JARVIS_DEBUG_AUDIO_DIR", "/somewhere")
    monkeypatch.setenv("GIT_DIR", "/elsewhere/.git")
    monkeypatch.setenv("GIT_AUTHOR_NAME", "Someone")
    monkeypatch.setenv("GIT_SSH_COMMAND", "ssh -i key")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/home/owner/.gitconfig")
    monkeypatch.delenv("GIT_CONFIG_NOSYSTEM", raising=False)
    pytester.makeini("[pytest]\nmarkers =\n    local: needs the real PC\n")
    pytester.makeconftest(_CONFTEST.read_text(encoding="utf-8"))
    pytester.makepyfile(_INNER_ENVIRONMENT)

    result = pytester.runpytest_inprocess("-p", "no:cacheprovider")

    result.assert_outcomes(passed=1)
