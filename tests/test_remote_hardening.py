"""Review findings on the model server on another machine (ADR 0020): each one pinned by a test.

Nothing here reaches a real server, resolver, terminal of the owner or systemd: requests are
answered by httpx mock transports, names by a patched resolver, the clock by a fake one.
"""

import argparse
import contextlib
import io
import json
import logging
import os
import socket
import ssl
import subprocess
import sys
import termios
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from strata_fixture import engine_run_config, install_engine
from test_remote import KEY, LISTED, TUNNEL

from jarvis import app, audio, cli, config, paths, services, update
from jarvis.about import facts
from jarvis.agent import Limits
from jarvis.model import REMOTE_PROBE_CACHE_S, ModelStatus, RemoteModelError, StrataModel
from jarvis.secretfile import remote_key_input, write_private


def _listing(_request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"data": [LISTED]})


def _inactive(_unit: str) -> str:
    return "inactive"


def _active(_unit: str) -> str:
    return "active"


def _only_strata_active(unit: str) -> str:
    return "active" if "strata" in unit else "inactive"


def _which(command: str) -> str:
    return f"/usr/bin/{command}"


def _no_notes(_cfg: config.Config) -> list[str]:
    return []


def _no_systemctl(*_args: str) -> int:
    return 0


def _no_models(_engine: str) -> None:
    return


def _stub_subprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    def run(_cmd: list[str], **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(returncode=0, stdout="")

    monkeypatch.setattr(cli.subprocess, "run", run)


def _remote(**settings: object) -> config.Config:
    return config.Config(
        strata=config.StrataConfig.model_validate({"mode": "remote", "url": TUNNEL, **settings}),
        spotify=config.SpotifyConfig(client_id="abc"),
    )


def _fake_strata(monkeypatch: pytest.MonkeyPatch, transport: httpx.MockTransport) -> None:
    real = app.strata_model

    def make(cfg: config.Config, key: str) -> StrataModel:
        return replace(real(cfg, key), transport=transport)

    monkeypatch.setattr(app, "strata_model", make)
    monkeypatch.setattr(cli, "strata_model", make)


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point every XDG folder at `tmp_path` and stand in for systemd."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setattr(cli, "_unit_state", _inactive)
    monkeypatch.setattr(cli, "_systemctl", _no_systemctl)
    return tmp_path


pytestmark = pytest.mark.usefixtures("home")

DNS_NAME = "https://model.example.test"


class Clock:
    """A clock that moves only when a test or a `sleep` moves it."""

    def __init__(self) -> None:
        super().__init__()
        self.now = 1000.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def _resolving(monkeypatch: pytest.MonkeyPatch, *addresses: str) -> list[str]:
    """Answer every name lookup with `addresses`; return the names asked for."""
    asked: list[str] = []

    def resolve(host: str) -> list[str]:
        asked.append(host)
        return list(addresses)

    monkeypatch.setattr(config, "resolve_addresses", resolve)
    return asked


def _model(
    cfg: config.Config, transport: httpx.MockTransport, clock: Clock | None = None
) -> StrataModel:
    clock = clock or Clock()
    return replace(app.strata_model(cfg, KEY), transport=transport, clock=clock, sleep=clock.sleep)


def _refusing(clock: Clock, seconds: float, seen: list[httpx.Request]) -> httpx.MockTransport:
    def refuse(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        clock.now += seconds
        raise httpx.ConnectError("offline", request=request)

    return httpx.MockTransport(refuse)


# 1. The local facts line is dev's, byte for byte.


def test_the_local_facts_line_is_devs_and_only_remote_words_it_differently() -> None:
    limits = Limits(tool_text_chars=3000)
    local = facts(config.Config(), "voice", limits)
    assert "GPL-3.0-or-later, running only on this PC; one agent, no hosted model." in local
    remote = facts(_remote(), "voice", limits)
    assert "running only on this PC" not in remote
    assert "one agent on this PC, no hosted model" in remote


# 2. A down backend costs a short, bounded silence.


def test_a_remote_probe_makes_one_short_attempt() -> None:
    clock = Clock()
    seen: list[httpx.Request] = []
    model = _model(_remote(), _refusing(clock, 2.0, seen), clock)

    status = model.probe()

    assert not status.reachable
    assert len(seen) == 1
    timeout = seen[0].extensions["timeout"]
    assert timeout["connect"] + timeout["read"] <= 3.0


def test_a_recent_probe_is_reused_for_a_few_seconds_whether_it_worked_or_not() -> None:
    clock = Clock()
    answers = [httpx.Response(200, json={"data": [LISTED]}), httpx.ConnectError("offline")]
    seen: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        answer = answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    model = _model(_remote(), httpx.MockTransport(respond), clock)
    assert model.probe().reachable
    clock.now += REMOTE_PROBE_CACHE_S - 1
    assert model.probe().reachable
    assert len(seen) == 1
    clock.now += 2  # past the cache
    assert not model.probe().reachable
    assert len(seen) == 2
    assert not model.probe().reachable  # the failure is cached too
    assert len(seen) == 2
    assert not model.is_loaded()
    assert len(seen) == 2


def test_a_turn_with_the_backend_down_waits_one_probe_and_one_attempt() -> None:
    clock = Clock()
    seen: list[httpx.Request] = []
    model = _model(_remote(), _refusing(clock, 2.0, seen), clock)

    assert model.is_loaded() is None  # what the agent asks at the top of a turn
    with pytest.raises(RemoteModelError):
        model.chat([], [])

    assert [r.method for r in seen] == ["GET", "POST"]
    assert clock.now - 1000.0 <= 4.0


def test_a_remote_connection_that_drops_is_retried_once_after_a_short_bounded_pause() -> None:
    clock = Clock()
    seen: list[httpx.Request] = []
    model = _model(_remote(), _refusing(clock, 0.0, seen), clock)

    with pytest.raises(RemoteModelError):
        model.chat([], [])

    assert [r.method for r in seen] == ["POST", "POST"]
    assert len(clock.slept) == 1
    assert 0 < clock.slept[0] <= 1.0


# 3. `jarvis status` takes `running` from the unit and never waits long for a down backend.


def test_status_of_a_remote_client_with_the_backend_down_still_says_it_runs(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config.save(_remote())
    services.write_remote_key(config.load(), KEY)
    monkeypatch.setattr(cli, "_unit_state", _active)

    def slow(_command: str, **_kwargs: object) -> None:  # no answer: the service is still probing
        return None

    monkeypatch.setattr(cli.control, "send", slow)

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    _fake_strata(monkeypatch, httpx.MockTransport(refuse))
    assert cli.main(["status", "--json"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["running"] is True
    assert shown["model_reachable"] is False


def test_status_of_a_stopped_remote_client_says_not_running(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config.save(_remote())
    services.write_remote_key(config.load(), KEY)
    _fake_strata(
        monkeypatch,
        httpx.MockTransport(_listing),
    )
    assert cli.main(["status"]) == 0
    assert capsys.readouterr().out.startswith("jarvis: not running, ")


def test_the_services_remote_status_is_a_cached_short_probe() -> None:
    clock = Clock()
    seen: list[httpx.Request] = []
    model = _model(_remote(), _refusing(clock, 2.0, seen), clock)
    report = cli._remote_status(_remote(), model)  # pyright: ignore[reportPrivateUsage]
    assert report is not None

    first = report()
    second = report()

    assert first["model_reachable"] is False
    assert second == first
    assert len(seen) == 1
    assert clock.now - 1000.0 <= 3.0


# 4. A DNS name is held to the private-address rule IP literals get.


@pytest.mark.parametrize(
    "addresses", [["8.8.8.8"], ["192.168.1.20", "8.8.8.8"], ["2001:db8::1"], []]
)
def test_a_name_that_resolves_anywhere_public_is_refused_by_the_client(
    monkeypatch: pytest.MonkeyPatch, addresses: list[str]
) -> None:
    _resolving(monkeypatch, *addresses)
    seen: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"data": [LISTED]})

    model = _model(_remote(url=DNS_NAME), httpx.MockTransport(respond))
    status = model.probe()
    assert not status.reachable
    assert "private network" in status.issue
    with pytest.raises(RemoteModelError, match="private network"):
        model.chat([], [])
    assert not seen


@pytest.mark.parametrize(
    "addresses",
    [["192.168.1.20"], ["100.101.102.103"], ["10.0.0.5", "fd00::20"], ["127.0.0.1", "::1"]],
)
def test_a_name_that_resolves_only_to_private_addresses_is_used(
    monkeypatch: pytest.MonkeyPatch, addresses: list[str]
) -> None:
    asked = _resolving(monkeypatch, *addresses)
    model = _model(
        _remote(url=DNS_NAME),
        httpx.MockTransport(_listing),
    )
    assert model.probe().reachable
    assert asked == ["model.example.test"]


def test_https_to_localhost_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    asked = _resolving(monkeypatch, "127.0.0.1", "::1")
    cfg = _remote(url="https://localhost:8443")
    assert cfg.model_base_url() == "https://localhost:8443/v1"
    model = _model(cfg, httpx.MockTransport(_listing))
    assert model.probe().reachable
    assert asked == ["localhost"]


def test_an_ip_literal_is_never_resolved(monkeypatch: pytest.MonkeyPatch) -> None:
    asked = _resolving(monkeypatch, "8.8.8.8")
    model = _model(
        _remote(),
        httpx.MockTransport(_listing),
    )
    assert model.probe().reachable
    assert not asked


def test_setup_refuses_a_name_that_resolves_to_a_public_address(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _resolving(monkeypatch, "93.184.216.34")
    monkeypatch.setattr(sys, "stdin", io.StringIO(KEY + "\n"))

    assert cli.main(["setup", "--remote", DNS_NAME]) == 2

    err = capsys.readouterr().err
    assert "private network" in err
    assert KEY not in err
    assert not paths.config_file().exists()
    assert not paths.remote_strata_key_file().exists()


def test_setup_accepts_a_name_that_resolves_privately(monkeypatch: pytest.MonkeyPatch) -> None:
    _resolving(monkeypatch, "100.101.102.103")
    _stub_subprocess(monkeypatch)
    monkeypatch.setattr(sys, "stdin", io.StringIO(KEY + "\n"))

    assert cli.main(["setup", "--remote", DNS_NAME]) == 0
    assert config.load().strata.url == DNS_NAME


def test_the_doctor_says_when_the_name_resolves_to_a_public_address(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_subprocess(monkeypatch)
    monkeypatch.setattr(audio, "installed_models", _no_models)
    _resolving(monkeypatch, "93.184.216.34")
    config.save(_remote(url=DNS_NAME))
    services.write_remote_key(config.load(), KEY)
    _fake_strata(
        monkeypatch,
        httpx.MockTransport(_listing),
    )
    report = cli.doctor_report(argparse.Namespace(essential=False, wait=False, no_service=False))
    refused = [check for check in report.checks if "private network" in check[1]]
    assert len(refused) == 1
    assert not refused[0][0]


# 5. `strata.tls_ca` is what the client trusts, and nothing else.


def _certificate(folder: Path, name: str) -> tuple[Path, Path]:
    """Write a self-signed localhost certificate usable as its own CA; return it and its key."""
    certificate, key = folder / f"{name}.pem", folder / f"{name}.key"
    subprocess.run(
        [
            "openssl", "req", "-x509", "-newkey", "ec", "-pkeyopt", "ec_paramgen_curve:prime256v1",
            "-nodes", "-keyout", str(key), "-out", str(certificate), "-days", "2",
            "-subj", f"/CN={name}", "-addext", "subjectAltName=DNS:localhost,IP:127.0.0.1",
            "-addext", "basicConstraints=critical,CA:TRUE",
            "-addext", "keyUsage=critical,keyCertSign,digitalSignature",
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )  # fmt: skip
    return certificate, key


def _https_server(certificate: Path, key: Path, connections: int) -> tuple[int, threading.Thread]:
    """Answer GET /v1/models over TLS on loopback until `connections` clients have come."""
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certificate, key)
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    listener.settimeout(10)
    body = json.dumps({"data": [LISTED]}).encode()

    def serve() -> None:
        with listener:
            for _ in range(connections):
                try:
                    connection, _address = listener.accept()
                except OSError:
                    return
                with connection:
                    try:
                        with context.wrap_socket(connection, server_side=True) as tls:
                            tls.settimeout(5)
                            tls.recv(4096)
                            tls.sendall(
                                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                                b"Connection: close\r\nContent-Length: %d\r\n\r\n%s"
                                % (len(body), body)
                            )
                    except ssl.SSLError, OSError:
                        continue  # a client that refused the certificate

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    return listener.getsockname()[1], thread


def test_the_ca_file_is_the_clients_tls_trust_and_a_foreign_certificate_is_refused(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    served, key = _certificate(home, "served")
    other, _other_key = _certificate(home, "other")
    _resolving(monkeypatch, "127.0.0.1")
    port, thread = _https_server(served, key, connections=2)
    url = f"https://localhost:{port}"
    seen: list[str] = []
    real = ssl.create_default_context

    def create(*, cafile: str | None = None) -> ssl.SSLContext:
        seen.append(cafile or "")
        return real(cafile=cafile)

    monkeypatch.setattr(ssl, "create_default_context", create)

    trusting = app.strata_model(_remote(url=url, tls_ca=str(served)), KEY)
    assert trusting.probe().reachable
    assert seen == [str(served)]

    foreign = app.strata_model(_remote(url=url, tls_ca=str(other)), KEY)
    refused = foreign.probe()
    assert not refused.reachable
    assert KEY not in refused.report("m")
    thread.join(timeout=15)


def test_a_client_without_a_ca_does_not_trust_a_self_signed_server(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    served, key = _certificate(home, "served")
    _resolving(monkeypatch, "127.0.0.1")
    port, thread = _https_server(served, key, connections=1)

    model = app.strata_model(_remote(url=f"https://localhost:{port}"), KEY)

    assert not model.probe().reachable
    thread.join(timeout=15)


# 6. Plain HTTP beyond loopback on the backend is warned about, where the owner reads.


def _backend(home: Path, listen: str) -> config.Config:
    install_engine(home / "data")
    return config.Config(
        strata=config.StrataConfig.model_validate(
            {"mode": "backend", "run_config": engine_run_config(home), "listen": listen}
        )
    )


def test_backend_setup_on_a_lan_address_warns_that_the_key_crosses_the_network(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _stub_subprocess(monkeypatch)
    install_engine(home / "data")
    run = engine_run_config(home)
    assert (
        cli.main(["setup", "--backend", "--run-config", str(run), "--listen", "192.168.1.20"]) == 0
    )
    out = capsys.readouterr().out
    assert "unencrypted" in out
    assert "SSH tunnel" in out
    assert "HTTPS" in out


def test_backend_setup_on_loopback_has_no_warning(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _stub_subprocess(monkeypatch)
    install_engine(home / "data")
    run = engine_run_config(home)
    assert cli.main(["setup", "--backend", "--run-config", str(run)]) == 0
    assert "unencrypted" not in capsys.readouterr().out


def test_the_backend_doctor_warns_when_it_listens_beyond_loopback(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_subprocess(monkeypatch)
    write_private(paths.backend_strata_key_file(), f"STRATA_API_KEY={KEY}\n")
    _fake_strata(
        monkeypatch,
        httpx.MockTransport(_listing),
    )
    args = argparse.Namespace(essential=False, wait=False, no_service=False)

    config.save(_backend(home, "192.168.1.20"))
    notes = " ".join(cli.doctor_report(args).notes)
    assert "unencrypted" in notes
    assert "SSH tunnel" in notes

    config.save(_backend(home, "127.0.0.1"))
    assert "unencrypted" not in " ".join(cli.doctor_report(args).notes)


# 7. A tunnel does not survive a reboot, and the doctor says how to raise it again.


def test_an_unreachable_remote_behind_a_tunnel_says_the_tunnel_ends_at_reboot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config.save(_remote())
    services.write_remote_key(config.load(), KEY)

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    _fake_strata(monkeypatch, httpx.MockTransport(refuse))
    ok, _what, fix = cli._split_model_check(config.load())  # pyright: ignore[reportPrivateUsage]
    assert not ok
    assert "reboot" in fix
    assert "logout" in fix
    assert "ssh -fNT -o ExitOnForwardFailure=yes -L 127.0.0.1:18080:" in fix
    assert "jarvis setup --backend" in fix


def test_a_remote_without_a_tunnel_gets_no_tunnel_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    _resolving(monkeypatch, "192.168.1.20")
    config.save(_remote(url=DNS_NAME))
    services.write_remote_key(config.load(), KEY)

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    _fake_strata(monkeypatch, httpx.MockTransport(refuse))
    _ok, _what, fix = cli._split_model_check(config.load())  # pyright: ignore[reportPrivateUsage]
    assert "reboot" not in fix


# 8. An update does not roll back because the remote backend is down.


def _update_calls(
    monkeypatch: pytest.MonkeyPatch, *, strata_unit_active: bool
) -> list[tuple[str, ...]]:
    calls: list[tuple[str, ...]] = []

    def run(
        args: tuple[str, ...], *, cwd: Path, timeout: float, show: bool = False
    ) -> subprocess.CompletedProcess[str]:
        del cwd, timeout, show
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, "", "")

    def state(_root: Path, unit: str) -> str:
        return "active" if unit == services.JARVIS_UNIT or strata_unit_active else "inactive"

    monkeypatch.setattr(update, "_run", run)
    monkeypatch.setattr(update.shutil, "which", _which)
    monkeypatch.setattr(update, "_unit_state", state)
    return calls


def _waited_for(calls: list[tuple[str, ...]]) -> tuple[str, ...]:
    (ready,) = (args[4:] for args in calls if "ready" in args)
    return ready


@pytest.mark.parametrize(("remote", "waits_for_strata"), [(True, False), (False, True)])
def test_the_update_waits_for_strata_only_when_strata_is_local(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, remote: bool, waits_for_strata: bool
) -> None:
    calls = _update_calls(monkeypatch, strata_unit_active=False)
    checkout = SimpleNamespace(root=tmp_path, branch="main")
    action = update.Apply(checkout, "a" * 40, "b" * 40, remote=remote)  # pyright: ignore[reportArgumentType]

    action._ensure_ready()  # pyright: ignore[reportPrivateUsage]

    assert ("strata" in _waited_for(calls)) is waits_for_strata
    assert "voice" in _waited_for(calls)


def test_a_remote_client_with_a_leftover_local_strata_unit_still_skips_the_wait(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _update_calls(monkeypatch, strata_unit_active=True)
    checkout = SimpleNamespace(root=tmp_path, branch="main")
    action = update.Apply(checkout, "a" * 40, "b" * 40, remote=True)  # pyright: ignore[reportArgumentType]
    assert action.strata

    action._ensure_ready()  # pyright: ignore[reportPrivateUsage]

    assert "strata" not in _waited_for(calls)


# 9. The key reader, the terminal prompt, a leftover local server.


@pytest.mark.parametrize(
    ("text", "key"),
    [
        (f"{KEY}\n", KEY),
        (f"STRATA_API_KEY={KEY}\n", KEY),
        (f"# written by jarvis\nOTHER=1\nSTRATA_API_KEY={KEY}\nTAIL=2\n", KEY),
        (f'STRATA_API_KEY="{KEY}"\n', KEY),
        (f"OTHER=1\n{KEY}\n", None),
        (f"{KEY} extra\n", None),
        ("", None),
        ("STRATA_API_KEY=\n", None),
        ("STRATA_API_KEY=two words\n", None),
        ("x" * 70000, None),
    ],
)
def test_the_key_is_found_wherever_the_backends_file_has_it(text: str, key: str | None) -> None:
    assert remote_key_input(text) == key


def test_setup_reads_the_key_from_the_whole_backend_file_on_stdin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_subprocess(monkeypatch)
    monkeypatch.setattr(sys, "stdin", io.StringIO(f"OTHER=1\nSTRATA_API_KEY={KEY}\n"))
    assert cli.main(["setup", "--remote", TUNNEL]) == 0
    assert services.read_key(config.load()) == KEY


class _Terminal:
    @staticmethod
    def isatty() -> bool:
        return True


def _pty(monkeypatch: pytest.MonkeyPatch, typed: bytes) -> tuple[int, int]:
    """Stand in for /dev/tty with a pseudo-terminal; `typed` arrives once the prompt is shown."""
    master, slave = os.openpty()
    monkeypatch.setattr(cli, "_TTY", Path(os.ttyname(slave)))
    monkeypatch.setattr(sys, "stdin", _Terminal())

    def type_when_asked() -> None:
        shown = b""
        with contextlib.suppress(OSError):  # the test ended, and closed the terminal, first
            while b"Strata API key" not in shown:
                shown += os.read(master, 256)
            os.write(master, typed)

    threading.Thread(target=type_when_asked, daemon=True).start()
    return master, slave


def test_the_tty_prompt_reads_the_key_unechoed_and_restores_echo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    master, slave = _pty(monkeypatch, f"{KEY}\n".encode())
    try:
        assert cli._read_remote_key() == KEY  # pyright: ignore[reportPrivateUsage]
        assert termios.tcgetattr(slave)[3] & termios.ECHO
        os.set_blocking(master, False)
        echoed = b""
        with contextlib.suppress(BlockingIOError):
            echoed = os.read(master, 4096)
        assert KEY.encode() not in echoed
    finally:
        os.close(master)
        os.close(slave)


def test_the_tty_prompt_restores_echo_when_the_read_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    master, slave = _pty(monkeypatch, b"\x04")  # end of input: nothing typed
    try:
        with pytest.raises(config.ConfigError):
            cli._read_remote_key()  # pyright: ignore[reportPrivateUsage]
        assert termios.tcgetattr(slave)[3] & termios.ECHO
    finally:
        os.close(master)
        os.close(slave)


def test_a_terminal_that_refuses_termios_is_a_setup_error_not_a_traceback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    master, slave = _pty(monkeypatch, f"{KEY}\n".encode())

    def broken(_terminal: object) -> list[Any]:
        raise termios.error(25, "Inappropriate ioctl for device")

    monkeypatch.setattr(termios, "tcgetattr", broken)
    try:
        assert cli.main(["setup", "--remote", TUNNEL]) == 2
    finally:
        os.close(master)
        os.close(slave)
    err = capsys.readouterr().err
    assert "stdin" in err
    assert KEY not in err
    assert not paths.config_file().exists()


def test_the_doctor_says_when_a_local_strata_still_runs_on_a_remote_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_subprocess(monkeypatch)
    monkeypatch.setattr(audio, "installed_models", _no_models)
    config.save(_remote())
    services.write_remote_key(config.load(), KEY)
    _fake_strata(
        monkeypatch,
        httpx.MockTransport(_listing),
    )
    args = argparse.Namespace(essential=False, wait=False, no_service=False)

    monkeypatch.setattr(cli, "_unit_state", _only_strata_active)
    notes = " ".join(cli.doctor_report(args).notes)
    assert "systemctl --user disable --now jarvis-strata.service" in notes

    monkeypatch.setattr(cli, "_unit_state", _inactive)
    assert "jarvis-strata" not in " ".join(cli.doctor_report(args).notes)


# Test gaps: the key never leaks, the agent formats, the SSH target.

SENTINEL = "sentinel-key-9f3a"


def test_the_key_never_reaches_logs_audit_or_error_text(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    caplog.set_level(logging.DEBUG)
    cfg = _remote()
    config.save(cfg)
    services.write_remote_key(cfg, SENTINEL)

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"offline {request.headers['Authorization']}", request=request)

    model = replace(app.strata_model(cfg, SENTINEL), transport=httpx.MockTransport(refuse))
    status = model.probe()
    with pytest.raises(RemoteModelError) as error:
        model.chat([], [])
    texts = [status.report("m"), str(error.value), repr(model), repr(status)]
    assert not any(SENTINEL in text for text in texts)

    _fake_strata(monkeypatch, httpx.MockTransport(refuse))
    assert cli.main(["ask", "hello"]) == 0
    seen = capsys.readouterr()
    assert "remote model server" in seen.out
    assert SENTINEL not in seen.out + seen.err
    audit = paths.audit_file()
    assert audit.exists()
    assert SENTINEL not in audit.read_text(encoding="utf-8")
    assert SENTINEL not in " ".join(r.getMessage() for r in caplog.records)


def test_a_refused_key_is_not_echoed_in_the_setup_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO(f"{SENTINEL} and more\n"))
    assert cli.main(["setup", "--remote", TUNNEL]) == 2
    seen = capsys.readouterr()
    assert SENTINEL not in seen.out + seen.err


def _one_json_line(capsys: pytest.CaptureFixture[str]) -> dict[str, Any]:
    out = capsys.readouterr().out
    assert out.count("\n") == 1, out
    return json.loads(out)


def test_status_and_doctor_answer_an_agent_in_one_line_in_remote_mode(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _stub_subprocess(monkeypatch)
    monkeypatch.setattr(audio, "installed_models", _no_models)
    monkeypatch.delenv("JARVIS_AGENT_FORMAT", raising=False)
    config.save(_remote())
    services.write_remote_key(config.load(), KEY)
    _fake_strata(
        monkeypatch,
        httpx.MockTransport(_listing),
    )
    capsys.readouterr()

    assert cli.main(["status", "--agent"]) == 0
    status = _one_json_line(capsys)
    assert status["command"] == "status"
    assert status["ok"] is True

    cli.main(["doctor", "--agent"])
    doctor = _one_json_line(capsys)
    assert doctor["command"] == "doctor"
    assert "ok" in doctor
    assert KEY not in json.dumps(doctor)


def test_status_and_doctor_answer_an_agent_in_one_line_in_backend_mode(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _stub_subprocess(monkeypatch)
    monkeypatch.setattr(cli, "_memlock_notes", _no_notes)
    config.save(_backend(home, "127.0.0.1"))
    write_private(paths.backend_strata_key_file(), f"STRATA_API_KEY={KEY}\n")
    _fake_strata(
        monkeypatch,
        httpx.MockTransport(_listing),
    )
    capsys.readouterr()

    assert cli.main(["status", "--agent"]) == 0
    assert _one_json_line(capsys)["command"] == "status"

    cli.main(["doctor", "--agent"])
    doctor = _one_json_line(capsys)
    assert doctor["command"] == "doctor"
    assert KEY not in json.dumps(doctor)


@pytest.mark.parametrize(
    "target",
    ["-oProxyCommand=x", "-J evil", "host name", "host;rm", "user@-oProxyCommand=x", "$(id)"],
)
def test_an_ssh_target_that_could_be_an_option_or_a_command_is_refused(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], target: str
) -> None:
    _stub_subprocess(monkeypatch)
    install_engine(home / "data")
    run = engine_run_config(home)
    assert cli.main(["setup", "--backend", "--run-config", str(run), f"--ssh-target={target}"]) == 2
    assert "--ssh-target" in capsys.readouterr().err
    assert not services.strata_profile_file().exists()


def test_a_good_ssh_target_is_quoted_after_a_double_dash(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _stub_subprocess(monkeypatch)
    install_engine(home / "data")
    run = engine_run_config(home)
    assert (
        cli.main(
            [
                "setup",
                "--backend",
                "--run-config",
                str(run),
                "--ssh-target",
                "me@model.example.test",
            ]
        )
        == 0
    )
    assert "-- me@model.example.test" in capsys.readouterr().out


def test_a_probe_status_cannot_carry_a_url_or_key() -> None:
    status = ModelStatus(reachable=False, latency_ms=3, issue="unreachable")
    assert "http" not in status.report("m")
