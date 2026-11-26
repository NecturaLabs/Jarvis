"""The model server on another machine (ADR 0020): roles, the client's server, the backend's setup.

Nothing here reaches a real server: the client's requests are answered by httpx mock transports,
and systemd by stand-ins. The SSH recipe is inspected as text.
"""

import argparse
import io
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from strata_fixture import engine_run_config, install_engine

from jarvis import app, audio, cli, config, modes, paths, services
from jarvis.model import ModelError, StrataModel
from jarvis.secretfile import read_api_key, write_private


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point every XDG folder at `tmp_path` and stand in for systemd."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    def inactive(_unit: str) -> str:
        return "inactive"

    def systemctl(*_args: str) -> int:
        return 0

    monkeypatch.setattr(cli, "_unit_state", inactive)
    monkeypatch.setattr(cli, "_systemctl", systemctl)
    return tmp_path


pytestmark = pytest.mark.usefixtures("home")


def _stub_subprocess(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    calls: list[list[str]] = []

    def run(cmd: list[str], **_kwargs: object) -> SimpleNamespace:
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stdout="")

    monkeypatch.setattr(cli.subprocess, "run", run)
    return calls


def _stub_systemctl(monkeypatch: pytest.MonkeyPatch, calls: list[list[str]]) -> None:
    def systemctl(*args: str) -> int:
        calls.append(["systemctl", "--user", *args])
        return 0

    monkeypatch.setattr(cli, "_systemctl", systemctl)


# The default stays all-local, byte for byte.

# What `jarvis setup` wrote on dev (9399669). This is the unchanged local setup fixture.
DEV_CONFIG = """\
schema_version = 1

[strata]
vram_reserve_mib = 12000
run_config = "<home>/install/strata-q4.json"

[spotify]
client_id = "abc"
"""
DEV_PROFILE = """\
{
 "exe": "component:strata-engine",
 "args": [
  "--pack",
  "<home>/packs/q4",
  "--native",
  "<home>/models/q4.gguf",
  "--expert-cache",
  "auto",
  "--prefill",
  "auto",
  "--max-context",
  "24576",
  "--kv",
  "int8",
  "--vision",
  "--vram-reserve-mib",
  "12000"
 ],
 "cwd": "<home>/data/strata/terminal/current",
 "tokenizer": "<home>/packs/q4/tokenizer",
 "model_name": "flash-q4",
 "log": "<home>/state/jarvis/strata.log",
 "lib_dirs": [],
 "port": 8080,
 "backend": "hip",
 "gpu": 0,
 "gpus_asked": true,
 "args_written": [
  "--pack",
  "<home>/packs/q4",
  "--native",
  "<home>/models/q4.gguf",
  "--expert-cache",
  "auto",
  "--prefill",
  "auto",
  "--max-context",
  "24576",
  "--kv",
  "int8",
  "--vision",
  "--vram-reserve-mib",
  "12000"
 ],
 "host": "127.0.0.1",
 "open_browser": false,
 "parallel": 1,
 "lazy_load": true,
 "idle_unload_s": 86400,
 "reasoning_budget_tokens": 512,
 "vision": {
  "exe": "<home>/data/strata/components/engine/0.1.0/strata-vision",
  "mmproj": "<home>/models/mmproj.gguf",
  "model": "<home>/models/q4.gguf",
  "gpu": false,
  "max_tokens": 300,
  "threads": 8,
  "elastic": true,
  "elastic_idle_s": 120
 }
}
"""
DEV_STRATA_UNIT = """\
[Unit]
Description=Strata model server for Jarvis
PartOf=graphical-session.target
After=graphical-session.target
StartLimitIntervalSec=15min
StartLimitBurst=5
ConditionPathExists=<home>/models/q4.gguf

[Service]
Type=exec
Environment=STRATA_DATA_DIR=<home>/data/strata
Environment=XDG_DATA_HOME=<home>/data
Environment=PIPEWIRE_REMOTE=/nonexistent
Environment=PULSE_SERVER=unix:/nonexistent
EnvironmentFile=<home>/state/jarvis/strata.env
ExecStartPre=<bin>/python -I -m jarvis.ramguard --config <home>/config/jarvis/config.toml \
--profile <home>/config/jarvis/strata-profile.json
ExecStart=<home>/data/strata/terminal/current/bin/strata serve --profile \
<home>/config/jarvis/strata-profile.json \\
  --ready-file <home>/state/jarvis/strata-ready.json
Restart=on-failure
RestartPreventExitStatus=75
RestartSec=5
RestartSteps=5
RestartMaxDelaySec=120
MemoryHigh=40G
MemoryMax=44G
LimitMEMLOCK=infinity
MemorySwapMax=0
CPUWeight=50
TimeoutStopSec=45
NoNewPrivileges=yes
LockPersonality=yes
RestrictSUIDSGID=yes
UMask=0077

[Install]
WantedBy=graphical-session.target
"""
DEV_VOICE_UNIT = """\
[Unit]
Description=Jarvis voice agent
PartOf=graphical-session.target
After=graphical-session.target pipewire.service wireplumber.service jarvis-strata.service
Wants=jarvis-strata.service
StartLimitIntervalSec=15min
StartLimitBurst=5

[Service]
Type=exec
Environment=XDG_CONFIG_HOME=<home>/config
Environment=XDG_DATA_HOME=<home>/data
Environment=XDG_STATE_HOME=<home>/state
ExecStart=<bin>/jarvis run
Restart=on-failure
RestartPreventExitStatus=2
RestartSec=5
RestartSteps=5
RestartMaxDelaySec=120
MemoryMax=4G
CPUWeight=50
TimeoutStopSec=15
NoNewPrivileges=yes
LockPersonality=yes
RestrictSUIDSGID=yes
SystemCallArchitectures=native

[Install]
WantedBy=graphical-session.target
"""


def test_a_default_setup_writes_what_dev_wrote(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _stub_subprocess(monkeypatch)

    def nowhere(_name: str) -> str | None:
        return None

    monkeypatch.setattr(services.shutil, "which", nowhere)  # the Strata under the home
    monkeypatch.setattr(services.os, "cpu_count", lambda: 16)  # the reader's threads
    run = engine_run_config(home)
    install_engine(home / "data")
    config.save(config.Config(strata=config.StrataConfig(vram_reserve_mib=12000)))

    assert cli.main(["setup", "--run-config", str(run), "--spotify-client-id", "abc"]) == 0

    def read(path: Path) -> str:
        text = path.read_text(encoding="utf-8")
        return text.replace(str(Path(sys.executable).parent), "<bin>").replace(str(home), "<home>")

    units = home / "config" / "systemd" / "user"
    assert read(home / "config" / "jarvis" / "config.toml") == DEV_CONFIG
    assert read(home / "config" / "jarvis" / "strata-profile.json") == DEV_PROFILE
    assert read(units / "jarvis-strata.service") == DEV_STRATA_UNIT
    assert read(units / "jarvis.service") == DEV_VOICE_UNIT
    assert sorted(p.name for p in units.iterdir()) == ["jarvis-strata.service", "jarvis.service"]
    assert calls == [["systemctl", "--user", "daemon-reload"]]


TUNNEL = "http://127.0.0.1:18080"
KEY = "fixture-key"
LISTED = {"id": "flash", "status": {"value": "loaded"}, "context_length": 8192, "vision": True}


def _remote(**settings: object) -> config.Config:
    return config.Config(
        strata=config.StrataConfig.model_validate({"mode": "remote", "url": TUNNEL, **settings}),
        spotify=config.SpotifyConfig(client_id="abc"),
    )


@pytest.mark.parametrize(
    ("url", "base"),
    [
        (TUNNEL + "/", TUNNEL + "/v1"),
        ("http://[::1]:18080/v1", "http://[::1]:18080/v1"),
        ("https://model.example.test/strata/v1/", "https://model.example.test/strata/v1"),
    ],
)
def test_remote_base_url(url: str, base: str) -> None:
    assert _remote(url=url).model_base_url() == base


@pytest.mark.parametrize(
    "url",
    [
        "",
        "ftp://192.168.1.20",
        "http://user:secret@127.0.0.1:18080",
        TUNNEL + "/v1?key=secret",
        TUNNEL + "/v1#key=secret",
        TUNNEL + "/v1\nx",
        TUNNEL + ":1",
        "http://:8080",
        "http://0.0.0.0:8080",
        "http://224.0.0.1:8080",
        "http://192.0.2.1:8080",
        "http://model.example.test:8080",
        "https://8.8.8.8:8080",
    ],
)
def test_refuses_unsafe_or_invalid_remote_urls(url: str) -> None:
    with pytest.raises(ValueError, match=r"strata\.url|control characters") as exc:
        _remote(url=url, allow_plain_http=True)
    assert "secret" not in str(exc.value)
    assert "extra_forbidden" not in str(exc.value)


@pytest.mark.parametrize("host", ["192.168.1.20", "100.101.102.103", "[fd00::20]"])
def test_plain_http_to_a_private_ip_is_opt_in(host: str) -> None:
    url = f"http://{host}:8080"
    with pytest.raises(ValueError, match="allow_plain_http"):
        _remote(url=url)
    assert _remote(url=url, allow_plain_http=True).model_base_url() == f"{url}/v1"


def test_local_config_does_not_accept_remote_settings() -> None:
    with pytest.raises(ValueError, match="need remote mode"):
        config.StrataConfig.model_validate({"url": TUNNEL})


def test_remote_uses_its_own_url_and_key(monkeypatch: pytest.MonkeyPatch) -> None:
    config.save(config.Config(model=config.ModelConfig(base_url="http://127.0.0.1:18081/v1")))
    write_private(paths.strata_key_file(), "STRATA_API_KEY=keep-local\n")
    calls = _stub_subprocess(monkeypatch)
    monkeypatch.setattr(sys, "stdin", io.StringIO(KEY + "\n"))

    assert cli.main(["setup", "--remote", TUNNEL, "--enable"]) == 0

    cfg = config.load()
    assert cfg.strata.mode == "remote"
    assert cfg.model_base_url() == TUNNEL + "/v1"
    assert read_api_key(paths.strata_key_file()) == "keep-local"
    key_file = paths.remote_strata_key_file()
    assert services.read_key(cfg) == KEY
    assert key_file.stat().st_mode & 0o777 == 0o600
    assert KEY not in paths.config_file().read_text()
    assert not services.strata_profile_file().exists()
    unit = services.unit_dir() / services.JARVIS_UNIT
    assert services.STRATA_UNIT not in unit.read_text()
    assert list(unit.parent.iterdir()) == [unit]
    assert all(services.STRATA_UNIT not in call for call in calls)
    assert calls[-1] == ["systemctl", "--user", "enable", "--now", services.JARVIS_UNIT]


def test_remote_setup_rejects_invalid_key_before_writing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO(KEY + " extra\n"))
    assert cli.main(["setup", "--remote", TUNNEL]) == 2
    assert KEY not in capsys.readouterr().err
    assert not paths.config_file().exists()
    assert not services.unit_dir().exists()


@pytest.mark.parametrize(
    ("listen", "extra"),
    [("127.0.0.1", []), ("192.168.1.20", ["--listen", "192.168.1.20"])],
)
def test_backend_setup_writes_only_the_backend(
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    listen: str,
    extra: list[str],
) -> None:
    calls = _stub_subprocess(monkeypatch)
    run = engine_run_config(home)
    install_engine(home / "data")

    assert (
        cli.main(
            [
                "setup",
                "--backend",
                "--enable",
                "--run-config",
                str(run),
                "--ssh-target",
                "model.example.test",
                *extra,
            ]
        )
        == 0
    )

    cfg = config.load()
    assert cfg.strata.mode == "backend"
    assert json.loads(services.strata_profile_file().read_text())["host"] == listen
    assert not (services.unit_dir() / services.JARVIS_UNIT).exists()
    unit = (services.unit_dir() / services.STRATA_UNIT).read_text()
    assert "WantedBy=default.target" in unit
    assert "graphical-session.target" not in unit
    key_file = paths.backend_strata_key_file()
    key = read_api_key(key_file)
    assert key
    assert key_file.stat().st_mode & 0o777 == 0o600
    assert str(key_file) in unit
    assert key not in unit
    assert key not in paths.config_file().read_text()
    report = capsys.readouterr().out
    assert f"jarvis setup --remote {TUNNEL}" in report
    assert f"18080:{listen}:8080" in report
    assert "model.example.test" in report
    assert key not in report
    assert all(services.JARVIS_UNIT not in call for call in calls)
    assert calls[-1] == ["systemctl", "--user", "enable", "--now", services.STRATA_UNIT]


@pytest.mark.parametrize(
    "listen",
    ["0.0.0.0", "::", "8.8.8.8", "224.0.0.1", "model.example.test"],  # ruff: ignore[hardcoded-bind-all-interfaces] - rejected bind
)
def test_backend_rejects_nonprivate_bind_addresses(listen: str) -> None:
    with pytest.raises(ValueError, match="listen"):
        config.StrataConfig.model_validate({"mode": "backend", "listen": listen})


def test_backend_setup_refuses_an_existing_voice_install(
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _stub_subprocess(monkeypatch)
    voice = services.unit_dir() / services.JARVIS_UNIT
    voice.parent.mkdir(parents=True)
    voice.write_text(DEV_VOICE_UNIT)

    assert cli.main(["setup", "--backend", "--run-config", str(engine_run_config(home))]) == 2
    assert not paths.config_file().exists()
    assert voice.read_text() == DEV_VOICE_UNIT
    assert not calls


def test_remote_lifecycle_never_touches_local_strata(monkeypatch: pytest.MonkeyPatch) -> None:
    config.save(_remote())
    calls = _stub_subprocess(monkeypatch)
    _stub_systemctl(monkeypatch, calls)

    def active(unit: str) -> str:
        calls.append(["is-active", unit])
        return "active"

    monkeypatch.setattr(cli, "_unit_state", active)
    root = services.unit_dir()
    root.mkdir(parents=True)
    (root / services.JARVIS_UNIT).write_text(DEV_VOICE_UNIT)
    stale = root / services.STRATA_UNIT
    stale.write_text(DEV_STRATA_UNIT)

    for cmd in ("start", "stop", "restart"):
        assert cli.main([cmd]) == 0
    assert cli.main(["unload"]) == 1
    assert all(services.STRATA_UNIT not in call for call in calls)
    assert stale.read_text() == DEV_STRATA_UNIT
    assert services.STRATA_UNIT not in (root / services.JARVIS_UNIT).read_text()
    assert not services.strata_profile_file().exists()


def _fake_strata(
    monkeypatch: pytest.MonkeyPatch,
    transport: httpx.MockTransport,
) -> None:
    real = app.strata_model

    def make(cfg: config.Config, key: str) -> StrataModel:
        return replace(real(cfg, key), transport=transport)

    monkeypatch.setattr(app, "strata_model", make)
    monkeypatch.setattr(cli, "strata_model", make)


def test_remote_doctor_and_status_use_only_authenticated_health_checks(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _stub_subprocess(monkeypatch)

    def no_models(_engine: str) -> None:
        return

    monkeypatch.setattr(audio, "installed_models", no_models)
    config.save(_remote(url="http://192.168.1.20:8080", allow_plain_http=True))
    services.write_remote_key(config.load(), KEY)
    seen: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"data": [LISTED]})

    _fake_strata(monkeypatch, httpx.MockTransport(respond))
    cli.main(["doctor"])
    doctor = capsys.readouterr().out
    assert "flash" in doctor
    assert "ms" in doctor
    assert "unencrypted" in doctor
    assert "Strata run config" not in doctor

    assert cli.main(["status", "--json"]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["model"] == "loaded"
    assert status["model_name"] == "flash"
    assert status["model_latency_ms"] >= 0
    assert status["model_reachable"] is True
    assert [(r.method, r.url.path, r.headers["Authorization"]) for r in seen] == [
        ("GET", "/v1/models", f"Bearer {KEY}"),
    ] * 2


def test_backend_doctor_checks_the_server_without_client_requirements(
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run = engine_run_config(home)
    config.save(
        config.Config(
            strata=config.StrataConfig.model_validate(
                {
                    "mode": "backend",
                    "run_config": run,
                }
            )
        )
    )
    _stub_subprocess(monkeypatch)
    write_private(paths.backend_strata_key_file(), f"STRATA_API_KEY={KEY}\n")
    _fake_strata(
        monkeypatch,
        httpx.MockTransport(lambda _request: httpx.Response(200, json={"data": [LISTED]})),
    )
    cli.main(["doctor"])
    output = capsys.readouterr().out
    assert "flash" in output
    assert "ms" in output
    assert "Spotify" not in output
    assert "Voxtype" not in output
    assert "wake" not in output


def test_remote_metadata_sets_limits_picture_support_and_about(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _remote()
    services.write_remote_key(cfg, KEY)
    seen: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.method == "GET":
            return httpx.Response(200, json={"data": [LISTED]})
        return httpx.Response(200, json={"choices": [{"message": {"content": "hello"}}]})

    _fake_strata(monkeypatch, httpx.MockTransport(respond))
    built = app.build(cfg, via="chat", units=cli.StrataUnits())
    assert built.modes is not None
    for mode in ("normal", "coding"):
        setup = built.modes.setup(mode)
        assert setup.limits.context_tokens == 8192
        assert setup.pictures
        about = setup.registry.get("about_jarvis")
        assert about is not None
        assert "flash" in about.run({}).text
        assert "8192" in about.run({}).text
    built.model.chat([], [])
    assert json.loads(seen[-1].content)["model"] == "flash"


def test_missing_remote_metadata_uses_the_local_client_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _remote(context=4096, remote_pictures=True)
    services.write_remote_key(cfg, KEY)
    _fake_strata(
        monkeypatch,
        httpx.MockTransport(
            lambda _request: httpx.Response(200, json={"data": [{"status": {"value": "loaded"}}]})
        ),
    )
    built = app.build(cfg, units=cli.StrataUnits())
    assert built.modes is not None
    setup = built.modes.setup("coding")
    assert setup.limits.context_tokens == 4096
    assert setup.pictures
    assert built.model.config.model == "jarvis"


def test_remote_coding_changes_client_limits_without_touching_strata(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls = _stub_subprocess(monkeypatch)
    cfg = _remote(context=4096)
    config.save(cfg)
    assert cli.main(["model", "coding"]) == 0
    assert "remote model stays the same" in capsys.readouterr().out.lower()
    assert modes.current() == "coding"
    assert modes.effective(cfg, "coding").chat.max_steps == cfg.coding.chat.max_steps
    assert modes.effective(cfg, "coding").strata.context == 4096
    assert not services.strata_profile_file().exists()
    assert not calls


def test_remote_connect_failure_is_tried_once_without_polling_or_loading() -> None:
    cfg = _remote()
    seen: list[httpx.Request] = []

    def refuse(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        assert request.extensions["timeout"]["connect"] <= 2.0
        raise httpx.ConnectError("offline", request=request)

    model = replace(app.strata_model(cfg, KEY), transport=httpx.MockTransport(refuse))
    assert model.probe().reachable is False
    assert not model.warm()
    assert not model.load()
    assert not model.unload()
    with pytest.raises(ModelError, match="remote model server"):
        model.chat([], [])
    # One probe, whose failure is remembered for a few seconds; the chat then does not wait out
    # the same dead server twice (a chat with no recent probe retries once: test_remote_hardening).
    assert [(r.method, r.url.path) for r in seen] == [
        ("GET", "/v1/models"),
        ("POST", "/v1/chat/completions"),
    ]


def test_a_redirect_cannot_send_the_remote_key_to_another_address() -> None:
    seen: list[str] = []

    def redirect(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.host)
        return httpx.Response(302, headers={"Location": "https://other.example.test/v1/models"})

    model = replace(app.strata_model(_remote(), KEY), transport=httpx.MockTransport(redirect))
    assert not model.probe().reachable
    assert seen == ["127.0.0.1"]


def test_remote_setup_without_a_url_reuses_the_existing_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config.save(_remote())
    _stub_subprocess(monkeypatch)
    services.write_remote_key(config.load(), KEY)

    assert cli.main(["setup", "--enable"]) == 0
    assert services.read_key(config.load()) == KEY


def test_backend_lifecycle_only_targets_strata(
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _stub_subprocess(monkeypatch)
    _stub_systemctl(monkeypatch, calls)
    run = engine_run_config(home)
    install_engine(home / "data")
    assert cli.main(["setup", "--backend", "--run-config", str(run)]) == 0
    for cmd in ("start", "stop", "restart"):
        assert cli.main([cmd]) == 0
    assert all(services.JARVIS_UNIT not in call for call in calls)
    assert [call[2] for call in calls if services.STRATA_UNIT in call] == [
        "start",
        "stop",
        "restart",
    ]


def test_an_unreachable_remote_model_prints_what_failed(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config.save(_remote())
    services.write_remote_key(config.load(), KEY)

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    _fake_strata(monkeypatch, httpx.MockTransport(refuse))
    assert cli.main(["ask", "hello"]) == 0
    assert "remote model server" in capsys.readouterr().out


def test_an_endpoint_change_cannot_send_the_previous_servers_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_subprocess(monkeypatch)
    monkeypatch.setattr(sys, "stdin", io.StringIO(KEY + "\n"))
    assert cli.main(["setup", "--remote", TUNNEL]) == 0

    _stub_subprocess(monkeypatch)
    config.save(_remote(url="http://127.0.0.1:18081"))

    assert services.read_key(config.load()) is None


def test_failed_voice_unit_write_does_not_publish_remote_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config.save(config.Config())
    monkeypatch.setattr(sys, "stdin", io.StringIO(KEY + "\n"))

    def full() -> Path:
        msg = "disk full"
        raise OSError(msg)

    monkeypatch.setattr(services, "write_voice_unit", full)
    assert cli.main(["setup", "--remote", TUNNEL]) == 2
    assert config.load().strata.mode == "local"


def test_setup_rejects_an_unusable_tls_ca(
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO(KEY + "\n"))
    assert (
        cli.main(
            ["setup", "--remote", "https://model.example.test", "--tls-ca", str(home / "gone.pem")]
        )
        == 2
    )
    assert not paths.config_file().exists()
    assert not paths.remote_strata_key_file().exists()


_DOCTOR_ARGS = argparse.Namespace(essential=False, wait=False, no_service=False)


def _no_models(_engine: str) -> None:
    return


def _no_checks(_cfg: config.Config) -> tuple[list[tuple[bool, str, str]], list[str]]:
    return [], []


def test_a_remote_client_has_no_run_config_to_ask_for() -> None:
    config.save(_remote())
    assert cli.main(["setup", "--has-run-config"]) == 0


def test_the_doctor_report_carries_the_remote_server_line_and_the_plain_http_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_subprocess(monkeypatch)
    monkeypatch.setattr(audio, "installed_models", _no_models)
    config.save(_remote(url="http://192.168.1.20:8080", allow_plain_http=True))
    services.write_remote_key(config.load(), KEY)
    _fake_strata(
        monkeypatch,
        httpx.MockTransport(lambda _request: httpx.Response(200, json={"data": [LISTED]})),
    )
    report = cli.doctor_report(_DOCTOR_ARGS)
    served = [check for check in report.checks if "flash" in check[1]]
    assert len(served) == 1
    assert served[0][0]
    assert not any("Strata server" in what for _, what, _ in report.checks)
    assert any("unencrypted" in note for note in report.notes)
    assert not any("Strata run config" in what for _, what, _ in report.checks)


def test_the_doctor_report_on_a_backend_leaves_out_the_client_checks(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_subprocess(monkeypatch)
    install_engine(home / "data")
    config.save(
        config.Config(
            strata=config.StrataConfig.model_validate(
                {"mode": "backend", "run_config": engine_run_config(home)}
            )
        )
    )
    write_private(paths.backend_strata_key_file(), f"STRATA_API_KEY={KEY}\n")
    _fake_strata(
        monkeypatch,
        httpx.MockTransport(lambda _request: httpx.Response(200, json={"data": [LISTED]})),
    )
    report = cli.doctor_report(_DOCTOR_ARGS)
    text = " ".join(what for _, what, _ in report.checks) + " ".join(report.notes)
    assert "flash" in text
    assert not any(word in text for word in ("Spotify", "wake", "Voxtype"))


@pytest.mark.parametrize(
    ("mode", "voice", "strata"), [("remote", True, False), ("backend", False, True)]
)
def test_the_update_health_step_writes_only_the_files_of_its_role(
    home: Path, monkeypatch: pytest.MonkeyPatch, mode: str, *, voice: bool, strata: bool
) -> None:
    install_engine(home / "data")
    settings: dict[str, object] = {"mode": mode}
    if mode == "remote":
        settings["url"] = TUNNEL
    else:
        settings["run_config"] = engine_run_config(home)
    config.save(config.Config(strata=config.StrataConfig.model_validate(settings)))
    monkeypatch.setattr(sys, "argv", ["jarvis", "health"])
    monkeypatch.setattr(cli, "essential_checks", _no_checks)
    cli.finish_update()
    assert (services.unit_dir() / services.JARVIS_UNIT).exists() is voice
    assert (services.unit_dir() / services.STRATA_UNIT).exists() is strata
    assert services.strata_profile_file().exists() is strata
