"""The guard that keeps the model's load from taking the memory a running program needs.

Regression: with a game running and 13 to 20 GB of RAM available, the model server was started and
loaded its experts (about 39 GB) into RAM; the kernel failed page allocations and the game died.
Every test fakes `/proc/meminfo` and the model file's size: nothing here reads the PC's memory.
"""

import json
import os
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import pytest
from strata_fixture import engine_run_config, install_engine

from jarvis import agent, cli, config, modes, paths, ramguard, services, update
from jarvis import audio as audio_module
from jarvis.config import Config, StrataConfig
from jarvis.model import ModelError, ModelMemoryError, StrataModel
from jarvis.services import STRATA_UNIT

if TYPE_CHECKING:
    from collections.abc import Callable

GIB = 1024**3
MODEL = 39_225_954_304  # a 39 GB model's main file, Strata says "about 39 GB" (whole KiB)


@pytest.fixture(autouse=True)  # ruff: ignore[pytest-fixture-autouse] - every test needs its own folders and a fake meminfo
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point every XDG folder at `tmp_path` and stand in for systemd."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setattr(ramguard, "CGROUP_ROOT", tmp_path / "cgroup")
    meminfo(tmp_path, monkeypatch, available=64 * GIB)
    return tmp_path


def _no_answer() -> str:
    return ""


def _no_models(_engine: str) -> frozenset[str] | None:
    return None


@pytest.fixture
def doctor_without_programs(monkeypatch: pytest.MonkeyPatch) -> None:
    """`jarvis doctor` asks systemd and Voxtype; here neither has an answer."""
    monkeypatch.setattr(cli, "_user_memlock_limit", _no_answer)
    monkeypatch.setattr(audio_module, "installed_models", _no_models)


def meminfo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, available: float) -> None:
    """Make `/proc/meminfo` say that `available` bytes are available."""
    path = tmp_path / "meminfo"
    path.write_text(
        f"MemTotal:       65536000 kB\nMemFree:         1000000 kB\n"
        f"MemAvailable:   {int(available) // 1024} kB\nBuffers:          100 kB\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(ramguard, "MEMINFO", path)


def big_model(tmp_path: Path, size: int = MODEL) -> Path:
    """Return a run config whose model file is `size` bytes long (sparse: it takes no disk)."""
    run = engine_run_config(tmp_path)
    data = json.loads(run.read_text(encoding="utf-8"))
    model = Path(data["args"][data["args"].index("--native") + 1])
    with model.open("wb") as file:
        file.truncate(size)
    return run


def local(tmp_path: Path, **strata: object) -> Config:
    """Save and return a config that serves the big model."""
    install_engine(tmp_path / "components")
    cfg = Config(
        strata=StrataConfig.model_validate(
            {
                "home": tmp_path / "strata-home",
                "components_parent": tmp_path / "components",
                "run_config": big_model(tmp_path),
                "vram_reserve_mib": 12000,
                **strata,
            }
        )
    )
    config.save(cfg)
    return cfg


def test_the_default_margin_is_four_gib_and_the_override_is_off() -> None:
    strata = StrataConfig()

    assert strata.min_free_ram_mib == 4096
    assert strata.allow_low_ram is False


def test_available_memory_is_what_the_kernel_counts_in_meminfo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    meminfo(tmp_path, monkeypatch, available=17 * GIB)

    assert ramguard.available_bytes() == 17 * GIB


def test_a_meminfo_that_cannot_be_read_gives_no_figure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ramguard, "MEMINFO", tmp_path / "nothing-here")

    assert ramguard.available_bytes() is None


def test_a_load_that_leaves_less_than_the_margin_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The defect: 17 GiB available, a 36.5 GiB model.
    cfg = local(tmp_path)
    meminfo(tmp_path, monkeypatch, available=17 * GIB)

    verdict = ramguard.check(ramguard.served_model(cfg), cfg.strata)

    assert verdict.short
    assert "36.5 GiB" in verdict.text
    assert "17.0 GiB" in verdict.text
    assert "strata.min_free_ram_mib" in verdict.text
    assert "strata.allow_low_ram" in verdict.text


def test_a_load_that_fits_with_the_margin_to_spare_is_allowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = local(tmp_path)
    meminfo(tmp_path, monkeypatch, available=MODEL + 4 * GIB)

    assert not ramguard.check(ramguard.served_model(cfg), cfg.strata).short


def test_a_load_one_byte_inside_the_margin_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = local(tmp_path, min_free_ram_mib=1024)
    meminfo(tmp_path, monkeypatch, available=MODEL + 1024**3 - 1)

    assert ramguard.check(ramguard.served_model(cfg), cfg.strata).short


def test_a_load_that_leaves_exactly_the_margin_is_allowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = local(tmp_path, min_free_ram_mib=1024)
    meminfo(tmp_path, monkeypatch, available=MODEL + GIB)

    assert not ramguard.check(ramguard.served_model(cfg), cfg.strata).short


def test_the_margin_is_the_owners_to_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = local(tmp_path, min_free_ram_mib=0)
    meminfo(tmp_path, monkeypatch, available=MODEL + 1)

    assert not ramguard.check(ramguard.served_model(cfg), cfg.strata).short


def test_the_override_loads_anyway(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = local(tmp_path, allow_low_ram=True)
    meminfo(tmp_path, monkeypatch, available=2 * GIB)

    assert not ramguard.reason(ramguard.served_model(cfg), cfg.strata)


def test_a_model_or_memory_that_is_not_known_blocks_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = local(tmp_path)
    meminfo(tmp_path, monkeypatch, available=1 * GIB)

    assert not ramguard.reason(None, cfg.strata)
    assert not ramguard.reason(tmp_path / "gone.gguf", cfg.strata)
    monkeypatch.setattr(ramguard, "MEMINFO", tmp_path / "nothing-here")
    assert not ramguard.reason(ramguard.served_model(cfg), cfg.strata)


def _server_memory(tmp_path: Path, anon: int) -> None:
    """Make the model server's cgroup say that it holds `anon` bytes of anonymous memory."""
    uid = ramguard.os.getuid()
    folder = (
        tmp_path / "cgroup" / "user.slice" / f"user-{uid}.slice" / f"user@{uid}.service"
        / "app.slice" / STRATA_UNIT
    )  # fmt: skip
    folder.mkdir(parents=True)
    (folder / "memory.stat").write_text(f"anon {anon}\nfile 123\nshmem 0\n", encoding="utf-8")


def test_the_memory_a_running_server_gives_back_counts_as_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A restart frees the loaded model first: its own memory is not what is in the way.
    cfg = local(tmp_path)
    meminfo(tmp_path, monkeypatch, available=20 * GIB)
    _server_memory(tmp_path, anon=MODEL)

    held = ramguard.held_by_unit(STRATA_UNIT)
    verdict = ramguard.check(ramguard.served_model(cfg), cfg.strata, credit=held)

    assert held == MODEL
    assert not verdict.short


def test_no_server_cgroup_gives_back_nothing() -> None:
    assert ramguard.held_by_unit(STRATA_UNIT) == 0


def test_the_unit_skips_the_start_when_the_model_would_not_fit(tmp_path: Path) -> None:
    cfg = local(tmp_path)

    text = services.strata_unit(cfg, paths.strata_key_file())

    lines = text.splitlines()
    pre = next(line for line in lines if line.startswith("ExecCondition="))
    assert "-m jarvis.ramguard" in pre
    assert f"--config {paths.config_file()}" in pre
    assert f"--profile {services.strata_profile_file()}" in pre
    assert lines.index(pre) < next(
        i for i, line in enumerate(lines) if line.startswith("ExecStart=")
    )


def _run_guard(capsys: pytest.CaptureFixture[str]) -> tuple[int, str]:
    code = ramguard.main(
        ["--config", str(paths.config_file()), "--profile", str(services.strata_profile_file())]
    )
    return code, capsys.readouterr().err


def test_the_units_check_refuses_with_a_reason_and_a_code_of_its_own(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = local(tmp_path)
    services.write_strata_profile(cfg)
    meminfo(tmp_path, monkeypatch, available=15 * GIB)

    code, said = _run_guard(capsys)

    assert code == ramguard.EXIT_SHORT
    assert "36.5 GiB" in said
    assert "15.0 GiB" in said


def test_the_units_check_lets_a_load_that_fits_start(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    services.write_strata_profile(local(tmp_path))

    assert _run_guard(capsys) == (0, "")


def test_the_units_check_reads_the_override_from_the_config_when_it_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = local(tmp_path)
    services.write_strata_profile(cfg)
    meminfo(tmp_path, monkeypatch, available=15 * GIB)
    assert _run_guard(capsys)[0] == ramguard.EXIT_SHORT

    config.save(
        cfg.model_copy(update={"strata": cfg.strata.model_copy(update={"allow_low_ram": True})})
    )

    assert _run_guard(capsys) == (0, "")


def test_the_units_check_serves_the_model_in_the_profile_not_the_one_of_the_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The profile is rewritten before a mode's server starts: it is what will load.
    cfg = local(tmp_path)
    services.write_strata_profile(cfg)
    profile = json.loads(services.strata_profile_file().read_text(encoding="utf-8"))
    small = tmp_path / "models" / "small.gguf"
    small.write_bytes(b"x")
    profile["args"][profile["args"].index("--native") + 1] = str(small)
    services.strata_profile_file().write_text(json.dumps(profile), encoding="utf-8")
    meminfo(tmp_path, monkeypatch, available=15 * GIB)

    assert _run_guard(capsys) == (0, "")


def test_the_units_check_blocks_nothing_when_it_cannot_tell(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _run_guard(capsys)[0] == 0  # no profile written yet


def _systemctl(monkeypatch: pytest.MonkeyPatch, states: dict[str, str]) -> list[tuple[str, ...]]:
    calls: list[tuple[str, ...]] = []

    def systemctl(*args: str) -> int:
        calls.append(args)
        return 0

    def state(unit: str) -> str:
        return states.get(unit, "inactive")

    monkeypatch.setattr(cli, "_systemctl", systemctl)
    monkeypatch.setattr(cli, "_unit_state", state)
    return calls


def test_start_refuses_to_load_into_too_little_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    local(tmp_path)
    meminfo(tmp_path, monkeypatch, available=17 * GIB)
    calls = _systemctl(monkeypatch, {})

    assert cli.main(["start"]) == 1

    assert not any(call[0] == "start" for call in calls)
    err = capsys.readouterr().err
    assert "not enough free memory" in err
    assert "strata.allow_low_ram" in err


def test_start_goes_ahead_when_the_model_fits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local(tmp_path)
    calls = _systemctl(monkeypatch, {})

    assert cli.main(["start"]) == 0

    assert ("start", STRATA_UNIT, "jarvis.service") in calls


def test_start_goes_ahead_with_the_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local(tmp_path, allow_low_ram=True)
    meminfo(tmp_path, monkeypatch, available=17 * GIB)
    calls = _systemctl(monkeypatch, {})

    assert cli.main(["start"]) == 0

    assert ("start", STRATA_UNIT, "jarvis.service") in calls


def test_start_leaves_a_server_that_already_runs_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Its model is loaded, so the memory it holds is not available: nothing is being loaded.
    local(tmp_path)
    meminfo(tmp_path, monkeypatch, available=2 * GIB)
    calls = _systemctl(monkeypatch, {STRATA_UNIT: "active"})

    assert cli.main(["start"]) == 0

    assert ("start", STRATA_UNIT, "jarvis.service") in calls


def _started(at: float | None) -> Callable[[str], float | None]:
    """Answer when the model server last started (None: it does not run)."""

    def started(_unit: str) -> float | None:
        return at

    return started


def _install(cfg: Config) -> None:
    """Write the profile and the unit as `jarvis setup` leaves them."""
    services.write_strata_profile(cfg)
    services.write_strata_unit(cfg, paths.strata_key_file())


def test_restart_that_would_load_into_too_little_memory_changes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    local(tmp_path)
    meminfo(tmp_path, monkeypatch, available=17 * GIB)
    calls = _systemctl(monkeypatch, {})
    monkeypatch.setattr(cli, "_started_at", _started(None))

    assert cli.main(["restart"]) == 1

    assert not [call for call in calls if call[0] in {"start", "restart"}]
    assert "not enough free memory" in capsys.readouterr().err


def test_restart_counts_the_memory_the_running_server_gives_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(local(tmp_path))
    meminfo(tmp_path, monkeypatch, available=8 * GIB)
    _server_memory(tmp_path, anon=MODEL)
    calls = _systemctl(monkeypatch, {STRATA_UNIT: "active"})
    monkeypatch.setattr(cli, "_started_at", _started(1.0))  # older than its files

    assert cli.main(["restart"]) == 0

    assert ("restart", STRATA_UNIT) in calls


def test_restart_of_a_server_that_runs_unchanged_is_not_a_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(local(tmp_path))
    meminfo(tmp_path, monkeypatch, available=2 * GIB)
    calls = _systemctl(monkeypatch, {STRATA_UNIT: "active"})
    monkeypatch.setattr(cli, "_started_at", _started(4_000_000_000.0))

    assert cli.main(["restart"]) == 0

    assert ("restart", "jarvis.service") in calls


def test_the_backend_unit_start_is_guarded_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = local(tmp_path)
    config.save(
        cfg.model_copy(update={"strata": cfg.strata.model_copy(update={"mode": "backend"})})
    )
    meminfo(tmp_path, monkeypatch, available=17 * GIB)
    calls = _systemctl(monkeypatch, {})

    assert cli.main(["start"]) == 1

    assert not any(call[0] == "start" for call in calls)
    assert "not enough free memory" in capsys.readouterr().err


def _doctor_lines(capsys: pytest.CaptureFixture[str]) -> list[str]:
    return [
        line
        for line in capsys.readouterr().out.splitlines()
        if "memory for the model" in line and line.split()[0] in {"ok", "FAIL", "note"}
    ]


@pytest.mark.usefixtures("doctor_without_programs")
def test_doctor_fails_the_memory_check_when_the_model_would_not_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    local(tmp_path)
    meminfo(tmp_path, monkeypatch, available=17 * GIB)
    _systemctl(monkeypatch, {})

    cli.main(["doctor", "--no-service"])

    (line,) = _doctor_lines(capsys)
    assert line.startswith("FAIL")
    assert "strata.allow_low_ram" in line


@pytest.mark.usefixtures("doctor_without_programs")
def test_doctor_passes_the_memory_check_when_it_fits(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    local(tmp_path)
    _systemctl(monkeypatch, {})

    cli.main(["doctor", "--no-service"])

    (line,) = _doctor_lines(capsys)
    assert line.startswith("ok")


@pytest.mark.usefixtures("doctor_without_programs")
def test_the_updates_health_check_does_not_fail_on_memory_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # An update must not be rolled back because a game is running.
    local(tmp_path)
    meminfo(tmp_path, monkeypatch, available=17 * GIB)
    _systemctl(monkeypatch, {})

    cli.main(["doctor", "--no-service", "--essential"])

    (line,) = _doctor_lines(capsys)
    assert line.startswith("note")


@dataclass
class _Units:
    running: bool = True
    restarts: int = 0

    def active(self) -> bool:
        return self.running

    def restart(self) -> bool:
        self.restarts += 1
        return True


@dataclass
class _Server:
    state: str | None = "loaded"
    loads: int = 0

    def status(self) -> str | None:
        return self.state

    def load(self) -> bool:
        self.loads += 1
        return True


class _Audit:
    def record(self, event: str, **fields: object) -> None:
        pass


def _coding_config(tmp_path: Path) -> Config:
    cfg = local(tmp_path)
    coder = json.loads(cfg.strata.run_config.read_text(encoding="utf-8"))
    big = tmp_path / "models" / "coder.gguf"
    with big.open("wb") as file:
        file.truncate(MODEL)
    coder["args"] = [str(big) if a.endswith("q4.gguf") else a for a in coder["args"]]
    coder["model_name"] = "flash-coder"
    path = tmp_path / "install" / "strata-coder.json"
    path.write_text(json.dumps(coder), encoding="utf-8")
    return cfg.model_copy(
        update={"strata": cfg.strata.model_copy(update={"coding_run_config": path})}
    )


def _switcher(cfg: Config, units: _Units, ram: Callable[[Config], str]) -> modes.Switcher:
    return modes.Switcher(cfg, units, lambda _cfg: _Server(), _Audit(), ram_reason=ram)


def test_a_switch_that_would_reload_into_too_little_memory_leaves_the_server_running(
    tmp_path: Path,
) -> None:
    cfg = _coding_config(tmp_path)
    services.write_strata_profile(cfg)
    services.write_strata_unit(cfg, paths.strata_key_file())
    before = services.strata_profile_file().read_text(encoding="utf-8")
    units = _Units()

    done = _switcher(cfg, units, lambda _served: "not enough free memory").switch("coding", "owner")

    assert units.restarts == 0
    assert done.mode == "normal"
    assert "not enough free memory" in done.error
    assert services.strata_profile_file().read_text(encoding="utf-8") == before
    assert modes.current() == "normal"


def test_a_switch_asks_about_the_mode_it_goes_to(tmp_path: Path) -> None:
    cfg = _coding_config(tmp_path)
    services.write_strata_profile(cfg)
    services.write_strata_unit(cfg, paths.strata_key_file())
    asked: list[str] = []

    def ram(served: Config) -> str:
        asked.append(services.model_name(served))
        return ""

    done = _switcher(cfg, _Units(), ram).switch("coding", "owner")

    assert asked == ["flash-coder"]
    assert done.mode == "coding"
    assert done.restarted


def test_a_switch_with_no_server_running_loads_nothing_and_asks_nothing(tmp_path: Path) -> None:
    cfg = _coding_config(tmp_path)
    services.write_strata_profile(cfg)
    services.write_strata_unit(cfg, paths.strata_key_file())

    def ram(_served: Config) -> str:
        pytest.fail("no model is loaded by this switch")

    assert _switcher(cfg, _Units(running=False), ram).switch("coding", "owner").mode == "coding"


def test_the_commands_switch_of_modes_is_guarded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _coding_config(tmp_path)
    config.save(cfg)
    services.write_strata_profile(cfg)
    services.write_strata_unit(cfg, paths.strata_key_file())
    meminfo(tmp_path, monkeypatch, available=17 * GIB)
    calls = _systemctl(monkeypatch, {STRATA_UNIT: "active"})

    said = cli.switch_model(cfg, "coding", lambda _line: None, "test")

    assert "enough free memory" in said
    assert not any(call[0] == "restart" for call in calls)


def _short() -> ramguard.Verdict:
    return ramguard.Verdict(MODEL, 17 * GIB, 4 * GIB, model=Path("/models/model.gguf"))


def _client(
    verdict: ramguard.Verdict,
    requests: list[str],
    *,
    state: str | None = "unloaded",
    model_file: Path = Path("/models/model.gguf"),
) -> StrataModel:
    """A client whose server lists the model as `state` (None: no server) and answers any load."""
    now = [0.0]

    def sleep(seconds: float) -> None:
        now[0] += seconds

    def clock() -> float:
        now[0] += 0.01
        return now[0]

    def answer(request: httpx.Request) -> httpx.Response:
        requests.append(request.url.path)
        if state is None:
            message = "connection refused"
            raise httpx.ConnectError(message)
        if request.url.path.endswith("/models"):
            listed = [] if state == "unloaded" else [{"id": "m", "status": {"value": state}}]
            return httpx.Response(200, json={"data": listed})
        return httpx.Response(200, json={})

    return StrataModel(
        config.ModelConfig(base_url="http://127.0.0.1:8080/v1"),
        "key",
        transport=httpx.MockTransport(answer),
        model_file=model_file,
        ram_guard=lambda _file: verdict,
        sleep=sleep,
        clock=clock,
    )


def big_model_file(tmp_path: Path) -> Path:
    """A model file that is there, for a client whose server is down."""
    path = tmp_path / "models" / "there.gguf"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def _asked(requests: list[str]) -> list[str]:
    """The requests that load or use the model, leaving out the listing that says what it is."""
    return [path for path in requests if not path.endswith("/models")]


def test_loading_the_model_is_refused_before_the_server_is_asked(
    caplog: pytest.LogCaptureFixture,
) -> None:
    requests: list[str] = []

    assert _client(_short(), requests).load() is False

    assert _asked(requests) == []
    assert "not enough free memory" in caplog.text


def test_warming_the_model_at_start_gives_up_at_once_when_it_would_not_fit(
    caplog: pytest.LogCaptureFixture,
) -> None:
    requests: list[str] = []
    waits: list[float] = []
    model = replace(_client(_short(), requests), sleep=waits.append)

    assert model.warm() is False

    assert _asked(requests) == []
    assert waits == []
    assert "not enough free memory" in caplog.text


def test_a_cold_request_does_not_make_the_server_load_into_too_little_memory() -> None:
    requests: list[str] = []

    with pytest.raises(ModelMemoryError, match="not enough free memory"):
        _client(_short(), requests).chat([], [], load_s=60.0)

    assert _asked(requests) == []


def test_a_request_to_a_loaded_model_is_not_guarded() -> None:
    requests: list[str] = []
    model = _client(_short(), requests)

    with pytest.raises(ModelError, match="without a message") as raised:
        model.chat([], [], load_s=0.0)

    assert not isinstance(raised.value, ModelMemoryError)
    assert _asked(requests) == ["/v1/chat/completions"]


def test_warming_beside_a_server_that_holds_the_model_is_not_refused() -> None:
    # The model is in memory, so MemAvailable already leaves it out: nothing is being loaded.
    requests: list[str] = []

    assert _client(_short(), requests, state="loaded").warm() is True

    assert "/v1/load" in _asked(requests)


def test_a_cold_request_at_a_server_that_holds_the_model_is_not_refused() -> None:
    requests: list[str] = []

    with pytest.raises(ModelError, match="without a message") as raised:
        _client(_short(), requests, state="loaded").chat([], [], load_s=60.0)

    assert not isinstance(raised.value, ModelMemoryError)


def test_a_request_during_a_load_that_is_under_way_is_not_refused() -> None:
    requests: list[str] = []
    model = _client(_short(), requests)

    with model._loads.asking(), pytest.raises(ModelError, match="without a message") as raised:  # pyright: ignore[reportPrivateUsage] - a load of this client that waits for its answer
        model.chat([], [], load_s=60.0)

    assert not isinstance(raised.value, ModelMemoryError)


def test_a_load_that_joins_one_under_way_is_not_refused() -> None:
    requests: list[str] = []
    model = _client(_short(), requests)

    with model._loads.asking():  # pyright: ignore[reportPrivateUsage] - a load of this client that waits for its answer
        assert model.load() is True

    assert _asked(requests) == []


def test_a_server_that_is_down_beside_too_little_memory_says_why(tmp_path: Path) -> None:
    # The unit's own check refused the start: the owner needs the reason and `jarvis start`.
    model = replace(_client(_short(), [], state=None), model_file=big_model_file(tmp_path))

    with pytest.raises(ModelMemoryError, match="not enough free memory"):
        model.chat([], [], load_s=0.0)


def test_a_server_that_is_down_with_memory_to_spare_is_just_down(tmp_path: Path) -> None:
    fits = ramguard.Verdict(MODEL, 60 * GIB, 4 * GIB)
    model = replace(_client(fits, [], state=None), model_file=big_model_file(tmp_path))

    with pytest.raises(ModelError, match="unreachable") as raised:
        model.chat([], [], load_s=0.0)

    assert not isinstance(raised.value, ModelMemoryError)


def test_the_owner_hears_why_the_model_did_not_load() -> None:
    said = agent._model_failure(  # pyright: ignore[reportPrivateUsage] - what is spoken for each model error
        ModelMemoryError("not enough free memory")
    )

    assert "memory" in said
    assert "jarvis start" in said
    assert said != agent._model_failure(  # pyright: ignore[reportPrivateUsage] - what is spoken for each model error
        Exception()
    )


def test_the_model_client_of_the_app_carries_the_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from jarvis import app  # ruff: ignore[import-outside-top-level] - loads the voice stack, which this test file otherwise skips

    cfg = local(tmp_path)
    meminfo(tmp_path, monkeypatch, available=17 * GIB)

    model = app.strata_model(cfg, "key")

    assert model.ram_guard is not None
    assert model.ram_guard(model.served_file()).short
    assert app.mode_model(cfg, "key").ram_guard is not None


def test_the_model_client_of_the_app_counts_the_memory_the_server_holds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A load under way, in this process or another, has already taken its memory from MemAvailable.
    from jarvis import app  # ruff: ignore[import-outside-top-level] - loads the voice stack, which this test file otherwise skips

    cfg = local(tmp_path)
    meminfo(tmp_path, monkeypatch, available=8 * GIB)
    _server_memory(tmp_path, anon=MODEL)

    model = app.strata_model(cfg, "key")

    assert model.ram_guard is not None
    assert not model.ram_guard(model.served_file()).short


def _audited() -> list[dict[str, object]]:
    path = paths.audit_file()
    if not path.is_file():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return [row for row in rows if row["event"] == "ram_refused"]


def test_a_refused_start_is_audited_with_the_figures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local(tmp_path)
    meminfo(tmp_path, monkeypatch, available=17 * GIB)
    _systemctl(monkeypatch, {})

    cli.main(["start"])

    (row,) = _audited()
    assert row["where"] == "start"
    assert row["need"] == MODEL
    assert row["available"] == 17 * GIB
    assert row["margin"] == 4 * GIB
    assert str(row["path"]).endswith("q4.gguf")


def test_a_refused_restart_and_a_start_that_fits_are_audited_as_they_were(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local(tmp_path)
    meminfo(tmp_path, monkeypatch, available=17 * GIB)
    _systemctl(monkeypatch, {})
    monkeypatch.setattr(cli, "_started_at", _started(None))

    cli.main(["restart"])
    meminfo(tmp_path, monkeypatch, available=64 * GIB)
    cli.main(["start"])

    assert [row["where"] for row in _audited()] == ["restart"]


def test_a_refused_switch_is_audited(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _coding_config(tmp_path)
    config.save(cfg)
    _install(cfg)
    meminfo(tmp_path, monkeypatch, available=17 * GIB)
    _systemctl(monkeypatch, {STRATA_UNIT: "active"})

    cli.switch_model(cfg, "coding", lambda _line: None, "test")

    (row,) = _audited()
    assert row["where"] == "mode switch"
    assert str(row["path"]).endswith("coder.gguf")


def test_the_units_refusal_is_audited(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    services.write_strata_profile(local(tmp_path))
    meminfo(tmp_path, monkeypatch, available=15 * GIB)

    assert _run_guard(capsys)[0] == ramguard.EXIT_SHORT

    (row,) = _audited()
    assert row["where"] == "unit"


def test_a_refused_client_load_is_audited_once_for_each_refusal() -> None:
    _client(_short(), []).load()

    (row,) = _audited()
    assert row["where"] == "model load"
    assert row["need"] == MODEL


def test_a_load_that_is_not_refused_is_not_audited() -> None:
    _client(_short(), [], state="loaded").warm()

    assert _audited() == []


def test_the_mode_switch_text_leads_with_what_the_model_can_say(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _coding_config(tmp_path)
    config.save(cfg)
    _install(cfg)
    meminfo(tmp_path, monkeypatch, available=17 * GIB)
    _systemctl(monkeypatch, {STRATA_UNIT: "active"})

    said = cli.switch_model(cfg, "coding", lambda _line: None, "test")

    assert said.startswith("There isn't enough free memory to load the coding model right now")
    assert "strata.allow_low_ram" in said


def test_a_refused_switch_leaves_the_servers_files_as_they_were(tmp_path: Path) -> None:
    # Rewriting them would make the next restart (and the idle check every minute) see a change
    # since the server started, and reload the model for nothing.
    cfg = _coding_config(tmp_path)
    _install(cfg)
    old = time.time() - 1000
    for path in (services.strata_profile_file(), services.unit_dir() / STRATA_UNIT):
        os.utime(path, (old, old))

    done = _switcher(cfg, _Units(), lambda _served: "no room").switch("coding", "owner")

    assert done.error
    assert not services.strata_changed_since(old + 500)
    assert modes.read_state().pending is None


def test_a_switch_that_changes_no_file_asks_nothing(tmp_path: Path) -> None:
    cfg = _coding_config(tmp_path)
    _install(cfg)

    def ram(_served: Config) -> str:
        pytest.fail("no restart follows, so nothing is loaded")

    assert _switcher(cfg, _Units(), ram).switch("normal", "owner").mode == "normal"


def test_the_unit_check_runs_isolated_and_a_refused_start_is_skipped_not_failed(
    tmp_path: Path,
) -> None:
    # An ExecCondition= that exits 1 to 254 skips the start without failing the unit, so systemd
    # does not retry it into the start limit (probed with transient units: no restarts).
    text = services.strata_unit(local(tmp_path), paths.strata_key_file())

    assert "ExecCondition=" + sys.executable + " -I -m jarvis.ramguard" in text
    assert "ExecStartPre" not in text
    assert "RestartPreventExitStatus=75" not in text
    assert 1 <= ramguard.EXIT_SHORT <= 254


def _inactive(_unit: str) -> str:
    return "inactive"


def _skipped_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, on: str
) -> list[tuple[str, ...]]:
    """Fit at the up-front check, then too little by the time systemd runs the unit's check.

    `systemctl` answers 0 and leaves the unit inactive, as it does for a start the unit skipped.
    """
    calls: list[tuple[str, ...]] = []

    def systemctl(*args: str) -> int:
        calls.append(args)
        if args[0] == on:
            meminfo(tmp_path, monkeypatch, available=17 * GIB)
        return 0

    monkeypatch.setattr(cli, "_systemctl", systemctl)
    monkeypatch.setattr(cli, "_unit_state", _inactive)
    return calls


def test_start_does_not_report_success_when_the_unit_skipped_its_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    local(tmp_path)
    calls = _skipped_start(tmp_path, monkeypatch, on="start")

    assert cli.main(["start"]) == 1

    assert "not enough free memory" in capsys.readouterr().err
    assert ("start", STRATA_UNIT, "jarvis.service") in calls


def test_restart_does_not_report_success_when_the_unit_skipped_its_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    local(tmp_path)
    calls = _skipped_start(tmp_path, monkeypatch, on="start")
    monkeypatch.setattr(cli, "_started_at", _started(None))

    assert cli.main(["restart"]) == 1

    assert "not enough free memory" in capsys.readouterr().err
    assert ("restart", "jarvis.service") not in calls


def test_a_backends_start_does_not_report_success_when_the_unit_skipped_its_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = local(tmp_path)
    config.save(
        cfg.model_copy(update={"strata": cfg.strata.model_copy(update={"mode": "backend"})})
    )
    _skipped_start(tmp_path, monkeypatch, on="start")

    assert cli.main(["start"]) == 1

    assert "not enough free memory" in capsys.readouterr().err


def test_an_update_does_not_restart_the_server_into_too_little_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    local(tmp_path)
    meminfo(tmp_path, monkeypatch, available=17 * GIB)

    assert update.no_room_for_the_model() is True

    err = capsys.readouterr().err
    assert "not restarted" in err
    assert "jarvis restart" in err
    assert [row["where"] for row in _audited()] == ["update"]


def test_an_update_restarts_the_server_when_it_fits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local(tmp_path)
    meminfo(tmp_path, monkeypatch, available=64 * GIB)

    assert update.no_room_for_the_model() is False
