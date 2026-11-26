"""Coding mode: what it lays over the config, the state every process shares, and the switch."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from strata_fixture import engine_run_config, install_engine, touch

from jarvis import modes, paths, services
from jarvis.config import CodingConfig, Config, StrataConfig

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))


pytestmark = pytest.mark.usefixtures("home")


def _coding_run_config(tmp_path: Path) -> Path:
    """Write a second run config, for a model of coding mode's own."""
    data = json.loads(engine_run_config(tmp_path).read_text(encoding="utf-8"))
    model = touch(tmp_path / "models" / "coder.gguf")
    data["args"] = [str(model) if a.endswith("q4.gguf") else a for a in data["args"]]
    data["model_name"] = "flash-coder"
    path = tmp_path / "install" / "strata-coder.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _config(tmp_path: Path, *, own_model: bool = True, **coding: object) -> Config:
    install_engine(tmp_path / "components")
    strata: dict[str, object] = {
        "home": tmp_path / "strata-home",
        "components_parent": tmp_path / "components",
        "run_config": engine_run_config(tmp_path),
        "vram_reserve_mib": 12000,
    }
    if own_model:
        strata["coding_run_config"] = _coding_run_config(tmp_path)
    return Config(
        strata=StrataConfig.model_validate(strata), coding=CodingConfig.model_validate(coding)
    )


@dataclass
class _Units:
    running: bool = True
    takes: bool = True
    restarts: int = 0

    def active(self) -> bool:
        return self.running

    def restart(self) -> bool:
        self.restarts += 1
        return self.takes


@dataclass
class _Server:
    """A model server in `state` (None: it does not answer), counting the loads asked of it."""

    state: str | None = "loaded"
    loads: int = 0
    on_load: Callable[[], None] = lambda: None

    def status(self) -> str | None:
        return self.state

    def load(self) -> bool:
        self.loads += 1
        self.on_load()
        self.state = "loaded"
        return True


@dataclass
class _Audit:
    events: list[tuple[str, dict[str, object]]] = field(
        default_factory=list[tuple[str, dict[str, object]]]
    )

    def record(self, event: str, **fields: object) -> None:
        self.events.append((event, fields))


@dataclass
class _Clock:
    now: float = 1000.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def _installed(cfg: Config) -> None:
    """Write the profile and the unit as `jarvis setup` leaves them, in the normal mode."""
    services.write_strata_profile(cfg)
    services.write_strata_unit(cfg, paths.strata_key_file())


def _switcher(
    cfg: Config,
    units: _Units,
    server: _Server,
    audit: _Audit,
    *,
    clock: _Clock | None = None,
    busy: Callable[[], bool] = lambda: False,
) -> modes.Switcher:
    clock = clock or _Clock()
    return modes.Switcher(
        cfg, units, lambda _cfg: server, audit, busy=busy, clock=clock, sleep=clock.sleep, now=clock
    )


def _served_model() -> str:
    profile = json.loads(services.strata_profile_file().read_text(encoding="utf-8"))
    args = profile["args"]
    return Path(args[args.index("--native") + 1]).name


def test_the_normal_mode_runs_the_config_as_it_is(tmp_path: Path) -> None:
    cfg = _config(tmp_path)

    assert modes.effective(cfg, "normal") is cfg
    assert modes.current() == "normal"


def test_coding_mode_lays_its_limits_over_the_chat_and_keeps_the_rest(tmp_path: Path) -> None:
    cfg = _config(tmp_path)

    coding = modes.effective(cfg, "coding")

    assert coding.chat.reply_tokens == 8192
    assert coding.chat.max_steps == 60
    assert coding.chat.thinking == "medium"
    assert coding.chat.thinking_max == cfg.chat.thinking_max  # unset there: the chat's own
    assert coding.chat.resume == cfg.chat.resume
    assert coding.chat.verbosity == cfg.chat.verbosity
    assert coding.strata.run_config == cfg.strata.coding_run_config
    assert coding.strata.context == 65536
    assert coding.strata.vram_reserve_mib == 12000  # unset in [coding]: [strata]'s
    assert coding.model.temperature == cfg.model.temperature


def test_a_spoken_request_in_coding_mode_stays_a_quick_command_with_room_to_write(
    tmp_path: Path,
) -> None:
    cfg = _config(tmp_path)

    voice = modes.effective(cfg, "coding").agent

    assert voice.memory_tokens == cfg.agent.memory_tokens
    assert voice.memory_idle_s == cfg.agent.memory_idle_s
    assert voice.tool_text_chars == cfg.agent.tool_text_chars
    assert voice.reply_tokens == 4096
    assert voice.max_seconds == pytest.approx(300.0)


def test_only_the_chat_lets_the_model_switch_modes_itself_by_default() -> None:
    coding = Config().coding

    assert coding.chat.auto is True
    assert coding.voice.auto is False


def test_what_the_owner_sets_for_coding_mode_wins(tmp_path: Path) -> None:
    cfg = _config(
        tmp_path,
        chat={"reply_tokens": 16000, "thinking": "high", "max_steps": None},
        temperature=0.6,
        vram_reserve_mib=4000,
        memory_max="30G",
        write_roots=["~/code"],
    )

    coding = modes.effective(cfg, "coding")

    assert coding.chat.reply_tokens == 16000
    assert coding.chat.max_steps == cfg.chat.max_steps
    assert coding.chat.thinking_max == "high"
    assert coding.model.temperature == pytest.approx(0.6)
    assert coding.strata.vram_reserve_mib == 4000
    assert coding.strata.memory_max == "30G"
    assert coding.files.write_roots == [*cfg.files.write_roots, "~/code"]


def test_a_thinking_ceiling_never_ends_below_where_coding_starts(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    low = cfg.model_copy(update={"chat": cfg.chat.model_copy(update={"thinking_max": "low"})})

    assert modes.effective(low, "coding").chat.thinking_max == "medium"


def test_without_a_coding_run_config_coding_mode_keeps_the_model(tmp_path: Path) -> None:
    cfg = _config(tmp_path, own_model=False)

    assert not modes.own_model(cfg)
    assert modes.effective(cfg, "coding").strata.run_config == cfg.strata.run_config
    assert modes.model_name(cfg, "coding") == "flash-q4"


def test_an_unreadable_state_file_is_the_normal_mode() -> None:
    modes.state_file().parent.mkdir(parents=True)
    modes.state_file().write_text("{not json", encoding="utf-8")

    assert modes.read_state() == modes.State()

    modes.state_file().write_text('{"mode": "turbo", "by": 7, "used": "x"}', encoding="utf-8")

    assert modes.read_state() == modes.State()


def test_a_switch_serves_the_coding_model_restarts_the_server_and_is_audited(
    tmp_path: Path,
) -> None:
    cfg = _config(tmp_path)
    _installed(cfg)
    units, server, audit = _Units(), _Server(), _Audit()
    said: list[str] = []

    done = _switcher(cfg, units, server, audit).switch("coding", "owner", said.append)

    assert (done.mode, done.changed, done.restarted, done.error) == ("coding", True, True, "")
    assert _served_model() == "coder.gguf"
    assert units.restarts == 1
    assert server.loads == 1
    assert modes.current() == "coding"
    assert said == ["Switching to the coding model, flash-coder: this can take a minute or two."]
    event, fields = audit.events[-1]
    assert event == "mode_switched"
    assert (fields["to"], fields["was"], fields["by"]) == ("coding", "normal", "owner")
    assert (fields["model"], fields["restarted"]) == ("flash-coder", True)


def test_the_file_the_unit_waits_for_follows_the_mode(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    _installed(cfg)

    _switcher(cfg, _Units(), _Server(), _Audit()).switch("coding", "owner")

    unit = (services.unit_dir() / services.STRATA_UNIT).read_text(encoding="utf-8")
    assert f"ConditionPathExists={tmp_path}/models/coder.gguf" in unit


def test_the_next_switch_says_how_long_the_model_took_to_load(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    _installed(cfg)
    clock = _Clock()
    server = _Server(on_load=lambda: clock.sleep(22.0))
    switcher = _switcher(cfg, _Units(), server, _Audit(), clock=clock)
    assert switcher.switch("coding", "owner").seconds == pytest.approx(22.0)
    switcher.switch("normal", "owner")
    said: list[str] = []

    switcher.switch("coding", "owner", said.append)

    assert said == ["Switching to the coding model, flash-coder: about 20 seconds."]


def test_a_switch_to_the_mode_jarvis_is_in_changes_nothing(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    _installed(cfg)
    units, audit = _Units(), _Audit()

    done = _switcher(cfg, units, _Server(), audit).switch("normal", "owner")

    assert (done.mode, done.changed, done.error) == ("normal", False, "")
    assert units.restarts == 0
    assert audit.events == []


def test_no_switch_while_another_process_answers_a_request(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    _installed(cfg)
    units = _Units()
    switcher = _switcher(cfg, units, _Server(), _Audit())

    with modes.Answering():  # as a chat window or `jarvis ask` holds it through a request
        cut = switcher.switch("coding", "owner")

    assert cut.error == modes.BUSY
    assert (modes.current(), units.restarts, _served_model()) == ("normal", 0, "q4.gguf")
    assert switcher.switch("coding", "owner").changed  # the request is over


def test_a_request_switches_the_mode_itself_by_setting_its_mark_aside(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    _installed(cfg)
    switcher = _switcher(cfg, _Units(), _Server(), _Audit())

    with modes.Answering() as mine:
        with mine.set_aside():
            done = switcher.switch("coding", "model")
        held_again = switcher.switch("normal", "owner")

    assert done.changed
    assert held_again.error == modes.BUSY


def test_a_request_inside_a_request_keeps_the_mark_until_the_outer_one_ends(
    tmp_path: Path,
) -> None:
    cfg = _config(tmp_path)
    _installed(cfg)
    switcher = _switcher(cfg, _Units(), _Server(), _Audit())
    mine = modes.Answering()

    with mine:
        with mine:
            pass
        still_held = switcher.switch("coding", "owner")

    assert still_held.error == modes.BUSY


def test_no_switch_while_a_request_is_in_flight(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    _installed(cfg)
    units = _Units()

    done = _switcher(cfg, units, _Server(), _Audit(), busy=lambda: True).switch("coding", "owner")

    assert done.error == modes.BUSY
    assert modes.current() == "normal"
    assert units.restarts == 0
    assert _served_model() == "q4.gguf"


def test_a_coding_run_config_that_cannot_be_served_leaves_everything_as_it_was(
    tmp_path: Path,
) -> None:
    cfg = _config(tmp_path)
    _installed(cfg)
    (tmp_path / "models" / "coder.gguf").unlink()
    units = _Units()

    done = _switcher(cfg, units, _Server(), _Audit()).switch("coding", "owner")

    assert "coder.gguf" in done.error
    assert done.mode == "normal"
    assert modes.current() == "normal"
    assert units.restarts == 0
    assert _served_model() == "q4.gguf"


def test_a_server_that_does_not_come_back_puts_the_mode_before_back(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    _installed(cfg)
    units, audit = _Units(), _Audit()
    server = _Server(state=None)  # never answers

    done = _switcher(cfg, units, server, audit).switch("coding", "owner")

    assert "did not come back" in done.error
    assert modes.current() == "normal"
    assert _served_model() == "q4.gguf"
    assert units.restarts == 2  # the switch, and the way back
    assert audit.events[-1][0] == "mode_switch_failed"


def test_a_model_that_was_not_in_memory_is_not_loaded_on_the_way_back(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    _installed(cfg)
    server = _Server(state="unloaded")
    switcher = _switcher(cfg, _Units(), server, _Audit())
    switcher.switch("coding", "owner")  # coding work follows at once: loaded
    assert server.loads == 1
    server.state = "unloaded"  # the owner freed it

    switcher.switch("normal", "owner")

    assert server.loads == 1


def test_coding_mode_on_the_same_model_and_context_needs_no_restart(tmp_path: Path) -> None:
    cfg = _config(tmp_path, own_model=False, context=24576, reasoning_budget_tokens=512)
    _installed(cfg)
    units = _Units()
    said: list[str] = []

    done = _switcher(cfg, units, _Server(), _Audit()).switch("coding", "owner", said.append)

    assert (done.changed, done.restarted) == (True, False)
    assert units.restarts == 0
    assert said == ["Switching to coding mode."]
    assert modes.current() == "coding"


def test_a_stopped_server_gets_its_new_profile_and_no_restart(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    _installed(cfg)
    units = _Units(running=False)

    done = _switcher(cfg, units, _Server(), _Audit()).switch("coding", "owner")

    assert (done.changed, done.restarted) == (True, False)
    assert units.restarts == 0
    assert _served_model() == "coder.gguf"


def test_coding_mode_left_unused_goes_back_to_normal(tmp_path: Path) -> None:
    cfg = _config(tmp_path, back_after_s=600.0)
    _installed(cfg)
    clock, audit = _Clock(), _Audit()
    switcher = _switcher(cfg, _Units(), _Server(), audit, clock=clock)
    switcher.switch("coding", "model")

    clock.sleep(599.0)
    assert switcher.back_if_idle() is None
    assert modes.current() == "coding"

    clock.sleep(2.0)
    back = switcher.back_if_idle()

    assert back is not None
    assert back.mode == "normal"
    assert modes.current() == "normal"
    assert audit.events[-1][1]["by"] == "idle"
    assert _served_model() == "q4.gguf"


def test_coding_mode_stays_when_going_back_is_switched_off(tmp_path: Path) -> None:
    cfg = _config(tmp_path, back_after_s=0.0)
    _installed(cfg)
    clock = _Clock()
    switcher = _switcher(cfg, _Units(), _Server(), _Audit(), clock=clock)
    switcher.switch("coding", "owner")

    clock.sleep(10**6)

    assert switcher.back_if_idle() is None
    assert modes.current() == "coding"


def test_using_coding_mode_starts_its_idle_time_again(tmp_path: Path) -> None:
    cfg = _config(tmp_path, back_after_s=600.0)
    _installed(cfg)
    clock = _Clock()
    switcher = _switcher(cfg, _Units(), _Server(), _Audit(), clock=clock)
    switcher.switch("coding", "owner")
    clock.sleep(500.0)

    modes.mark_used(clock.now)
    clock.sleep(500.0)

    assert switcher.back_if_idle() is None
    assert modes.current() == "coding"


def test_marking_use_in_the_normal_mode_writes_nothing() -> None:
    modes.mark_used()

    assert not modes.state_file().exists()
