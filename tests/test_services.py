"""The Strata profile and systemd units: their text, their quoting and how they are written."""

import json
import stat
import sys
from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError
from strata_fixture import engine_run_config, inkling_run_config, install_engine

from jarvis import services
from jarvis.config import Config, ConfigError, StrataConfig


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))


pytestmark = pytest.mark.usefixtures("home")


def _config(tmp_path: Path, *, engine: bool = True, **strata: object) -> Config:
    """Return a config on the main engine's run config, with an installed engine (`engine`)."""
    if engine:
        install_engine(tmp_path / "components")
    settings: dict[str, object] = {
        "home": tmp_path / "strata-home",
        "components_parent": tmp_path / "components",
        **strata,
    }
    if "run_config" not in settings:
        settings["run_config"] = engine_run_config(tmp_path)
        settings.setdefault("vram_reserve_mib", 12000)  # its engine's cache would fill the card
    return Config(strata=StrataConfig.model_validate(settings))


def _lines(text: str) -> list[str]:
    return text.splitlines()


def test_the_strata_unit_for_a_normal_config(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    key = tmp_path / "state" / "jarvis" / "strata.env"

    text = services.strata_unit(cfg, key)

    lines = _lines(text)
    assert f"ConditionPathExists={tmp_path}/models/q4.gguf" in lines
    assert f"Environment=STRATA_DATA_DIR={tmp_path}/strata-home" in lines
    assert f"Environment=XDG_DATA_HOME={tmp_path}/components" in lines
    assert f"EnvironmentFile={key}" in lines
    assert "Environment=PIPEWIRE_REMOTE=/nonexistent" in lines
    assert "MemoryHigh=40G" in lines
    assert "MemoryMax=44G" in lines
    assert any(line.startswith("ExecStart=") and " serve --profile " in line for line in lines)
    assert text.endswith("WantedBy=graphical-session.target\n")


def test_a_strata_program_that_is_not_an_absolute_path_never_reaches_the_unit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _config(tmp_path, home=Path("relative-strata"))

    def nowhere(_name: str) -> str | None:
        return None

    monkeypatch.setattr(services.shutil, "which", nowhere)

    with pytest.raises(ConfigError, match="absolute path"):
        services.strata_unit(cfg, tmp_path / "strata.env")


def test_a_jarvis_program_that_is_not_an_absolute_path_never_reaches_the_unit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "executable", "venv/bin/python")

    with pytest.raises(ConfigError, match="absolute path"):
        services.jarvis_unit()


def test_the_strata_unit_keeps_the_servers_own_notes_in_the_journal(tmp_path: Path) -> None:
    """Strata's per-request timings and its notes (an ignored prefix pin) are on its stdout."""
    text = services.strata_unit(_config(tmp_path), tmp_path / "strata.env")

    assert "--quiet" not in text


def test_paths_with_spaces_are_quoted_for_systemd(tmp_path: Path) -> None:
    spaced = tmp_path / "my models"
    run = engine_run_config(tmp_path, model=spaced / "q4 k.gguf")
    cfg = _config(tmp_path, home=spaced / "strata home", run_config=run)
    key = tmp_path / "my keys" / "strata.env"

    lines = _lines(services.strata_unit(cfg, key))

    assert f"ConditionPathExists={spaced}/q4 k.gguf" in lines
    assert f'Environment="STRATA_DATA_DIR={spaced}/strata home"' in lines
    assert f"EnvironmentFile={key}" in lines


def test_the_exec_line_quotes_a_program_path_with_a_space(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    program = tmp_path / "my apps" / "venv" / "bin" / "python"
    monkeypatch.setattr(sys, "executable", str(program))

    assert f'ExecStart="{program.parent}/jarvis" run' in _lines(services.jarvis_unit())


@pytest.mark.parametrize(
    ("raw", "written"),
    [
        ("/a b", '"/a b"'),
        ('/a"b', '"/a\\"b"'),
        ("/a\\b", '"/a\\\\b"'),
        ("/a$HOME", "/a$$HOME"),
        ("/a%h", "/a%%h"),
        ("/a;b", '"/a;b"'),
        ("/plain/path", "/plain/path"),
    ],
)
def test_exec_words_are_escaped_per_systemd_rules(raw: str, written: str) -> None:
    assert services.exec_word(raw) == written


@pytest.mark.parametrize(
    "program", ["-/bin/x", "@/bin/x", "+/bin/x", "!/bin/x", ":/bin/x", "bin/x", ""]
)
def test_a_program_that_is_not_an_absolute_path_never_starts_a_unit_line(program: str) -> None:
    with pytest.raises(ConfigError, match="absolute path"):
        services.exec_program(program, "program")


def test_an_absolute_program_is_written_as_an_exec_word() -> None:
    assert services.exec_program("/my apps/jarvis", "program") == '"/my apps/jarvis"'


@pytest.mark.parametrize("bad", ["/a\nExecStartPre=/bin/evil", "/a\rb", "/a\x00b", "/a\x7fb"])
def test_a_control_character_never_reaches_a_unit(tmp_path: Path, bad: str) -> None:
    cfg = _config(tmp_path)
    unchecked = cfg.model_copy(update={"strata": cfg.strata.model_copy(update={"home": Path(bad)})})

    with pytest.raises(ConfigError, match="control character"):
        services.strata_unit(unchecked, tmp_path / "strata.env")
    with pytest.raises(ConfigError, match="control character"):
        services.strata_unit(cfg, Path(bad))


def test_a_refused_value_writes_no_unit_at_all(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    unchecked = cfg.model_copy(
        update={"strata": cfg.strata.model_copy(update={"memory_max": "40G\nExecStart=/bin/evil"})}
    )

    with pytest.raises(ConfigError, match="memory_max"):
        services.write_units(unchecked, tmp_path / "strata.env")

    assert not services.unit_dir().exists() or list(services.unit_dir().iterdir()) == []


def test_a_value_ending_in_a_backslash_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="backslash"):
        services.strata_unit(_config(tmp_path), Path("/keys/strata.env\\"))


def test_the_model_condition_is_omitted_when_no_run_config_is_set(tmp_path: Path) -> None:
    cfg = Config(strata=StrataConfig(home=tmp_path / "s"))

    text = services.strata_unit(cfg, tmp_path / "strata.env")

    assert "ConditionPathExists" not in text


def test_a_setup_error_is_not_restarted() -> None:
    lines = _lines(services.jarvis_unit())

    assert "RestartPreventExitStatus=2" in lines
    assert "Restart=on-failure" in lines


def test_the_strata_profile_content(tmp_path: Path) -> None:
    cfg = _config(tmp_path, port=9100, context=4096, idle_unload_s=120)

    profile = json.loads(services.write_strata_profile(cfg).read_text(encoding="utf-8"))

    assert profile["port"] == 9100
    assert profile["host"] == "127.0.0.1"
    assert profile["lazy_load"] is True
    assert profile["idle_unload_s"] == 120
    assert profile["model_name"] == "flash-q4"
    assert profile["tokenizer"] == str(tmp_path / "packs" / "q4" / "tokenizer")
    assert profile["args"][profile["args"].index("--native") + 1] == str(
        tmp_path / "models" / "q4.gguf"
    )
    assert profile["args"][profile["args"].index("--max-context") + 1] == "4096"
    assert "api_key" not in json.dumps(profile)


def test_the_profile_keeps_the_models_own_settings_and_sets_jarvis_limits(tmp_path: Path) -> None:
    profile = json.loads(json.dumps(services.strata_profile(_config(tmp_path))))

    assert profile["exe"] == "component:strata-engine"
    assert profile["backend"] == "hip"
    args = profile["args"]
    for flag, value in (("--pack", str(tmp_path / "packs" / "q4")), ("--kv", "int8")):
        assert args[args.index(flag) + 1] == value  # Strata's own flags, untouched
    assert args[args.index("--prefill") + 1] == "auto"
    assert profile["args_written"] == args
    assert "args_derived" not in profile  # one launch's record, not the model's
    assert "engine_ready_s" not in profile
    assert profile["parallel"] == 1
    assert profile["log"] != "/strata/install/strata-q4.log"
    assert profile["cwd"] == str(tmp_path / "strata-home" / "terminal" / "current")


def test_an_engine_whose_cache_fills_the_card_is_held_back_by_the_reserve(tmp_path: Path) -> None:
    args = _args(services.strata_profile(_config(tmp_path, vram_reserve_mib=12000)))

    assert args[args.index("--expert-cache") + 1] == "auto"
    assert args[args.index("--vram-reserve-mib") + 1] == "12000"


def test_inklings_experts_on_the_card_follow_expert_vram_gib(tmp_path: Path) -> None:
    cfg = _config(tmp_path, run_config=inkling_run_config(tmp_path), expert_vram_gib=4)

    profile = json.loads(json.dumps(services.strata_profile(cfg)))

    args = profile["args"]
    assert profile["exe"] == "component:strata-inkling"
    assert (profile["arch"], profile["drafting"], profile["parallel"]) == (
        "inkling2_loop",
        "dspark",
        1,
    )
    assert args[args.index("--vram-experts-gb") + 1] == "4"
    assert "--ram-experts-gb" not in args  # the rest is the planner's, in RAM
    assert args[args.index("--vision") + 1] == str(tmp_path / "models" / "inkling-vision.strata")
    assert "vision" not in profile  # this engine reads pictures itself


def test_pictures_off_drops_the_reader_of_either_engine(tmp_path: Path) -> None:
    engine = json.loads(json.dumps(services.strata_profile(_config(tmp_path, pictures=False))))
    inkling = _config(tmp_path, run_config=inkling_run_config(tmp_path), pictures=False)
    inkling_args = _args(services.strata_profile(inkling))

    assert "--vision" not in engine["args"]
    assert "vision" not in engine
    assert "--vision" not in inkling_args
    assert str(tmp_path / "models" / "inkling-vision.strata") not in inkling_args
    assert not services.reads_pictures(_config(tmp_path, pictures=False))


def test_without_a_run_config_there_is_no_profile(tmp_path: Path) -> None:
    with pytest.raises(services.StrataSetupError, match="run_config is not set"):
        services.strata_profile(_config(tmp_path, run_config=Path()))


def test_a_run_config_naming_a_missing_file_is_a_setup_error(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    (tmp_path / "models" / "q4.gguf").unlink()

    with pytest.raises(services.StrataSetupError, match=r"missing: .*q4\.gguf.*strata launch"):
        services.strata_profile(cfg)


@pytest.mark.parametrize(
    "text",
    ["not json", "[]", '{"exe": "something-else", "args": []}', '{"exe": "component:strata-x"}'],
)
def test_a_file_that_is_not_a_strata_run_config_is_refused(tmp_path: Path, text: str) -> None:
    path = tmp_path / "bad.json"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(services.StrataSetupError, match=r"bad\.json"):
        services.strata_profile(_config(tmp_path, run_config=path))


def test_the_model_name_and_reader_come_from_the_run_config(tmp_path: Path) -> None:
    cfg = _config(tmp_path)

    assert services.model_name(cfg) == "flash-q4"
    assert services.reads_pictures(cfg)
    assert "run_config is not set" in services.model_name(_config(tmp_path, run_config=Path()))


def test_units_are_written_atomically_and_leave_no_temp_file(tmp_path: Path) -> None:
    written = services.write_units(_config(tmp_path), tmp_path / "strata.env")

    assert [p.name for p in written] == [services.STRATA_UNIT, services.JARVIS_UNIT]
    assert sorted(p.name for p in services.unit_dir().iterdir()) == sorted(
        [services.JARVIS_UNIT, services.STRATA_UNIT]
    )
    assert stat.S_IMODE(written[0].stat().st_mode) == 0o644
    assert written[1].read_text(encoding="utf-8") == services.jarvis_unit()


def test_rewriting_a_unit_replaces_it_whole(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    first = services.write_units(cfg, tmp_path / "strata.env")[0]
    first.write_text("stale\n", encoding="utf-8")

    services.write_units(cfg, tmp_path / "strata.env")

    assert "stale" not in first.read_text(encoding="utf-8")


def test_unchanged_strata_files_are_not_rewritten(tmp_path: Path) -> None:
    """jarvis restart restarts Strata when these files are newer than it; same text is no change."""
    cfg = _config(tmp_path)
    unit = services.write_units(cfg, tmp_path / "strata.env")[0]
    profile = services.write_strata_profile(cfg)
    before = (unit.stat().st_ino, profile.stat().st_ino)

    services.write_units(cfg, tmp_path / "strata.env")
    services.write_strata_profile(cfg)

    assert (unit.stat().st_ino, profile.stat().st_ino) == before


def test_a_failed_write_leaves_the_old_file_and_no_temp_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "unit" / "x.service"
    target.parent.mkdir()
    target.write_text("old\n", encoding="utf-8")

    def refuse(_self: Path, _target: object) -> None:
        msg = "disk full"
        raise OSError(msg)

    monkeypatch.setattr(Path, "replace", refuse)

    with pytest.raises(OSError, match="disk full"):
        services.write_atomic(target, "new\n")

    assert target.read_text(encoding="utf-8") == "old\n"
    assert [p.name for p in target.parent.iterdir()] == ["x.service"]


def test_the_profile_is_written_atomically(tmp_path: Path) -> None:
    path = services.write_strata_profile(_config(tmp_path))

    assert sorted(p.name for p in path.parent.iterdir()) == [path.name]


def _engine(tmp_path: Path, *versions: str) -> Path:
    root = tmp_path / "components" / "strata" / "components" / "engine"
    for version in versions:
        (root / version).mkdir(parents=True)
        (root / version / "strata").write_bytes(b"")
        (root / version / "strata-vision").write_bytes(b"")
    return root


def test_the_picture_reader_runs_on_the_processor_and_only_when_a_picture_comes(
    tmp_path: Path,
) -> None:
    engine = _engine(tmp_path, "0.1.60-x", "0.1.62-x")
    reader = tmp_path / "models" / "mmproj.gguf"
    cfg = _config(tmp_path, vision_idle_s=90)

    profile = json.loads(json.dumps(services.strata_profile(cfg)))

    assert "--vision" in profile["args"]
    vision = profile["vision"]
    assert vision["exe"] == str(engine / "0.1.62-x" / "strata-vision")
    assert vision["mmproj"] == str(reader)
    assert vision["model"] == str(tmp_path / "models" / "q4.gguf")
    assert vision["gpu"] is False  # in RAM, not in the VRAM games need
    assert vision["elastic"] is True  # started by the first picture, stopped when idle
    assert vision["elastic_idle_s"] == 90
    assert vision["max_tokens"] == 300  # Strata's own choice for the model
    assert 1 <= vision["threads"] <= 16


def test_a_picture_reader_without_the_engine_program_is_a_setup_error(tmp_path: Path) -> None:
    cfg = _config(tmp_path, engine=False)

    with pytest.raises(services.StrataSetupError, match="picture reader"):
        services.strata_profile(cfg)


def test_both_units_start_only_once_their_program_has_started(tmp_path: Path) -> None:
    # `Type=exec`: `systemctl start` fails when the program cannot be started (systemd.service).
    strata = _lines(services.strata_unit(_config(tmp_path), tmp_path / "strata.env"))
    jarvis = _lines(services.jarvis_unit())

    assert "Type=exec" in strata
    assert "Type=exec" in jarvis


def test_the_voice_service_starts_after_the_audio_session() -> None:
    after = next(line for line in _lines(services.jarvis_unit()) if line.startswith("After="))

    assert {"pipewire.service", "wireplumber.service"} <= set(after.removeprefix("After=").split())


def test_both_units_start_and_stop_with_the_desktop_session(tmp_path: Path) -> None:
    # #22: the session's start starts Jarvis and Strata; its end stops both, freeing the model.
    for text in (services.strata_unit(_config(tmp_path), tmp_path / "key"), services.jarvis_unit()):
        lines = _lines(text)
        assert "PartOf=graphical-session.target" in lines
        assert "WantedBy=graphical-session.target" in lines


def _section(text: str, name: str) -> list[str]:
    """Return the lines of one `[name]` section of a unit."""
    lines = _lines(text)
    start = lines.index(f"[{name}]") + 1
    end = next((i for i in range(start, len(lines)) if lines[i].startswith("[")), len(lines))
    return lines[start:end]


def _units(tmp_path: Path) -> dict[str, str]:
    return {
        "strata": services.strata_unit(_config(tmp_path), tmp_path / "key"),
        "jarvis": services.jarvis_unit(),
    }


_BOTH_HARDENED = (
    "NoNewPrivileges=yes",
    "LockPersonality=yes",
    "RestrictSUIDSGID=yes",
)
_VOICE_HARDENED = ("SystemCallArchitectures=native",)
# Left out on purpose (issue #33): each would break something a unit runs.
_NOT_SET = (
    "RestrictNamespaces",  # the Azure helper runs under `unshare --map-current-user --net`
    "MemoryDenyWriteExecute",  # onnxruntime and Strata's HIP path may JIT
    "ProtectHome",  # config, state and models live under home and on other drives
    "ProtectSystem",
    "PrivateNetwork",  # Spotify, Open-Meteo and web reads
    "IPAddressDeny",
    "RestrictRealtime",  # until pw-record and pw-play are shown not to need it
    # In a user unit these two make a private user namespace that Spotify, launched by the voice
    # service, would inherit.
    "ProtectKernelModules",
    "ProtectKernelLogs",
    "Nice",  # CPUWeight is the only priority change
    "CPUSchedulingPolicy",
    "IOSchedulingClass",
)


@pytest.mark.parametrize("option", _BOTH_HARDENED)
def test_both_units_are_hardened(tmp_path: Path, option: str) -> None:
    for text in _units(tmp_path).values():
        assert option in _section(text, "Service")


@pytest.mark.parametrize("option", _VOICE_HARDENED)
def test_only_the_voice_service_takes_the_kernel_and_architecture_options(
    tmp_path: Path, option: str
) -> None:
    units = _units(tmp_path)

    assert option in _section(units["jarvis"], "Service")
    assert option not in _lines(units["strata"])


@pytest.mark.parametrize("name", _NOT_SET)
def test_options_that_would_break_a_unit_are_not_set(tmp_path: Path, name: str) -> None:
    for text in _units(tmp_path).values():
        assert not any(line.startswith(f"{name}=") for line in _lines(text))


def test_only_strata_makes_what_it_creates_private(tmp_path: Path) -> None:
    # A file the file tools write for the owner keeps the mode it asks for (0644).
    units = _units(tmp_path)

    assert "UMask=0077" in _section(units["strata"], "Service")
    assert not any(line.startswith("UMask=") for line in _lines(units["jarvis"]))


def test_both_units_give_way_to_a_game_for_the_processor(tmp_path: Path) -> None:
    for text in _units(tmp_path).values():
        assert "CPUWeight=50" in _section(text, "Service")


def test_both_units_stop_restarting_after_a_persistent_failure(tmp_path: Path) -> None:
    # Five starts in 15 minutes: a unit that keeps failing stays `failed` (and shows in
    # `jarvis status`) instead of re-reading the model for ever. They belong to [Unit].
    for text in _units(tmp_path).values():
        unit = _section(text, "Unit")
        assert "StartLimitIntervalSec=15min" in unit
        assert "StartLimitBurst=5" in unit
        assert not any(line.startswith("StartLimit") for line in _section(text, "Service"))


@pytest.mark.parametrize(
    ("memory_max", "high", "limit"),
    [
        ("40G", "40G", "44G"),
        ("100K", "100K", "110K"),  # exact in the same unit
        ("10", "10", "11"),
        ("512M", "512M", "590558004"),  # not exact in any unit: rounded up, in bytes
    ],
)
def test_the_strata_memory_limit_sits_a_tenth_above_the_high_mark(
    tmp_path: Path, memory_max: str, high: str, limit: str
) -> None:
    text = services.strata_unit(_config(tmp_path, memory_max=memory_max), tmp_path / "key")

    lines = _lines(text)
    assert f"MemoryHigh={high}" in lines
    assert f"MemoryMax={limit}" in lines


def test_a_memory_size_that_is_no_size_never_reaches_the_unit(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    cfg = cfg.model_copy(update={"strata": cfg.strata.model_copy(update={"memory_max": "40GB"})})

    with pytest.raises(ConfigError, match="memory_max"):
        services.strata_unit(cfg, tmp_path / "key")


def test_the_voice_services_own_memory_limit_is_unchanged() -> None:
    lines = _lines(services.jarvis_unit())

    assert "MemoryMax=4G" in lines
    assert not any(line.startswith("MemoryHigh=") for line in lines)


def _args(profile: dict[str, object]) -> list[str]:
    args = profile["args"]
    assert isinstance(args, list)
    return [str(a) for a in cast("list[object]", args)]


def test_no_vram_reserve_is_passed_unless_one_is_set(tmp_path: Path) -> None:
    # On an engine that takes a size for its experts; the other needs one (see above).
    cfg = _config(tmp_path, run_config=inkling_run_config(tmp_path))
    args = _args(services.strata_profile(cfg))

    assert "--vram-reserve-mib" not in args


def test_a_vram_reserve_is_passed_to_strata(tmp_path: Path) -> None:
    args = _args(services.strata_profile(_config(tmp_path, vram_reserve_mib=1536)))

    assert args[args.index("--vram-reserve-mib") + 1] == "1536"


def test_the_whole_context_stays_in_vram_unless_kv_resident_is_set(tmp_path: Path) -> None:
    assert "--kv-resident" not in _args(services.strata_profile(_config(tmp_path)))


def test_kv_resident_keeps_that_much_of_the_context_in_vram_and_the_rest_in_ram(
    tmp_path: Path,
) -> None:
    args = _args(services.strata_profile(_config(tmp_path, context=65536, kv_resident=24576)))

    assert args[args.index("--kv-resident") + 1] == "24576"
    assert args[args.index("--max-context") + 1] == "65536"


@pytest.mark.parametrize("cells", [1, 20479])
def test_kv_resident_below_the_engines_minimum_is_refused(tmp_path: Path, cells: int) -> None:
    with pytest.raises(ValidationError, match="at least 20480"):
        _config(tmp_path, kv_resident=cells)


def _with_routing_profile(tmp_path: Path, *, learn_experts: bool = True) -> tuple[Config, Path]:
    """Return a config whose engine starts from a routing profile, and that profile."""
    cfg = _config(tmp_path, learn_experts=learn_experts)
    generic = tmp_path / "control" / "expert-profile.bin"
    generic.parent.mkdir(exist_ok=True)
    generic.write_bytes(b"generic")
    run = tmp_path / "install" / "strata-q4.json"
    data = json.loads(run.read_text(encoding="utf-8"))
    data["args"] += ["--expert-profile", str(generic)]
    run.write_text(json.dumps(data), encoding="utf-8")
    return cfg, generic


def test_an_engine_with_a_routing_profile_learns_jarvis_own_and_starts_from_it(
    tmp_path: Path,
) -> None:
    cfg, _ = _with_routing_profile(tmp_path)
    learned = str(services.expert_profile_file(cfg))

    args = _args(services.strata_profile(cfg))

    assert args[args.index("--expert-profile") + 1] == learned
    assert args[args.index("--expert-profile-save") + 1] == learned


def test_setup_starts_the_learned_profile_from_stratas_once_and_keeps_what_was_learned(
    tmp_path: Path,
) -> None:
    cfg, _ = _with_routing_profile(tmp_path)
    learned = services.expert_profile_file(cfg)

    path = services.write_strata_profile(cfg)
    first = path.read_text(encoding="utf-8")
    learned.write_bytes(b"learned from requests")  # what the engine saved
    services.write_strata_profile(cfg)

    assert learned.read_bytes() == b"learned from requests"
    assert path.read_text(encoding="utf-8") == first  # the profile never changes for it


def test_the_first_setup_copies_stratas_routing_profile(tmp_path: Path) -> None:
    cfg, _ = _with_routing_profile(tmp_path)

    services.write_strata_profile(cfg)

    assert services.expert_profile_file(cfg).read_bytes() == b"generic"


def test_learning_experts_is_left_out_when_off_or_when_the_engine_has_no_profile(
    tmp_path: Path,
) -> None:
    off, generic = _with_routing_profile(tmp_path, learn_experts=False)
    off_args = _args(services.strata_profile(off))

    assert "--expert-profile-save" not in off_args
    assert off_args[off_args.index("--expert-profile") + 1] == str(generic)
    assert "--expert-profile-save" not in _args(services.strata_profile(_config(tmp_path / "b")))
    services.write_strata_profile(off)
    assert not services.expert_profile_file(off).exists()


def test_pictures_stay_on_while_the_models_drive_is_not_mounted(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    (tmp_path / "models" / "q4.gguf").unlink()
    (tmp_path / "models" / "mmproj.gguf").unlink()

    assert services.reads_pictures(cfg)
    assert services.model_name(cfg) == "flash-q4"


def test_an_engine_whose_cache_fills_the_card_needs_a_vram_reserve(tmp_path: Path) -> None:
    with pytest.raises(services.StrataSetupError, match="vram_reserve_mib"):
        services.strata_profile(_config(tmp_path, vram_reserve_mib=0))


def test_the_learned_expert_profile_is_kept_per_model(tmp_path: Path) -> None:
    cfg, _ = _with_routing_profile(tmp_path)
    run = tmp_path / "install" / "strata-q4.json"
    data = json.loads(run.read_text(encoding="utf-8"))
    data["model_name"] = "other-model"
    other = tmp_path / "install" / "strata-other.json"
    other.write_text(json.dumps(data), encoding="utf-8")
    other_cfg = _config(tmp_path, run_config=other)

    assert services.expert_profile_file(cfg) != services.expert_profile_file(other_cfg)


@pytest.mark.parametrize("flag", ["--host", "--port"])
def test_a_run_config_that_picks_its_own_address_is_refused(tmp_path: Path, flag: str) -> None:
    cfg = _config(tmp_path)
    run = tmp_path / "install" / "strata-q4.json"
    data = json.loads(run.read_text(encoding="utf-8"))
    data["args"] += [flag, "192.0.2.1" if flag == "--host" else "9000"]
    run.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(services.StrataSetupError, match=flag):
        services.strata_profile(cfg)


def test_a_relative_routing_profile_is_a_setup_error(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    run = tmp_path / "install" / "strata-q4.json"
    data = json.loads(run.read_text(encoding="utf-8"))
    data["args"] += ["--expert-profile", "data/expert-profile.bin"]
    run.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(services.StrataSetupError, match="expert-profile"):
        services.write_strata_profile(cfg)


def test_a_bare_vision_flag_without_a_reader_is_dropped(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    run = tmp_path / "install" / "strata-q4.json"
    data = json.loads(run.read_text(encoding="utf-8"))
    del data["vision"]
    run.write_text(json.dumps(data), encoding="utf-8")

    assert "--vision" not in _args(services.strata_profile(cfg))
    assert not services.reads_pictures(cfg)


@pytest.mark.parametrize("arg", ["--host=192.0.2.1", "--port=9000"])
def test_an_address_given_with_an_equals_sign_is_refused_too(tmp_path: Path, arg: str) -> None:
    cfg = _config(tmp_path)
    run = tmp_path / "install" / "strata-q4.json"
    data = json.loads(run.read_text(encoding="utf-8"))
    data["args"].append(arg)
    run.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(services.StrataSetupError, match=arg.split("=", maxsplit=1)[0]):
        services.strata_profile(cfg)


def test_a_fixed_expert_cache_needs_no_reserve(tmp_path: Path) -> None:
    cfg = _config(tmp_path, vram_reserve_mib=0)
    run = tmp_path / "install" / "strata-q4.json"
    data = json.loads(run.read_text(encoding="utf-8"))
    at = data["args"].index("--expert-cache")
    data["args"][at + 1] = "2000"
    run.write_text(json.dumps(data), encoding="utf-8")

    args = _args(services.strata_profile(cfg))

    assert args[args.index("--expert-cache") + 1] == "2000"
