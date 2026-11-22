"""The Strata profile and systemd units: their text, their quoting and how they are written."""

import json
import stat
import sys
from pathlib import Path

import pytest

from jarvis import services
from jarvis.config import Config, ConfigError, StrataConfig


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))


pytestmark = pytest.mark.usefixtures("home")


def _config(tmp_path: Path, **strata: object) -> Config:
    settings: dict[str, object] = {
        "home": tmp_path / "strata-home",
        "components_parent": tmp_path / "components",
        "model_file": tmp_path / "models" / "q4.gguf",
        "tokenizer_dir": tmp_path / "tokenizer",
        **strata,
    }
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
    cfg = _config(tmp_path, home=spaced / "strata home", model_file=spaced / "q4 k.gguf")
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
    unchecked = cfg.model_copy(
        update={"strata": cfg.strata.model_copy(update={"model_file": Path(bad)})}
    )

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


def test_the_model_condition_is_omitted_when_no_model_is_configured(tmp_path: Path) -> None:
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
    assert profile["model_name"] == cfg.strata.model_id
    assert profile["tokenizer"] == str(tmp_path / "tokenizer")
    assert profile["args"][:2] == ["--native", str(tmp_path / "models" / "q4.gguf")]
    assert profile["args"][profile["args"].index("--max-context") + 1] == "4096"
    assert "api_key" not in json.dumps(profile)


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


def test_without_a_picture_reader_the_profile_reads_no_pictures(tmp_path: Path) -> None:
    profile = json.loads(json.dumps(services.strata_profile(_config(tmp_path))))

    assert "vision" not in profile
    assert "--vision" not in profile["args"]


def test_the_picture_reader_runs_on_the_processor_and_only_when_a_picture_comes(
    tmp_path: Path,
) -> None:
    engine = _engine(tmp_path, "0.1.60-x", "0.1.62-x")
    reader = tmp_path / "models" / "mmproj-F16.gguf"
    cfg = _config(tmp_path, vision_file=reader, vision_idle_s=90)

    profile = json.loads(json.dumps(services.strata_profile(cfg)))

    assert "--vision" in profile["args"]
    vision = profile["vision"]
    assert vision["exe"] == str(engine / "0.1.62-x" / "strata-vision")
    assert vision["mmproj"] == str(reader)
    assert vision["model"] == str(cfg.strata.model_file)
    assert vision["gpu"] is False  # in RAM, not in the VRAM games need
    assert vision["elastic"] is True  # started by the first picture, stopped when idle
    assert vision["elastic_idle_s"] == 90
    assert vision["max_tokens"] > 0
    assert 1 <= vision["threads"] <= 16


def test_a_picture_reader_without_the_engine_program_is_a_setup_error(tmp_path: Path) -> None:
    cfg = _config(tmp_path, vision_file=tmp_path / "models" / "mmproj-F16.gguf")

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
    "UMask=0077",
)
_VOICE_HARDENED = (
    "SystemCallArchitectures=native",
    "ProtectKernelModules=yes",
    "ProtectKernelLogs=yes",
)
# Left out on purpose (issue #33): each would break something a unit runs.
_NOT_SET = (
    "RestrictNamespaces",  # the Azure helper runs under `unshare --map-current-user --net`
    "MemoryDenyWriteExecute",  # onnxruntime and Strata's HIP path may JIT
    "ProtectHome",  # config, state and models live under home and on /data
    "ProtectSystem",
    "PrivateNetwork",  # Spotify, Open-Meteo and web reads
    "IPAddressDeny",
    "RestrictRealtime",  # until pw-record and pw-play are shown not to need it
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
    return [str(a) for a in args]  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]


def test_the_prompt_chunk_defaults_to_512(tmp_path: Path) -> None:
    args = _args(services.strata_profile(_config(tmp_path)))

    assert args[args.index("--prefill") + 1] == "512"


def test_the_prompt_chunk_may_be_left_to_strata(tmp_path: Path) -> None:
    args = _args(services.strata_profile(_config(tmp_path, prefill="auto")))

    assert args[args.index("--prefill") + 1] == "auto"


def test_no_vram_reserve_is_passed_unless_one_is_set(tmp_path: Path) -> None:
    args = _args(services.strata_profile(_config(tmp_path)))

    assert "--vram-reserve-mib" not in args


def test_a_vram_reserve_is_passed_to_strata(tmp_path: Path) -> None:
    args = _args(services.strata_profile(_config(tmp_path, vram_reserve_mib=1536)))

    assert args[args.index("--vram-reserve-mib") + 1] == "1536"
