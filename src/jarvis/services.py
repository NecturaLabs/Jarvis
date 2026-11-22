"""The Strata profile and the systemd user units Jarvis runs under.

`jarvis-strata.service` runs `strata serve` in the foreground with lazy loading: it starts without
the model, and `jarvis.service`, the voice loop, loads it when it starts. Strata's own notes and
per-request timings (no request text) go to the journal.
"""

import contextlib
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from jarvis import paths
from jarvis.config import ConfigError
from jarvis.files import write_atomic

if TYPE_CHECKING:
    from jarvis.config import Config

STRATA_UNIT = "jarvis-strata.service"
JARVIS_UNIT = "jarvis.service"

_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_NEEDS_QUOTES = re.compile(r"""[\s"'\\;]""")
_SETUP_ERROR = 2  # `jarvis run` exits with this when setup is missing; restarting cannot fix it


def _checked(value: str, what: str) -> str:
    """Return `value` when it is safe to write into a unit line.

    Raises:
        ConfigError: it holds a control character (a newline would add a directive) or ends in a
            backslash (which would continue the line).
    """
    if _CONTROL.search(value):
        msg = f"{what} holds a control character and cannot be written into a systemd unit"
        raise ConfigError(msg)
    if value.endswith("\\"):
        msg = f"{what} ends in a backslash and cannot be written into a systemd unit"
        raise ConfigError(msg)
    return value


def _quoted(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def exec_word(value: str, what: str = "value") -> str:
    """Return `value` as one word of an `ExecStart=` line (systemd.service(5), "Command lines").

    Specifiers (`%`) and environment references (`$`) are doubled so they stay literal; a word with
    whitespace, quotes, a backslash or a semicolon is double-quoted.
    """
    word = _checked(value, what).replace("%", "%%").replace("$", "$$")
    return _quoted(word) if _NEEDS_QUOTES.search(word) or not word else word


def exec_program(value: str, what: str = "program") -> str:
    """Return `value` as the program word that starts an `ExecStart=` line.

    systemd reads a leading `-`, `@`, `+`, `!` or `:` there as a prefix that changes how the
    command runs, so only an absolute path is accepted.

    Raises:
        ConfigError: `value` is not an absolute path, or cannot be written into a unit.
    """
    if not value.startswith("/"):
        msg = f"{what} must be an absolute path, not {value!r}"
        raise ConfigError(msg)
    return exec_word(value, what)


_SIZE = re.compile(r"(\d+)([KMGT]?)")
_SIZE_UNITS = {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
MEMORY_MAX_PERCENT = 110  # `MemoryMax` is this share of `MemoryHigh`


def memory_limit(memory_high: str) -> str:
    """Return the `MemoryMax` that sits a tenth above `memory_high` (a size such as `40G`).

    The result is in the largest unit that holds it exactly, else in bytes rounded up.

    Raises:
        ConfigError: `memory_high` is not a number with an optional `K`, `M`, `G` or `T`.
    """
    found = _SIZE.fullmatch(memory_high)
    if found is None:
        msg = f"memory_max must be a number with an optional K, M, G or T, not {memory_high!r}"
        raise ConfigError(msg)
    amount = int(found[1]) * _SIZE_UNITS[found[2]]
    limit = -(-amount * MEMORY_MAX_PERCENT // 100)
    for unit in ("T", "G", "M", "K"):
        if limit and limit % _SIZE_UNITS[unit] == 0:
            return f"{limit // _SIZE_UNITS[unit]}{unit}"
    return str(limit)


# Sandboxing both units take. Left out on purpose, with why: `RestrictNamespaces` (the Azure helper
# runs under `unshare --map-current-user --net`), `MemoryDenyWriteExecute` (onnxruntime and
# Strata's HIP path may JIT), `ProtectHome` and `ProtectSystem` (config, state and models live under
# home and on other disks), `PrivateNetwork` and `IPAddressDeny` (Spotify, Open-Meteo, web reads),
# `RestrictRealtime` (not shown safe for `pw-record` and `pw-play`), and on the voice service
# `ProtectKernelModules` and `ProtectKernelLogs` (in a user unit they put the service in a private
# user namespace, which every program it starts inherits, Spotify included) and `UMask=0077` (a file
# the file tools write for the owner keeps the mode it asks for).
_HARDENING = """\
NoNewPrivileges=yes
LockPersonality=yes
RestrictSUIDSGID=yes
"""
# Strata writes nothing for the owner: whatever it creates is private.
_HARDENING_STRATA = """\
UMask=0077
"""
# What only the voice service takes: it runs no model engine, whose GPU runtime is not shown to
# keep to this CPU's own system calls.
_HARDENING_VOICE = """\
SystemCallArchitectures=native
"""
# A game beside the units wins the processor: the weight is half the default (100).
_CPU_WEIGHT = 50
# A unit that fails five times in 15 minutes stays `failed` (`jarvis start` resets it) instead of
# restarting for ever; for Strata every round would read the model again.
_START_LIMIT = """\
StartLimitIntervalSec=15min
StartLimitBurst=5
"""


def _environment(name: str, value: str) -> str:
    """Return an `Environment=` line (specifiers doubled; `$` is literal there)."""
    assignment = f"{name}={_checked(value, name).replace('%', '%%')}"
    return f"Environment={_quoted(assignment) if _NEEDS_QUOTES.search(assignment) else assignment}"


def _path(value: object, what: str) -> str:
    """Return a path-valued setting: the rest of the line, so spaces need no quotes."""
    return _checked(str(value), what).replace("%", "%%")


def strata_profile_file() -> Path:
    """Return the generated Strata run config."""
    return paths.config_dir() / "strata-profile.json"


def strata_ready_file() -> Path:
    """Return where the Strata service writes its instance record."""
    return paths.state_dir() / "strata-ready.json"


READER = "strata-vision"  # Strata's picture reader on the processor, in each engine folder
READER_MAX_TOKENS = 300  # the tokens one picture may become (Strata's own default for this reader)
READER_THREADS_CAP = 16


class StrataSetupError(ConfigError):
    """Something the Strata profile needs is missing from the Strata install."""


def _reader(config: Config) -> Path:
    """Return the picture reader of the newest Strata engine folder.

    Strata moves the reader to the engine in use when it starts, so any installed one serves.

    Raises:
        StrataSetupError: no engine folder carries the reader.
    """
    engines = config.strata.components_parent / "strata" / "components" / "engine"
    found = sorted(engines.glob(f"*/{READER}")) if engines.is_dir() else []
    if not found:
        msg = f"no picture reader ({READER}) in {engines}; install the Strata engine first"
        raise StrataSetupError(msg)
    return found[-1]


def _vision(config: Config) -> dict[str, object]:
    """Return the profile's picture section: the reader on the processor, started on demand.

    It runs in RAM, not in the VRAM that games and the model's experts use, and only from the first
    picture until `vision_idle_s` without one ("elastic").
    """
    s = config.strata
    threads = max(1, min((os.cpu_count() or 8) // 2, READER_THREADS_CAP))
    return {
        "exe": str(_reader(config)),
        "mmproj": str(s.vision_file),
        "model": str(s.model_file),
        "gpu": False,
        "max_tokens": READER_MAX_TOKENS,
        "threads": threads,
        "elastic": True,
        "elastic_idle_s": s.vision_idle_s,
    }


def strata_profile(config: Config) -> dict[str, object]:
    """Return the Strata run config: Jarvis's model, capped VRAM, lazy load, idle unload.

    With `strata.vision_file` set it also reads pictures (`_vision`, which raises
    `StrataSetupError` when the Strata engine has no picture reader).
    """
    s = config.strata
    args = [
        "--native", str(s.model_file),
        "--max-context", str(s.context),
        "--prefill", str(s.prefill),
        "--mtp", "2",
        "--expert-vram", f"{s.expert_vram_gib:g}",
    ]  # fmt: skip
    if s.vram_reserve_mib:
        args += ["--vram-reserve-mib", str(s.vram_reserve_mib)]
    vision = _vision(config) if s.pictures else None
    if vision is not None:
        args.append("--vision")
    profile: dict[str, object] = {
        "exe": "component:strata-engine",
        "args": args,
        "args_written": args,
        "cwd": str(s.home / "terminal" / "current"),
        "tokenizer": str(s.tokenizer_dir),
        "model_name": s.model_id,
        "arch": "qwen35moe",
        "log": str(paths.state_dir() / "strata.log"),
        "lib_dirs": [],
        "port": s.port,
        "backend": "hip",
        "gpu": 0,
        "gpus_asked": True,
        "host": "127.0.0.1",
        "open_browser": False,
        "parallel": 1,
        "lazy_load": True,
        "idle_unload_s": s.idle_unload_s,
        "reasoning_budget_tokens": s.reasoning_budget_tokens,
    }
    if vision is not None:
        profile["vision"] = vision
    return profile


def _write_if_changed(path: Path, text: str) -> None:
    """Write `text` atomically unless the file already holds it.

    `jarvis restart` restarts Strata when its profile or unit is newer than the running server,
    so a file rewritten with the same text must keep its old time.
    """
    with contextlib.suppress(OSError, ValueError):  # missing or unreadable: write it
        if path.read_text(encoding="utf-8") == text:
            return
    write_atomic(path, text)


def write_strata_profile(config: Config) -> Path:
    """Write the Strata run config (no secrets: the key comes from the unit's environment)."""
    path = strata_profile_file()
    _write_if_changed(path, json.dumps(strata_profile(config), indent=1) + "\n")
    return path


def _strata_bin(config: Config) -> str:
    candidate = config.strata.home / "terminal" / "current" / "bin" / "strata"
    return str(candidate) if candidate.exists() else (shutil.which("strata") or str(candidate))


def strata_unit(config: Config, key_file: Path) -> str:
    """Return the `jarvis-strata.service` unit text.

    A value that cannot be written safely into a unit (a control character in it) is a ConfigError.
    """
    s = config.strata
    condition = (
        f"ConditionPathExists={_path(s.model_file, 'model_file')}\n"
        if s.model_file != Path()
        else ""
    )
    start = " ".join(
        [
            exec_program(_strata_bin(config), "strata executable"),
            "serve --profile",
            exec_word(str(strata_profile_file()), "profile file"),
            "\\\n  --ready-file",
            exec_word(str(strata_ready_file()), "ready file"),
        ]
    )
    memory = _path(s.memory_max, "memory_max")
    limit = memory_limit(memory)
    return f"""[Unit]
Description=Strata model server for Jarvis
PartOf=graphical-session.target
After=graphical-session.target
{_START_LIMIT}{condition}
[Service]
Type=exec
{_environment("STRATA_DATA_DIR", str(s.home))}
{_environment("XDG_DATA_HOME", str(s.components_parent))}
Environment=PIPEWIRE_REMOTE=/nonexistent
Environment=PULSE_SERVER=unix:/nonexistent
EnvironmentFile={_path(key_file, "key file")}
ExecStart={start}
Restart=on-failure
RestartSec=5
RestartSteps=5
RestartMaxDelaySec=120
MemoryHigh={memory}
MemoryMax={limit}
CPUWeight={_CPU_WEIGHT}
TimeoutStopSec=45
{_HARDENING}{_HARDENING_STRATA}
[Install]
WantedBy=graphical-session.target
"""


def jarvis_unit() -> str:
    """Return the `jarvis.service` unit text (the voice loop)."""
    exe = exec_program(str(Path(sys.executable).parent / "jarvis"), "jarvis executable")
    return f"""[Unit]
Description=Jarvis voice agent
PartOf=graphical-session.target
After=graphical-session.target pipewire.service wireplumber.service {STRATA_UNIT}
Wants={STRATA_UNIT}
{_START_LIMIT}
[Service]
Type=exec
ExecStart={exe} run
Restart=on-failure
RestartPreventExitStatus={_SETUP_ERROR}
RestartSec=5
RestartSteps=5
RestartMaxDelaySec=120
MemoryMax=4G
CPUWeight={_CPU_WEIGHT}
TimeoutStopSec=15
{_HARDENING}{_HARDENING_VOICE}
[Install]
WantedBy=graphical-session.target
"""


def unit_dir() -> Path:
    """Return the user unit directory."""
    return paths.xdg_config_home() / "systemd" / "user"


def write_units(config: Config, key_file: Path) -> list[Path]:
    """Write both unit files and return their paths.

    Both texts are rendered, and so checked, before the first file is written: a ConfigError leaves
    the existing units untouched.
    """
    texts = ((STRATA_UNIT, strata_unit(config, key_file)), (JARVIS_UNIT, jarvis_unit()))
    directory = unit_dir()
    written: list[Path] = []
    for name, text in texts:
        path = directory / name
        _write_if_changed(path, text)
        written.append(path)
    return written
