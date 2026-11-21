"""The Strata profile and the systemd user units Jarvis runs under.

`jarvis-strata.service` runs `strata serve` in the foreground with lazy loading, so the model stays
unloaded after boot until the first real request. `jarvis.service` runs the voice loop.
"""

import json
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


def strata_profile(config: Config) -> dict[str, object]:
    """Return the Strata run config: Jarvis's model, capped VRAM, lazy load, idle unload."""
    s = config.strata
    args = [
        "--native", str(s.model_file),
        "--max-context", str(s.context),
        "--prefill", str(s.prefill),
        "--mtp", "2",
        "--expert-vram", f"{s.expert_vram_gib:g}",
    ]  # fmt: skip
    return {
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


def write_strata_profile(config: Config) -> Path:
    """Write the Strata run config (no secrets: the key comes from the unit's environment)."""
    path = strata_profile_file()
    write_atomic(path, json.dumps(strata_profile(config), indent=1) + "\n")
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
            "--quiet",
        ]
    )
    memory = _path(s.memory_max, "memory_max")
    return f"""[Unit]
Description=Strata model server for Jarvis (lazy load, idle unload)
PartOf=graphical-session.target
After=graphical-session.target
{condition}
[Service]
Type=simple
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
MemoryMax={memory}
TimeoutStopSec=45

[Install]
WantedBy=graphical-session.target
"""


def jarvis_unit() -> str:
    """Return the `jarvis.service` unit text (the voice loop)."""
    exe = exec_program(str(Path(sys.executable).parent / "jarvis"), "jarvis executable")
    return f"""[Unit]
Description=Jarvis voice agent
PartOf=graphical-session.target
After=graphical-session.target pipewire.service {STRATA_UNIT}
Wants={STRATA_UNIT}

[Service]
Type=simple
ExecStart={exe} run
Restart=on-failure
RestartPreventExitStatus={_SETUP_ERROR}
RestartSec=5
RestartSteps=5
RestartMaxDelaySec=120
MemoryMax=4G
TimeoutStopSec=15

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
        write_atomic(path, text)
        written.append(path)
    return written
