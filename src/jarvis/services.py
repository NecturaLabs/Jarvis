"""The Strata profile and the systemd user units Jarvis runs under.

`jarvis-strata.service` runs `strata serve` in the foreground with lazy loading, so the model stays
unloaded after boot until the first real request. `jarvis.service` runs the voice loop.
"""

import json
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from jarvis import paths

if TYPE_CHECKING:
    from jarvis.config import Config

STRATA_UNIT = "jarvis-strata.service"
JARVIS_UNIT = "jarvis.service"


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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(strata_profile(config), indent=1) + "\n", encoding="utf-8")
    return path


def _strata_bin(config: Config) -> str:
    candidate = config.strata.home / "terminal" / "current" / "bin" / "strata"
    return str(candidate) if candidate.exists() else (shutil.which("strata") or str(candidate))


def strata_unit(config: Config, key_file: Path) -> str:
    """Return the `jarvis-strata.service` unit text."""
    s = config.strata
    return f"""[Unit]
Description=Strata model server for Jarvis (lazy load, idle unload)
PartOf=graphical-session.target
After=graphical-session.target
ConditionPathExists={s.model_file}

[Service]
Type=simple
Environment=STRATA_DATA_DIR={s.home}
Environment=XDG_DATA_HOME={s.components_parent}
Environment=PIPEWIRE_REMOTE=/nonexistent
Environment=PULSE_SERVER=unix:/nonexistent
EnvironmentFile={key_file}
ExecStart={_strata_bin(config)} serve --profile {strata_profile_file()} \\
  --ready-file {strata_ready_file()} --quiet
Restart=on-failure
RestartSec=5
RestartSteps=5
RestartMaxDelaySec=120
MemoryHigh={s.memory_max}
MemoryMax={s.memory_max}
TimeoutStopSec=45

[Install]
WantedBy=graphical-session.target
"""


def jarvis_unit() -> str:
    """Return the `jarvis.service` unit text (the voice loop)."""
    exe = Path(sys.executable).parent / "jarvis"
    return f"""[Unit]
Description=Jarvis voice agent
PartOf=graphical-session.target
After=graphical-session.target pipewire.service {STRATA_UNIT}
Wants={STRATA_UNIT}

[Service]
Type=simple
ExecStart={exe} run
Restart=on-failure
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
    """Write both unit files and return their paths."""
    directory = unit_dir()
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, text in ((STRATA_UNIT, strata_unit(config, key_file)), (JARVIS_UNIT, jarvis_unit())):
        path = directory / name
        path.write_text(text, encoding="utf-8")
        written.append(path)
    return written
