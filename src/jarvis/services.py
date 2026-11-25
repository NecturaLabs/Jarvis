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
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

from jarvis import paths
from jarvis.config import ConfigError
from jarvis.files import write_atomic, write_atomic_bytes

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
# `strata.keep_in_ram`: the unit's memory is never swapped, and the engine may lock its model in RAM
# (it asks for that limit itself). A user unit gets a limit only up to its user manager's hard
# one, 8 MiB unless that was raised (docs/configuration.md), and starts with that; the swap
# setting holds either way.
_KEEP_IN_RAM = """\
LimitMEMLOCK=infinity
MemorySwapMax=0
"""
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


def expert_profile_file(config: Config) -> Path:
    """Return where the engine keeps what it learned about the experts Jarvis's requests use.

    One file per model, so a profile learned on one model never starts another.
    """
    name = re.sub(r"[^A-Za-z0-9._-]", "_", read_run_config(config).name) or "model"
    return paths.state_dir() / f"strata-expert-profile-{name}.bin"


def strata_ready_file() -> Path:
    """Return where the Strata service writes its instance record."""
    return paths.state_dir() / "strata-ready.json"


READER = "strata-vision"  # Strata's picture reader on the processor, in each engine folder
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


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Strata's own run config for a model (`strata-<model>.json`, written by `strata launch`).

    Everything model-specific comes from it: the engine, its arguments, the tokenizer, the
    architecture and the picture reader. Jarvis only sets its own limits on top (`strata_profile`).
    """

    path: Path
    data: dict[str, object]
    args: tuple[str, ...]

    def value_of(self, flag: str) -> str | None:
        """Return the value after `flag`, or None when the flag is absent or takes no value."""
        if flag not in self.args:
            return None
        at = self.args.index(flag) + 1
        value = self.args[at] if at < len(self.args) else ""
        return None if not value or value.startswith("--") else value

    @property
    def user_args(self) -> tuple[str, ...]:
        """The engine options set by hand, which Strata applies on top of `args` at every start."""
        raw = self.data.get("user_args")
        return tuple(str(a) for a in cast("list[object]", raw)) if isinstance(raw, list) else ()

    @property
    def loads_projection(self) -> bool:
        """Whether the engine loads a control vector, such as Strata's speed projection."""
        return bool(set(PROJECTION_FLAGS) & {*self.args, *self.user_args})

    @property
    def model(self) -> Path:
        """The model file the engine loads (`--native`)."""
        native = self.value_of("--native")
        return Path(native) if native else Path()

    @property
    def expert_profile(self) -> Path | None:
        """The routing profile the engine starts from (`--expert-profile`), if it takes one."""
        value = self.value_of("--expert-profile")
        return Path(value) if value else None

    @property
    def name(self) -> str:
        """The model's name in Strata."""
        return str(self.data.get("model_name", ""))

    @property
    def picture_reader(self) -> Path | None:
        """The picture reader's file, or None when the model is set up without one.

        One engine runs a reader beside it (`"vision"` with its `mmproj`); another reads pictures
        itself (`--vision FILE`).
        """
        vision = self.data.get("vision")
        if isinstance(vision, dict):
            mmproj = cast("dict[str, object]", vision).get("mmproj")
            return Path(str(mmproj)) if mmproj else None
        value = self.value_of("--vision")
        return Path(value) if value else None


def read_run_config(config: Config) -> RunConfig:
    """Read `strata.run_config`, without checking that the files it names are there.

    A model drive that is not mounted yet still gives its model's path (`missing_files` checks).

    Raises:
        StrataSetupError: it is unset or unreadable, is not a Strata run config, or picks its own
            address (`--host`, `--port`: Jarvis serves on 127.0.0.1 at `strata.port`).
    """
    path = config.strata.run_config
    if path == Path():
        msg = (
            "strata.run_config is not set: run `strata launch` for the model, then "
            "`jarvis setup --run-config <its strata-<model>.json>`"
        )
        raise StrataSetupError(msg)
    try:
        loaded: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        msg = f"cannot read the Strata run config {path}: {exc}"
        raise StrataSetupError(msg) from exc
    data = cast("dict[str, object]", loaded) if isinstance(loaded, dict) else {}
    exe, raw = data.get("exe"), data.get("args")
    if not (isinstance(exe, str) and exe.startswith("component:strata-") and isinstance(raw, list)):
        msg = f"{path} is not a Strata run config (no Strata engine and arguments in it)"
        raise StrataSetupError(msg)
    run = RunConfig(path, data, tuple(str(a) for a in cast("list[object]", raw)))
    for flag in ("--host", "--port"):
        if any(a == flag or a.startswith(f"{flag}=") for a in run.args):
            msg = f"{path} sets its own {flag}; Jarvis serves on 127.0.0.1 at strata.port"
            raise StrataSetupError(msg)
    return run


def fills_the_card(run: RunConfig) -> bool:
    """Whether the engine's expert cache takes all free VRAM (`--expert-cache auto`)."""
    return run.value_of("--expert-cache") == "auto"


def missing_files(run: RunConfig) -> list[str]:
    """Return the files the run config names that are not there (a drive not mounted, a move)."""
    named = [*run.args, *(str(f) for f in (run.picture_reader,) if f is not None)]
    return sorted({a for a in named if a.startswith("/") and not Path(a).exists()})


def _require_files(run: RunConfig) -> None:
    """Raise `StrataSetupError` when a file the run config names is missing.

    Raises:
        StrataSetupError: naming the missing files.
    """
    missing = missing_files(run)
    if missing:
        msg = (
            f"the Strata run config {run.path} names files that are missing: {', '.join(missing)}; "
            "mount the model's drive, or run `strata launch` for the model again, then "
            "`jarvis setup`"
        )
        raise StrataSetupError(msg)


def reads_pictures(config: Config) -> bool:
    """Whether Jarvis serves its model with a picture reader (`strata.pictures` and one set up)."""
    if not config.strata.pictures:
        return False
    try:
        return read_run_config(config).picture_reader is not None
    except StrataSetupError:
        return False


def model_name(config: Config) -> str:
    """Return the served model's name, or what is wrong with its run config."""
    try:
        return read_run_config(config).name or "unnamed model"
    except StrataSetupError as exc:
        return f"no model ({exc})"


def learns_experts(config: Config, run: RunConfig) -> bool:
    """Whether the engine learns Jarvis's experts: `learn_experts`, and a profile to start from."""
    return config.strata.learn_experts and run.expert_profile is not None


def _with(args: list[str], flag: str, value: str) -> list[str]:
    """Return `args` with `flag` set to `value`, replacing its old value or adding both."""
    if flag in args:
        at = args.index(flag)
        return [*args[: at + 1], value, *args[at + 2 :]]
    return [*args, flag, value]


def _without(args: list[str], flag: str) -> list[str]:
    """Return `args` without `flag` and its value, if it has one."""
    if flag not in args:
        return args
    at = args.index(flag)
    valued = at + 1 < len(args) and not args[at + 1].startswith("--")
    return [*args[:at], *args[at + (2 if valued else 1) :]]


# The control-vector options Strata keeps together (its `USER_KEPT_FLAGS`): the experimental speed
# projection, which Strata ships for some models, is loaded through them.
PROJECTION_FLAGS = (
    "--control-vector",
    "--control-vector-scaled",
    "--control-vector-layer-range",
    "--cvec-mode",
    "--cvec-dir",
)


def _without_projection(args: list[str] | tuple[str, ...]) -> list[str]:
    """Return `args` without the control-vector options and every value after each."""
    kept: list[str] = []
    dropping = False
    for arg in args:
        if arg.startswith("--"):
            dropping = arg in PROJECTION_FLAGS
        if not dropping:
            kept.append(arg)
    return kept


def _tuned_args(config: Config, run: RunConfig) -> list[str]:
    """Return the run config's arguments with Jarvis's context and VRAM limits.

    A limit is set through whichever flag the model's engine takes, as Strata's setup wrote it:
    `--expert-vram` or `--vram-experts-gb` take `expert_vram_gib` (the experts kept on the card,
    the rest go to RAM); an engine whose expert cache fills the card (`--expert-cache`) is held back
    by `vram_reserve_mib`, the VRAM it leaves free. An engine that starts from a routing profile
    learns Jarvis's own (`learn_experts`, `expert_profile_file`). Without `strata.pictures`, or
    without a reader set up, `--vision` is dropped.

    Raises:
        StrataSetupError: the engine's expert cache would fill the card and no reserve is set.
    """
    s = config.strata
    if fills_the_card(run) and not s.vram_reserve_mib:
        msg = (
            "this model's engine fills the card with its expert cache (--expert-cache), which "
            "leaves no VRAM for a game: set strata.vram_reserve_mib (12000 holds "
            "Qwen3.8-Flash-Next at about 8 GiB on a 20 GB card, see docs/models.md)"
        )
        raise StrataSetupError(msg)
    args = _with(list(run.args), "--max-context", str(s.context))
    for flag in ("--expert-vram", "--vram-experts-gb"):
        if flag in args:
            args = _with(args, flag, f"{s.expert_vram_gib:g}")
    args = _without(args, "--ram-experts-gb")  # the planner puts what the card does not hold in RAM
    if s.vram_reserve_mib:
        args = _with(args, "--vram-reserve-mib", str(s.vram_reserve_mib))
    if s.kv_resident:
        args = _with(args, "--kv-resident", str(s.kv_resident))
    if learns_experts(config, run):
        learned = str(expert_profile_file(config))  # `write_strata_profile` starts it from Strata's
        args = _with(_with(args, "--expert-profile", learned), "--expert-profile-save", learned)
    if not s.pictures or run.picture_reader is None:
        args = _without(args, "--vision")
    if not s.speed_projection:
        args = _without_projection(args)
    return args


def _vision(config: Config, run: RunConfig) -> dict[str, object] | None:
    """Return the profile's picture reader beside the engine, on the processor and on demand.

    It runs in RAM, not in the VRAM that games and the model's experts use, and only from the first
    picture until `vision_idle_s` without one ("elastic"). None: no reader beside the engine.
    """
    vision = run.data.get("vision")
    if not config.strata.pictures or not isinstance(vision, dict):
        return None
    threads = max(1, min((os.cpu_count() or 8) // 2, READER_THREADS_CAP))
    return {
        **cast("dict[str, object]", vision),
        "exe": str(_reader(config)),
        "gpu": False,
        "threads": threads,
        "elastic": True,
        "elastic_idle_s": config.strata.vision_idle_s,
    }


# Keys of Strata's run config that describe one launch, not the model; Jarvis sets its own.
_LAUNCH_KEYS = ("args_written", "args_derived", "engine_ready_s", "vision")


def strata_profile(config: Config) -> dict[str, object]:
    """Return the Strata run config Jarvis serves: the model's own, with Jarvis's limits on top.

    Jarvis's part: one request at a time on 127.0.0.1, lazy load, idle unload, the reasoning cap,
    its log, and the context and VRAM limits (`_tuned_args`). `StrataSetupError` comes from a run
    config that cannot be used (`read_run_config`) and from a missing picture reader (`_reader`).
    """
    s = config.strata
    run = read_run_config(config)
    _require_files(run)
    args = _tuned_args(config, run)
    profile: dict[str, object] = {
        **{key: value for key, value in run.data.items() if key not in _LAUNCH_KEYS},
        "args": args,
        "args_written": args,
        "cwd": str(s.home / "terminal" / "current"),
        "log": str(paths.state_dir() / "strata.log"),
        "port": s.port,
        "host": "127.0.0.1",
        "open_browser": False,
        "parallel": 1,
        "lazy_load": True,
        "idle_unload_s": s.idle_unload_s,
        "reasoning_budget_tokens": s.reasoning_budget_tokens,
    }
    if s.effort_position:
        profile["effort_position"] = s.effort_position
    if not s.speed_projection and run.user_args:
        profile["user_args"] = _without_projection(run.user_args)
    vision = _vision(config, run)
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
    """Write the Strata run config (no secrets: the key comes from the unit's environment).

    When the engine learns Jarvis's experts and has learned nothing yet, it starts from a copy of
    the run config's own routing profile, so the profile never has to change once it has.

    Raises:
        StrataSetupError: as `strata_profile`, or that routing profile is not an absolute file.
    """
    path = strata_profile_file()
    profile = strata_profile(config)
    run = read_run_config(config)
    seed = run.expert_profile
    learned = expert_profile_file(config)
    if learns_experts(config, run) and seed is not None and not learned.is_file():
        if not (seed.is_absolute() and seed.is_file()):
            msg = f"the run config's --expert-profile {seed} is not a file Jarvis can start from"
            raise StrataSetupError(msg)
        write_atomic_bytes(learned, seed.read_bytes())
    _write_if_changed(path, json.dumps(profile, indent=1) + "\n")
    return path


def _strata_bin(config: Config) -> str:
    candidate = config.strata.home / "terminal" / "current" / "bin" / "strata"
    return str(candidate) if candidate.exists() else (shutil.which("strata") or str(candidate))


def strata_unit(config: Config, key_file: Path) -> str:
    """Return the `jarvis-strata.service` unit text.

    A value that cannot be written safely into a unit (a control character in it) is a ConfigError.
    """
    s = config.strata
    model = read_run_config(config).model if s.run_config != Path() else Path()
    condition = f"ConditionPathExists={_path(model, 'model file')}\n" if model != Path() else ""
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
    in_ram = _KEEP_IN_RAM if s.keep_in_ram else ""
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
{in_ram}CPUWeight={_CPU_WEIGHT}
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


def write_strata_unit(config: Config, key_file: Path) -> Path:
    """Write `jarvis-strata.service` alone and return its path; a ConfigError writes nothing."""
    path = unit_dir() / STRATA_UNIT
    _write_if_changed(path, strata_unit(config, key_file))
    return path


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
