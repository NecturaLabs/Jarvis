"""Whether the model fits in the memory that is free beside what is already running.

Strata loads the model's experts into RAM. Started with a game running and too little memory
available, the kernel fails page allocations and kills a running program. So before the model
server starts or restarts, and before anything asks it to load, Jarvis compares the model file's
size with `MemAvailable` (the kernel's own figure for what a program can take without swapping,
reclaimable page cache included) and refuses when the load would leave less than
`strata.min_free_ram_mib` free. `strata.allow_low_ram` loads anyway.

`python -m jarvis.ramguard` is the check the model server's unit runs before it starts
(`ExecStartPre=`): it reads the config and the profile when it runs, so the key and the mode in
force are never stale.
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from jarvis import config, paths, services
from jarvis.audit import AuditLog
from jarvis.jsonutil import obj

if TYPE_CHECKING:
    from collections.abc import Sequence

MEMINFO = Path("/proc/meminfo")
CGROUP_ROOT = Path("/sys/fs/cgroup")
EXIT_SHORT = services.RAM_SHORT_EXIT  # EX_TEMPFAIL: the unit's check refuses a start now
_KIB = 1024
_MIB = 1024**2
_GIB = 1024**3


def available_bytes() -> int | None:
    """Return `MemAvailable` in bytes, or None when `/proc/meminfo` cannot be read."""
    try:
        text = MEMINFO.read_text(encoding="utf-8")
    except OSError, UnicodeDecodeError:
        return None
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        if key == "MemAvailable":
            try:
                return int(rest.split()[0]) * _KIB
            except ValueError, IndexError:
                return None
    return None


def held_by_unit(unit: str) -> int:
    """Return the anonymous memory in bytes that `unit` holds, which stopping it gives back.

    A server that runs holds its loaded model, so `MemAvailable` leaves it out; a restart frees it
    before the new load. Page cache is not counted: `MemAvailable` already counts what the kernel
    can reclaim. Zero when the unit has no cgroup or its memory cannot be read.
    """
    uid = os.getuid()
    base = CGROUP_ROOT / "user.slice" / f"user-{uid}.slice" / f"user@{uid}.service"
    try:
        for stat in base.glob(f"**/{unit}/memory.stat"):
            for line in stat.read_text(encoding="utf-8").splitlines():
                key, _, value = line.partition(" ")
                if key == "anon":
                    return int(value)
    except OSError, ValueError:
        return 0
    return 0


def _gib(size: float) -> str:
    return f"{size / _GIB:.1f} GiB"


@dataclass(frozen=True, slots=True)
class Verdict:
    """What the model needs against what is available, with the margin to keep.

    `need` and `available` are None when they could not be found out; the load is then not
    refused (`jarvis doctor` and the unit's other checks name a missing model file).
    """

    need: int | None
    available: int | None
    margin: int
    allowed: bool = False
    model: Path | None = None

    @property
    def known(self) -> bool:
        """Whether both figures are known."""
        return self.need is not None and self.available is not None

    @property
    def short(self) -> bool:
        """Whether the load is refused: it would leave less than the margin and is not allowed."""
        if self.need is None or self.available is None or self.allowed:
            return False
        return self.available - self.need < self.margin

    @property
    def summary(self) -> str:
        """The figures in one line; empty when they are not known."""
        if self.need is None or self.available is None:
            return ""
        figures = (
            f"needs about {_gib(self.need)}, {_gib(self.available)} available, "
            f"{_gib(self.margin)} kept free"
        )
        return f"{figures}; loading anyway: strata.allow_low_ram is on" if self.allowed else figures

    @property
    def text(self) -> str:
        """The reason a load is refused, and what to do about it."""
        return (
            f"not enough free memory to load the model: it {self.summary} "
            "(strata.min_free_ram_mib). Close a game or other heavy program and try again, "
            "or set strata.allow_low_ram = true to load anyway."
        )


def check(model: Path | None, strata: config.StrataConfig, *, credit: int = 0) -> Verdict:
    """Return whether `model` fits in the available memory.

    `credit` is memory a stopping server gives back before the load (`held_by_unit`). The need is
    the size of the model file, which Strata reads into RAM.
    """
    need: int | None = None
    if model is not None and model != Path():
        try:
            need = model.stat().st_size
        except OSError:
            need = None
    free = available_bytes()
    return Verdict(
        need,
        None if free is None else free + credit,
        strata.min_free_ram_mib * _MIB,
        allowed=strata.allow_low_ram,
        model=model,
    )


def reason(model: Path | None, strata: config.StrataConfig, *, credit: int = 0) -> str:
    """Return why the model should not load now, or "" when it may."""
    verdict = check(model, strata, credit=credit)
    return verdict.text if verdict.short else ""


def refusal(where: str, verdict: Verdict) -> str:
    """Audit a refused load (`ram_refused`: where, need, available, margin, the model file).

    Returns the reason, or "" when `verdict` refuses nothing and nothing is recorded.
    """
    if not verdict.short:
        return ""
    AuditLog(paths.audit_file()).record(
        "ram_refused",
        where=where,
        need=verdict.need,
        available=verdict.available,
        margin=verdict.margin,
        path=str(verdict.model or ""),
    )
    return verdict.text


def served_model(cfg: config.Config) -> Path | None:
    """Return the model file the run config of `cfg` names, or None when it is not known."""
    try:
        model = services.read_run_config(cfg).model
    except services.StrataSetupError:
        return None
    return model if model != Path() else None


def server_verdict(model: Path | None, strata: config.StrataConfig) -> Verdict:
    """Return whether `model` fits beside the memory the model server already holds.

    What the server holds (a loaded model, or one still loading) is no longer in `MemAvailable`,
    so it counts as free: the same figure whether the model is out of memory, going in or in.
    """
    return check(model, strata, credit=held_by_unit(services.STRATA_UNIT))


def reload_reason(cfg: config.Config, where: str) -> str:
    """Return why the model server must not be restarted on `cfg`'s model now, or "".

    The memory the running server holds is counted as free: the restart gives it back first. A
    refusal is audited as coming from `where`.
    """
    return refusal(where, server_verdict(served_model(cfg), cfg.strata))


def switch_reason(cfg: config.Config) -> str:
    """Return why a switch of modes must not restart the server on `cfg`'s model, or ""."""
    return reload_reason(cfg, "mode switch")


def _profile_model(path: Path) -> Path | None:
    """Return the model the written profile serves, or None when there is none to read."""
    try:
        data = obj(json.loads(path.read_text(encoding="utf-8")))
    except OSError, ValueError:
        return None
    run = services.RunConfig(path, data, tuple(str(a) for a in data.get("args", ())))
    return run.model if run.model != Path() else None


def main(argv: Sequence[str] | None = None) -> int:
    """Run the unit's check: 0 to start the server, `EXIT_SHORT` (with the reason) not to."""
    parser = argparse.ArgumentParser(prog="python -m jarvis.ramguard")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        cfg = config.load(args.config)
    except config.ConfigError:
        return 0  # the server's other checks and `jarvis doctor` say what is wrong with it
    why = refusal("unit", check(_profile_model(args.profile), cfg.strata))
    if why:
        print(f"jarvis: {why}", file=sys.stderr)
        return EXIT_SHORT
    return 0


if __name__ == "__main__":
    sys.exit(main())
