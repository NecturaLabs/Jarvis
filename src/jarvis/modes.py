"""Coding mode: which mode Jarvis is in, what it changes, and the switch between the two.

In the normal mode Jarvis serves `strata.run_config` with the limits of `[agent]` and `[chat]`. In
coding mode it serves `strata.coding_run_config` (when one is set), with `[coding.chat]` laid over
`[chat]` and `[coding.voice]` over `[agent]`. The mode is kept in a small state file every Jarvis
process reads, so the voice service, a chat window and `jarvis model` agree on it.

One model server runs, with one profile, so one model is in memory at a time: a switch that
changes the profile restarts the server, which ends the old engine before the new one starts.
Every switch is audited, and one taking a restart tells the owner first.
"""

import contextlib
import fcntl
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, cast

from jarvis import paths, services
from jarvis.config import EFFORTS, CodingChatConfig, ConfigError
from jarvis.files import write_atomic

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from jarvis.config import AgentConfig, Config

type Mode = Literal["normal", "coding"]
MODES: tuple[Mode, ...] = ("normal", "coding")
type By = Literal["owner", "model", "idle"]

READY_S = 60.0  # how long a restarted model server may take to answer
READY_POLL_S = 0.5
BUSY = "Jarvis is answering a request right now; switch once it has finished."
_LIMITS = tuple(name for name in CodingChatConfig.model_fields if name != "auto")


def state_file() -> Path:
    """Return where the mode is kept."""
    return paths.state_dir() / "mode.json"


def _lock_file() -> Path:
    return paths.state_dir() / "mode.lock"


@dataclass(frozen=True, slots=True)
class State:
    """The mode, who set it, when coding mode was last used and how long each mode took to load.

    `used` is Unix time, so every process reads the same idle time. `load_s` holds the seconds the
    last switch into each mode waited for its model, for the line that tells the owner.
    """

    mode: Mode = "normal"
    by: By = "owner"
    used: float = 0.0
    load_s: dict[str, float] = field(default_factory=dict[str, float])


def read_state() -> State:
    """Return the saved state; a missing or unreadable file is the normal mode."""
    try:
        loaded: object = json.loads(state_file().read_text(encoding="utf-8"))
    except OSError, ValueError:
        return State()
    data = cast("dict[str, object]", loaded) if isinstance(loaded, dict) else {}
    used, loads = data.get("used"), data.get("load_s")
    said = data.get("by")
    by: By = "model" if said == "model" else "idle" if said == "idle" else "owner"
    return State(
        "coding" if data.get("mode") == "coding" else "normal",
        by,
        float(used) if isinstance(used, int | float) and not isinstance(used, bool) else 0.0,
        {
            str(name): float(seconds)
            for name, seconds in cast("dict[object, object]", loads).items()
            if isinstance(seconds, int | float) and not isinstance(seconds, bool)
        }
        if isinstance(loads, dict)
        else {},
    )


def _write_state(state: State) -> None:
    text = json.dumps(
        {"mode": state.mode, "by": state.by, "used": state.used, "load_s": state.load_s}
    )
    write_atomic(state_file(), text + "\n", 0o600)


def current() -> Mode:
    """Return the mode Jarvis is in."""
    return read_state().mode


def effective(config: Config, mode: Mode) -> Config:
    """Return the config as `mode` runs it: itself, or with `[coding]` laid over it.

    In coding mode `[coding.chat]` is laid over `[chat]` and `[coding.voice]` over `[agent]` (a
    limit unset there stays the channel's own), the model server takes its context and, where
    set, its run config, VRAM limits and memory, the model its temperature, and the files their
    extra write roots.
    """
    if mode != "coding":
        return config
    coding = config.coding
    strata: dict[str, object] = {
        "context": coding.context,
        "reasoning_budget_tokens": coding.reasoning_budget_tokens,
    }
    if config.strata.coding_run_config != Path():
        strata["run_config"] = config.strata.coding_run_config
    if coding.vram_reserve_mib:
        strata["vram_reserve_mib"] = coding.vram_reserve_mib
    if coding.expert_vram_gib is not None:
        strata["expert_vram_gib"] = coding.expert_vram_gib
    if coding.memory_max:
        strata["memory_max"] = coding.memory_max
    model = {} if coding.temperature is None else {"temperature": coding.temperature}
    roots = list(dict.fromkeys([*config.files.write_roots, *coding.write_roots]))
    return config.model_copy(
        update={
            "agent": config.agent.model_copy(update=_overlay(config.agent, coding.voice)),
            "chat": config.chat.model_copy(update=_overlay(config.chat, coding.chat)),
            "strata": config.strata.model_copy(update=strata),
            "model": config.model.model_copy(update=model),
            "files": config.files.model_copy(update={"write_roots": roots}),
        }
    )


def _overlay(section: AgentConfig, over: CodingChatConfig) -> dict[str, object]:
    """Return the limits `over` sets, for laying over `section`; the ceiling never ends below."""
    limits: dict[str, object] = {
        name: value for name in _LIMITS if (value := getattr(over, name)) is not None
    }
    start = over.thinking or section.thinking
    ceiling = over.thinking_max or section.thinking_max
    if EFFORTS.index(ceiling) < EFFORTS.index(start):
        limits["thinking_max"] = start
    return limits


def own_model(config: Config) -> bool:
    """Whether coding mode serves a model of its own (`strata.coding_run_config`)."""
    return config.strata.coding_run_config not in {Path(), config.strata.run_config}


def model_name(config: Config, mode: Mode) -> str:
    """Return the name of the model `mode` serves, from its run config."""
    return services.model_name(effective(config, mode))


def mark_used(now: float | None = None) -> None:
    """Note that coding mode was used now, so the idle time before going back starts again.

    Skipped while a switch holds the lock: the switch stamps the time itself.
    """
    with _locked(wait=False) as held:
        state = read_state()
        if held and state.mode == "coding":
            stamp = time.time() if now is None else now
            _write_state(State(state.mode, state.by, stamp, state.load_s))


def idle_too_long(config: Config, now: float | None = None) -> bool:
    """Whether coding mode has gone unused for `coding.back_after_s` (never when that is 0)."""
    state = read_state()
    after = config.coding.back_after_s
    now = time.time() if now is None else now
    return state.mode == "coding" and after > 0 and now - state.used > after


@contextlib.contextmanager
def _locked(*, wait: bool = True) -> Generator[bool]:
    """Hold the mode's lock: one process changes the mode or its state at a time.

    Without `wait`, a lock another process holds is not waited for: False is yielded instead.
    """
    path = _lock_file()
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB))
        except BlockingIOError:
            yield False
        else:
            yield True
    finally:
        os.close(fd)


class Units(Protocol):
    """The model server's systemd unit, as the switch needs it."""

    def active(self) -> bool:
        """Return whether the model server runs."""
        ...

    def restart(self) -> bool:
        """Restart the model server, after reloading its unit; return whether systemd took it."""
        ...


class Server(Protocol):
    """The model server's client, as the switch needs it (`StrataModel`)."""

    def status(self) -> str | None:
        """Return the model's state, or None while the server does not answer."""
        ...

    def load(self) -> bool:
        """Load the model and wait for it; return whether the server accepted."""
        ...


class Audit(Protocol):
    """Where a switch is recorded (`AuditLog`)."""

    def record(self, event: str, **fields: object) -> None:
        """Record one event."""
        ...


@dataclass(frozen=True, slots=True)
class Switched:
    """How a switch ended: the mode now, and `error` when it could not be made.

    `changed` is false when Jarvis already was in that mode; `restarted` says the model server
    was restarted, and `seconds` how long the switch took, the model's load included.
    """

    mode: Mode
    changed: bool = False
    restarted: bool = False
    seconds: float = 0.0
    error: str = ""


def _texts() -> tuple[str | None, str | None]:
    """Return the model server's profile and unit as they are on disk."""
    found: list[str | None] = []
    for path in (services.strata_profile_file(), services.unit_dir() / services.STRATA_UNIT):
        try:
            found.append(path.read_text(encoding="utf-8"))
        except OSError, ValueError:
            found.append(None)
    return found[0], found[1]


def announcement(config: Config, to: Mode, *, restart: bool) -> str:
    """Return the line that tells the owner about a switch before it is made.

    When the model server restarts it says how long the model took to load the last time, once
    that is known.
    """
    name = "coding" if to == "coding" else "usual"
    if not restart:
        return f"Switching to {'coding' if to == 'coding' else 'normal'} mode."
    waited = read_state().load_s.get(to)
    model = model_name(config, to)
    wait = (
        f"about {max(5, round(waited / 5) * 5):.0f} seconds"
        if waited
        else "this can take a minute or two"
    )
    return f"Switching to the {name} model, {model}: {wait}."


class Switcher:
    """Switches the mode for one Jarvis process: the files, the model server, the state, the audit.

    `units` is the model server's unit, `server` makes the client of a config's model server, and
    `busy` says whether another request is in flight (a switch would cut its model call).
    """

    def __init__(  # ruff: ignore[too-many-arguments] - collaborators, each a test seam
        self,
        config: Config,
        units: Units,
        server: Callable[[Config], Server],
        audit: Audit,
        *,
        busy: Callable[[], bool] = lambda: False,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], float] = time.time,
    ) -> None:
        """Switch `config`'s Jarvis; `clock` and `sleep` time the wait, `now` stamps the state."""
        super().__init__()
        self._config = config
        self._units = units
        self._server = server
        self._audit = audit
        self._busy = busy
        self._clock = clock
        self._sleep = sleep
        self._now = now

    def needs_restart(self, to: Mode) -> bool:
        """Return whether switching to `to` changes what the model server runs.

        A coding mode with the same run config, context and limits as the normal one changes only
        Jarvis's own settings. A config that cannot be served counts as a change (`switch` says
        what is wrong with it).
        """
        try:
            profile = services.strata_profile(effective(self._config, to))
        except ConfigError:
            return True
        return json.dumps(profile, indent=1) + "\n" != _texts()[0]

    def switch(self, to: Mode, by: By, say: Callable[[str], None] = lambda _line: None) -> Switched:
        """Put Jarvis in mode `to`; `say` gets the line for the owner before a restart.

        Nothing changes when Jarvis is in that mode already, when another request is in flight
        (`busy`), or when the mode's run config cannot be served. When the restarted server does
        not come back, the files and the server go back to the mode before.
        """
        if self._busy():
            return Switched(current(), error=BUSY)
        with _locked():
            state = read_state()
            if state.mode == to:
                return Switched(to)
            began = self._clock()
            before = _texts()
            try:
                restart = self._write(to, before)
            except (ConfigError, OSError) as exc:
                return Switched(state.mode, error=f"{to} mode cannot be served: {exc}")
            running = restart and self._units.active()
            if running:
                say(announcement(self._config, to, restart=True))
                if not self._restart(to, state):
                    self._write_back(state.mode)
                    return Switched(state.mode, restarted=True, error=self._failed(to, state))
            elif not restart:
                say(announcement(self._config, to, restart=False))
            seconds = round(self._clock() - began, 1)
            loads = {**state.load_s, to: seconds} if running else state.load_s
            _write_state(State(to, by, self._now(), loads))
            self._audit.record(
                "mode_switched",
                to=to,
                was=state.mode,
                by=by,
                model=model_name(self._config, to),
                restarted=running,
                seconds=seconds,
            )
            return Switched(to, changed=True, restarted=running, seconds=seconds)

    def back_if_idle(self) -> Switched | None:
        """Go back to the normal mode when coding mode went unused too long; else do nothing."""
        if not idle_too_long(self._config, self._now()):
            return None
        return self.switch("normal", "idle")

    def _write(self, to: Mode, before: tuple[str | None, str | None]) -> bool:
        """Write the model server's profile and unit for `to`; return whether either changed.

        Before `jarvis setup` has written the unit there is no model server to change.
        """
        config = effective(self._config, to)
        services.strata_profile(config)  # refuses a run config that cannot be served
        if before[1] is None:
            return False
        services.write_strata_profile(config)
        services.write_strata_unit(config, paths.strata_key_file())
        return _texts() != before

    def _write_back(self, mode: Mode) -> None:
        with contextlib.suppress(ConfigError, OSError):
            self._write(mode, _texts())
            self._units.restart()

    def _restart(self, to: Mode, state: State) -> bool:
        """Restart the model server on the new profile and load its model; return whether it did.

        The model is loaded again when it was in memory before, and always for coding mode, which
        is switched to for work that follows at once.
        """
        was_loaded = self._server(effective(self._config, state.mode)).status() == "loaded"
        if not self._units.restart():
            return False
        server = self._server(effective(self._config, to))
        deadline = self._clock() + READY_S
        while server.status() is None:
            if self._clock() > deadline:
                return False
            self._sleep(READY_POLL_S)
        return server.load() if was_loaded or to == "coding" else True

    def _failed(self, to: Mode, state: State) -> str:
        self._audit.record("mode_switch_failed", to=to, was=state.mode)
        return (
            f"The model server did not come back in {to} mode, so Jarvis is in {state.mode} "
            "mode still; `jarvis doctor` and `jarvis logs` say more."
        )
