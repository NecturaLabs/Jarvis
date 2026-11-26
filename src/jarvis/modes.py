"""Coding mode: which mode Jarvis is in, what it changes, and the switch between the two.

In the normal mode Jarvis serves `strata.run_config` with the limits of `[agent]` and `[chat]`. In
coding mode it serves `strata.coding_run_config` (when one is set), with `[coding.chat]` laid over
`[chat]` and `[coding.voice]` over `[agent]`. The mode is kept in a small state file every Jarvis
process reads, so the voice service, a chat window and `jarvis model` agree on it.

One model server runs, with one profile, so one model is in memory at a time: a switch that
changes the profile restarts the server, which ends the old engine before the new one starts.
Every switch is audited, one taking a restart tells the owner first, and none is made while any
Jarvis process answers a request (`Answering`).
"""

import contextlib
import dataclasses
import fcntl
import json
import math
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, Self, cast

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
BUSY = "Jarvis is answering a request. Try again when it has finished."
_LIMITS = tuple(name for name in CodingChatConfig.model_fields if name != "auto")
# `[strata]` keys coding mode may set for itself; unset (None) keeps `[strata]`'s.
_STRATA_KEYS = ("kv_resident", "keep_in_ram", "pictures", "effort_position")


def state_file() -> Path:
    """Return where the mode is kept."""
    return paths.state_dir() / "mode.json"


def _lock_file() -> Path:
    return paths.state_dir() / "mode.lock"


@dataclass(frozen=True, slots=True)
class State:
    """The mode, who set it, when coding mode was last used and how long each mode took to load.

    `used` is Unix time, so every process reads the same idle time. `load_s` holds the seconds the
    last switch into each mode waited for its model, for the line that tells the owner. `pending`
    is the mode a switch is being made to, noted before the model server's files are touched and
    cleared when the switch ends: one still there was left by a switch that was killed, so the
    files and the server may be either mode's.
    """

    mode: Mode = "normal"
    by: By = "owner"
    used: float = 0.0
    load_s: dict[str, float] = field(default_factory=dict[str, float])
    pending: Mode | None = None


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
    half = data.get("pending")
    pending: Mode | None = "coding" if half == "coding" else "normal" if half == "normal" else None
    return State(
        "coding" if data.get("mode") == "coding" else "normal",
        by,
        _finite(used) or 0.0,
        {
            str(name): seconds
            for name, value in cast("dict[object, object]", loads).items()
            if (seconds := _finite(value)) is not None
        }
        if isinstance(loads, dict)
        else {},
        pending,
    )


def _finite(value: object) -> float | None:
    """Return `value` when it is a finite number, else None (JSON also reads `Infinity`, `NaN`)."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value) if math.isfinite(value) else None


def _write_state(state: State) -> None:
    data: dict[str, object] = {
        "mode": state.mode,
        "by": state.by,
        "used": state.used,
        "load_s": state.load_s,
    }
    if state.pending is not None:
        data["pending"] = state.pending
    text = json.dumps(data)
    write_atomic(state_file(), text + "\n", 0o600)


def current() -> Mode:
    """Return the mode Jarvis is in."""
    return read_state().mode


def effective(config: Config, mode: Mode) -> Config:
    """Return the config as `mode` runs it: itself, or with `[coding]` laid over it.

    In coding mode `[coding.chat]` is laid over `[chat]` and `[coding.voice]` over `[agent]` (a
    limit unset there stays the channel's own), the model server takes its context and, where
    set, its run config, VRAM limits, memory, context cells in VRAM, swap, picture reader and
    effort position, the model its temperature, and the files their
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
    for name in _STRATA_KEYS:
        if (value := getattr(coding, name)) is not None:
            strata[name] = value
    model = {} if coding.temperature is None else {"temperature": coding.temperature}
    roots = list(dict.fromkeys([*config.files.write_roots, *coding.write_roots]))
    return config.model_copy(
        update={
            "agent": config.agent.model_copy(update=_overlay(config.agent, coding.voice)),
            "chat": config.chat.model_copy(update=_overlay(config.chat, coding.chat)),
            "strata": config.strata.model_copy(update=strata)
            if config.strata.mode == "local"
            else config.strata,
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
    strata = config.strata
    return (
        strata.mode == "local"
        and strata.coding_run_config != Path()
        and os.path.realpath(strata.coding_run_config) != os.path.realpath(strata.run_config)
    )


def model_name(config: Config, mode: Mode) -> str:
    """Return the name of the model `mode` serves, from its run config."""
    return services.model_name(effective(config, mode))


def mark_used(now: float | None = None) -> None:
    """Note that coding mode was used now, so the idle time before going back starts again.

    It waits for a switch that holds the lock, so the time is never dropped: a return to the
    normal mode for idleness looks at it under that lock.
    """
    with locked():
        state = read_state()
        if state.mode == "coding":
            stamp = time.time() if now is None else now
            _write_state(dataclasses.replace(state, used=stamp))


def idle_too_long(config: Config, now: float | None = None) -> bool:
    """Whether coding mode has gone unused for `coding.back_after_s` (never when that is 0)."""
    state = read_state()
    after = config.coding.back_after_s
    now = time.time() if now is None else now
    return state.mode == "coding" and after > 0 and now - state.used > after


@contextlib.contextmanager
def locked() -> Generator[None]:
    """Hold the mode's lock while the model server's files are written or it is restarted.

    `jarvis setup` and `jarvis restart` hold it, so neither writes the files in the middle of a
    switch, and no switch begins in the middle of theirs.
    """
    with _locked():
        yield


@contextlib.contextmanager
def nobody_answering() -> Generator[bool]:
    """Keep requests from beginning while held; yield False when one is in flight already.

    For a restart of the model server outside a switch (`jarvis restart`), which would cut a
    request's model call like a switch does.
    """
    with _nobody_answering() as free:
        yield free


@contextlib.contextmanager
def _locked(*, wait: bool = True) -> Generator[bool]:
    """Hold the mode's lock: one process changes the mode or its state at a time.

    Without `wait`, a lock another process holds is not waited for: False is yielded instead.
    """
    fd = _open_lock(_lock_file())
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB))
        except BlockingIOError:
            yield False
        else:
            yield True
    finally:
        os.close(fd)


def _answering_file() -> Path:
    return paths.state_dir() / "mode-answering.lock"


def _open_lock(path: Path) -> int:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    return os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)


class Answering:
    """One process's mark that it is answering a request, for as long as it is held.

    A switch restarts the model server, which would cut the model call of a request in flight in
    any Jarvis process: the voice service, a chat window, `jarvis ask`. Each holds this mark (a
    shared lock on one small file) through its request, and a switch is made only while nobody
    holds one. A request that begins during a switch waits here until the model server is back.
    """

    def __init__(self) -> None:
        """Hold nothing yet."""
        super().__init__()
        self._fd: int | None = None
        self._depth = 0
        self._own = threading.Lock()  # a summary or a prompt read ahead may run beside a request

    def __enter__(self) -> Self:
        """Hold the mark; a request inside a request keeps the one already held."""
        with self._own:
            if self._depth == 0:
                fd = _open_lock(_answering_file())
                try:
                    fcntl.flock(fd, fcntl.LOCK_SH)
                except BaseException:
                    os.close(fd)
                    raise
                self._fd = fd
            self._depth += 1
        return self

    def __exit__(self, *_exc: object) -> None:
        """Let the mark go once the outermost request ends."""
        with self._own:
            self._depth -= 1
            if self._depth == 0 and self._fd is not None:
                os.close(self._fd)
                self._fd = None

    def release(self) -> bool:
        """Let the mark go for a switch this request makes itself; return whether it was let go.

        `hold` takes it again. It stays held, and the switch is refused as busy, while another
        call of this process holds it too (a summary, the prompt read ahead): the switch would
        restart the model server under that call.
        """
        with self._own:
            if self._fd is None or self._depth != 1:
                return False
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            return True

    def hold(self) -> None:
        """Hold the mark again after `release` (nothing to do when no request is being answered).

        A switch calls it (`Switcher.switch`'s `rehold`) while it still keeps other switches out,
        so none is made between this request's own switch and the rest of the request.
        """
        with self._own:
            if self._fd is not None:
                fcntl.flock(self._fd, fcntl.LOCK_SH)


@contextlib.contextmanager
def _nobody_answering() -> Generator[bool]:
    with _try_lock(_answering_file(), fcntl.LOCK_EX) as free:
        yield free


@contextlib.contextmanager
def voice_request() -> Generator[bool]:
    """Mark the voice request from capture through playback, unless maintenance is underway."""
    with _try_lock(paths.state_dir() / "voice-request.lock", fcntl.LOCK_SH) as admitted:
        yield admitted


@contextlib.contextmanager
def voice_idle() -> Generator[bool]:
    """Keep voice requests out while applying an update; yield False if one was admitted."""
    with _try_lock(paths.state_dir() / "voice-request.lock", fcntl.LOCK_EX) as idle:
        yield idle


@contextlib.contextmanager
def _try_lock(path: Path, mode: int) -> Generator[bool]:
    fd = _open_lock(path)
    try:
        try:
            fcntl.flock(fd, mode | fcntl.LOCK_NB)
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
    if config.strata.mode == "remote":
        return f"Switching to {to} mode on this client. The remote model stays the same."
    name = "coding" if to == "coding" else "usual"
    if not restart:
        return f"Switching to {'coding' if to == 'coding' else 'normal'} mode."
    waited = read_state().load_s.get(to)
    model = model_name(config, to)
    wait = (
        f"It takes about {max(5, round(waited / 5) * 5):.0f} seconds"
        if waited
        else "It can take a minute or two"
    )
    return f"Switching to the {name} model, {model}. {wait}."


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
        Jarvis's own settings. Only the profile is compared: a mode that changes only the unit
        (`keep_in_ram`, `memory_max`) still restarts the server when `switch` writes it. A config
        that cannot be served counts as a change (`switch` says what is wrong with it).
        """
        if self._config.strata.mode != "local":
            return False
        try:
            profile = services.strata_profile(effective(self._config, to))
        except ConfigError:
            return True
        return json.dumps(profile, indent=1) + "\n" != _texts()[0]

    def switch(
        self,
        to: Mode,
        by: By,
        say: Callable[[str], None] = lambda _line: None,
        *,
        rehold: Callable[[], None] = lambda: None,
        only_if: Callable[[], bool] = lambda: True,
    ) -> Switched:
        """Put Jarvis in mode `to`; `say` gets the line for the owner before a restart.

        `rehold` is called once requests may begin again and before another switch can: a request
        that makes the switch itself takes its mark back there (`Answering.hold`), so it goes on
        in the mode it switched to. `only_if` is asked with every lock held and no request in
        flight; when it says no, nothing changes.

        Nothing changes when Jarvis is in that mode already (and the model server's files are that
        mode's), when a request is in flight in any
        Jarvis process (`busy`, `Answering`), or when the mode's run config cannot be served. When
        the restarted server does not come back, the files and the server go back to the mode
        before.
        """
        if self._config.strata.mode == "backend":
            return Switched(
                read_state().mode, error="Change the served model in the backend setup."
            )
        with _locked():
            try:
                state = read_state()
                if state.mode == to and not self._out_of_step(state):
                    return Switched(to)
                # Requests are held off first, then asked about: none begins between the two.
                with _nobody_answering() as free:
                    if not free or self._busy():
                        return Switched(state.mode, error=BUSY)
                    if not only_if():
                        return Switched(read_state().mode)
                    return self._switch(to, by, say, state)
            finally:
                rehold()

    def _out_of_step(self, state: State) -> bool:
        """Whether the model server may not be running the mode in force.

        A switch that was killed leaves its `pending` note, or files that are not the mode's;
        asking for the mode again then puts the files and the server right.
        """
        return self._config.strata.mode == "local" and (
            state.pending is not None
            or (_texts()[0] is not None and self.needs_restart(state.mode))
        )

    def _switch(self, to: Mode, by: By, say: Callable[[str], None], state: State) -> Switched:
        """Make the switch from `state` to `to`, with the lock held and no request in flight."""
        began = self._clock()
        if self._config.strata.mode == "remote":
            say(announcement(self._config, to, restart=False))
            return self._remember(to, by, state, began, restarted=False)
        before = _texts()
        try:
            services.strata_profile(effective(self._config, to))  # refused before anything changes
            # Noted before the files change: a switch killed from here on is seen and made whole.
            _write_state(dataclasses.replace(state, pending=to))
            # After a killed switch the server may run either mode's files: it is restarted.
            restart = self._write(to, before) or (
                state.pending is not None and before[1] is not None
            )
        except (ConfigError, OSError) as exc:
            # The server was not restarted: with its files back it runs what it ran before.
            if self._write_back(state.mode, restart=False):
                self._settle(state)  # a note an earlier killed switch left stays
            return Switched(state.mode, error=f"{to} mode cannot be served: {exc}")
        running = restart and self._units.active()
        if running:
            say(announcement(self._config, to, restart=True))
            if not self._restart(to, state):
                if self._write_back(state.mode, restart=True):
                    self._settle(dataclasses.replace(state, pending=None))
                return Switched(state.mode, restarted=True, error=self._failed(to, state))
        elif not restart:
            say(announcement(self._config, to, restart=False))
        return self._remember(to, by, state, began, restarted=running)

    def _remember(
        self, to: Mode, by: By, state: State, began: float, *, restarted: bool
    ) -> Switched:
        seconds = round(self._clock() - began, 1)
        loads = {**state.load_s, to: seconds} if restarted else state.load_s
        _write_state(State(to, by, self._now(), loads))
        self._audit.record(
            "mode_switched",
            to=to,
            was=state.mode,
            by=by,
            model=model_name(self._config, to),
            restarted=restarted,
            seconds=seconds,
        )
        return Switched(to, changed=True, restarted=restarted, seconds=seconds)

    def idle(self) -> bool:
        """Whether coding mode went unused too long, so `back_if_idle` would leave it."""
        return idle_too_long(self._config, self._now())

    def back_if_idle(
        self,
        say: Callable[[str], None] = lambda _line: None,
        *,
        rehold: Callable[[], None] = lambda: None,
    ) -> Switched | None:
        """Go back to the normal mode when coding mode went unused too long; else do nothing.

        `say` gets the line for the owner and `rehold` is called as in `switch`.
        """
        if not self.idle():
            return None
        # Asked again with the locks held: a request that ended meanwhile stamped its use.
        return self.switch(
            "normal",
            "idle",
            say,
            rehold=rehold,
            only_if=self.idle,
        )

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

    def _write_back(self, mode: Mode, *, restart: bool) -> bool:
        """Put the files of `mode` back after a failed switch; return whether they are back.

        With `restart` the server is restarted on them too, and it counts.
        """
        try:
            self._write(mode, _texts())
        except ConfigError, OSError:
            return False
        return self._units.restart() if restart else True

    @staticmethod
    def _settle(state: State) -> None:
        """Note how a failed switch left things, once files and server are the old mode's again.

        Its note stays when they could not be put right: asking for a mode then restarts.
        """
        with contextlib.suppress(OSError):
            _write_state(state)

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
            f"The model server did not come back in {to} mode, so Jarvis stays in {state.mode} "
            "mode. See jarvis doctor and jarvis logs."
        )
