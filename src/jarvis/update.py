"""Update the installed checkout and its locked environment as one service operation."""

import contextlib
import fcntl
import os
import re
import shutil
import sys
import time
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from jarvis import config, modes, paths, services
from jarvis.audit import AuditLog, last_event
from jarvis.files import write_atomic, write_atomic_bytes

if TYPE_CHECKING:
    from argparse import Namespace
    from collections.abc import Callable, Generator, Mapping

    type BeforeApply = Callable[[config.Config, AuditLog], None]

TIMER = "jarvis-update.timer"
JOB = "jarvis-update.service"
TIMER_ENV = "JARVIS_UPDATE_TIMER"
UV_ENV = "JARVIS_UPDATE_UV"
BUSY = 75
GIT_S = 120
SYNC_S = 900
SYSTEMCTL_S = 120
HEALTH_S = 60
WAIT_S = 30

_TIMER_TEXT = """\
[Unit]
Description=Check for and apply Jarvis updates daily
[Timer]
OnCalendar=daily
RandomizedDelaySec=2h
Persistent=true
[Install]
WantedBy=timers.target
"""
_WORKER = "from jarvis.cli import finish_update; raise SystemExit(finish_update())"


class UpdateError(Exception):
    """An update step that could not complete."""


class BusyError(UpdateError):
    """An update or a request is already in progress."""


# These variables can redirect Git to another checkout or rewrite its view of ancestry.
_GIT_CONTEXT = {
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_COMMON_DIR",
    "GIT_INDEX_FILE",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_NAMESPACE",
    "GIT_GRAFT_FILE",
    "GIT_SHALLOW_FILE",
    "GIT_REPLACE_REF_BASE",
    "GIT_CONFIG",
    "GIT_CONFIG_PARAMETERS",
    "GIT_CONFIG_COUNT",
}


class Finished(Protocol):
    """What a program the update started left behind."""

    @property
    def returncode(self) -> int:
        """Its exit status."""
        ...

    @property
    def stdout(self) -> str:
        """What it wrote to its output, unless it was shown."""
        ...

    @property
    def stderr(self) -> str:
        """What it wrote to its errors, unless they were shown."""
        ...


class Launch(Protocol):
    """Starts one program and waits for it (`jarvis.cli` gives the one used).

    It raises `UpdateError` when the program runs past `timeout`.
    """

    def __call__(
        self,
        args: tuple[str, ...],
        *,
        cwd: Path,
        timeout: float,
        env: Mapping[str, str],
        show: bool,
    ) -> Finished:
        """Run `args` in `cwd` with exactly `env`; with `show`, its output goes to the terminal."""
        ...


# Programs are started from the CLI, the launcher for its own commands (AGENTS.md, Boundaries).
_LAUNCH: ContextVar[Launch] = ContextVar("jarvis_update_launch")


def _run(args: tuple[str, ...], *, cwd: Path, timeout: float, show: bool = False) -> Finished:
    env = {key: value for key, value in os.environ.items() if key not in _GIT_CONTEXT}
    env.update(
        GIT_TERMINAL_PROMPT="0",
        GIT_NO_REPLACE_OBJECTS="1",
        UV_PROJECT_ENVIRONMENT=str(cwd / ".venv"),
    )
    try:
        launch = _LAUNCH.get()
    except LookupError:
        msg = "updates run only through the jarvis command"
        raise UpdateError(msg) from None
    return launch(args, cwd=cwd, timeout=timeout, env=env, show=show)


def _git(root: Path, *args: str) -> Finished:
    return _run(("git", *args), cwd=root, timeout=GIT_S)


def _read_git(root: Path, *args: str) -> str:
    done = _git(root, *args)
    if done.returncode:
        msg = f"Git could not read the checkout ({args[0]})"
        raise UpdateError(msg)
    return done.stdout.strip()


def _root() -> Path:
    venv = Path(sys.prefix)
    root = venv.parent
    if venv.name != ".venv" or not (root / ".git").exists():
        msg = "run update from the Jarvis checkout's .venv/bin/jarvis"
        raise UpdateError(msg)
    return root.resolve()


@dataclass(frozen=True, slots=True)
class Checkout:
    """The checked-out release branch and its configured upstream."""

    root: Path
    branch: str
    remote: str

    @classmethod
    def installed(cls, branch: str) -> Checkout:
        """Locate the installed checkout and require the requested branch to track itself.

        Raises:
            UpdateError: The checkout does not track the configured upstream.
        """
        root = _root()
        ref = f"refs/heads/{branch}"
        if _git(root, "check-ref-format", ref).returncode:
            msg = "set update.branch to a valid Git branch name"
            raise UpdateError(msg)
        current = _git(root, "symbolic-ref", "--quiet", "HEAD")
        if current.returncode or current.stdout.strip() != ref:
            msg = f"checkout is not on the configured branch {branch}; switch it yourself"
            raise UpdateError(msg)
        remote = _git(root, "config", "--get", f"branch.{branch}.remote").stdout.strip()
        upstream = _git(root, "config", "--get", f"branch.{branch}.merge").stdout.strip()
        if (
            upstream != ref
            or not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_./-]*", remote)
            or _git(root, "remote", "get-url", remote).returncode
        ):
            msg = f"set {branch} to track the same branch on a named Git remote"
            raise UpdateError(msg)
        return cls(root, branch, remote)

    @property
    def remote_ref(self) -> str:
        """Where Git records the tip of the release branch."""
        return f"refs/remotes/{self.remote}/{self.branch}"

    def clean_head(self) -> str:
        """Return HEAD after checking there are no tracked edits or untracked files.

        Raises:
            UpdateError: The checkout has local edits.
        """
        if _read_git(self.root, "status", "--porcelain", "--untracked-files=all"):
            msg = "checkout is dirty; commit or move your changes before updating"
            raise UpdateError(msg)
        return _read_git(self.root, "rev-parse", "--verify", "HEAD")

    def fetch(self) -> str:
        """Fetch exactly the configured upstream branch, without tags or submodules.

        Raises:
            UpdateError: The remote branch could not be fetched.
        """
        fetched = _git(
            self.root,
            "fetch",
            "--no-tags",
            "--no-recurse-submodules",
            "--no-auto-maintenance",
            "--no-write-fetch-head",
            "--refmap=",
            self.remote,
            f"+refs/heads/{self.branch}:{self.remote_ref}",
        )
        if fetched.returncode:
            msg = f"could not fetch {self.remote}/{self.branch}; check its URL and connection"
            raise UpdateError(msg)
        return _read_git(self.root, "rev-parse", "--verify", f"{self.remote_ref}^{{commit}}")

    def fast_forward(self, before: str, target: str) -> None:
        """Require a forward-only update path.

        Raises:
            UpdateError: The checkout has commits absent from the remote branch.
        """
        if _git(self.root, "merge-base", "--is-ancestor", before, target).returncode:
            msg = "the checkout is ahead of or diverged from the remote; fast-forwards only"
            raise UpdateError(msg)


@contextlib.contextmanager
def _locked() -> Generator[None]:
    path = paths.state_dir() / "update.lock"
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            msg = "another update is in progress"
            raise BusyError(msg) from exc
        yield
    finally:
        os.close(fd)


def _checked(root: Path, argv: tuple[str, ...], phase: str, timeout: float) -> None:
    result = _run(argv, cwd=root, timeout=timeout)
    if result.returncode:
        for output in (result.stdout, result.stderr):
            if output:
                print(output.rstrip(), file=sys.stderr)
        raise UpdateError(phase)


def _systemctl(root: Path, *args: str) -> None:
    _checked(root, ("systemctl", "--user", *args), f"systemd could not {args[0]}", SYSTEMCTL_S)


def _unit_state(root: Path, unit: str) -> str:
    result = _run(
        ("systemctl", "--user", "show", "--property=ActiveState", "--value", unit),
        cwd=root,
        timeout=SYSTEMCTL_S,
    )
    state = result.stdout.strip()
    if result.returncode or state not in {"active", "inactive", "failed"}:
        msg = f"could not establish whether {unit} is running (try again once services settle)"
        raise UpdateError(msg)
    return state


def _uv() -> str:
    found = os.environ.get(UV_ENV) if os.environ.get(TIMER_ENV) == "1" else shutil.which("uv")
    if found is None:
        candidate = Path.home() / ".local" / "bin" / "uv"
        if os.access(candidate, os.X_OK):
            return str(candidate)
        msg = "uv is missing; install it with sudo pacman -S uv"
        raise UpdateError(msg)
    return str(Path(found).absolute())


def _sync(root: Path, uv: str) -> None:
    _checked(root, (uv, "sync", "--locked"), "uv sync --locked failed", SYNC_S)


def _worker(root: Path, phase: str, *options: str) -> None:
    python = root / ".venv" / "bin" / "python"
    _checked(
        root,
        (str(python), "-c", _WORKER, phase, *options),
        f"updated Jarvis failed its {phase} check",
        HEALTH_S,
    )


def wait_for[T](check: Callable[[], T | None]) -> T | None:
    """Poll a read-only service check until it returns a value or startup times out."""
    until = time.monotonic() + WAIT_S
    while (found := check()) is None and time.monotonic() < until:
        time.sleep(0.25)
    return found


@dataclass(frozen=True, slots=True)
class SavedFile:
    """Contents of one generated file at the start of the update."""

    path: Path
    contents: bytes | None

    @classmethod
    def read(cls, path: Path) -> SavedFile:
        """Read the old file, or note that it was absent."""
        try:
            return cls(path, path.read_bytes())
        except FileNotFoundError:
            return cls(path, None)

    def changed(self) -> bool:
        """Whether another operation has written different contents."""
        return self.contents != self.read(self.path).contents

    def restore(self) -> None:
        """Put the original contents back if the file changed."""
        if self.changed():
            if self.contents is None:
                self.path.unlink(missing_ok=True)
            else:
                write_atomic_bytes(self.path, self.contents)


_ERROR = (UpdateError, config.ConfigError, OSError)


class Apply:
    """Keep enough state to restore the prior checkout, files and services on failure."""

    def __init__(
        self, checkout: Checkout, before: str, target: str, *, remote: bool = False
    ) -> None:
        """Capture the current revision, generated files and running services.

        `remote` is a client of a model server on another machine: it waits for none here.
        """
        super().__init__()
        self.remote = remote
        self.checkout = checkout
        self.before = before
        self.target = target
        self.root = checkout.root
        self.uv = _uv()
        self.voice = _unit_state(self.root, services.JARVIS_UNIT) == "active"
        self.strata = _unit_state(self.root, services.STRATA_UNIT) == "active"
        unit_dir = services.unit_dir()
        self.profile = SavedFile.read(services.strata_profile_file())
        self.strata_unit = SavedFile.read(unit_dir / services.STRATA_UNIT)
        self.voice_unit = SavedFile.read(unit_dir / services.JARVIS_UNIT)
        self.started_voice = False

    def apply(self) -> None:
        """Install and validate the forward update, preserving which services were running."""
        if self.voice:
            _systemctl(self.root, "stop", services.JARVIS_UNIT)
        _checked(
            self.root,
            ("git", "merge", "--ff-only", "--no-overwrite-ignore", self.target),
            "Git could not fast-forward the checkout",
            GIT_S,
        )
        _sync(self.root, self.uv)
        _worker(self.root, "health")
        reload_strata = self.strata and self._strata_stale()
        if self.strata_unit.changed() or self.voice_unit.changed() or reload_strata:
            _systemctl(self.root, "daemon-reload")
        if reload_strata:
            _systemctl(self.root, "restart", services.STRATA_UNIT)
        self._start_voice()
        self._ensure_ready()

    def _strata_stale(self) -> bool:
        if self.profile.changed() or self.strata_unit.changed():
            return True
        stamp = _run(
            (
                "systemctl",
                "--user",
                "show",
                services.STRATA_UNIT,
                "-p",
                "ActiveEnterTimestamp",
                "--timestamp=us+utc",
                "--value",
            ),
            cwd=self.root,
            timeout=SYSTEMCTL_S,
        )
        started = services.parse_systemd_time(stamp.stdout) if not stamp.returncode else None
        return started is not None and services.strata_changed_since(started)

    def _ensure_ready(self) -> None:
        ready = [
            name
            for name, on in (
                ("strata", not self.remote and (self.strata or self.voice)),
                ("voice", self.voice),
            )
            if on
        ]
        if ready:
            _worker(self.root, "ready", *ready)

    def _start_voice(self) -> None:
        if self.voice:
            if _unit_state(self.root, services.JARVIS_UNIT) == "failed":
                _systemctl(self.root, "reset-failed", services.JARVIS_UNIT)
            self.started_voice = True
            _systemctl(self.root, "start", services.JARVIS_UNIT)

    def restore(self) -> None:
        """Restore the old checkout and environment without discarding intervening edits.

        Raises:
            UpdateError: The checkout changed or the update could not be undone.
        """
        if Checkout.installed(self.checkout.branch) != self.checkout:
            msg = "the configured remote changed during the update"
            raise UpdateError(msg)
        head = self.checkout.clean_head()
        if head not in {self.before, self.target}:
            msg = "HEAD changed during the update; restore the previous commit manually"
            raise UpdateError(msg)
        if self.started_voice:
            _systemctl(self.root, "stop", services.JARVIS_UNIT)
        if head == self.target:
            _checked(
                self.root,
                ("git", "reset", "--keep", self.before),
                "Git could not restore the previous commit",
                GIT_S,
            )
            _sync(self.root, self.uv)
        self._restore_services()

    def _restore_services(self) -> None:
        """Put the generated files back and the services as they were before the update."""
        units_changed = self.strata_unit.changed() or self.voice_unit.changed()
        strata_changed = self.profile.changed() or self.strata_unit.changed()
        for saved in (self.profile, self.strata_unit, self.voice_unit):
            saved.restore()
        if units_changed:
            _systemctl(self.root, "daemon-reload")
        if self.strata:
            state = _unit_state(self.root, services.STRATA_UNIT)
            if state == "failed":
                _systemctl(self.root, "reset-failed", services.STRATA_UNIT)
            if strata_changed or state != "active":
                _systemctl(self.root, "restart", services.STRATA_UNIT)
        elif self.started_voice and _unit_state(self.root, services.STRATA_UNIT) == "active":
            _systemctl(self.root, "stop", services.STRATA_UNIT)
        self._start_voice()
        self._ensure_ready()


def _apply(
    checkout: Checkout, before: str, target: str, audit: AuditLog, *, remote: bool = False
) -> int:
    with modes.locked(), modes.nobody_answering() as free, modes.voice_idle() as idle:
        if not free or not idle:
            msg = "Jarvis is handling a request; try the update again when it is idle"
            raise BusyError(msg)
        if Checkout.installed(checkout.branch) != checkout or checkout.clean_head() != before:
            msg = "the checkout changed while fetching; run the update check again"
            raise UpdateError(msg)
        action = Apply(checkout, before, target, remote=remote)
        try:
            action.apply()
        except (*_ERROR, KeyboardInterrupt) as exc:
            print(f"jarvis: update failed ({exc or 'interrupted'})", file=sys.stderr)
            try:
                action.restore()
            except _ERROR as restore:
                print(
                    f"jarvis: rollback failed ({restore}); previous commit was {before}",
                    file=sys.stderr,
                )
                audit.record("update", result="rollback failed", before=before, target=target)
            else:
                print(f"jarvis: restored {before[:12]}", file=sys.stderr)
                audit.record("update", result="rolled back", before=before, target=target)
            return 1
    print(f"jarvis: updated {before[:12]} to {target[:12]}")
    audit.record("update", result="applied", before=before, target=target)
    return 0


def _timer(cfg: config.Config, *, on: bool) -> None:
    directory = services.unit_dir()
    timer, job = directory / TIMER, directory / JOB
    if on:
        root = Checkout.installed(cfg.update.branch).root
        exe = services.exec_program(str(root / ".venv" / "bin" / "jarvis"))
        uv = services.environment_line(UV_ENV, _uv())
        job_text = f"""\
[Unit]
Description=Update Jarvis when idle
[Service]
Type=oneshot
Environment={TIMER_ENV}=1
{services.xdg_environment()}{uv}
ExecStart={exe} update --apply --quiet
SuccessExitStatus={BUSY}
"""
        write_atomic(job, job_text)
        write_atomic(timer, _TIMER_TEXT)
        _systemctl(root, "daemon-reload")
        _systemctl(root, "enable", "--now", TIMER)
    elif timer.exists() or job.exists() or cfg.update.auto:
        root = Path.home()
        if timer.exists():
            _systemctl(root, "disable", "--now", TIMER)
        timer.unlink(missing_ok=True)
        job.unlink(missing_ok=True)
        _systemctl(root, "daemon-reload")
    saved = config.load()
    config.save(saved.model_copy(update={"update": saved.update.model_copy(update={"auto": on})}))


def command(args: Namespace, launch: Launch, before_apply: BeforeApply | None = None) -> int:
    """Check for updates, apply one, or manage the automatic update timer.

    Every program it needs (git, uv, systemctl, the updated Jarvis) is started by `launch`.
    `before_apply` runs first when an update is applied (the plugins', which `jarvis.cli` gives).
    """
    token = _LAUNCH.set(launch)
    try:
        return _logged(args, before_apply)
    finally:
        _LAUNCH.reset(token)


def _logged(args: Namespace, before_apply: BeforeApply | None) -> int:
    audit = AuditLog(paths.audit_file())
    try:
        with _locked():
            return _command(args, audit, before_apply)
    except BusyError as exc:
        if not args.quiet:
            print(f"jarvis: {exc}", file=sys.stderr)
        audit.record("update", result="busy")
        return BUSY
    except _ERROR as exc:
        print(f"jarvis: {exc}", file=sys.stderr)
        audit.record("update", result="refused")
        return 1


def _command(args: Namespace, audit: AuditLog, before_apply: BeforeApply | None) -> int:
    cfg = config.load()
    if args.auto is not None:
        _timer(cfg, on=args.auto == "on")
        audit.record("update", result=f"auto {args.auto}")
        if not args.quiet:
            print(f"jarvis: automatic updates {args.auto}")
        return 0
    if os.environ.get(TIMER_ENV) == "1" and not cfg.update.auto:
        audit.record("update", result="auto disabled")
        return 0
    if args.apply and before_apply is not None:
        before_apply(cfg, audit)  # before Jarvis's own update, whose restart then reads its result
    checkout = Checkout.installed(cfg.update.branch)
    before = checkout.clean_head()
    target = checkout.fetch()
    checkout.fast_forward(before, target)
    if before == target:
        if not args.quiet:
            print(f"jarvis: up to date at {before[:12]}")
        audit.record("update", result="up to date", before=before)
        return 0
    if not args.quiet:
        print(f"jarvis: {checkout.remote}/{checkout.branch} {before[:12]} → {target[:12]}")
        print(_read_git(checkout.root, "log", "--oneline", f"{before}..{target}"))
        print(_read_git(checkout.root, "diff", "--stat", before, target))
    if args.apply:
        return _apply(checkout, before, target, audit, remote=cfg.strata.mode == "remote")
    audit.record("update", result="available", before=before, target=target)
    return 0


def _available(cfg: config.Config) -> str:
    try:
        checkout = Checkout.installed(cfg.update.branch)
        before = _read_git(checkout.root, "rev-parse", "HEAD")
        known = _read_git(checkout.root, "rev-parse", "--verify", checkout.remote_ref)
        checkout.fast_forward(before, known)
    except _ERROR as exc:
        return f"unknown ({exc})"
    return "up to date as of the last fetch" if before == known else f"yes ({known[:12]})"


def _last_result() -> str:
    row = last_event(paths.audit_file(), "update")
    return (
        f"{row.get('result', 'unknown')} ({row.get('ts', 'unknown time')})"
        if row
        else "none recorded"
    )


def doctor_notes(cfg: config.Config, launch: Launch) -> list[str]:
    """Report cached update availability, the opt-in flag and the most recent audit result."""
    token = _LAUNCH.set(launch)
    try:
        available = _available(cfg)
    finally:
        _LAUNCH.reset(token)
    state = "off"
    if cfg.update.auto:
        state = (
            "on"
            if (services.unit_dir() / TIMER).is_file()
            else "on (run jarvis update --auto on to install timer)"
        )
    return [
        f"note update available: {available}",
        f"note automatic updates: {state}",
        f"note last update result: {_last_result()}",
    ]
