"""Download and update plugin marketplaces with Git, and `jarvis plugins` (docs/adr/0027).

Jarvis follows the latest of each repository, never a pinned version. A marketplace (and a plugin
that lives in a repository of its own) is cloned once, shallow, and later fast-forwarded to the
default branch's newest commit. Git never runs anything from the repository: hooks are off, only
https is allowed, links are not written (so none can point outside the folder), submodules are
not fetched, and Git reads none of the owner's own configuration. Jarvis starts Git only through
the launcher the CLI gives it (`cli.launch_for_update`), as `jarvis update` does.

Each change is audited with the commit before and after.
"""

import contextlib
import fcntl
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from jarvis import config, paths, plugins
from jarvis.audit import AuditLog, last_event
from jarvis.update import UpdateError

if TYPE_CHECKING:
    from argparse import Namespace
    from collections.abc import Callable, Generator, Iterable

    from jarvis.update import Launch

CLONE_S = 180
GIT_S = 60
FETCH_DEPTH = 100  # enough history to see that a new commit builds on the one held
MAX_CLONE_BYTES = 50 * 2**20  # a repository bigger than this (`.git` included) is not used
_OTHERS_WRITE = 0o022
# Everything Git is told on every call: no hooks, https only, no links written, no submodules.
SAFE = (
    "-c",
    "core.hooksPath=/dev/null",
    "-c",
    "protocol.allow=never",
    "-c",
    "protocol.https.allow=always",
    "-c",
    "core.symlinks=false",
    "-c",
    "core.fsmonitor=false",
    "-c",
    "submodule.recurse=false",
)
# What Git may inherit: where it is, its home, the network settings. Nothing else of the
# owner's environment (their Git configuration, credentials and `GIT_*` settings stay out).
INHERITED = (
    "PATH",
    "HOME",
    "TMPDIR",
    "LANG",
    "HTTPS_PROXY",
    "https_proxy",
    "HTTP_PROXY",
    "http_proxy",
    "ALL_PROXY",
    "all_proxy",
    "NO_PROXY",
    "no_proxy",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
)
_NOT_FOUND = "git is not installed (sudo pacman -S git)"
_REF = re.compile(rf"{plugins.NAME.pattern}@{plugins.NAME.pattern}")


class PluginError(Exception):
    """A plugin step that could not complete."""


@dataclass(frozen=True, slots=True)
class Change:
    """What one download or update did to one repository."""

    name: str
    action: Literal["downloaded", "updated", "current", "failed"]
    before: str = ""
    after: str = ""
    error: str = ""

    def line(self) -> str:
        """Return the line `jarvis plugins update` prints for it."""
        match self.action:
            case "downloaded":
                return f"{self.name}: downloaded {self.after[:12]}"
            case "updated":
                return f"{self.name}: {self.before[:12]} → {self.after[:12]}"
            case "current":
                return f"{self.name}: up to date at {self.after[:12]}"
            case "failed":
                return f"{self.name}: failed ({self.error})"


def _env() -> dict[str, str]:
    env = {name: os.environ[name] for name in INHERITED if name in os.environ}
    env.update(
        GIT_TERMINAL_PROMPT="0",
        GIT_CONFIG_GLOBAL="/dev/null",
        GIT_CONFIG_NOSYSTEM="1",
        GIT_ALLOW_PROTOCOL="https",
        GIT_NO_REPLACE_OBJECTS="1",
        LC_ALL="C",
    )
    return env


class Git:
    """Runs Git with the safe settings through the launcher the CLI gave."""

    def __init__(self, launch: Launch) -> None:
        """Run every command through `launch`."""
        super().__init__()
        self._launch = launch

    def run(self, cwd: Path, *args: str, timeout: float = GIT_S) -> tuple[int, str]:
        """Return Git's exit status and its output (errors included).

        Raises:
            PluginError: Git is missing or ran past `timeout`.
        """
        try:
            done = self._launch(
                ("git", *SAFE, *args), cwd=cwd, timeout=timeout, env=_env(), show=False
            )
        except FileNotFoundError as error:
            raise PluginError(_NOT_FOUND) from error
        except (OSError, UpdateError) as error:
            raise PluginError(plugins.tidy(str(error), 200)) from error
        return done.returncode, (done.stdout or "").strip() or (done.stderr or "").strip()

    def read(self, cwd: Path, *args: str, timeout: float = GIT_S) -> str:
        """Return what Git printed.

        Raises:
            PluginError: Git failed.
        """
        code, out = self.run(cwd, *args, timeout=timeout)
        if code:
            raise PluginError(f"git {args[0]} failed: {plugins.tidy(out, 200)}")
        return out

    def head(self, folder: Path) -> tuple[str, str]:
        """Return the commit a downloaded repository is at, and its date."""
        commit, _, date = self.read(folder, "log", "-1", "--format=%H %cI").partition(" ")
        return commit, date


def _tree_bytes(folder: Path) -> int:
    """Return the bytes of every file under `folder`, `.git` included; no link is followed."""
    total = 0
    for here, _dirs, names in os.walk(folder):
        for name in names:
            with contextlib.suppress(OSError):
                total += (Path(here) / name).lstat().st_size
    return total


def _only_yours(folder: Path) -> None:
    """Take group and other write permission off `folder` and all that is in it.

    Jarvis reads only what nobody else can write (`plugins.read_file`), whatever the umask was.
    """
    for here, dirs, names in os.walk(folder):
        for name in (*dirs, *names):
            path = Path(here) / name
            if not path.is_symlink():
                path.chmod(path.stat().st_mode & ~_OTHERS_WRITE)
    folder.chmod(folder.stat().st_mode & ~_OTHERS_WRITE)


def _partial(destination: Path) -> Path:
    return destination.parent / f".{destination.name}.partial"


def _fill(
    git: Git, url: str, partial: Path, check: Callable[[Path], None] | None, depth: int
) -> str:
    """Clone `url` into `partial`, judge it and return its commit.

    Raises:
        PluginError: Git failed, or the copy is too big or did not pass `check`.
    """
    git.read(
        partial.parent,
        "clone",
        "--depth",
        str(depth),
        "--filter=blob:limit=1m",
        "--no-recurse-submodules",
        "--single-branch",
        "--no-tags",
        "--quiet",
        "--",
        url,
        str(partial),
        timeout=CLONE_S,
    )
    if _tree_bytes(partial) > MAX_CLONE_BYTES:
        msg = f"the repository is bigger than {MAX_CLONE_BYTES // 2**20} MB and is not used"
        raise PluginError(msg)
    _only_yours(partial)
    if check is not None:
        check(partial)
    return git.head(partial)[0]


def _download(
    git: Git, url: str, partial: Path, check: Callable[[Path], None] | None, depth: int
) -> str:
    """Clone `url` into the new folder `partial`, check it and return its commit.

    A copy bigger than `MAX_CLONE_BYTES` or failing `check` is refused and removed.

    Raises:
        PluginError: Git failed, or the copy was refused.
        OSError: a folder or permission could not be set.
    """
    parent = partial.parent
    parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    for folder in (parent, parent.parent):
        folder.chmod(folder.stat().st_mode & ~_OTHERS_WRITE)
    shutil.rmtree(partial, ignore_errors=True)
    try:
        return _fill(git, url, partial, check, depth)
    except PluginError, OSError:
        shutil.rmtree(partial, ignore_errors=True)
        raise


def _clone(
    git: Git, url: str, destination: Path, check: Callable[[Path], None] | None = None
) -> str:
    """Download `url` to `destination` (which must not exist) and return its commit.

    The clone is made beside it and moved into place once it passed `check`, so a failure leaves
    nothing behind and a reader never sees half of it. Git failing, or the copy being refused,
    raises `PluginError`.

    Raises:
        OSError: a folder or permission could not be set, or the move failed.
    """
    partial = _partial(destination)
    commit = _download(git, url, partial, check, depth=1)
    try:
        partial.rename(destination)
    except OSError:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return commit


def _replace(git: Git, folder: Path, partial: Path, before: str) -> str:
    """Swap the new copy in `partial` for `folder`, if it builds on the commit `before`.

    Returns the new commit, which is `before` when nothing changed (the copy is then dropped).

    Raises:
        PluginError: the new commit does not build on `before`.
    """
    after = git.head(partial)[0]
    if after == before:
        shutil.rmtree(partial, ignore_errors=True)
    elif git.run(partial, "merge-base", "--is-ancestor", before, after)[0]:
        msg = (
            f"upstream no longer builds on {before[:12]} (rewritten, or over {FETCH_DEPTH} "
            f"commits ahead); delete {folder} and update again"
        )
        raise PluginError(msg)
    else:
        _swap(folder, partial)
    return after


def _refresh(
    git: Git, folder: Path, url: str, check: Callable[[Path], None] | None = None
) -> tuple[str, str]:
    """Move a downloaded repository to the newest commit of its default branch.

    The newest commit is cloned into a new folder beside the old one, checked, and only then
    swapped in, so what is read is always the old copy whole or the new one whole. Returns the
    commit it was at and the one it is at now.

    Raises:
        PluginError: the clone failed, or the new commit does not build on the old one.
        OSError: a folder or permission could not be set, or the swap failed.
    """
    before = git.head(folder)[0]
    partial = _partial(folder)
    _download(git, url, partial, check, depth=FETCH_DEPTH)
    try:
        return before, _replace(git, folder, partial, before)
    except PluginError, OSError:
        shutil.rmtree(partial, ignore_errors=True)
        raise


def _old(folder: Path) -> Path:
    return folder.parent / f".{folder.name}.old"


def _swap(folder: Path, fresh: Path) -> None:
    """Put the folder `fresh` where `folder` is, keeping the old one if that fails.

    The old copy stays beside as `.<name>.old` until the next swap, so a reader that opened it
    before the swap still reads it whole.

    Raises:
        OSError: the swap failed.
    """
    old = _old(folder)
    shutil.rmtree(old, ignore_errors=True)
    folder.rename(old)
    try:
        fresh.rename(folder)
    except OSError:
        old.rename(folder)
        raise


def _is_marketplace(folder: Path) -> None:
    """Check that `folder` holds a marketplace Jarvis can use.

    Raises:
        PluginError: it does not.
    """
    try:
        plugins.read_marketplace_at(folder.parent, folder.name)
    except FileNotFoundError as error:
        msg = "it is not a marketplace (no .claude-plugin/marketplace.json)"
        raise PluginError(msg) from error
    except plugins.MarketplaceError as error:
        raise PluginError(str(error)) from error


@contextlib.contextmanager
def _locked() -> Generator[None]:
    path = paths.state_dir() / "plugins.lock"
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            msg = "another plugin update is in progress"
            raise PluginError(msg) from error
        yield
    finally:
        os.close(fd)


def _bring(  # ruff: ignore[too-many-arguments] - one repository: where, from where, whether to fetch news
    git: Git,
    name: str,
    folder: Path,
    url: str,
    *,
    update: bool,
    check: Callable[[Path], None] | None = None,
) -> Change:
    """Download a repository when it is missing, else (with `update`) move it to the newest."""
    try:
        if folder.exists() or folder.is_symlink():
            if not update:
                return Change(name, "current", after=git.head(folder)[0])
            before, after = _refresh(git, folder, url, check)
            return Change(name, "current" if before == after else "updated", before, after)
        return Change(name, "downloaded", after=_clone(git, url, folder, check))
    except (PluginError, OSError) as error:
        return Change(name, "failed", error=plugins.tidy(str(error), 200))


def _wanted(settings: config.PluginsConfig, root: Path) -> Iterable[tuple[str, Path, str, bool]]:
    """Yield each repository the settings call for: name, folder, address, is it a marketplace."""
    for spec in settings.marketplaces:
        url = plugins.address(spec)
        yield spec, root.joinpath(*plugins.marketplace_parts(spec)), url, True
    seen: set[str] = set()
    for spec in settings.marketplaces:
        try:
            market = plugins.read_marketplace(root, spec)
        except FileNotFoundError, plugins.MarketplaceError:
            continue
        for ref in settings.enabled:
            name, _, called = ref.partition("@")
            entry = market.entry(name) if called == market.name else None
            if entry is not None and entry.source.url and entry.source.url not in seen:
                seen.add(entry.source.url)
                folder = root.joinpath(*plugins.plugin_parts(spec, entry.source))
                yield ref, folder, entry.source.url, False


def sync(
    settings: config.PluginsConfig,
    data_dir: Path,
    launch: Launch,
    audit: AuditLog,
    *,
    update: bool,
) -> list[Change]:
    """Download what the enabled plugins call for that is missing and, with `update`, update it.

    Nothing is done, and nothing fetched, while no plugin is enabled.

    A repository that fails does not stop the others. Every change and failure is audited with the
    commits before and after. Another update running raises `PluginError`.
    """
    changes: list[Change] = []
    if not settings.enabled:
        return changes  # nothing is downloaded until a plugin is enabled
    git = Git(launch)
    root = plugins.plugins_root(data_dir)
    with _locked():
        for name, folder, url, market in _wanted(settings, root):
            change = _bring(
                git, name, folder, url, update=update, check=_is_marketplace if market else None
            )
            changes.append(change)
            audit.record(
                "plugin",
                name=name,
                result=change.action,
                before=change.before,
                after=change.after,
                error=change.error,
            )
    return changes


def update_all(cfg: config.Config, launch: Launch, audit: AuditLog) -> None:
    """Bring the plugins up to date for `jarvis update --apply`; a failure is said, never raised."""
    if not cfg.plugins.enabled or not cfg.plugins.auto_update:
        return
    try:
        changes = sync(cfg.plugins, cfg.data_dir, launch, audit, update=True)
    except (PluginError, OSError) as error:
        print(f"jarvis: plugins not updated ({error})", file=sys.stderr)
        audit.record("plugin", name="all", result="failed", error=plugins.tidy(str(error), 200))
        return
    for change in changes:
        if change.action == "failed":
            print(f"jarvis: plugin {change.line()}", file=sys.stderr)
        elif change.action != "current":
            print(f"jarvis: plugin {change.line()}")


# -- the command and doctor ---------------------------------------------------------------------


def _inventory(cfg: config.Config, git: Git) -> list[dict[str, Any]]:
    """Return what is downloaded and enabled, for `jarvis plugins` and its agent format."""
    root = plugins.plugins_root(cfg.data_dir)
    loaded = plugins.load(cfg.plugins, cfg.data_dir)
    found: list[dict[str, Any]] = []
    for spec in cfg.plugins.marketplaces:
        row: dict[str, Any] = {"source": spec, "downloaded": False, "plugins": []}
        found.append(row)
        try:
            market = plugins.read_marketplace(root, spec)
        except FileNotFoundError:
            continue
        except plugins.MarketplaceError as error:
            row["error"] = str(error)
            continue
        folder = root.joinpath(*plugins.marketplace_parts(spec))
        row |= {"name": market.name, "downloaded": True, "refused": list(market.refused)}
        with contextlib.suppress(PluginError):
            row["commit"], row["date"] = git.head(folder)
        for entry in market.entries:
            skills = [
                skill.name for skill in loaded.skills if skill.name.startswith(f"{entry.name}:")
            ]
            parts = plugins.plugin_parts(spec, entry.source)
            row["plugins"].append(
                {
                    "name": entry.name,
                    "description": entry.description,
                    "enabled": f"{entry.name}@{market.name}" in cfg.plugins.enabled,
                    "skills": skills,
                    "ignored": plugins.ignored_parts(root, parts),
                }
            )
    return found


def inventory(cfg: config.Config, launch: Launch) -> list[dict[str, Any]]:
    """Return the marketplaces and plugins as `jarvis plugins` shows them."""
    return _inventory(cfg, Git(launch))


def _list(cfg: config.Config, launch: Launch) -> int:
    rows = inventory(cfg, launch)
    if not rows:
        print("No plugin marketplaces are set. See docs/configuration.md, [plugins].")
    for row in rows:
        if not row["downloaded"]:
            print(f"{row['source']}: not downloaded yet (installing one of its plugins does)")
            if row.get("error"):
                print(f"  {row['error']}")
            continue
        when = f" at {row['commit'][:12]} ({row['date'][:10]})" if row.get("commit") else ""
        print(f"{row['name']} ({row['source']}){when}")
        for plugin in row["plugins"]:
            state = (
                f"enabled, {len(plugin['skills'])} skills" if plugin["enabled"] else "not enabled"
            )
            print(f"  {plugin['name']}: {state}")
            if plugin["ignored"]:
                print(f"    never used or run: {', '.join(plugin['ignored'])}")
        for why in row["refused"]:
            print(f"  left out: {why}")
    print(
        "Jarvis uses only a plugin's skills: its hooks, commands, agents, MCP servers and "
        "scripts are never run."
    )
    return 0


def _update(cfg: config.Config, launch: Launch) -> int:
    if not cfg.plugins.enabled:
        print(
            "No plugin is enabled, so nothing is downloaded. `jarvis plugins install` enables one."
        )
        return 0
    audit = AuditLog(paths.audit_file())
    try:
        changes = sync(cfg.plugins, cfg.data_dir, launch, audit, update=True)
    except PluginError as error:
        print(f"jarvis: {error}", file=sys.stderr)
        return 1
    for change in changes:
        print(change.line(), file=sys.stderr if change.action == "failed" else sys.stdout)
    if any(change.action in {"downloaded", "updated"} for change in changes):
        print("A new chat window or `jarvis restart` takes the new plugin text.")
    return int(any(change.action == "failed" for change in changes))


def _save(**changes: object) -> None:
    """Write `[plugins]` settings into the config file, leaving the rest of it as it is."""
    current = config.load()
    config.save(current.model_copy(update={"plugins": current.plugins.model_copy(update=changes)}))


def _hinted(spec: str, called: str) -> bool:
    """Return whether a setting's repository name is the marketplace name looked for."""
    return spec.rstrip("/").removesuffix(".git").rsplit("/", 1)[-1] == called


def _find(
    cfg: config.Config, ref: str, git: Git, audit: AuditLog
) -> tuple[str, plugins.Marketplace, plugins.Entry, list[str]]:
    """Return what a `plugin@marketplace` names, and which marketplaces were downloaded to find it.

    The answer is the marketplace's setting, the marketplace, the entry and the settings of the
    marketplaces downloaded.

    Marketplaces already downloaded are asked first, then those whose repository is called like
    the marketplace; one is downloaded only while the name is still unfound.

    Raises:
        PluginError: no set marketplace has that name, or it lists no such plugin.
    """
    name, _, called = ref.partition("@")
    root = plugins.plugins_root(cfg.data_dir)

    def folder(spec: str) -> Path:
        return root.joinpath(*plugins.marketplace_parts(spec))

    have = [spec for spec in cfg.plugins.marketplaces if folder(spec).exists()]
    missing = [spec for spec in cfg.plugins.marketplaces if spec not in have]
    missing.sort(key=lambda spec: not _hinted(spec, called))
    failures: list[str] = []
    fetched: list[str] = []
    for spec in (*have, *missing):
        if spec in missing:
            change = _bring(
                git, spec, folder(spec), plugins.address(spec), update=False, check=_is_marketplace
            )
            if change.action == "failed":
                failures.append(f"{spec}: {change.error}")
                continue
            fetched.append(spec)
            audit.record("plugin", name=spec, result="downloaded", after=change.after)
        try:
            market = plugins.read_marketplace(root, spec)
        except (FileNotFoundError, plugins.MarketplaceError) as error:
            failures.append(f"{spec}: {error}")
            continue
        if market.name == called:
            entry = market.entry(name)
            if entry is None:
                held = ", ".join(entry.name for entry in market.entries) or "none"
                msg = f"marketplace {called} lists no plugin {name} (it lists: {held})"
                raise PluginError(msg)
            return spec, market, entry, fetched
    msg = f"no marketplace named {called} is set under [plugins] in config.toml"
    raise PluginError(msg + "".join(f"; {failure}" for failure in failures))


def _fetch_for(
    cfg: config.Config, ref: str, git: Git, audit: AuditLog
) -> tuple[str, plugins.Marketplace, plugins.Entry, list[str]]:
    """Download what installing `ref` needs: its marketplace and, if apart, its repository.

    Raises:
        PluginError: it could not be found or downloaded.
    """
    with _locked():
        spec, market, entry, fetched = _find(cfg, ref, git, audit)
        if entry.source.url:
            folder = plugins.plugins_root(cfg.data_dir).joinpath(
                *plugins.plugin_parts(spec, entry.source)
            )
            change = _bring(git, ref, folder, entry.source.url, update=False)
            if change.action == "failed":
                raise PluginError(change.error)
        audit.record("plugin", name=ref, result="installed")
    return spec, market, entry, fetched


def _install(ref: str, cfg: config.Config, launch: Launch) -> int:
    try:
        spec, market, entry, fetched = _fetch_for(
            cfg, ref, Git(launch), AuditLog(paths.audit_file())
        )
    except PluginError as error:
        print(f"jarvis: {error}", file=sys.stderr)
        return 1
    if fetched:
        print(f"Downloaded marketplaces: {', '.join(fetched)}.")
    enabled = list(cfg.plugins.enabled)
    if ref not in enabled:
        enabled.append(ref)
    _save(enabled=enabled)
    now = config.load()
    held = [
        s.name
        for s in plugins.load(now.plugins, now.data_dir).skills
        if s.name.startswith(f"{entry.name}:")
    ]
    print(f"{ref} is enabled: {len(held)} skills ({', '.join(held) or 'none'}).")
    ignored = plugins.ignored_parts(
        plugins.plugins_root(cfg.data_dir), plugins.plugin_parts(spec, entry.source)
    )
    if ignored:
        print(f"Never used or run: {', '.join(ignored)}.")
    print(
        "Its text reaches the model only when it reads a skill; to put one in every request add "
        f"it under [plugins.always_on] in config.toml. {market.name} is kept up to date by "
        "`jarvis update --apply` and `jarvis plugins update`."
    )
    return 0


def _uninstall(ref: str, cfg: config.Config) -> int:
    if ref not in cfg.plugins.enabled:
        print(f"jarvis: {ref} is not enabled", file=sys.stderr)
        return 2
    name = ref.partition("@")[0]
    root = plugins.plugins_root(cfg.data_dir)
    left = [other for other in cfg.plugins.enabled if other != ref]
    for spec in cfg.plugins.marketplaces:
        with contextlib.suppress(FileNotFoundError, plugins.MarketplaceError):
            market = plugins.read_marketplace(root, spec)
            entry = market.entry(name) if ref.endswith(f"@{market.name}") else None
            if entry is not None and entry.source.url:
                shared = any(
                    (other := market.entry(o.partition("@")[0])) is not None
                    and other.source.url == entry.source.url
                    for o in left
                    if o.endswith(f"@{market.name}")
                )
                if not shared:
                    gone = root.joinpath(*plugins.plugin_parts(spec, entry.source))
                    shutil.rmtree(gone, ignore_errors=True)
                    shutil.rmtree(_old(gone), ignore_errors=True)
    on = cfg.plugins.always_on
    _save(
        enabled=left,
        always_on=on.model_copy(
            update={
                "chat": [skill for skill in on.chat if not skill.startswith(f"{name}:")],
                "voice": [skill for skill in on.voice if not skill.startswith(f"{name}:")],
            }
        ),
        always_on_args={
            skill: arg
            for skill, arg in cfg.plugins.always_on_args.items()
            if not skill.startswith(f"{name}:")
        },
    )
    AuditLog(paths.audit_file()).record("plugin", name=ref, result="uninstalled")
    print(f"{ref} is disabled; its skills are gone from the next chat window or `jarvis restart`.")
    return 0


def command(args: Namespace, launch: Launch) -> int:
    """Run `jarvis plugins`: list, install, uninstall or update."""
    cfg = config.load()
    match args.action:
        case "update":
            return _update(cfg, launch)
        case "install" | "uninstall":
            if not args.target or not _REF.fullmatch(args.target):
                print(f"jarvis: {args.action} takes PLUGIN@MARKETPLACE", file=sys.stderr)
                return 2
            if args.action == "install":
                return _install(args.target, cfg, launch)
            return _uninstall(args.target, cfg)
        case _:
            return _list(cfg, launch)


def doctor_notes(cfg: config.Config, launch: Launch) -> list[str]:
    """Return `jarvis doctor`'s notes on plugins: commits, failed updates and skills left out."""
    if not cfg.plugins.enabled:
        return []
    git = Git(launch)
    notes: list[str] = []
    for row in _inventory(cfg, git):
        if row["downloaded"] and row.get("commit"):
            notes.append(
                f"note plugin marketplace {row['name']}: {row['commit'][:12]} of {row['date'][:10]}"
            )
        last = last_event(paths.audit_file(), "plugin", name=row["source"])
        if last and last.get("result") == "failed":
            notes.append(
                f"note plugin {row['source']} could not be updated at {last.get('ts')}: "
                f"{last.get('error', 'unknown')}"
            )
    notes.append(
        "note plugin updates: "
        + (
            "with jarvis update --apply"
            if cfg.plugins.auto_update
            else "only jarvis plugins update"
        )
    )
    return [*notes, *plugins.doctor_notes(cfg.plugins, cfg.data_dir)]
