"""Plugins in the Claude Code marketplace format: Jarvis reads their skills and nothing else.

A marketplace is a Git repository with `.claude-plugin/marketplace.json` naming its plugins and
where each lives (a folder in the same repository, a GitHub repository or an https address). A
plugin's `skills/<name>/SKILL.md` files are the only part Jarvis uses. Its hooks, commands, agents,
MCP servers, scripts and settings are never read and never run (docs/adr/0027).

The repositories live in Jarvis's own folder, `<data_dir>/plugins/`: `marketplaces/<host--owner--
repo>/` for each marketplace and `sources/<...>/` for a plugin that lives in another repository.
Jarvis never reads any other agent's plugins or settings. `pluginsync.py` fetches and updates them;
this module only reads what is there, once per process like the owner's skills, so the prompt and
`read_skill` agree. Each skill is `plugin:skill`. A skill named under `always_on` for a channel has
its text in that channel's instructions on every request; the others are listed and read when
needed. Plugin text is the owner's choice of third-party instructions: it guides the model and
never widens what an operation may do, since every boundary is in code before an operation runs.

Only regular files and folders of the owner's are read, opened without following a link, within
the size limits below. `jarvis doctor` says what was left out and why.
"""

import errno
import hashlib
import json
import os
import re
import stat
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlsplit

from jarvis import config
from jarvis.skills import Skill, third_party

if TYPE_CHECKING:
    from collections.abc import Generator

PLUGINS_DIR = "plugins"
MARKETPLACE_FILE = (".claude-plugin", "marketplace.json")
SKILL_FILE = "SKILL.md"
NAME = re.compile(r"[a-z0-9][a-z0-9-]{0,39}")
_RELATIVE = re.compile(r"[A-Za-z0-9_-][A-Za-z0-9._-]*")
_FRONT = re.compile(r"\A---\n(?P<head>.*?)\n---[ \t]*(?:\n(?P<body>.*))?\Z", re.DOTALL)
_KEY = re.compile(r"^(?P<key>[A-Za-z][A-Za-z0-9_-]*):[ \t]*(?P<value>.*)$")
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
_BLOCK = re.compile(r"[>|][+-]?")

MAX_JSON_BYTES = 262_144  # a marketplace file
MAX_ENTRIES = 100  # plugins read from one marketplace
MAX_SKILL_BYTES = 16_000  # one SKILL.md, read whole as one tool result
MAX_SKILLS_PER_PLUGIN = 30
MAX_PLUGIN_SKILLS = 60  # all plugins together
MAX_DESCRIPTION_CHARS = 120  # one skill's line in the prompt
MAX_DESCRIPTIONS_CHARS = 1500  # all plugin skills' lines together
MAX_ALWAYS_ON_BYTES = 8_000  # one always-on skill's body
MAX_ALWAYS_ON_TOTAL_BYTES = 12_000  # one channel's always-on bodies together
MAX_NOTES = 20  # the most problems `jarvis doctor` names
MAX_DIR_NAME = 100
DIGEST_CHARS = 12
MIN_QUOTED = 2  # the two quotes of an empty value
# What a plugin may hold that Jarvis never uses; `jarvis plugins list` says which are there.
IGNORED = (
    "hooks",
    "commands",
    "agents",
    "scripts",
    "output-styles",
    ".mcp.json",
    "mcp.json",
    ".lsp.json",
    "settings.json",
)
CHANNELS = ("chat", "voice")


class MarketplaceError(ValueError):
    """A marketplace file Jarvis cannot use."""


@dataclass(frozen=True, slots=True)
class Source:
    """Where a plugin lives: a Git address to clone, or a folder of its marketplace."""

    url: str = ""
    path: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Entry:
    """One plugin a marketplace lists."""

    name: str
    description: str
    source: Source


@dataclass(frozen=True, slots=True)
class Marketplace:
    """A marketplace file read: its name, its usable plugins and why any entry was refused."""

    name: str
    entries: tuple[Entry, ...]
    refused: tuple[str, ...] = ()

    def entry(self, name: str) -> Entry | None:
        """Return the plugin called `name`, if the marketplace lists it."""
        return next((entry for entry in self.entries if entry.name == name), None)


def tidy(text: str, limit: int) -> str:
    """Return `text` on one line, without control characters, cut to `limit` characters."""
    one = " ".join(_CONTROL.sub(" ", text).split())
    return one if len(one) <= limit else one[: limit - 1].rstrip() + "…"


def dir_name(url: str) -> str:
    """Return the folder name for a Git address: readable, and never the same for two addresses.

    It is the host and path made safe as one path component, then a short digest of the address
    (host in lower case), so `a/b--c` and `a--b/c` do not share a folder.
    """
    parts = urlsplit(url)
    full = f"{(parts.hostname or '').lower()}{parts.path}"
    flat = re.sub(r"[^A-Za-z0-9._-]+", "-", full.strip("/")).strip(".-") or "source"
    digest = hashlib.sha256(full.encode()).hexdigest()[:DIGEST_CHARS]
    return f"{flat[: MAX_DIR_NAME - DIGEST_CHARS - 1]}-{digest}"


def address(spec: str) -> str:
    """Return the https address of a marketplace or plugin source given as `owner/repo` or a URL.

    One that is neither (see `config.marketplace_url`) raises `ValueError`.
    """
    return config.marketplace_url(spec).removesuffix("/")


def marketplace_parts(spec: str) -> tuple[str, str]:
    """Return a marketplace's folder under the plugins folder, as path components."""
    return ("marketplaces", dir_name(address(spec)))


def plugin_parts(spec: str, source: Source) -> tuple[str, ...]:
    """Return a plugin's folder under the plugins folder, as path components."""
    if source.url:
        return ("sources", dir_name(source.url))
    return (*marketplace_parts(spec), *source.path)


def plugins_root(data_dir: Path) -> Path:
    """Return the folder Jarvis keeps its plugins in."""
    return data_dir / PLUGINS_DIR


@contextmanager
def descend(start: int, *parts: str) -> Generator[int]:
    """Open the folder `parts` below the open folder `start` and yield its descriptor.

    No link is followed, and each folder must be the owner's and writable by nobody else. A part
    that is missing raises `FileNotFoundError`; one that is a link, not a folder or not the
    owner's, `ValueError`. `start` stays open.
    """
    fd = start
    owned = False
    try:
        for part in parts:
            following = _open_dir(fd, part)
            if owned:
                os.close(fd)
            fd, owned = following, True
        yield fd
    finally:
        if owned:
            os.close(fd)


@contextmanager
def open_tree(root: Path, *parts: str) -> Generator[int]:
    """Open the plugins folder `root` and the folder `parts` below it, as `descend` does.

    A link higher up than `root` is followed, as for the owner's config folder; `root` itself
    is held to the same rules as the folders below it.
    """
    start = _open_dir(None, root)
    try:
        with descend(start, *parts) as fd:
            yield fd
    finally:
        os.close(start)


def _yours_alone(info: os.stat_result) -> bool:
    """Return whether a folder or file is the owner's and writable by nobody else."""
    return info.st_uid == os.getuid() and not info.st_mode & (stat.S_IWGRP | stat.S_IWOTH)


def _open_dir(parent: int | None, name: str | Path) -> int:
    """Open one folder without following a link.

    Raises:
        FileNotFoundError: it is not there.
        ValueError: it is a link, is not a folder, is not yours alone or cannot be opened.
    """
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    try:
        fd = os.open(name, flags, dir_fd=parent)
    except FileNotFoundError:
        raise
    except OSError as error:
        if error.errno == errno.ELOOP:
            msg = f"{name} is a link, which is not followed"
        elif error.errno == errno.ENOTDIR:
            msg = f"{name} is not a folder"
        else:
            msg = f"{name} could not be opened ({error.strerror})"
        raise ValueError(msg) from error
    if not _yours_alone(os.fstat(fd)):
        os.close(fd)
        msg = f"{name} is not yours alone (another user owns it, or others can write to it)"
        raise ValueError(msg)
    return fd


def read_file(folder: int, name: str, limit: int) -> str:
    """Return a file's text, read without following a link.

    Raises:
        FileNotFoundError: it is not there.
        ValueError: it is not a regular file of the owner's, is writable by others, is bigger
            than `limit` bytes or is not UTF-8 text.
    """
    try:
        fd = os.open(
            name,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_NOCTTY | os.O_CLOEXEC,
            dir_fd=folder,
        )
    except FileNotFoundError:
        raise
    except OSError as error:
        msg = "a link, which is not followed" if error.errno == errno.ELOOP else "unreadable"
        raise ValueError(msg) from error
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            msg = "not a regular file"
            raise ValueError(msg)
        if not _yours_alone(info):
            msg = "not yours alone (another user owns it, or others can write to it)"
            raise ValueError(msg)
        if info.st_size > limit:
            msg = f"bigger than {limit} bytes"
            raise ValueError(msg)
        data = os.read(fd, limit + 1)
    finally:
        os.close(fd)
    if len(data) > limit:
        msg = f"bigger than {limit} bytes"
        raise ValueError(msg)
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as error:
        msg = "not UTF-8 text"
        raise ValueError(msg) from error


def _source(raw: object) -> Source:
    """Return where a plugin entry says the plugin lives.

    Raises:
        ValueError: the source is of a kind Jarvis does not use, or is not safe.
        TypeError: it is neither text nor an object.
    """
    if isinstance(raw, str):
        if raw != "." and not raw.startswith("./"):
            msg = "a folder source must start with ./"
            raise ValueError(msg)
        parts = tuple(part for part in raw.split("/")[1:] if part)
        if not all(_RELATIVE.fullmatch(part) and part not in {".", ".."} for part in parts):
            msg = "a folder source may only name plain folders inside the marketplace"
            raise ValueError(msg)
        return Source(path=parts)
    if not isinstance(raw, dict):
        msg = "the source is neither a folder nor a Git repository"
        raise TypeError(msg)
    fields = cast("dict[str, Any]", raw)
    if "ref" in fields or "sha" in fields:
        msg = "it pins a version, and Jarvis follows the latest of a repository"
        raise ValueError(msg)
    kind = fields.get("source")
    if kind == "github" and isinstance(fields.get("repo"), str):
        if not re.fullmatch(config.GITHUB_REPO.pattern, str(fields["repo"])):
            msg = "the GitHub repository is not owner/repo"
            raise ValueError(msg)
        return Source(url=address(str(fields["repo"])))
    if kind == "url" and isinstance(fields.get("url"), str):
        return Source(url=address(str(fields["url"])))
    msg = "only folder, github and https url sources are used"
    raise ValueError(msg)


def parse_marketplace(text: str) -> Marketplace:
    """Return the marketplace a `marketplace.json` describes.

    Raises:
        MarketplaceError: the file is not a marketplace Jarvis can use. A plugin entry that is not
            usable is listed in `refused` instead.
    """
    try:
        raw = cast("object", json.loads(text))
    except ValueError as error:
        msg = "marketplace.json is not valid JSON"
        raise MarketplaceError(msg) from error
    if not isinstance(raw, dict):
        msg = "marketplace.json is not an object"
        raise MarketplaceError(msg)
    top = cast("dict[str, Any]", raw)
    name = top.get("name")
    if not isinstance(name, str) or not NAME.fullmatch(name):
        msg = "marketplace.json needs a name of lowercase letters, digits and hyphens"
        raise MarketplaceError(msg)
    listed = top.get("plugins")
    if not isinstance(listed, list):
        msg = "marketplace.json has no plugins list"
        raise MarketplaceError(msg)
    entries: list[Entry] = []
    refused: list[str] = []
    for item in cast("list[object]", listed)[:MAX_ENTRIES]:
        fields = cast("dict[str, Any]", item) if isinstance(item, dict) else {}
        called = fields.get("name")
        if not isinstance(called, str) or not NAME.fullmatch(called):
            refused.append("an entry without a valid name")
            continue
        if any(entry.name == called for entry in entries):
            refused.append(f"{called}: listed twice")
            continue
        try:
            source = _source(fields.get("source"))
        except (TypeError, ValueError) as error:
            refused.append(f"{called}: {error}")
            continue
        described = fields.get("description")
        entries.append(
            Entry(called, tidy(described, 200) if isinstance(described, str) else "", source)
        )
    if len(cast("list[object]", listed)) > MAX_ENTRIES:
        refused.append(f"only the first {MAX_ENTRIES} entries are read")
    return Marketplace(name, tuple(entries), tuple(refused))


def _marketplace_in(folder: int) -> Marketplace:
    """Return the marketplace in the open folder.

    Raises:
        MarketplaceError: the file cannot be used.
    """
    try:
        with descend(folder, MARKETPLACE_FILE[0]) as inner:
            return parse_marketplace(read_file(inner, MARKETPLACE_FILE[1], MAX_JSON_BYTES))
    except MarketplaceError:
        raise
    except ValueError as error:
        raise MarketplaceError(str(error)) from error


def read_marketplace_at(root: Path, *parts: str) -> Marketplace:
    """Return the marketplace in the folder `parts` below `root`.

    Raises:
        MarketplaceError: the file cannot be used. A missing folder or file is the
            `FileNotFoundError` it raises.
    """
    try:
        with open_tree(root, *parts) as folder:
            return _marketplace_in(folder)
    except MarketplaceError:
        raise
    except ValueError as error:
        raise MarketplaceError(str(error)) from error


def read_marketplace(root: Path, spec: str) -> Marketplace:
    """Return the marketplace downloaded for `spec` under the plugins folder `root`.

    One that is not downloaded raises `FileNotFoundError`; a file that cannot be used,
    `MarketplaceError`.
    """
    return read_marketplace_at(root, *marketplace_parts(spec))


def ignored_parts(root: Path, parts: tuple[str, ...]) -> list[str]:
    """Return the parts of a plugin Jarvis never uses that it holds (`IGNORED`)."""
    try:
        with open_tree(root, *parts) as folder:
            held = set(os.listdir(folder))  # ruff: ignore[os-listdir] - a folder opened without a link
    except OSError, ValueError:
        return []
    return [name for name in IGNORED if name in held]


def front_matter(text: str) -> tuple[dict[str, str], str] | None:
    """Return a skill file's front matter keys and its body, or None without a front matter.

    Reads what skills use: `key: value` lines, and a value given as a `>` or `|` block.
    """
    parts = _FRONT.match(text.removeprefix("﻿").replace("\r\n", "\n"))
    if parts is None:
        return None
    keys: dict[str, str] = {}
    lines = parts["head"].split("\n")
    index = 0
    while index < len(lines):
        found = _KEY.match(lines[index])
        index += 1
        if found is None:
            continue
        value = found["value"].strip()
        block: list[str] = []
        while index < len(lines) and (not lines[index].strip() or lines[index][0] in " \t"):
            block.append(lines[index].strip())
            index += 1
        if _BLOCK.fullmatch(value):
            value = ("\n" if value.startswith("|") else " ").join(block).strip()
        elif len(value) >= MIN_QUOTED and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        keys[found["key"]] = value
    return keys, parts["body"] or ""


@dataclass(frozen=True, slots=True)
class Loaded:
    """The plugin skills read, why anything was left out, and what each channel has always on."""

    skills: tuple[Skill, ...] = ()
    notes: tuple[str, ...] = ()
    sections: tuple[tuple[str, str], ...] = ()  # (skill, its text for the prompt) of each on
    active: tuple[tuple[str, tuple[str, ...]], ...] = ()  # each channel's always-on skills

    def always_on(self, channel: str) -> tuple[str, ...]:
        """Return the skills whose whole text is in `channel`'s instructions."""
        return next((ids for name, ids in self.active if name == channel), ())

    def text(self, channel: str) -> str:
        """Return what `channel`'s instructions gain: each always-on skill, headed by its name."""
        shown = dict(self.sections)
        return "".join(f"\n{shown[name]}" for name in self.always_on(channel))


def _skills_of(base: int, plugin: str) -> tuple[list[Skill], list[str]]:
    """Return the skills the open plugin folder holds and why any was left out."""
    found: list[Skill] = []
    try:
        with descend(base, "skills") as folder:
            names = sorted(os.listdir(folder))  # ruff: ignore[os-listdir] - opened without a link
            reasons = {
                name: _skill_into(folder, name, plugin, found)
                for name in names[:MAX_SKILLS_PER_PLUGIN]
            }
    except FileNotFoundError:
        return found, [f"{plugin}: has no skills folder"]
    except ValueError as error:
        return found, [f"{plugin}: {error}"]
    notes = [f"{plugin}:{tidy(name, 40)}: {why}" for name, why in reasons.items() if why]
    if len(names) > MAX_SKILLS_PER_PLUGIN:
        notes.append(f"{plugin}: only {MAX_SKILLS_PER_PLUGIN} skills are read")
    return found, notes


def _skill_into(skills_fd: int, name: str, plugin: str, found: list[Skill]) -> str:
    """Add one skill folder's skill to `found`, or return why it was left out."""
    if not NAME.fullmatch(name):
        return "the folder name must be lowercase letters, digits and hyphens (at most 40)"
    try:
        with descend(skills_fd, name) as folder:
            text = read_file(folder, SKILL_FILE, MAX_SKILL_BYTES)
    except FileNotFoundError:
        return "has no SKILL.md"
    except ValueError as error:
        return str(error)
    head = front_matter(text)
    description = head[0].get("description", "").strip() if head else ""
    if head is None or not description:
        return "SKILL.md does not open with a --- block holding a description"
    body = _CONTROL.sub("", head[1]).strip()
    if not body:
        return "SKILL.md has no instructions"
    shown = tidy(description, MAX_DESCRIPTION_CHARS)
    found.append(Skill(f"{plugin}:{name}", shown, body + "\n", plugin=True))
    return ""


@dataclass(frozen=True, slots=True)
class _Market:
    """A downloaded marketplace: the setting that names it, what it lists, its open folder."""

    spec: str
    read: Marketplace
    folder: int


def _markets(
    settings: config.PluginsConfig, root: Path, notes: list[str], stack: ExitStack
) -> dict[str, _Market]:
    """Return each downloaded marketplace by its name, its folder held open on `stack`.

    Holding the folder open means a plugin's skills are read from the very copy its marketplace
    file was, even if an update replaces the folder meanwhile.
    """
    market: dict[str, _Market] = {}
    for spec in settings.marketplaces:
        try:
            folder = stack.enter_context(open_tree(root, *marketplace_parts(spec)))
            read = _marketplace_in(folder)
        except FileNotFoundError:
            notes.append(f"marketplace {spec} is not downloaded: run `jarvis plugins update`")
        except ValueError as error:  # a marketplace that cannot be used is one such error
            notes.append(f"marketplace {spec}: {error}")
        else:
            if read.name in market:
                notes.append(f"marketplace {spec} has the name {read.name}, which is taken")
            else:
                market[read.name] = _Market(spec, read, folder)
            notes.extend(f"marketplace {read.name}: left out {why}" for why in read.refused)
    return market


def _plugin_skills(
    root: Path, market: _Market, entry: Entry, stack: ExitStack
) -> tuple[list[Skill], list[str]]:
    """Return a plugin's skills, read from its own open folder, and the problems."""
    try:
        if entry.source.url:
            base = stack.enter_context(open_tree(root, "sources", dir_name(entry.source.url)))
        else:
            base = stack.enter_context(descend(market.folder, *entry.source.path))
    except FileNotFoundError:
        return [], [f"{entry.name}: is not downloaded: run `jarvis plugins update`"]
    except ValueError as error:
        return [], [f"{entry.name}: {error}"]
    return _skills_of(base, entry.name)


def load(settings: config.PluginsConfig, data_dir: Path) -> Loaded:
    """Return the skills of every enabled plugin downloaded under `data_dir`, and the problems.

    A plugin is read from folders opened once, so an update that lands meanwhile cannot mix its
    old and new files.
    """
    root = plugins_root(data_dir)
    notes: list[str] = []
    found: list[Skill] = []
    room = MAX_DESCRIPTIONS_CHARS
    with ExitStack() as stack:
        market = _markets(settings, root, notes, stack) if settings.enabled else {}
        for ref in settings.enabled:
            name, _, called = ref.partition("@")
            if called not in market:
                notes.append(f"{ref}: its marketplace is not added")
                continue
            entry = market[called].read.entry(name)
            if entry is None:
                notes.append(f"{ref}: marketplace {called} lists no plugin {name}")
                continue
            got, why = _plugin_skills(root, market[called], entry, stack)
            notes.extend(why)
            for skill in got:
                if len(found) >= MAX_PLUGIN_SKILLS:
                    notes.append(f"only {MAX_PLUGIN_SKILLS} plugin skills are read")
                    break
                if len(skill.description) > room:
                    notes.append(
                        f"{skill.name}: all descriptions together may be "
                        f"{MAX_DESCRIPTIONS_CHARS} characters"
                    )
                    continue
                found.append(skill)
                room -= len(skill.description)
    return _arranged(settings, tuple(found), notes)


def _arranged(settings: config.PluginsConfig, found: tuple[Skill, ...], notes: list[str]) -> Loaded:
    """Decide each channel's always-on skills, within the size limits."""
    by_name = {skill.name: skill for skill in found}
    sections: dict[str, str] = {}
    active: list[tuple[str, tuple[str, ...]]] = []
    for channel in CHANNELS:
        total = 0
        on: list[str] = []
        for name in cast("list[str]", getattr(settings.always_on, channel)):
            skill = by_name.get(name)
            if skill is None:
                notes.append(f"always_on.{channel} names {name}, which is not loaded")
                continue
            size = len(skill.body.encode())
            if size > MAX_ALWAYS_ON_BYTES:
                notes.append(
                    f"{name} is too long to be always on in the {channel} ({size} bytes, at most "
                    f"{MAX_ALWAYS_ON_BYTES}): it is listed and read with read_skill instead"
                )
            elif total + size > MAX_ALWAYS_ON_TOTAL_BYTES:
                notes.append(
                    f"{name} is not always on in the {channel}: all always-on skills together "
                    f"may be {MAX_ALWAYS_ON_TOTAL_BYTES} bytes"
                )
            else:
                total += size
                on.append(name)
                argument = settings.always_on_args.get(name, "")
                sections[name] = third_party(skill, argument)
        active.append((channel, tuple(on)))
    return Loaded(found, tuple(notes), tuple(sections.items()), tuple(active))


@cache
def _loaded_at(data_dir: str, settings: str) -> Loaded:
    return load(config.PluginsConfig.model_validate_json(settings), Path(data_dir))


def loaded(settings: config.PluginsConfig, data_dir: Path) -> Loaded:
    """Return `load`'s answer, read once per process and settings.

    Every prompt and `read_skill` of one process then agree; a new download counts from the next
    start of the voice service or chat window.
    """
    return _loaded_at(str(data_dir), settings.model_dump_json())


def doctor_notes(settings: config.PluginsConfig, data_dir: Path) -> list[str]:
    """Return `jarvis doctor`'s notes on the plugin skills: how many, and what was left out."""
    now = load(settings, data_dir)
    notes = [f"note plugin skills: {len(now.skills)} read from {plugins_root(data_dir)}"]
    if not now.skills:
        notes = []
    shown = [f"note plugin: {line}" for line in now.notes[:MAX_NOTES]]
    if len(now.notes) > MAX_NOTES:
        shown.append(
            f"note {len(now.notes) - MAX_NOTES} more plugin problems left out of this list"
        )
    return [*notes, *shown]
