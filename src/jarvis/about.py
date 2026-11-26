"""What Jarvis knows about itself: its version, model, voice, settings and where its files are.

`about_jarvis` answers from the config and Strata's run config, worked out in code, so the model
never guesses which model it is or what it may do. `read_jarvis_file` reads one of Jarvis's own
configuration files, picked by name from a fixed list (no path comes from the model): the config,
Strata's run config and the profile Jarvis serves from it, Jarvis's own Voxtype settings and the two
systemd units. None of them holds a secret (the Strata key and the Spotify tokens live in the state
folder, which is never read), and a value under a name that looks like one is still hidden. What
the files say reaches the model as `file` text, like any file on the PC: an action after it needs a
yes, and so does text sent off the PC. Neither operation changes anything: Jarvis never edits its
own configuration, as it could then lift its own boundaries.
"""

import os
import re
import stat
from pathlib import Path
from typing import TYPE_CHECKING, Any

from jarvis import __version__, paths, services
from jarvis.operations import Operation, OperationError, Result
from jarvis.policy import Origin

if TYPE_CHECKING:
    from collections.abc import Callable

    from jarvis.agent import Limits
    from jarvis.config import Config

MAX_FILE_BYTES = 256 * 1024
HIDDEN = "(hidden)"
# A setting (`name = value`, `"name": value`, `Name=value`) and an `Environment=NAME=value` line.
_SETTING = re.compile(r"""^(?P<lead>\s*["']?(?P<name>[\w.-]+)["']?\s*[=:]\s*)(?P<value>\S.*)$""")
# One argument of a JSON list on its own line: `"--api-key",`.
_FLAG = re.compile(r"""^\s*["']?(?P<name>--[\w-]+)["']?,?\s*$""")
_VARIABLE = re.compile(r"""^["']?(?P<name>\w+)=""")
_SECRET_WORDS = frozenset({"key", "apikey", "token", "secret", "password", "passwd", "credentials"})
_SECTION = re.compile(r"^\s*\[([^\]\n]+)\]\s*$", re.MULTILINE)
# What a remembered request keeps, as `facts` says it.
_DETAIL = {
    "condensed": "condensed (each request, its reply, actions and look-ups)",
    "full": "in full (with its tool calls and their results)",
}


def jarvis_files(config: Config) -> dict[str, Path]:
    """Return Jarvis's own configuration files by the names `read_jarvis_file` takes."""
    if config.strata.mode == "remote":
        return {
            "config": paths.config_file(),
            "voxtype": paths.config_dir() / "voxtype.toml",
            "voice_unit": services.unit_dir() / services.JARVIS_UNIT,
        }
    return {
        "config": paths.config_file(),
        "strata_run_config": config.strata.run_config,
        "strata_profile": services.strata_profile_file(),
        "voxtype": paths.config_dir() / "voxtype.toml",
        "voice_unit": services.unit_dir() / services.JARVIS_UNIT,
        "strata_unit": services.unit_dir() / services.STRATA_UNIT,
    }


def _secret_name(name: str) -> bool:
    """Whether a setting's name says it holds a secret (`api_key`, `refresh-token`, `apiKey`)."""
    words = re.split(r"[_.-]|(?<=[a-z])(?=[A-Z])", name)
    return any(word.casefold() in _SECRET_WORDS for word in words)


def _holds_secret(setting: re.Match[str]) -> bool:
    if _secret_name(setting.group("name")):
        return True
    variable = _VARIABLE.match(setting.group("value"))
    return (
        setting.group("name").casefold() == "environment"
        and variable is not None
        and _secret_name(variable.group("name"))
    )


def hide_secrets(text: str) -> str:
    """Return `text` with the value of every setting whose name looks like a secret's hidden."""
    lines: list[str] = []
    after_flag = False  # the line before was a secret's flag in a list of arguments
    for line in text.split("\n"):
        found = _SETTING.match(line)
        shown = line
        if after_flag and line.strip():
            shown = line[: len(line) - len(line.lstrip())] + HIDDEN
        elif found is not None and _holds_secret(found):
            shown = found.group("lead") + HIDDEN
        flag = _FLAG.match(line)
        after_flag = flag is not None and _secret_name(flag.group("name"))
        lines.append(shown)
    return "\n".join(lines)


def section_of(text: str, section: str) -> str | None:
    """Return one `[section]` of a TOML or unit file with its heading, or None when it has none.

    A section's own tables (`[files]` holds `[files.something]`) come with it.
    """
    wanted = section.strip().strip("[]").strip().casefold()
    headings = list(_SECTION.finditer(text))
    kept: list[str] = []
    for at, heading in enumerate(headings):
        name = heading.group(1).strip().casefold()
        if name == wanted or name.startswith(wanted + "."):
            end = headings[at + 1].start() if at + 1 < len(headings) else len(text)
            kept.append(text[heading.start() : end].strip("\n"))
    return "\n\n".join(kept) if kept else None


def _yes(value: object) -> str:
    return "on" if value else "off"


def _model_lines(config: Config) -> list[str]:
    if config.strata.mode == "remote":
        reader = (
            "with picture support" if config.strata.remote_pictures else "without picture support"
        )
        return [
            (
                f"Model: {config.model.model}, served by Strata on the owner's network, "
                f"context {config.strata.context} tokens, {reader}."
            )
        ]
    try:
        run = services.read_run_config(config)
    except services.StrataSetupError as exc:
        return [f"Model: none usable ({exc})."]
    reader = "with a picture reader" if services.reads_pictures(config) else "no picture reader"
    engine = str(run.data.get("exe", "")).removeprefix("component:") or "unknown engine"
    strata = config.strata
    served = (
        f"Model: {run.name or 'unnamed'} (file {run.model.name or 'not named'}), served by Strata"
        f" ({engine}) on 127.0.0.1:{strata.port}, context {strata.context} tokens, {reader},"
        f" kept loaded for {strata.idle_unload_s} s when idle."
    )
    return [served, f"Strata run config: {run.path}."]


def facts(config: Config, channel: str, limits: Limits) -> str:
    """Return what Jarvis is and how it is set up, one fact per line."""
    audio, bounds, files = config.audio, config.boundaries, config.files
    if audio.wake_engine == "azure":
        wake = f"the Azure keyword model {audio.azure_keyword_model or '(not set)'}"
    else:
        wake = "LiveKit models " + ", ".join(audio.wake_words)
    overrides = ", ".join(f"{op} {cls}" for op, cls in sorted(bounds.classes.items())) or "none"
    replies = "spoken" if channel == "voice" else "written"
    engine = f", engine {audio.transcribe_engine}" if audio.transcribe_engine else ""
    asking = "off (bypass_consent)" if bounds.bypass_consent else "on"
    where = (
        "one agent on this PC, no hosted model"
        if config.strata.mode == "remote"
        else "running only on this PC; one agent, no hosted model"
    )
    lines = [
        (
            f"Name: {config.persona.name}. Program: Jarvis {__version__}, free software under"
            f" GPL-3.0-or-later, {where}."
        ),
        f"This conversation: the {channel}; replies are {replies}.",
        *_model_lines(config),
        f"Speech to text: Voxtype, model {audio.transcribe_model}{engine}.",
        f"Voice: {audio.voice}, speed {audio.speech_speed}.",
        (
            f"Wake word: {wake}; follow-up without it for {audio.follow_up_s} s; speaking over"
            f" Jarvis {_yes(audio.barge_in)}; listening at the lock screen"
            f" {_yes(audio.listen_when_locked)}; PC audio {_yes(audio.listen_to_pc)}."
        ),
        (
            f"Asking before acting: {asking}; a strong action always waits for the confirm key."
            f" Owner's class overrides: {overrides}. At most {bounds.max_actions} actions in"
            f" {bounds.actions_window_s:.0f} s."
        ),
        (
            f"Files: read in {', '.join(files.read_roots) or 'nowhere'}; written in"
            f" {', '.join(files.write_roots) or 'nowhere'}. Your workspace {files.workspace} is"
            " always readable and writable, and a bare file name is saved there. Shell command:"
            f" {_yes(config.shell.enabled)}."
        ),
        (
            f"Limits here: {limits.max_steps} model calls and {limits.max_tool_calls} tool calls"
            f" per request, {limits.reply_tokens} tokens per reply, {limits.memory_tokens} tokens"
            f" of remembered conversation, kept {_DETAIL[limits.memory_detail]}."
        ),
        (
            f"Config file: {paths.config_file()}. State (tokens, audit log; never read):"
            f" {paths.state_dir()}. Models and voices: {config.data_dir}."
        ),
        "Readable with read_jarvis_file: " + ", ".join(jarvis_files(config)) + ".",
    ]
    return "\n".join(lines)


def _read_regular(path: Path) -> bytes:
    """Return the start of a regular file, opened without following a link.

    Raises:
        OperationError: it is not there, not a regular file or cannot be read.
    """
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        msg = f"cannot read {path}: {exc.strerror or exc}"
        raise OperationError(msg) from exc
    with os.fdopen(fd, "rb") as handle:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            msg = f"{path} is not a file"
            raise OperationError(msg)
        try:
            return handle.read(MAX_FILE_BYTES + 1)
        except OSError as exc:
            msg = f"cannot read {path}: {exc.strerror or exc}"
            raise OperationError(msg) from exc


class AboutOps:
    """`about_jarvis` and `read_jarvis_file`, for one channel."""

    def __init__(
        self,
        config: Config,
        channel: str,
        limits: Limits,
        *,
        refused: Callable[[Path], bool] = lambda _path: False,
        files: Callable[[], dict[str, Path]] | None = None,
    ) -> None:
        """Answer for `channel` with its `limits`.

        `refused` says whether a file, resolved through its links, is one the `[files]` boundary
        never reads (a secret, a `files.deny` entry, Jarvis's own state); `files` replaces
        `jarvis_files` (tests).
        """
        super().__init__()
        self._config = config
        self._channel = channel
        self._limits = limits
        self._refused = refused
        self._files = files or (lambda: jarvis_files(config))

    def about(self, _args: dict[str, Any]) -> Result:
        """Say what Jarvis is and how it is set up."""
        return Result(facts(self._config, self._channel, self._limits), Origin.TOOL_STATUS)

    def read(self, args: dict[str, Any]) -> Result:
        """Read one of Jarvis's own configuration files, or one section of it.

        Raises:
            OperationError: the file is not there or cannot be read, or has no such section.
        """
        name = str(args["file"])
        path = self._files().get(name, Path())
        if path == Path():
            msg = f"{name} is not set up"
            raise OperationError(msg)
        resolved = Path(os.path.realpath(path))
        if self._refused(resolved):
            msg = f"{path} is kept private"
            raise OperationError(msg)
        data = _read_regular(resolved)
        text = hide_secrets(data[:MAX_FILE_BYTES].decode("utf-8", errors="replace"))
        section = str(args.get("section") or "").strip()
        if section:
            found = section_of(text, section)
            if found is None:
                msg = f"{path} has no [{section.strip('[]')}] section"
                raise OperationError(msg)
            text = found
        room = self._limits.tool_text_chars - len(str(path)) - 100
        if len(text) > room:
            text = (
                text[:room] + f"\n[cut at {room} characters: ask for one section, such as [audio]]"
            )
        return Result(f"{path}:\n{text}", Origin.FILE)

    def operations(self) -> list[Operation]:
        """Return the two operations; neither changes anything."""
        return [
            Operation(
                "about_jarvis",
                "Your own facts: version, model, voice, wake word, boundaries, limits, files.",
                {"type": "object", "properties": {}, "additionalProperties": False},
                self.about,
                side_effect=False,
            ),
            Operation(
                "read_jarvis_file",
                "Read one of your own config files, or one [section] of it.",
                {
                    "type": "object",
                    "properties": {
                        "file": {"type": "string", "enum": list(jarvis_files(self._config))},
                        "section": {"type": "string", "maxLength": 60},
                    },
                    "required": ["file"],
                    "additionalProperties": False,
                },
                self.read,
                side_effect=False,
            ),
        ]
