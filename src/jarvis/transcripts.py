"""Export retained conversations and existing recordings to a private ZIP."""

import contextlib
import json
import os
import re
import shutil
import stat
import tempfile
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from zipfile import ZIP_DEFLATED, ZipFile

from jarvis.activity import VOICE_VIA
from jarvis.audit import GENERATIONS
from jarvis.chat.archive import FIRST_VERSION, LOG_SUFFIX, MEMORY_SUFFIX
from jarvis.jsonutil import obj, objs
from jarvis.secretfile import read_private

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping
    from pathlib import Path

_RECORDING = re.compile(r"(?:utterance|output)-\d+\.wav")
_SESSION = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")
_BOUNDARY = frozenset({"service_started", "memory_cleared", "memory_expired"})


@dataclass(frozen=True, slots=True)
class Export:
    """Number of conversation transcripts and recording files written."""

    transcripts: int
    user_recordings: int
    jarvis_recordings: int


def _rows(text: str) -> Iterator[dict[str, object]]:
    for line in text.splitlines():
        # A partial last line can be left by a crash or an export during a write.
        with contextlib.suppress(ValueError, RecursionError):
            yield obj(json.loads(line))


def _text_entry(row: Mapping[str, object], *, scheduled: bool = False) -> str:
    event = row.get("event")
    if not isinstance(event, str) or event not in {"request", "reply"}:
        return ""
    stamp, text = row.get("ts", ""), row.get("text", "")
    prefix = f"[{stamp}] " if stamp else ""
    if event == "reply":
        status = " [stopped]" if row.get("stopped") else ""
        return f"{prefix}Jarvis: {text or '[no reply]'}{status}\n"
    if scheduled:
        speaker = "Scheduled request"
    elif row.get("source") == "note":
        speaker = "Jarvis note"
    elif row.get("origin") == "pc_audio":
        speaker = "PC audio"
    else:
        speaker = "Owner" if row.get("via") else "Request"
    return f"{prefix}{speaker}: {text}\n"


def _chat_transcripts(folder: Path) -> Iterator[tuple[str, list[str]]]:
    for path in sorted(folder.glob("*.json")):
        if path.name.endswith(MEMORY_SUFFIX) or _SESSION.fullmatch(path.stem) is None:
            continue
        try:
            metadata = obj(json.loads(read_private(path) or "{}"))
        except ValueError, RecursionError:
            continue
        rows = (
            iter(objs(metadata.get("log")))
            if metadata.get("version") == FIRST_VERSION
            else _rows(read_private(folder / f"{path.stem}{LOG_SUFFIX}") or "")
        )
        if lines := [line for row in rows if (line := _text_entry(row | {"via": "chat"}))]:
            yield f"chat-{path.stem}", lines


def _retained_acp(state: Path) -> set[str]:
    return {
        path.stem
        for path in (state / "acp-sessions").glob("jarvis-*.json")
        if _SESSION.fullmatch(path.stem) and read_private(path) is not None
    }


def _current_inode(path: Path) -> tuple[int, int] | None:
    with contextlib.suppress(FileNotFoundError):
        info = path.stat(follow_symlinks=False)
        return info.st_dev, info.st_ino
    return None


def _audit_rows(state: Path) -> Iterator[dict[str, object]]:
    current = state / "audit.jsonl"
    logs = [state / f"audit.jsonl.{n}" for n in range(GENERATIONS, 0, -1)]
    paths = [*logs, current]
    # A rotation changes the path identities. Retry so entries cannot be copied twice or lost.
    for _ in range(3):
        before = [_current_inode(path) for path in paths]
        snapshot = [read_private(path) or "" for path in paths]
        if before == [_current_inode(path) for path in paths]:
            for text in snapshot:
                yield from _rows(text)
            return
    msg = "the audit log kept rotating during the export"
    raise OSError(msg)


@dataclass(slots=True)
class _VoiceSessions:
    members: set[str]
    transcripts: dict[str, list[str]] = field(default_factory=dict[str, list[str]])
    session: int = 1
    scheduled: bool = False

    def record(self, row: dict[str, object]) -> None:
        name = f"voice-{self.session:03d}"
        event = row.get("event")
        if isinstance(event, str) and event in _BOUNDARY:
            self.session += name in self.transcripts
            if event == "service_started":
                self.scheduled = False
            return
        if event == "schedule_due":
            self.scheduled = row.get("action") == "do"
        if event == "recording_saved":
            self._recording(name, row)
        if line := _text_entry(row, scheduled=self.scheduled):
            lines = self.transcripts.setdefault(name, [])
            lines.append(line)
            if event == "request":
                member = f"audio/user/{row.get('input_recording')}"
                if member in self.members:
                    lines.append(f"Owner WAV: ../{member}\n")
                self.scheduled = False

    def _recording(self, name: str, row: dict[str, object]) -> None:
        member = f"audio/jarvis/{row.get('filename')}"
        if row.get("speaker") == "jarvis" and member in self.members and name in self.transcripts:
            self.transcripts[name].append(f"Jarvis WAV saved: ../{member}\n")


def _audit_transcripts(state: Path, audio_members: set[str]) -> dict[str, list[str]]:
    saved_acp = _retained_acp(state)
    voice = _VoiceSessions(audio_members)
    transcripts: dict[str, list[str]] = {}
    for row in _audit_rows(state):
        via = row.get("via")
        if via == VOICE_VIA:
            voice.record(row)
            continue
        name = _audit_name(via, row.get("session"), saved_acp)
        if name and (line := _text_entry(row)):
            transcripts.setdefault(name, []).append(line)
    return transcripts | voice.transcripts


def _audit_name(via: object, tag: object, saved_acp: set[str]) -> str:
    if via == "acp" and isinstance(tag, str) and tag in saved_acp:
        return f"acp-{tag}"
    if via == "ask":
        return f"ask-{tag}" if isinstance(tag, str) and _SESSION.fullmatch(tag) else "ask"
    # Older untagged entries have no reliable session. Chat retention is its own archive.
    return "legacy" if via is None else ""


def _recordings(archive: ZipFile, folder: Path, source: str, prefix: str) -> int:
    try:
        directory = os.open(folder, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except FileNotFoundError:
        return 0
    try:
        return _copy_wavs(archive, folder, directory, source, prefix)
    finally:
        os.close(directory)


def _copy_wavs(archive: ZipFile, folder: Path, directory: int, source: str, prefix: str) -> int:
    copied = 0
    for name in sorted(os.listdir(directory)):
        if not name.startswith(f"{prefix}-") or _RECORDING.fullmatch(name) is None:
            continue
        try:
            flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
            fd = os.open(name, flags, dir_fd=directory)
        except FileNotFoundError:
            continue
        with os.fdopen(fd, "rb") as recording:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
                msg = f"{folder / name} must be a private regular file of your own"
                raise PermissionError(msg)
            with archive.open(f"audio/{source}/{name}", "w", force_zip64=True) as target:
                shutil.copyfileobj(recording, target)
        copied += 1
    return copied


def export(state: Path, destination: Path, *, input_recordings: Path | None = None) -> Export:
    """Save retained text and available WAVs without replacing a file or changing the sources.

    Raises:
        FileExistsError: The destination already exists.
        FileNotFoundError: The requested microphone recordings folder does not exist.
    """
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    if input_recordings is not None and not input_recordings.is_dir():
        msg = f"no microphone recordings folder at {input_recordings}"
        raise FileNotFoundError(msg)
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".jarvis-archive-") as temp:
        with ZipFile(temp, "w", compression=ZIP_DEFLATED) as archive:
            user = (
                _recordings(archive, input_recordings, "user", "utterance")
                if input_recordings is not None
                else 0
            )
            jarvis = _recordings(archive, state / "recordings", "jarvis", "output")
            transcripts = dict(_chat_transcripts(state / "chat-archive"))
            transcripts.update(_audit_transcripts(state, set(archive.namelist())))
            for name, lines in transcripts.items():
                archive.writestr(f"transcripts/{name}.txt", "".join(lines))
            archive.writestr(
                "README.txt",
                "Each transcript covers a saved chat, ACP session, ask invocation, "
                "or voice conversation.\n"
                "Voice conversations split on service startup or a memory reset.\n"
                "Older ask entries without a session tag are in ask.txt.\n"
                "Older log entries with no channel tag are in legacy.txt.\n"
                "Times are UTC when present. WAV names contain Unix timestamps in nanoseconds.\n"
                "WAVs are grouped by speaker. Transcripts link to WAVs named in the log.\n"
                "A Jarvis WAV can include output from earlier sessions.\n",
            )
        temp.flush()
        # A second exporter may finish first. Publish the complete file or leave its file alone.
        os.link(temp.name, destination, follow_symlinks=False)
    return Export(len(transcripts), user, jarvis)
