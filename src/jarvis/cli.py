"""Command-line entry point for `jarvis`."""

import argparse
import contextlib
import json
import logging
import math
import os
import random
import re
import secrets
import shutil
import signal
import subprocess
import sys
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from jarvis import __version__, config, control, models, modes, paths, services
from jarvis.activity import AuditTail
from jarvis.agent import Declined, Hooks, NoReason
from jarvis.app import (
    SetupError,
    build,
    chat_memory_file,
    chat_pictures,
    echoes_hint,
    spotify_operations,
    strata_model,
    system_prompt,
    voice_busy,
)
from jarvis.audit import AuditLog
from jarvis.chat import clipboard
from jarvis.chat.archive import ChatArchive
from jarvis.chat.history import InputHistory
from jarvis.confirmkey import ConfirmKey
from jarvis.files import write_atomic
from jarvis.operations import OperationError
from jarvis.paths import spotify_token_file, strata_key_file
from jarvis.pc.hearing import Listening, PcHearing
from jarvis.policy import OpClass, is_yes
from jarvis.schedule import JobStore
from jarvis.secretfile import ensure_api_key, read_api_key, read_private
from jarvis.services import JARVIS_UNIT, STRATA_UNIT, reads_pictures
from jarvis.spotify.auth import SCOPES, SpotifyAuthError, TokenStore, login, redirect_uri
from jarvis.sysinfo import Sampler
from jarvis.toolswitch import WORDS as TOOL_WORDS
from jarvis.toolswitch import ToolSetting, tool_settings, with_class
from jarvis.web import browser

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    import numpy as np
    from numpy.typing import NDArray
    from pydantic import BaseModel

    from jarvis.app import AgentModes
    from jarvis.audio import FallbackWake, Speaker, VoxtypeWatch, WakeWord
    from jarvis.chat.app import SharedSetting
    from jarvis.model import StrataModel
    from jarvis.voice import VoiceLoop

log = logging.getLogger(__name__)

WAKE_DIR = "wake"
WAKE_SUFFIX = ".onnx"
VAD_MODEL = "vad/silero_vad.onnx"
AZURE_PYTHON = "azure-wake/venv/bin/python"
AZURE_CHECK_S = 30.0
SYSTEMCTL_S = 120.0  # a start or stop; Strata takes up to 45 s to stop (TimeoutStopSec)
SYSTEMCTL_QUERY_S = 10.0  # is-active, show
WAKE_NAME = re.compile(r"\bjarvis\b", re.IGNORECASE)  # what an Azure model also wakes on


def _azure_files(cfg: config.Config) -> tuple[Path, Path]:
    """Return the Azure keyword model file and the Python that runs its helper."""
    model = Path(cfg.audio.azure_keyword_model).expanduser()
    python = (
        Path(cfg.audio.azure_python).expanduser()
        if cfg.audio.azure_python
        else cfg.data_dir / AZURE_PYTHON
    )
    return model, python


def wake_files(cfg: config.Config) -> list[Path]:
    """Return the model file of each wake word `audio.wake_words` turns on."""
    folder = cfg.data_dir / WAKE_DIR
    return [folder / f"{name}{WAKE_SUFFIX}" for name in cfg.audio.wake_words]


def livekit_wake(cfg: config.Config) -> WakeWord:
    """Return LiveKit's detector for the wake words that are on (it skips any unusable file)."""
    from jarvis.audio import WakeWord  # ruff: ignore[import-outside-top-level] - loads numpy and ONNX Runtime, which most commands never need

    return WakeWord(
        wake_files(cfg),
        cfg.audio.wake_threshold,
        cfg.audio.wake_thresholds,
        over_speech=cfg.audio.barge_in_threshold,
    )


def wake_word(cfg: config.Config) -> WakeWord | FallbackWake:
    """Return the wake word detector `audio.wake_engine` names.

    An Azure model that cannot start (no model file, no helper Python) falls back to LiveKit's,
    so a switch never leaves Jarvis deaf; `jarvis doctor` says why.
    """
    from jarvis.audio import (  # ruff: ignore[import-outside-top-level] - loads numpy and ONNX Runtime, which most commands never need
        AudioError,
        AzureWake,
        FallbackWake,
        azure_wake_command,
    )

    def livekit() -> WakeWord:
        return livekit_wake(cfg)

    if cfg.audio.wake_engine == "azure":
        model, python = _azure_files(cfg)
        if not cfg.audio.azure_keyword_model or not model.is_file():
            log.warning("no Azure keyword model at %r; using LiveKit's wake word", str(model))
        else:
            try:
                wake = AzureWake(azure_wake_command(python, model))
            except AudioError as exc:
                log.warning("the Azure wake word could not start (%s); using LiveKit's", exc)
            else:
                log.info("listening with the Azure keyword model %s", model.name)
                return FallbackWake(wake, livekit)
    return livekit()


def _app_names(value: str) -> list[str]:
    """Return the app names of a comma-separated `--apps` value.

    Raises:
        argparse.ArgumentTypeError: a name is empty.
    """
    names = [name.strip() for name in value.split(",")]
    if not all(names):
        msg = "name at least one app and no empty names (to hear all PC audio use --all-output)"
        raise argparse.ArgumentTypeError(msg)
    return names


def _wake_names(value: str) -> list[str]:
    """Return the model names of a comma-separated `--words` value.

    Raises:
        argparse.ArgumentTypeError: a name is empty.
    """
    names = [name.strip().removesuffix(WAKE_SUFFIX) for name in value.split(",")]
    if not all(names):
        msg = "name at least one wake word model and no empty names"
        raise argparse.ArgumentTypeError(msg)
    return names


# What `jarvis help` lists, by group, with each command's one-line summary.
COMMAND_GROUPS: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = (
    (
        "Everyday",
        (
            ("status", "show what Jarvis is doing and whether the model is loaded"),
            ("pause", "let go of the microphone and free the model until resume"),
            ("resume", "listen again after pause"),
            ("hush", "stop Jarvis speaking now"),
            ("confirm", "allow the action Jarvis is asking about (bind it to a key)"),
            ("chat", "type requests and watch what Jarvis does, live"),
            ("ask", "send one typed request, as if spoken"),
            ("model", "show the model in use, or switch mode: normal or coding"),
        ),
    ),
    (
        "Running Jarvis",
        (
            ("start", "start Jarvis: the voice service and its model server"),
            ("stop", "stop Jarvis and free the model"),
            ("restart", "restart Jarvis after a config change"),
            ("logs", "follow Jarvis's live log (Ctrl+C to leave)"),
            ("unload", "free the model's memory now"),
            ("doctor", "check every part without loading the model"),
        ),
    ),
    (
        "Settings",
        (
            ("wake", "show or switch the wake word model (livekit or azure)"),
            ("consent", "show or set whether Jarvis asks before acting"),
            ("tools", "list the tools, or switch one on, off or to asking first"),
            ("speed", "show or set how fast Jarvis speaks (1.0 is normal)"),
            ("pc-audio", "let Jarvis take requests from PC audio (Discord by default)"),
        ),
    ),
    (
        "First-time setup",
        (
            ("setup", "write the config, the Strata profile and the user units"),
            ("spotify-login", "connect Jarvis to your Spotify account"),
        ),
    ),
    (
        "Other",
        (
            ("help", "show this list"),
            ("run", "run the voice loop in this terminal (jarvis.service runs it)"),
            ("speech-test", "record your own voice and score how well it is heard"),
        ),
    ),
)
_NAME_WIDTH = 15
HELP_WIDTH = 80
# What a command's own help adds to its one-line summary.
_DETAILS = {
    "restart": (
        "Restarts the voice service. A stopped model server is started; a running one is "
        "restarted only when its settings changed since it started, so the model stays loaded "
        "otherwise."
    ),
    "consent": (
        "With no argument, shows the setting. A strong action always waits for the confirm "
        "key, and an operation set to deny is always refused."
    ),
    "wake": "With no argument and no option, shows the settings and the wake word files.",
    "model": (
        "With no argument, shows the mode Jarvis is in, each mode's model and whether it is "
        "loaded. Coding mode gives programming work more room and, when setup was given a "
        "coding run config, its own model: the model server then restarts, so only one model "
        "is in memory at a time."
    ),
    "tools": (
        "With no argument, lists every tool and its setting. A tool that is off is not offered "
        "to the model at all. The voice service restarts to take a change; an open chat takes "
        "it the next time it opens."
    ),
    "speech-test": (
        "record shows one card at a time and saves what you say, through Jarvis's own "
        "microphone and voice detection, with the words you confirm or type. score "
        "transcribes the set again with the current speech-to-text settings, and compare says "
        "whether two saved scores differ beyond chance. The set stays in its folder, "
        "readable by you only."
    ),
    "pc-audio": (
        "With no argument and no option, shows the settings. --apps and --all-output change "
        "which audio is heard without turning listening on or off."
    ),
}


def _commands_help() -> str:
    """Return the grouped list of commands `jarvis help` ends with."""
    lines = ["commands:"]
    for group, commands in COMMAND_GROUPS:
        lines.append(f"  {group}")
        lines.extend(f"    {name:{_NAME_WIDTH}} {summary}" for name, summary in commands)
        lines.append("")
    lines.extend(
        (
            "Run `jarvis <command> --help` for a command's options.",
            "Running `jarvis` alone shows what is running, then follows the live log while",
            "Jarvis runs and the output is a terminal.",
        )
    )
    return "\n".join(lines)


def _description(name: str, summary: str) -> str:
    """Return a command's own help text: its summary as a sentence, then any details."""
    text = summary[:1].upper() + summary[1:] + "."
    return f"{text} {_DETAILS[name]}" if name in _DETAILS else text


class _Formatter(argparse.HelpFormatter):
    """Help at most `HELP_WIDTH` columns wide, narrower on a narrow terminal."""

    def __init__(self, prog: str) -> None:
        width = min(HELP_WIDTH, shutil.get_terminal_size().columns)
        super().__init__(prog, width=width, max_help_position=30)


class _ListFormatter(argparse.RawDescriptionHelpFormatter, _Formatter):
    """The top-level help: the grouped list of commands keeps its own layout."""


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser for the `jarvis` command."""
    parser = argparse.ArgumentParser(
        prog="jarvis",
        usage="jarvis [<command> [options]]",
        description="Jarvis, a local voice agent for Omarchy.",
        epilog=_commands_help(),
        formatter_class=_ListFormatter,
        suggest_on_error=True,
        color=False,
    )
    parser.add_argument(
        "--version", action="version", version=f"jarvis {__version__}", help="show the version"
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>", help=argparse.SUPPRESS)
    made = {
        name: sub.add_parser(
            name,
            help=summary,
            description=_description(name, summary),
            formatter_class=_Formatter,
            color=False,
        )
        for _group, commands in COMMAND_GROUPS
        for name, summary in commands
    }

    setup = made["setup"]
    setup.add_argument(
        "--spotify-client-id", default=None, metavar="ID", help="your Spotify app's client ID"
    )
    setup.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        metavar="PATH",
        help="where wake, VAD and voice models live",
    )
    setup.add_argument(
        "--strata-home",
        type=Path,
        default=None,
        metavar="PATH",
        help="Strata's data folder (STRATA_DATA_DIR)",
    )
    setup.add_argument(
        "--strata-components-parent",
        type=Path,
        default=None,
        metavar="PATH",
        help="the folder holding Strata's components (its XDG_DATA_HOME)",
    )
    setup.add_argument(
        "--run-config",
        type=Path,
        default=None,
        metavar="PATH",
        help="Strata's run config for the model (the strata-<model>.json `strata launch` writes)",
    )
    setup.add_argument(
        "--coding-run-config",
        type=Path,
        default=None,
        metavar="PATH",
        help="Strata's run config for coding mode's own model (an empty value unsets it)",
    )
    setup.add_argument(
        "--mic",
        default=None,
        metavar="NODE",
        help="PipeWire source node name (empty: the default source)",
    )
    setup.add_argument("--enable", action="store_true", help="enable and start the user units")

    made["model"].add_argument(
        "mode", nargs="?", choices=modes.MODES, metavar="MODE", help="normal or coding"
    )
    made["spotify-login"].add_argument(
        "--no-browser",
        action="store_true",
        help="print the login link to open yourself instead of opening a browser; it still "
        "waits for the login",
    )

    test = made["speech-test"]
    test.add_argument("action", choices=("record", "score", "compare"), help="what to do")
    test.add_argument(
        "scores", nargs="*", type=Path, metavar="SCORE", help="compare: two files saved by score"
    )
    test.add_argument(
        "--dir",
        type=Path,
        default=None,
        metavar="PATH",
        help="the set's folder (default: speech-test in the data folder)",
    )
    test.add_argument(
        "--count", type=_positive, default=40, metavar="N", help="record: how many cards (40)"
    )
    test.add_argument(
        "--music", action="store_true", help="record: mark these as said with music playing"
    )
    test.add_argument(
        "--out", type=Path, default=None, metavar="FILE", help="score: save each request's result"
    )
    test.add_argument(
        "--engine",
        default=None,
        metavar="NAME",
        help="score: another Voxtype engine; give its model too (a .gguf file for parakeet, "
        "cohere or gguf)",
    )
    test.add_argument("--model", default=None, metavar="NAME", help="score: another model")
    test.add_argument("--no-hint", action="store_true", help="score: without the vocabulary hint")

    ask = made["ask"]
    ask.add_argument("text", nargs="+", metavar="words", help="the request, as you would say it")
    ask.add_argument(
        "--yes",
        action="store_true",
        help="answer yes to Jarvis's questions (default: ask here); it cannot press the "
        "confirm key a strong action waits for, and a browser click or typing is still asked here",
    )

    made["status"].add_argument(
        "--json", action="store_true", help="print one JSON object (for a status bar)"
    )
    wake = made["wake"]
    wake.add_argument(
        "engine", nargs="?", choices=["livekit", "azure"], help="the wake word engine to use"
    )
    wake.add_argument("--model", metavar="FILE", help="the Azure custom keyword .table file")
    wake.add_argument(
        "--words",
        type=_wake_names,
        metavar="NAME,...",
        help=(
            "LiveKit wake word models to listen for, by file name without .onnx, "
            "e.g. hey_jarvis,yo_jarvis"
        ),
    )
    made["consent"].add_argument(
        "mode",
        nargs="?",
        choices=["bypass", "ask"],
        help="ask: ask for a yes before a risky action or one after outside text; bypass: act "
        "without asking (a strong action still waits for the confirm key)",
    )
    made["speed"].add_argument(
        "value",
        nargs="?",
        type=_speech_speed,
        help=f"{SPEEDS}; 1 is the voice's own pace",
    )
    tool = made["tools"]
    tool.add_argument("name", nargs="?", help="the tool, as `jarvis tools` lists it")
    tool.add_argument(
        "setting",
        nargs="?",
        choices=[*TOOL_WORDS, "default"],
        help="on: runs; ask: asks first; key: needs the confirm key; off: not offered; "
        "default: as Jarvis ships it",
    )
    pc = made["pc-audio"]
    pc.add_argument("state", nargs="?", choices=["on", "off"], help="turn listening on or off")
    which = pc.add_mutually_exclusive_group()
    which.add_argument(
        "--apps",
        type=_app_names,
        metavar="A,B",
        help="apps to listen to, comma-separated, by process or app name (--apps discord,firefox)",
    )
    which.add_argument(
        "--all-output",
        action="store_true",
        help="listen to all PC audio, not only named apps (anything playing can speak to Jarvis)",
    )
    return parser


def _load() -> config.Config:
    try:
        return config.load()
    except config.ConfigError as exc:
        print(f"jarvis: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


def cmd_setup(args: argparse.Namespace) -> int:
    """Write config, key, Strata profile and units; keep every value already set unless given."""
    current = _load()
    spotify = current.spotify.model_copy(
        update={"client_id": args.spotify_client_id} if args.spotify_client_id is not None else {}
    )
    strata_updates = {
        key: value
        for key, value in {
            "home": args.strata_home,
            "components_parent": args.strata_components_parent,
            "run_config": args.run_config,
            "coding_run_config": args.coding_run_config,
        }.items()
        if value is not None
    }
    audio = current.audio.model_copy(update={"source": args.mic} if args.mic is not None else {})
    updated = current.model_copy(
        update={
            "spotify": spotify,
            "strata": current.strata.model_copy(update=strata_updates),
            "audio": audio,
            **({"data_dir": args.data_dir} if args.data_dir is not None else {}),
        }
    )
    try:
        updated = config.Config.model_validate(updated.model_dump())
    except ValueError as exc:
        print(f"jarvis: setup value refused: {exc}", file=sys.stderr)
        return 2
    if args.coding_run_config is not None and updated.strata.run_config != Path():
        try:  # a coding run config that cannot be served is refused before anything is saved
            services.strata_profile(modes.effective(updated, "coding"))
        except config.ConfigError as exc:
            print(f"jarvis: {exc}", file=sys.stderr)
            return 2
    config.save(updated)
    ensure_api_key(strata_key_file())
    if updated.strata.run_config == Path():
        print(f"config: {paths.config_file()}")
        print(
            "next: run `strata launch` for the model, then "
            "`jarvis setup --run-config <the strata-<model>.json it writes>`"
        )
        return 0
    try:
        services.strata_profile(
            modes.effective(updated, "coding")
        )  # a change of the usual one can break it too
        with modes.locked():  # not in the middle of a switch of modes, which writes them too
            served = modes.effective(updated, modes.current())  # a setup in coding mode keeps it
            profile = services.write_strata_profile(served)
            units = services.write_units(served, strata_key_file())
    except config.ConfigError as exc:
        print(f"jarvis: {exc}", file=sys.stderr)
        return 2
    print(f"config: {paths.config_file()}\nstrata profile: {profile}")
    print("units: " + ", ".join(str(u) for u in units))
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=False, timeout=SYSTEMCTL_S)
    if args.enable:
        units = [services.STRATA_UNIT, services.JARVIS_UNIT]
        # A unit stopped by its start limit refuses to start until its failed state is cleared.
        subprocess.run(
            ["systemctl", "--user", "reset-failed", *units], check=False, timeout=SYSTEMCTL_S
        )
        subprocess.run(
            ["systemctl", "--user", "enable", "--now", *units], check=False, timeout=SYSTEMCTL_S
        )
    if not updated.spotify.client_id:
        print("next: jarvis setup --spotify-client-id <your app's client ID>")
    elif read_private(spotify_token_file()) is None:
        print("next: jarvis spotify-login")
    return 0


def cmd_spotify_login(args: argparse.Namespace) -> int:
    """Run the PKCE login and store the tokens."""
    cfg = _load()
    if not cfg.spotify.client_id:
        print(
            "jarvis: set the client ID first: jarvis setup --spotify-client-id <id>",
            file=sys.stderr,
        )
        return 2

    def show(url: str) -> None:
        print(f"Redirect URI your app must list: {redirect_uri(cfg.spotify.redirect_port)}")
        print(f"Open this link and allow access:\n\n{url}\n")
        if not args.no_browser and shutil.which("xdg-open"):
            subprocess.Popen(
                ["xdg-open", url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )

    try:
        login(cfg.spotify.client_id, cfg.spotify.redirect_port, spotify_token_file(), show)
    except (SpotifyAuthError, OSError) as exc:
        print(f"jarvis: Spotify login failed: {exc}", file=sys.stderr)
        return 1
    print("Spotify connected.")
    return 0


def _terminal_confirm(question: str) -> bool | Declined:
    """Ask on the terminal; a typed `y` is a yes, as are the spoken ones (`is_yes`)."""
    try:
        answer = input(f"{question} [y/n] ")
    except EOFError:
        return Declined(NoReason.NO_ANSWER)  # no terminal to answer on

    return answer.strip().casefold() == "y" or is_yes(answer)


def cmd_ask(args: argparse.Namespace) -> int:
    """Answer one typed request and print the reply."""
    try:
        jarvis = build(_load(), via="ask", units=StrataUnits())
    except SetupError as exc:
        print(f"jarvis: {exc}", file=sys.stderr)
        return 2
    confirm: Callable[[str], bool | Declined] = (
        (lambda q: print(q, "-> yes") is None) if args.yes else _terminal_confirm
    )
    # `--yes` cannot answer for a click or typing in the browser, which always asks (ADR 0016).
    always = _terminal_confirm if args.yes else None
    reply = jarvis.agent.handle(
        " ".join(args.text),
        Hooks(confirm=confirm, confirm_always=always, waking=lambda: print("(waking up)")),
    )
    print(reply)
    return 0


def _chat_status(cfg: config.Config) -> str:
    voice = _unit_state(JARVIS_UNIT)
    model = _strata(cfg).status() if _unit_state(STRATA_UNIT) == "active" else None
    return f"voice {voice} · model {model or 'server not running'}"


def _chat_details(cfg: config.Config) -> list[tuple[str, str]]:
    """Return what /status adds to the services: versions, the model and where things live."""
    mode = modes.current()
    served = modes.effective(cfg, mode)
    chat = served.chat
    return [
        ("jarvis", __version__),
        ("mode", f"{mode} (/model switches)"),
        ("model", f"{services.model_name(served)}, {served.strata.context:,} tokens of context"),
        ("replies", f"up to {chat.reply_tokens:,} tokens, {chat.max_steps} model calls"),
        ("memory", f"{chat.memory_detail}, up to {chat.memory_tokens:,} tokens"),
        ("config", str(paths.config_file())),
        ("conversation", str(chat_memory_file())),
        ("archive", f"{chat_archive_dir()}, up to {cfg.chat.archive_keep} conversations"),
        ("audit log", str(paths.audit_file())),
    ]


@dataclass(frozen=True, slots=True)
class _Shared:
    """One of Jarvis's own settings the chat's /config shows beside its own."""

    section: str  # the config section it is in; dotted for a section inside one ("coding.chat")
    key: str
    label: str
    values: tuple[object, ...]
    heading: str  # the heading /config shows it under
    about: str
    risky: str = ""  # what turning it on means, said before it is turned on


SHARED_SETTINGS: tuple[_Shared, ...] = (
    _Shared(
        "audio",
        "speech_speed",
        "Speaking speed",
        (0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0),
        "Voice",
        "How fast Jarvis speaks its replies; 1.0 is the voice's own pace.",
    ),
    _Shared(
        "audio",
        "listen_to_pc",
        "Take requests from PC audio",
        (False, True),
        "Listening",
        "Jarvis also hears requests spoken in the PC's own audio, such as someone in a call or "
        "a video. Whatever they ask to be done always needs your yes.",
        "Jarvis will take requests from voices in the PC's audio, such as a call or a video; "
        "each of their actions still needs your yes.",
    ),
    _Shared(
        "audio",
        "pc_audio_recall_s",
        "Recall PC audio (seconds)",
        (0, 10, 30, 60),
        "Listening",
        "Keep the last seconds of PC audio in memory so Jarvis can hear what was just said "
        "(needs PC audio on); 0 keeps none.",
        "Jarvis will keep the last seconds of the PC's audio in memory.",
    ),
    _Shared(
        "browser",
        "idle_close_s",
        "Close the browser after (seconds)",
        (30.0, 60.0, 120.0, 300.0),
        "Chat",
        "How long the headless browser stays open after its last use, in the chat and by "
        "voice; then it closes with its profile and pages. The chat takes it when it opens again.",
    ),
    _Shared(
        "agent",
        "browser",
        "Headless browser by voice",
        (False, True),
        "Voice",
        "Spoken requests can open and read pages in the headless browser too; every click or "
        "typing there asks you first. Needs agent-browser.",
    ),
    _Shared(
        "boundaries",
        "bypass_consent",
        "Act without asking first",
        (False, True),
        "Safety",
        "On: Jarvis acts without asking for your yes, even after reading a web page or a file. "
        "A tool that needs the confirm key still waits for it, and a tool that is off stays off.",
        "Jarvis will act without asking you first, even when a web page or a file could have "
        "steered it.",
    ),
    _Shared(
        "coding",
        "back_after_s",
        "Leave coding mode after (seconds)",
        (0.0, 600.0, 1800.0, 3600.0, 14400.0),
        "Mode",
        "Jarvis goes back to normal mode after coding mode went unused this long; 0 stays in "
        "coding mode until you switch (/model normal).",
    ),
    _Shared(
        "coding.chat",
        "auto",
        "Switch mode by itself in the chat",
        (False, True),
        "Mode",
        "The model moves a request that is programming work to coding mode, and back when you "
        "say the work is over. Off: only /model and `jarvis model` switch. Takes effect when "
        "the chat next starts.",
    ),
    _Shared(
        "coding.voice",
        "auto",
        "Switch mode by itself for speech",
        (False, True),
        "Mode",
        "The same for spoken requests. With a coding model of its own, a switch loads another "
        "model, so the request that caused it waits in silence for the load.",
        "Spoken requests that are programming work will switch the mode by themselves, and "
        "Jarvis is silent while a separate coding model loads.",
    ),
)


def _section(cfg: BaseModel, dotted: str) -> BaseModel:
    """Return the config section a dotted name such as "coding.chat" names."""
    found = cfg
    for part in dotted.split("."):
        found = getattr(found, part)
    return found


def _with_setting(model: BaseModel, dotted: str, name: str, value: object) -> BaseModel:
    """Return `model` with `name` set to `value` in its section `dotted`, each level copied."""
    head, _, rest = dotted.partition(".")
    inner = getattr(model, head)
    changed = (
        _with_setting(inner, rest, name, value) if rest else inner.model_copy(update={name: value})
    )
    return model.model_copy(update={head: changed})


def _shared_settings() -> list[SharedSetting]:
    """Return the shared settings with their values in the config on disk."""
    from jarvis.chat.app import SharedSetting

    cfg = config.load()
    return [
        SharedSetting(
            f"{item.section}.{item.key}",
            item.label,
            item.values,
            getattr(_section(cfg, item.section), item.key),
            item.heading,
            item.about,
            item.risky,
        )
        for item in SHARED_SETTINGS
    ]


def _set_shared(key: str, value: object) -> None:
    """Write one shared setting from the chat into the config on disk and audit it.

    The voice service is restarted once, when the chat's settings close (`_apply_from_chat`).

    Raises:
        ConfigError: no shared setting has that key, or the value is not one of its choices.
    """
    found = next((s for s in SHARED_SETTINGS if f"{s.section}.{s.key}" == key), None)
    if found is None or value not in found.values:
        msg = f"the chat cannot set {key} to {value!r}"
        raise config.ConfigError(msg)
    cfg = config.load()
    was = getattr(_section(cfg, found.section), found.key)
    changed = _with_setting(cfg, found.section, found.key, value)
    config.save(config.Config.model_validate(changed.model_dump()))
    audit = AuditLog(paths.audit_file())
    if key == "boundaries.bypass_consent":  # recorded as `jarvis consent` records it
        audit.record("consent_changed", bypass_consent=value, source="chat")
    else:
        audit.record("setting_changed", setting=key, was=was, to=value, source="chat")


def _apply_from_chat() -> None:
    """Restart the voice service so it takes what the chat's settings changed.

    Raises:
        RestartError: the restart failed, so the voice service runs on the old settings.
    """
    if not _restart_if_running():
        raise RestartError(RESTART_FAILED)


def chat_archive_dir() -> Path:
    """Return where the chat keeps every conversation for /resume and reading back."""
    return paths.state_dir() / "chat-archive"


def _chat_archive(keep: int) -> ChatArchive | None:
    """Return the chat's archive, keeping `keep` conversations; None when it keeps none.

    With 0, the conversations an earlier setting kept are deleted, as a lower cap deletes the
    oldest past it.
    """
    archive = ChatArchive(chat_archive_dir(), keep)
    if keep:
        return archive
    try:
        gone = archive.forget()
    except OSError as exc:
        log.warning("the chat archive could not be deleted: %s", exc)
    else:
        if gone:
            log.info("deleted %d archived chat conversations: chat.archive_keep is 0", gone)
    return None


def _save_chat_settings(chat: config.ChatConfig, followed: AgentModes | None = None) -> None:
    """Write the chat's settings into the config file as it is now on disk, and audit them.

    `followed`, the running chat's modes, takes them too, so a change of mode keeps them.
    """
    if followed is not None:
        followed.restyle(chat)
    current = config.load()
    config.save(current.model_copy(update={"chat": chat}))
    was, now = current.chat.model_dump(), chat.model_dump()
    audit = AuditLog(paths.audit_file())
    for key in sorted(k for k in now if now[k] != was.get(k)):
        audit.record(
            "setting_changed", setting=f"chat.{key}", was=was.get(key), to=now[key], source="chat"
        )


def _screenshot_folders(home: Path, env: Mapping[str, str]) -> tuple[Path, ...]:
    """Return where Omarchy saves screenshots (its setting first), then the Pictures folder."""
    chosen = [env.get(name, "") for name in ("OMASNAP_SCREENSHOT_DIR", "OMARCHY_SCREENSHOT_DIR")]
    folders = [Path(value).expanduser() for value in chosen if value]
    folders += [home / "Pictures" / "Screenshots", home / "Pictures"]
    return tuple(dict.fromkeys(folders))


def _chat_prompt(chat: config.ChatConfig) -> str:
    """Render the chat's instructions for these settings, in the mode Jarvis is in now."""
    mode = modes.current()
    cfg = modes.effective(config.load().model_copy(update={"chat": chat}), mode)
    return system_prompt(cfg, "chat", coding=mode == "coding")


def cmd_chat(_args: argparse.Namespace) -> int:
    """Open the chat window: typed requests, and everything Jarvis does, live."""
    from jarvis.chat.app import ChatApp, Services

    cfg = _load()
    session_id = secrets.token_hex(8)
    agent = None
    followed = None
    unavailable = ""
    try:
        built = build(cfg, via="chat", session=session_id, units=StrataUnits())
        agent, followed = built.agent, built.modes
    except SetupError as exc:
        unavailable = str(exc)
    documents = Path.home() / "Documents"
    services = Services(
        status=lambda: _chat_status(cfg),
        details=lambda: _chat_details(cfg),
        stats=Sampler().sample,
        save_settings=lambda chat: _save_chat_settings(chat, followed),
        system_prompt=_chat_prompt,
        tools=lambda: _tool_settings(config.load()),
        next_tool_setting=_next_tool_setting,
        # The config as it is now: `jarvis setup` or /config may have changed coding mode since.
        model_report=lambda: model_report(config.load()),
        switch_model=lambda to, say: switch_model(config.load(), to, say, "chat"),
        shared=_shared_settings,
        set_shared=_set_shared,
        apply=_apply_from_chat,
        history=InputHistory(paths.state_dir() / "chat-history", cfg.chat.history_size),
        archive=_chat_archive(cfg.chat.archive_keep),
        export_dir=documents if documents.is_dir() else Path.home(),
        paste=clipboard.read,
        pictures=reads_pictures(cfg),
        screenshots=_screenshot_folders(Path.home(), os.environ),
        shown=chat_pictures(),
        twelve_hour=cfg.clock.twelve_hour,
    )
    ChatApp(
        AuditTail(paths.audit_file()),
        agent,
        unavailable=unavailable,
        name=cfg.persona.name,
        settings=cfg.chat,
        services=services,
        session_id=session_id,
    ).run()
    return 0


type _Transcriber = Callable[[NDArray[np.int16]], str]


def _pc_listening(
    cfg: config.Config, answer: _Transcriber, *, paused: Callable[[], bool]
) -> Listening | None:
    """Return how the voice service hears PC audio for `pc_audio_listen`; None while it is off.

    It reads with the unhinted speech-to-text: the vocabulary hint names the owner's tools and
    music, which another person's words in a call need not hold. A say-nothing hallucination
    (`is_phantom`) counts as no speech. A listen ends once `paused` is true.
    """
    if not cfg.audio.listen_to_pc:
        return None
    import numpy as np  # ruff: ignore[import-outside-top-level] - loaded only by the voice service

    from jarvis.audio import AudioError, is_phantom  # ruff: ignore[import-outside-top-level] - loads numpy and ONNX Runtime, which most commands never need

    def heard(frames: Sequence[NDArray[np.int16]]) -> str:
        try:
            text = answer(np.concatenate(frames))
        except AudioError as exc:
            msg = f"speech-to-text failed: {exc}"
            raise OperationError(msg) from exc
        return "" if is_phantom(text) else text

    return Listening(PcHearing(cfg.audio.pc_audio_recall_s, paused=paused), heard)


def _transcribers(
    cfg: config.Config, hint: Callable[[], str], settings_file: str = "voxtype.toml"
) -> tuple[_Transcriber, _Transcriber]:
    """Return the request transcriber (with the vocabulary hint) and the yes/no one (without).

    Both run Voxtype with Jarvis's own settings, written here to `settings_file` in Jarvis's
    config folder; the owner's Voxtype config is never touched. Voxtype rereads the file for
    each run, so a command trying another engine beside a running Jarvis writes its own.
    """
    from jarvis.audio import transcribe, voxtype_settings  # ruff: ignore[import-outside-top-level] - loads numpy and ONNX Runtime, which most commands never need

    audio = cfg.audio
    engine, model = audio.transcribe_engine, audio.transcribe_model
    settings = paths.config_dir() / settings_file
    write_atomic(
        settings,
        voxtype_settings(
            audio.transcribe_threads,
            short_window=audio.transcribe_short_window,
            engine=engine,
            model=model,
            backend=audio.transcribe_backend,
        ),
    )
    # Only Whisper reads `--initial-prompt`; another engine would ignore the hint.
    hinted = audio.vocabulary_hint and engine in {"", "whisper"}

    def request(samples: NDArray[np.int16]) -> str:
        primed = hint() if hinted else ""
        text = transcribe(samples, engine, model, primed, settings=settings)
        if primed and echoes_hint(text, primed):
            log.info("ignored a transcript that only repeats the vocabulary hint: %r", text)
            return ""
        return text

    def answer(samples: NDArray[np.int16]) -> str:
        return transcribe(samples, engine, model, "", settings=settings)

    return request, answer


def _never() -> bool:
    return False


def _speaker(cfg: config.Config) -> Speaker:
    """Return the speaker, which hears its own wake word only when barge-in is on."""
    from jarvis.audio import (  # ruff: ignore[import-outside-top-level] - loads numpy and ONNX Runtime, which most commands never need
        OwnVoiceCheck,
        Speaker,
        make_synthesizer,
    )

    data = cfg.data_dir
    own_wake = None
    if cfg.audio.barge_in:
        own_wake = OwnVoiceCheck(
            livekit_wake(cfg),
            WAKE_NAME if cfg.audio.wake_engine == "azure" else None,
        )
    synthesizer = make_synthesizer(
        data, cfg.audio.voice, cfg.audio.speech_speed, phoneme_map=cfg.audio.kokoro_phoneme_map
    )
    return Speaker(synthesizer, own_wake=own_wake)


def _serve_controls(handler: control.Controls, audit: AuditLog) -> control.ControlServer:
    """Start the control socket; a run without one still listens, so a failure is only logged."""
    controls = control.ControlServer(handler, audit=audit)
    try:
        controls.start()
    except OSError as exc:
        log.warning("no control socket (status, pause and resume will not reach this run): %s", exc)
    return controls


def cmd_run(_args: argparse.Namespace) -> int:  # ruff: ignore[too-many-locals] - the voice service's parts, wired together here
    """Run the voice loop until stopped."""
    from jarvis.audio import (  # ruff: ignore[import-outside-top-level] - loads numpy and ONNX Runtime, which most commands never need
        MIC_STALL_S,
        AppAudio,
        FileWarmer,
        Microphone,
        PausableMic,
        Vad,
        VoxtypeWatch,
        audio_playing_elsewhere,
        screen_locked,
    )
    from jarvis.voice import PcListener, VoiceLoop  # ruff: ignore[import-outside-top-level] - imports jarvis.audio, which loads numpy and ONNX Runtime

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = _load()
    # The hint is the agent's, built after the transcribers that `pc_audio_listen` needs; it is
    # first asked for once a request is heard.
    request, answer = _transcribers(cfg, lambda: jarvis.hint())  # ruff: ignore[unnecessary-lambda] - `jarvis` is built below
    mic = PausableMic(Microphone(cfg.audio.source, stall_s=MIC_STALL_S))
    listening = _pc_listening(cfg, answer, paused=lambda: mic.paused)
    try:
        jarvis = build(cfg, units=StrataUnits(), listening=listening)
    except SetupError as exc:
        print(f"jarvis: {exc}", file=sys.stderr)
        return 2
    speaker = _speaker(cfg)
    pc_audio = AppAudio(cfg.audio.pc_audio_apps) if cfg.audio.listen_to_pc else None
    stop_requested = threading.Event()
    refresh_stop = threading.Event()

    def stop(_signum: int, _frame: object) -> None:
        # A signal handler interrupts whatever the main thread holds (the speaker's lock, say),
        # so it only raises the flag; the stopper thread does the stopping.
        stop_requested.set()

    def stopper() -> None:
        stop_requested.wait()
        refresh_stop.set()
        speaker.stop()
        mic.stop()
        if pc_audio is not None:
            pc_audio.stop()
        if listening is not None:
            listening.hearing.stop()

    threading.Thread(target=stopper, name="jarvis-stopper", daemon=True).start()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    # Off the request path: the Spotify login and library (for the hint and matching) are kept
    # warm, so the first request does not wait for them.
    jarvis.ops.start_refresh(refresh_stop)
    speech_model = FileWarmer(_speech_model_file(cfg.audio))
    mic.start()
    wakes = [wake_word(cfg), *([wake_word(cfg)] if pc_audio is not None else [])]

    def after_sleep() -> None:
        handler.after_sleep()  # made below, once the loop it controls exists

    def prewarm() -> None:
        speech_model.start()  # read from the disk while the request is still being said
        handler.prewarm()  # made below, as above

    keys = ConfirmKey(audit=jarvis.audit)  # pressed through the socket, waited on by the loop

    voxtype = VoxtypeWatch()  # started with the loop, so nothing before can leave it running
    loop = VoiceLoop(
        jarvis.agent,
        mic,
        wakes[0],
        Vad(cfg.data_dir / VAD_MODEL),
        speaker,
        cfg.audio,
        request,
        prewarm=prewarm,
        pc=None
        if pc_audio is None
        else PcListener(
            pc_audio,
            wakes[-1],
            Vad(cfg.data_dir / VAD_MODEL),
            None if listening is None else listening.hearing,
        ),
        transcribe_answer=answer,
        names=(cfg.persona.name,),
        dictating=voxtype.dictating,
        meeting=voxtype.meeting,
        media_playing=audio_playing_elsewhere,
        audit=jarvis.audit,
        paused=lambda: mic.paused,
        until_resumed=mic.wait_resumed,
        reopen=mic.recover,
        locked=_never if cfg.audio.listen_when_locked else screen_locked,
        after_sleep=after_sleep,
        keys=keys,
        due=JobStore(paths.schedule_file()).take_due,
    )
    handler = control.Controls(
        loop, mic, jarvis.model, jarvis.audit, keys=keys, prime=jarvis.agent.prime
    )
    handler.warm_at_start()  # the model loads now, so the first request does not wait for it
    if jarvis.switcher is not None:
        threading.Thread(
            target=_back_when_idle,
            args=(jarvis.switcher, loop, refresh_stop),
            name="jarvis-mode-idle",
            daemon=True,
        ).start()
    speech_model.start()
    jarvis.audit.record("service_started", bypass_consent=cfg.boundaries.bypass_consent)
    return _run_until_stopped(
        loop, stop_requested, _serve_controls(handler, jarvis.audit), voxtype, wakes
    )


MODE_CHECK_S = 60.0  # how often the voice service looks whether coding mode idled too long


def _back_when_idle(switcher: modes.Switcher, loop: VoiceLoop, stop: threading.Event) -> None:
    """Return to the normal mode once coding mode went unused too long, between requests.

    The thread sleeps between looks, and a look reads one small file.
    """
    while not stop.wait(MODE_CHECK_S):
        if str(loop.state) == "idle":
            switcher.back_if_idle()


def _run_until_stopped(
    loop: VoiceLoop,
    stop_requested: threading.Event,
    controls: control.ControlServer,
    voxtype: VoxtypeWatch,
    wakes: list[WakeWord | FallbackWake],
) -> int:
    """Run `loop` with Voxtype's status followed, then end what the run started."""
    from jarvis.audio import FallbackWake  # ruff: ignore[import-outside-top-level] - loads numpy and ONNX Runtime, which most commands never need

    try:
        voxtype.start()
        loop.run(stop_requested.is_set)
    except Exception:
        if stop_requested.is_set():
            return 0
        raise
    finally:
        controls.close()
        voxtype.stop()
        for wake in wakes:  # an Azure helper ends with its input; end it now, not at exit
            if isinstance(wake, FallbackWake):
                wake.close()
    return 0


def _model_files(cfg: config.Config) -> list[tuple[str, Path]]:
    """Return the (role, file) pairs the doctor can check against a known release."""
    from jarvis.audio import KOKORO_PREFIX, voice_files  # ruff: ignore[import-outside-top-level] - loads numpy and ONNX Runtime, which most commands never need

    found = [*(("wake", path) for path in wake_files(cfg)), ("vad", cfg.data_dir / VAD_MODEL)]
    if cfg.audio.voice.startswith(KOKORO_PREFIX):
        model, voices = voice_files(cfg.data_dir, cfg.audio.voice)
        found += [("kokoro-model", model), ("kokoro-voices", voices)]
    with contextlib.suppress(services.StrataSetupError):  # `_model_checks` reports it
        reader = services.read_run_config(cfg).picture_reader
        if cfg.strata.pictures and reader is not None:
            found.append(("vision", reader))
    return found


def _release_lines(cfg: config.Config) -> tuple[list[str], list[tuple[bool, str, str]]]:
    """Hash the model files that exist; return the report lines and any unreadable-file failures."""
    lines: list[str] = []
    failures: list[tuple[bool, str, str]] = []
    for role, path in _model_files(cfg):
        if not path.is_file():
            continue
        try:
            verdict = models.verify(role, path)
        except OSError as exc:
            failures.append(
                (False, f"read {path} ({exc.strerror or exc})", "check its permissions")
            )
            continue
        known = verdict.known
        if known is None:
            lines.append(f"note {path.name}: not a known release (sha256 {verdict.digest})")
        else:
            release = f"{known.project} {known.version} ({known.licence})"
            lines.append(f"{'ok':4} {path.name}: {release}, sha256 verified")
    return lines, failures


def _model_checks(cfg: config.Config) -> list[tuple[bool, str, str]]:
    """Return the checks that the files Strata loads are there (and its run config usable)."""
    try:
        run = services.read_run_config(cfg)
    except services.StrataSetupError as exc:
        return [(False, f"Strata run config {cfg.strata.run_config}", str(exc))]
    others = [f for f in services.missing_files(run) if Path(f) != run.model]
    checks = [
        (run.model.is_file(), f"model {run.model}", "mount its drive"),
        (
            not others,
            f"files the run config names{': ' + ', '.join(others) + ' missing' if others else ''}",
            "run `strata launch` for the model again, then jarvis setup",
        ),
    ]
    if services.learns_experts(cfg, run):
        learned = services.expert_profile_file(cfg)
        usable = learned.is_file() and learned.stat().st_size > 0
        checks.append((usable, f"expert profile {learned}", "run jarvis setup"))
    if services.fills_the_card(run) and not cfg.strata.vram_reserve_mib:
        checks.append(
            (False, "VRAM left for games", "set strata.vram_reserve_mib (docs/models.md)")
        )
    return checks


def _coding_checks(cfg: config.Config) -> list[tuple[bool, str, str]]:
    """Return the model checks for coding mode's own run config, when one is set."""
    if cfg.strata.coding_run_config == Path():
        return []
    try:
        services.strata_profile(modes.effective(cfg, "coding"))
    except config.ConfigError as exc:
        usable = [(False, "coding mode: a profile the model server can run", str(exc))]
    else:
        usable = []
    checks = _model_checks(modes.effective(cfg, "coding"))
    fixed = "jarvis setup --coding-run-config <its strata-<model>.json>"
    return [
        *(
            (ok, f"coding mode: {what}", fix.replace("jarvis setup", fixed))
            for ok, what, fix in checks
        ),
        *usable,
    ]


def _speech_model_file(audio: config.AudioConfig) -> Path | None:
    """Return the file Voxtype reads the speech model from, or None when Jarvis cannot tell.

    A Parakeet, Cohere or GGUF model is the file the config names. A Whisper model is that too, or
    the file Voxtype downloaded for its name into its data folder, which Jarvis reads and never
    writes. Other engines keep their models in their own forms.
    """
    from jarvis.audio import FILE_ENGINES  # ruff: ignore[import-outside-top-level] - loads numpy and ONNX Runtime, which most commands never need

    engine, model = audio.transcribe_engine, audio.transcribe_model
    whisper = engine in {"", "whisper"}
    if not model or not (whisper or engine in FILE_ENGINES):
        return None
    if model.startswith("/"):
        return Path(model)
    return paths.xdg_data_home() / "voxtype" / "models" / f"ggml-{model}.bin" if whisper else None


def _whisper_model(cfg: config.Config) -> list[tuple[bool, str, str]]:
    """Return the check that the speech model Jarvis asks Voxtype for is there.

    That is a downloaded Whisper model, or the file a Parakeet, Cohere or GGUF engine is given;
    other engines keep their models in their own forms, so Voxtype itself reports on those.
    """
    from jarvis import audio as audio_module  # ruff: ignore[import-outside-top-level] - loads numpy and ONNX Runtime, which most commands never need

    audio = cfg.audio
    name = audio.transcribe_model
    if audio_module.model_in_settings(audio.transcribe_engine, name):
        fix = "download it, or point audio.transcribe_model at the model file"
        return [(Path(name).is_file(), f"speech model {name}", fix)]
    if audio.transcribe_engine not in {"", "whisper"} or not name:
        return []
    fix = "download it with `voxtype setup model`"
    installed = None if name.startswith("/") else audio_module.installed_models("whisper")
    if installed is not None:
        found = name in installed
        said = "installed" if found else "not installed"
        return [(found, f"speech model {name} (Voxtype reports it {said})", fix)]
    model = _speech_model_file(audio)
    return [(model is not None and model.is_file(), f"speech model {model}", fix)]


def _model_url(cfg: config.Config) -> list[tuple[bool, str, str]]:
    """Return the check that a saved model URL still points at Strata's port.

    A URL saved by an older version keeps the port it was saved with; after `strata.port` is
    changed by hand, Jarvis would talk to the old port.
    """
    url = cfg.model.base_url
    if not url or urlsplit(url).port == cfg.strata.port:
        return []
    return [
        (
            False,
            f"model.base_url {url} is not on Strata's port {cfg.strata.port}",
            "remove model.base_url from the config, or set it to Strata's port",
        )
    ]


RENEW_WARNING_DAYS = 14


def _spotify_login(cfg: config.Config) -> tuple[bool, str, str]:
    """Return the doctor check for the Spotify login, saying how long Spotify still honours it.

    Spotify ends a login six months after it was granted; refreshing does not extend it.
    """
    fix = "jarvis spotify-login"
    tokens = TokenStore(cfg.spotify.client_id, spotify_token_file())
    if not tokens.has_login():
        return False, "Spotify login", fix
    days = tokens.days_left()
    if days is None:
        return True, "Spotify login (expiry unknown: logged in before Jarvis kept the date)", fix
    if days <= 0:
        return False, "Spotify login expired", fix
    soon = f"; renew soon with `{fix}`" if days <= RENEW_WARNING_DAYS else ""
    return True, f"Spotify login ({days} days left{soon})", fix


def _spotify_scope_notes(cfg: config.Config) -> list[str]:
    """Note a stored Spotify login that lacks permissions Jarvis now asks for (all else works)."""
    granted = TokenStore(cfg.spotify.client_id, spotify_token_file()).granted()
    if granted is None:
        return []
    missing = sorted(set(SCOPES) - granted)
    if not missing:
        return []
    note = (
        f"note Spotify login lacks {', '.join(missing)}: run `jarvis spotify-login` once to "
        "allow what needs them"
    )
    return [note]


def _tool_checks(cfg: config.Config) -> tuple[list[tuple[bool, str, str]], list[str]]:
    """Return the checks for the programs Jarvis runs, and notes for the optional ones."""
    needed = ["pw-record", "pw-play", "wpctl", "voxtype"]
    if cfg.audio.listen_to_pc or not cfg.audio.follow_up_over_audio:
        needed.append(
            "pw-dump"
        )  # who is playing: PC audio's source, and follow-ups shut over audio
    checks = [(shutil.which(t) is not None, f"tool {t}", f"install {t}") for t in needed]
    notes: list[str] = []
    if not shutil.which("playerctl"):
        notes.append(
            "note tool playerctl missing: other media players can't be paused or skipped "
            "(`pacman -S playerctl`)"
        )
    if cfg.chat.browser or cfg.agent.browser:
        if not shutil.which(browser.PROGRAM):
            notes.append(
                f"note tool {browser.PROGRAM} missing: the headless browser tools are not offered "
                "(see docs/configuration.md, `[browser]`)"
            )
        elif not browser.chrome_found(Path.home()):
            notes.append(
                f"note tool {browser.PROGRAM} has no Chrome to drive: its tools fail until "
                f"`{browser.PROGRAM} install` downloads one"
            )
    return checks, notes


def _user_memlock_limit() -> str:
    """Return the hard limit on locked memory of this user's service manager, or "" if unknown."""
    try:
        done = subprocess.run(
            ["systemctl", "show", f"user@{os.getuid()}.service", "-p", "LimitMEMLOCK", "--value"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except OSError, subprocess.TimeoutExpired:
        return ""
    return done.stdout.strip() if done.returncode == 0 else ""


def _model_bytes(cfg: config.Config) -> float:
    """Return the size of the model file the run config names; infinite when it is not known."""
    try:
        model = services.read_run_config(cfg).model
        return model.stat().st_size if model != Path() else float("inf")
    except services.StrataSetupError, OSError:
        return float("inf")


def _memlock_notes(cfg: config.Config) -> list[str]:
    """Note that the model cannot be locked in RAM when the user manager's limit keeps it out."""
    limit = _user_memlock_limit()
    if not cfg.strata.keep_in_ram or limit in {"", "infinity"}:
        return []
    if limit.isdigit() and int(limit) >= _model_bytes(cfg):
        return []  # the whole model fits under the limit
    note = (
        f"note the model can't be locked in RAM: your user services may lock {limit} bytes at "
        "most (it still stays out of swap); see "
        "docs/configuration.md#keeping-the-model-in-ram to raise it"
    )
    return [note]


def cmd_doctor(_args: argparse.Namespace) -> int:
    """Print one line per check with its fix; never loads the model.

    Model files found on disk are also hashed and named by the release they match; a file that
    matches no known release is only a note.
    """
    cfg = _load()
    from jarvis.audio import voice_files  # ruff: ignore[import-outside-top-level] - loads numpy and ONNX Runtime, which most commands never need

    wake_checks, wake_notes = _wake_checks(cfg)
    tool_checks, tool_notes = _tool_checks(cfg)
    files = [
        cfg.data_dir / VAD_MODEL,
        *voice_files(cfg.data_dir, cfg.audio.voice),
    ]
    state = _strata(cfg).status() if read_api_key(strata_key_file()) else None
    release_lines, unreadable = _release_lines(cfg)
    checks: list[tuple[bool, str, str]] = [
        (paths.config_file().exists(), f"config {paths.config_file()}", "run jarvis setup"),
        (bool(cfg.spotify.client_id), "Spotify client ID", "jarvis setup --spotify-client-id <id>"),
        _spotify_login(cfg),
        *wake_checks,
        *((f.exists(), f"file {f}", "see docs/models.md") for f in files),
        *unreadable,
        *_whisper_model(cfg),
        *_model_url(cfg),
        *_model_checks(cfg),
        *_coding_checks(cfg),
        *tool_checks,
        *_source_checks(cfg),
        *_azure_checks(cfg),
        *_lock_checks(cfg),
        (
            state is not None,
            f"Strata server ({state or 'unreachable'})",
            "jarvis start (it clears a start limit first)",
        ),
    ]
    for ok, what, fix in checks:
        print(f"{'ok' if ok else 'FAIL':4} {what}" + ("" if ok else f"  -> {fix}"))
    notes = [
        *wake_notes,
        *tool_notes,
        *_spotify_scope_notes(cfg),
        *_memlock_notes(cfg),
    ]
    for line in [*notes, *release_lines]:
        print(line)
    failed = sum(not ok for ok, _, _ in checks)
    verified = sum(line.startswith("ok ") for line in release_lines)
    print(f"{len(checks) + verified} checks, {failed} failed")
    return 0 if not failed else 1


def _wake_checks(cfg: config.Config) -> tuple[list[tuple[bool, str, str]], list[str]]:
    """Return the checks on the wake word files that are on, and notes for the rest.

    Jarvis listens with the models it finds, so only having none fails; a phrase that is on but
    not trained yet is a note. A file that is there but does not load, as the runtime would load
    it, fails too, and so does a note for each `wake_thresholds` name that is not on.
    """
    from jarvis.audio import wake_load_errors  # ruff: ignore[import-outside-top-level] - loads numpy and ONNX Runtime, which most commands never need

    files = wake_files(cfg)
    folder = cfg.data_dir / WAKE_DIR
    found = [path for path in files if path.is_file()]
    names = ", ".join(path.stem for path in found)
    check = (
        bool(found),
        f"wake word files ({names})" if found else f"wake word files: none found in {folder}",
        "see docs/models.md",
    )
    checks: list[tuple[bool, str, str]] = [check]
    if found:
        checks.extend(
            (False, f"wake word {path.stem} does not load: {reason}", "see docs/models.md")
            for path, reason in wake_load_errors(found).items()
        )
    notes = [
        f"note: wake word {path.stem} is on but has no file {path}; train it (docs/models.md) "
        "or drop it with `jarvis wake --words`"
        for path in files
        if path not in found
    ]
    notes.extend(
        f"note: wake_thresholds names {name}, which is not in wake_words, so it does nothing; "
        "check the spelling or turn it on with `jarvis wake --words`"
        for name in cfg.audio.wake_thresholds
        if name not in cfg.audio.wake_words
    )
    return checks, notes


def _source_checks(cfg: config.Config) -> list[tuple[bool, str, str]]:
    """Return the check that a configured `audio.source` is a microphone PipeWire lists.

    A recorder is not pinned to its source, so PipeWire may move it to the default microphone
    instead of failing; the check is what tells a mistyped name from a working one. With no
    source set, the default microphone is used and there is nothing to check.
    """
    from jarvis.audio import pw_dump, source_found  # ruff: ignore[import-outside-top-level] - loads numpy and ONNX Runtime, which most commands never need

    source = cfg.audio.source
    if not source:
        return []
    graph = pw_dump()
    if not graph:  # a running PipeWire always lists its own nodes
        return [
            (
                False,
                f"audio.source {source} not checked: PipeWire's graph (pw-dump) could not be read",
                "check that PipeWire runs: `systemctl --user status pipewire`",
            )
        ]
    found = source_found(graph, source)
    return [
        (
            found,
            f"audio.source {source}" + ("" if found else " matches no microphone in pw-dump"),
            "list microphones with `pw-dump | grep node.name`, or clear audio.source",
        )
    ]


def _lock_checks(cfg: config.Config) -> list[tuple[bool, str, str]]:
    """Return the check that the lock screen can be asked about, when Jarvis must not listen there.

    With `audio.listen_when_locked` off, a lock state that cannot be read keeps Jarvis deaf.
    """
    from jarvis.audio import lock_state  # ruff: ignore[import-outside-top-level] - loads numpy and ONNX Runtime, which most commands never need

    if cfg.audio.listen_when_locked:
        return []
    state = lock_state()
    shown = "unknown" if state is None else ("locked" if state else "unlocked")
    fix = "run Jarvis in the Omarchy session, or turn audio.listen_when_locked on"
    return [(state is not None, f"lock screen state ({shown})", fix)]


def _azure_checks(cfg: config.Config) -> list[tuple[bool, str, str]]:
    """Return the checks for the Azure wake word when `audio.wake_engine` is "azure".

    The last one runs the helper itself, with no network, to try the model as Jarvis would.
    """
    from jarvis.audio import azure_env, azure_wake_command  # ruff: ignore[import-outside-top-level] - loads numpy and ONNX Runtime, which most commands never need

    if cfg.audio.wake_engine != "azure":
        return []
    model, python = _azure_files(cfg)
    have_model = bool(cfg.audio.azure_keyword_model) and model.is_file()
    checks = [
        (
            have_model,
            f"Azure keyword model {cfg.audio.azure_keyword_model or '(not set)'}",
            "jarvis wake azure --model <file>.table",
        ),
        (
            python.is_file(),
            f"Azure wake Python {python}",
            "create it as docs/configuration.md says (Azure wake word)",
        ),
    ]
    if not (have_model and python.is_file()):
        return checks
    try:
        done = subprocess.run(
            azure_wake_command(python, model, check=True),
            capture_output=True,
            text=True,
            env=azure_env(),
            timeout=AZURE_CHECK_S,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        reason = str(exc)
    else:
        reason = "" if done.returncode == 0 else (done.stderr.strip().splitlines() or ["?"])[0]
    checks.append(
        (
            not reason,
            "the Azure model loads in its helper, with no network"
            + (f" ({reason[:160]})" if reason else ""),
            (
                "a namespace needs user.max_user_namespaces above 0; a rejected model needs "
                "a fresh download from Speech Studio"
            ),
        )
    )
    return checks


def _strata(cfg: config.Config) -> StrataModel:
    return strata_model(cfg, read_api_key(strata_key_file()) or "")


class StrataUnits:
    """The model server's unit, for a switch of modes (`modes.Units`)."""

    @staticmethod
    def active() -> bool:
        """Return whether the model server runs."""
        return _unit_state(STRATA_UNIT) in {"active", "activating", "reloading"}

    @staticmethod
    def restart() -> bool:
        """Reload the units and restart the model server; return whether systemd took it."""
        _systemctl("daemon-reload")
        _reset_failed(STRATA_UNIT)
        return _systemctl("restart", STRATA_UNIT) == 0


def _switcher(cfg: config.Config, source: str) -> modes.Switcher:
    """Return the switch of modes for a command or a window, audited as coming from `source`."""
    audit = AuditLog(paths.audit_file(), tags={"via": source})
    return modes.Switcher(cfg, StrataUnits(), _strata, audit, busy=voice_busy)


def model_report(cfg: config.Config) -> list[str]:
    """Return what `jarvis model` and the chat's /model show: the mode and each mode's model."""
    state = modes.read_state()
    running = _unit_state(STRATA_UNIT) == "active"
    loaded = (_strata(cfg).status() if running else None) or "server not running"
    lines = [f"mode: {state.mode} (model {loaded})"]
    for mode in modes.MODES:
        served = modes.effective(cfg, mode)
        mark = "*" if mode == state.mode else " "
        lines.append(
            f"{mark} {mode}: {services.model_name(served)}, {served.strata.context:,} tokens of "
            "context"
        )
    if not modes.own_model(cfg):
        lines.append(
            "coding mode has no model of its own (strata.coding_run_config is unset): it "
            "changes the limits, the context and the prompt"
        )
    after = cfg.coding.back_after_s
    if state.mode == "coding" and after:
        lines.append(f"back to normal after {after / 60:.0f} minutes unused")
    lines.append("switch with: model normal | model coding")
    return lines


def switch_model(cfg: config.Config, to: str, say: Callable[[str], None], source: str) -> str:
    """Switch to mode `to` for the owner and return the line that says how it went."""
    if to not in modes.MODES:
        return f"No mode {to}: it is normal or coding."
    wanted: modes.Mode = "coding" if to == "coding" else "normal"
    done = _switcher(cfg, source).switch(wanted, "owner", say)
    if done.error:
        return done.error
    name = modes.model_name(cfg, done.mode)
    if not done.changed:
        return f"Already in {done.mode} mode ({name})."
    took = f", loaded in {done.seconds:.0f} s" if done.restarted else ""
    return f"Now in {done.mode} mode: {name}{took}."


def cmd_model(args: argparse.Namespace) -> int:
    """Show the mode and the models, or switch between normal and coding mode."""
    cfg = _load()
    if args.mode is None:
        print("\n".join(model_report(cfg)))
        return 0
    line = switch_model(cfg, args.mode, print, "cli")
    print(line)
    return 0 if line.startswith(("Now in", "Already in")) else 1


# How a model server that cannot be reached is named to the owner.
_MODEL_SHOWN = {
    "unreachable": "server unreachable",
    "missing": "file missing (is its drive mounted?)",
}


def cmd_status(args: argparse.Namespace) -> int:
    """Print what the running Jarvis is doing, or the model state when it is not running."""
    reply = control.send("status")
    running = reply is not None
    if reply is not None and reply.get("ok"):
        shown = {key: value for key, value in reply.items() if key != "ok"}
    else:
        strata = _strata(_load())
        model = strata.status() or control.unreachable_model(strata)
        shown = {"state": "unavailable", "paused": False, "model": model}
        if reply is not None:
            shown["error"] = str(reply.get("error", "no answer"))
    shown["running"] = running
    if args.json:  # a status bar reads the state from the JSON, whatever it is
        print(json.dumps(shown))
        return 0
    if "error" in shown:
        print(f"jarvis: running but not answering ({shown['error']}), model: {shown['model']}")
        return 1
    model = _MODEL_SHOWN.get(str(shown["model"]), shown["model"])
    if running:
        state = shown.get("state")
        print(
            f"jarvis: {state}" if state == "model loading" else f"jarvis: {state}, model: {model}"
        )
        return 0
    print(f"jarvis: not running, model: {model}")
    return 0 if shown["model"] not in _MODEL_SHOWN else 1


def _control(command: str, done: Callable[[dict[str, object]], str]) -> int:
    reply = control.send(command)
    if reply is None:
        print("jarvis: not running (start it with `jarvis start`)")
        return 1
    if not reply.get("ok"):
        print(f"jarvis: {reply.get('error', 'the command failed')}")
        return 1
    print(f"jarvis: {done(reply)}")
    return 0


def cmd_pause(_args: argparse.Namespace) -> int:
    """Let go of the microphone and free the model until `resume`."""
    return _control("pause", lambda r: "paused" if r.get("changed") else "already paused")


def cmd_resume(_args: argparse.Namespace) -> int:
    """Listen again after `pause`."""
    return _control("resume", lambda r: "listening again" if r.get("changed") else "was not paused")


def cmd_hush(_args: argparse.Namespace) -> int:
    """Stop Jarvis speaking now."""
    return _control("hush", lambda _r: "hushed")


def cmd_confirm(_args: argparse.Namespace) -> int:
    """Press the confirm key: allow the one action Jarvis is waiting on, if any."""
    reply = control.send("confirm")
    if reply is None:
        print("jarvis: not running (start it with `jarvis start`)")
        return 1
    if not reply.get("ok"):
        print(f"jarvis: {reply.get('error', 'the command failed')}")
        return 1
    confirmed = reply.get("confirmed")
    if not isinstance(confirmed, str):
        print(f"jarvis: {reply.get('reason', 'nothing was confirmed')}")
        return 1
    shown = "".join(
        ch if ch.isprintable() else " " for ch in confirmed
    )  # the model chose its words
    print(f'jarvis: confirmed: "{shown}"')
    return 0


def cmd_unload(_args: argparse.Namespace) -> int:
    """Free the model now."""
    ok = _strata(_load()).unload()
    print("model unloaded" if ok else "could not unload (busy or unreachable)")
    return 0 if ok else 1


def _systemctl(*args: str) -> int:
    """Run `systemctl --user` with `args`; a call that hangs is given up after SYSTEMCTL_S."""
    try:
        return subprocess.run(
            ["systemctl", "--user", *args], check=False, timeout=SYSTEMCTL_S
        ).returncode
    except subprocess.TimeoutExpired:
        print(f"jarvis: systemctl {' '.join(args)} did not finish in {SYSTEMCTL_S:.0f} s")
        return 1


def _unit_state(unit: str) -> str:
    try:
        done = subprocess.run(
            ["systemctl", "--user", "is-active", unit],
            capture_output=True,
            text=True,
            check=False,
            timeout=SYSTEMCTL_QUERY_S,
        )
    except subprocess.TimeoutExpired:
        return "unknown"
    return done.stdout.strip() or "unknown"


def _reset_failed(*units: str) -> None:
    """Clear the failed state (and start limit) of the units that are in it, before a start.

    A unit stopped by its start limit refuses to start until reset. Only a failed unit is reset:
    systemd unloads a stopped unit that nothing needs, and `reset-failed` on it prints an error.
    """
    failed = [unit for unit in units if _unit_state(unit) == "failed"]
    if failed:
        _systemctl("reset-failed", *failed)


def cmd_help(_args: argparse.Namespace) -> int:
    """Print the list of commands, as `jarvis --help` does."""
    build_parser().print_help()
    return 0


def cmd_start(_args: argparse.Namespace) -> int:
    """Start both services."""
    _reset_failed(STRATA_UNIT, JARVIS_UNIT)
    code = _systemctl("start", STRATA_UNIT, JARVIS_UNIT)
    print(f"jarvis: {_unit_state(JARVIS_UNIT)}, model server: {_unit_state(STRATA_UNIT)}")
    return code


def cmd_stop(_args: argparse.Namespace) -> int:
    """Stop the voice service, then the model server (which frees the model)."""
    code = _systemctl("stop", JARVIS_UNIT, STRATA_UNIT)
    print(f"jarvis: {_unit_state(JARVIS_UNIT)}, model server: {_unit_state(STRATA_UNIT)}")
    return code


def parse_systemd_time(text: str) -> float | None:
    """Return a `systemctl show --timestamp=us+utc` time as Unix time, or None when unset."""
    stamp = text.strip()
    if not stamp.endswith(" UTC"):
        return None
    try:
        when = datetime.strptime(
            stamp.removesuffix(" UTC") + " +0000", "%a %Y-%m-%d %H:%M:%S.%f %z"
        )
    except ValueError:
        return None
    return when.timestamp()


def _started_at(unit: str) -> float | None:
    """Return when `unit` last became active (Unix time, to the microsecond), or None if stopped.

    Whole seconds are not enough: `jarvis setup` writes Strata's files and starts it within the
    same second, and those files must not count as newer than the server.
    """
    show = ["show", unit, "-p", "ActiveEnterTimestamp", "--timestamp=us+utc", "--value"]
    try:
        done = subprocess.run(
            ["systemctl", "--user", *show],
            capture_output=True,
            text=True,
            check=False,
            timeout=SYSTEMCTL_QUERY_S,
        )
    except subprocess.TimeoutExpired:
        return None
    if _unit_state(unit) != "active":
        return None
    return parse_systemd_time(done.stdout)


def _strata_changed_since(started: float) -> bool:
    """Return whether Strata's profile or unit was rewritten after it started."""
    files = (services.strata_profile_file(), services.unit_dir() / services.STRATA_UNIT)
    return any(path.exists() and path.stat().st_mtime > started for path in files)


def _refresh_strata() -> None:
    """Write Strata's profile and unit from the config, so a setting changed there reaches it.

    Files that already hold the same text keep their time, so an unchanged Strata is not
    restarted. Only `jarvis setup` writes the voice unit, which names the environment Jarvis
    runs from, and before setup has written Strata's unit nothing is written. A config that
    cannot be read or built, or a file that cannot be written, is said and Jarvis restarts anyway.
    """
    try:
        cfg = config.load()
        if cfg.strata.run_config == Path() or not (services.unit_dir() / STRATA_UNIT).is_file():
            return
        served = modes.effective(cfg, modes.current())
        services.write_strata_profile(served)
        services.write_strata_unit(served, strata_key_file())
    except (config.ConfigError, OSError) as exc:
        print(f"jarvis: the model server's settings were left as they were: {exc}")
        return
    _systemctl("daemon-reload")


def cmd_restart(_args: argparse.Namespace) -> int:
    """Restart the voice service; restart Strata too when its settings changed since it started.

    Strata's profile and unit are written from the config first. A running Strata is otherwise
    left alone, so the model stays loaded. No switch of modes is made meanwhile, and while no
    request is being answered none begins until Strata is back. The restart is the owner's own,
    so it is made even when a request is being answered (in a chat window, say): it says that
    it cut one, and requests are not held off then. It is also made when the modes' lock cannot
    be opened, which is said.
    """
    _reset_failed(STRATA_UNIT, JARVIS_UNIT)
    with contextlib.ExitStack() as held:
        try:
            held.enter_context(modes.locked())
            free = held.enter_context(modes.nobody_answering())
        except OSError as exc:
            print(f"jarvis: restarting without the modes' lock: {exc.strerror or exc}")
            free = True
        _refresh_strata()
        started = _started_at(STRATA_UNIT)
        changed = started is not None and _strata_changed_since(started)
        cut = changed and not free
        code = _systemctl("restart" if changed else "start", STRATA_UNIT)
    code = code or _systemctl("restart", JARVIS_UNIT)
    print(f"jarvis: {_unit_state(JARVIS_UNIT)}" + (", model server restarted" if changed else ""))
    if cut:
        print("jarvis: a request that was being answered was cut off by the model server's restart")
    return code


def cmd_logs(_args: argparse.Namespace) -> int:
    """Follow both services' journals."""
    cmd = [
        "journalctl",
        "--user",
        "-u",
        JARVIS_UNIT,
        "-u",
        STRATA_UNIT,
        "-f",
        "-n",
        "30",
        "-o",
        "cat",
    ]
    try:
        return subprocess.run(cmd, check=False).returncode
    except KeyboardInterrupt:
        return 0


def _pc_audio_summary(audio: config.AudioConfig) -> str:
    if not audio.listen_to_pc:
        return "off"
    apps = ", ".join(audio.pc_audio_apps) or "all PC audio"
    return f"on ({apps}; a yes lasts {audio.pc_audio_grant_s / 60:g} min)"


RESTART_FAILED = (
    "the voice service did not restart, so it still runs on the old settings; see `jarvis logs`"
)


class RestartError(RuntimeError):
    """The voice service did not restart after a change, so it runs on the old settings."""


def _restart_if_running() -> bool:
    """Restart the voice service if it runs; return False when that restart failed."""
    if _unit_state(JARVIS_UNIT) != "active":
        return True
    # A restart counts against the start limit (5 in 15 min): clear it first, so a few
    # config changes in a row do not leave the voice service stopped.
    _systemctl("reset-failed", JARVIS_UNIT)
    if _systemctl("restart", JARVIS_UNIT) != 0:
        print(f"jarvis: {RESTART_FAILED}", file=sys.stderr)
        return False
    return True


CONSENT_SAID = {
    True: "consent: bypassed (Jarvis acts without asking; a strong action still waits for the "
    "confirm key, and `deny` still refuses)",
    False: "consent: asks (a risky action, or one after outside text, needs your yes)",
}


def cmd_consent(args: argparse.Namespace) -> int:
    """Show or set whether Jarvis asks for a yes before acting."""
    cfg = _load()
    if args.mode is not None:
        bypass = args.mode == "bypass"
        cfg = cfg.model_copy(
            update={"boundaries": cfg.boundaries.model_copy(update={"bypass_consent": bypass})}
        )
        config.save(cfg)
        AuditLog(paths.audit_file()).record("consent_changed", bypass_consent=bypass)
        _restart_if_running()
    print(CONSENT_SAID[cfg.boundaries.bypass_consent])
    return 0


SPEEDS = f"{config.MIN_SPEECH_SPEED:g} to {config.MAX_SPEECH_SPEED:g}"


def _speech_speed(text: str) -> float:
    """Return a speaking speed from the command line.

    Raises:
        ArgumentTypeError: it is no number in the range, and says what it takes.
    """
    try:
        value = float(text)
    except ValueError:
        value = math.nan
    if not config.MIN_SPEECH_SPEED <= value <= config.MAX_SPEECH_SPEED:  # NaN fails it too
        msg = (
            f"a speed from {config.MIN_SPEECH_SPEED:g} to {config.MAX_SPEECH_SPEED:g}, such as 1.25"
        )
        raise argparse.ArgumentTypeError(msg)
    return value


def cmd_speed(args: argparse.Namespace) -> int:
    """Show or set how fast Jarvis speaks; a change restarts the voice service."""
    cfg = _load()
    if args.value is not None:
        audio = cfg.audio.model_copy(update={"speech_speed": args.value})
        cfg = cfg.model_copy(update={"audio": audio})
        config.save(cfg)
        _restart_if_running()
    print(f"speech speed: {cfg.audio.speech_speed:g}")
    return 0


def _tool_settings(cfg: config.Config) -> list[ToolSetting]:
    return tool_settings(cfg, spotify_operations(cfg))


def set_tool(
    name: str, cls: OpClass | None, *, source: str = "cli", restart: bool = True
) -> ToolSetting:
    """Set one tool's class in the config on disk (None: as shipped), audit it, restart voice.

    `source` says where the change came from in the audit log; the chat passes `restart=False`
    and restarts the voice service once, when its settings close.

    Returns the tool's new setting.

    Raises:
        ConfigError: no tool has that name, or it does not allow that class.
    """
    cfg = config.load()
    found = next((s for s in _tool_settings(cfg) if s.name == name), None)
    if found is None:
        msg = f"no tool is called {name!r}; `jarvis tools` lists them"
        raise config.ConfigError(msg)
    changed = with_class(cfg, found, found.shipped if cls is None else cls)
    now = next(s for s in _tool_settings(changed) if s.name == name)
    config.save(changed)
    AuditLog(paths.audit_file()).record(
        "tool_class_changed", op=name, was=found.state.value, to=now.state.value, source=source
    )
    if restart:
        _restart_if_running()
    return now


def _next_tool_setting(name: str) -> ToolSetting:
    """Move one tool on to its next setting (on, asks first, needs the key, off), for the chat.

    Raises:
        ConfigError: as `set_tool`.
    """
    found = next((s for s in _tool_settings(config.load()) if s.name == name), None)
    if found is None:
        msg = f"no tool is called {name!r}"
        raise config.ConfigError(msg)
    return set_tool(name, found.following(), source="chat", restart=False)


def cmd_tools(args: argparse.Namespace) -> int:
    """List the tools and their settings, or set one."""
    cfg = _load()
    if args.name is not None and args.setting is None:
        print("say how to set it: on, ask, key, off or default", file=sys.stderr)
        return 2
    if args.name is not None:
        try:
            now = set_tool(
                args.name, None if args.setting == "default" else TOOL_WORDS[args.setting]
            )
        except config.ConfigError as exc:
            print(f"jarvis: {exc}", file=sys.stderr)
            return 2
        print(f"{now.name}: {now.label}")
        return 0
    rows = _tool_settings(cfg)
    width = max(len(s.name) for s in rows)
    label = max(len(s.label) for s in rows)
    for s in rows:
        mark = "" if s.state is s.shipped else " *"
        print(f"{s.name:<{width}}  {s.label + mark:<{label + 2}}  {s.summary}")
    print("* set in your config; `jarvis tools NAME default` puts it back")
    return 0


def cmd_pc_audio(args: argparse.Namespace) -> int:
    """Show or set whether Jarvis takes requests heard in PC audio, and from which apps."""
    cfg = _load()
    if args.state is None and args.apps is None and not args.all_output:
        print(f"pc audio: {_pc_audio_summary(cfg.audio)}")
        return 0
    update: dict[str, object] = {}
    if args.state is not None:
        update["listen_to_pc"] = args.state == "on"
    if args.apps is not None:
        update["pc_audio_apps"] = args.apps
    elif args.all_output:
        update["pc_audio_apps"] = []
    new = cfg.model_copy(update={"audio": cfg.audio.model_copy(update=update)})
    config.save(config.Config.model_validate(new.model_dump()))
    print(f"pc audio: {_pc_audio_summary(new.audio)}")
    _restart_if_running()
    return 0


def cmd_wake(args: argparse.Namespace) -> int:
    """Show or set which wake word model Jarvis listens with: LiveKit's or an Azure keyword."""
    cfg = _load()
    update: dict[str, object] = {}
    if args.engine is not None:
        update["wake_engine"] = args.engine
    if args.model is not None:
        update["azure_keyword_model"] = str(Path(args.model).expanduser().resolve())
    if args.words is not None:
        update["wake_words"] = args.words
    if not update:
        print(f"wake word: {_wake_summary(cfg.audio)}")
        _print_wake_models(cfg)
        return 0
    new = cfg.model_copy(update={"audio": cfg.audio.model_copy(update=update)})
    try:
        new = config.Config.model_validate(new.model_dump())
    except ValueError as exc:
        print(f"jarvis: wake value refused: {exc}", file=sys.stderr)
        return 2
    config.save(new)
    print(f"wake word: {_wake_summary(new.audio)}")
    _print_wake_models(new)
    if new.audio.wake_engine == "azure" and not Path(new.audio.azure_keyword_model).is_file():
        print(
            "no Azure keyword model file found: Jarvis listens with LiveKit's until "
            "`jarvis wake azure --model FILE` names one"
        )
    _restart_if_running()
    return 0


def _wake_summary(audio: config.AudioConfig) -> str:
    if audio.wake_engine == "azure":
        return f"Azure keyword model {audio.azure_keyword_model or '(not set)'}"
    own = "".join(
        f"; {name} {audio.wake_thresholds[name]}"
        for name in audio.wake_words
        if name in audio.wake_thresholds
    )
    return f"LiveKit {', '.join(audio.wake_words)} (threshold {audio.wake_threshold}{own})"


def _print_wake_models(cfg: config.Config) -> None:
    """List each LiveKit wake word model: on or off, and any that is on but has no file."""
    folder = cfg.data_dir / WAKE_DIR
    found = sorted(
        path.stem
        for path in folder.glob(f"*{WAKE_SUFFIX}")
        if path.is_file() and re.fullmatch(config.WAKE_WORD_PATTERN, path.stem)
    )
    for name in sorted({*found, *cfg.audio.wake_words}):
        state = "on" if name in cfg.audio.wake_words else "off"
        own = cfg.audio.wake_thresholds.get(name)
        threshold = "" if own is None else f" (threshold {own})"
        missing = "" if name in found else f" (no file {folder / name}{WAKE_SUFFIX})"
        print(f"  {state:3} {name}{threshold}{missing}")


# How a systemd unit's state is named to the owner.
_UNIT_SHOWN = {"active": "running", "inactive": "stopped", "activating": "starting"}


def cmd_overview(args: argparse.Namespace) -> int:
    """Show what is running and the model state, then follow the live log."""
    voice, server = _unit_state(JARVIS_UNIT), _unit_state(STRATA_UNIT)
    state = _strata(_load()).status() if server == "active" else None
    shown = (_UNIT_SHOWN.get(unit, unit) for unit in (voice, server))
    print("jarvis: {}   model server: {}".format(*shown), end="")
    print(f"   model: {state}" if state else "")
    cfg = _load()
    print(f"pc audio: {_pc_audio_summary(cfg.audio)}")
    ok, login, fix = _spotify_login(cfg)
    print(login if ok else f"{login}: run {fix}")
    print("All commands: jarvis help")
    if voice != "active":
        print("Jarvis is not running; start it with: jarvis start")
        return 0
    if not sys.stdout.isatty():
        return 0
    print("live log (Ctrl+C to leave; Jarvis keeps running):")
    return cmd_logs(args)


LISTEN_WAIT_S = 8.0  # how long a speech-test card waits for speech to start


def _positive(text: str) -> int:
    """Parse a count of at least one.

    Raises:
        ArgumentTypeError: `text` is not a whole number of at least 1.
    """
    try:
        value = int(text)
    except ValueError:
        value = 0
    if value < 1:
        msg = f"not a count of at least 1: {text!r}"
        raise argparse.ArgumentTypeError(msg)
    return value


def _listening() -> bool:
    """Whether a running Jarvis may hold the microphone: it runs, and has not said it is paused."""
    status = control.send("status", timeout=1.0)
    return status is not None and status.get("paused") is not True


def _listener(
    cfg: config.Config,
) -> tuple[Callable[[], NDArray[np.int16] | None], Callable[[], None]]:
    """Return a capture of one request as Jarvis takes it after the wake word, and its stop."""
    from jarvis.audio import MIC_STALL_S, Microphone, Vad  # ruff: ignore[import-outside-top-level] - loads numpy and ONNX Runtime, which most commands never need
    from jarvis.voice import capture_utterance  # ruff: ignore[import-outside-top-level] - imports jarvis.audio

    mic = Microphone(cfg.audio.source, stall_s=MIC_STALL_S)
    vad = Vad(cfg.data_dir / VAD_MODEL)
    settings = cfg.audio.model_copy(
        update={"no_speech_timeout_s": max(cfg.audio.no_speech_timeout_s, LISTEN_WAIT_S)}
    )
    mic.start()

    def listen() -> NDArray[np.int16] | None:
        mic.drain()
        print("  listening…")
        return capture_utterance(mic.frames(), vad, settings).samples

    return listen, mic.stop


def _compare_scores(scores: list[Path]) -> int:
    """Print whether two saved scores differ beyond chance; return the exit code."""
    from jarvis import speechtest  # ruff: ignore[import-outside-top-level] - imports jarvis.audio

    if len(scores) != 2:  # ruff: ignore[magic-value-comparison] - a pair
        print("jarvis: compare takes two score files saved by `score --out`", file=sys.stderr)
        return 2
    try:
        first, second = (json.loads(path.read_text(encoding="utf-8")) for path in scores)
        print(speechtest.compare(first, second))
    except (OSError, ValueError) as exc:  # ScoreError is a ValueError
        print(f"jarvis: a score could not be read: {exc}", file=sys.stderr)
        return 2
    return 0


def cmd_speech_test(args: argparse.Namespace) -> int:
    """Record a private set of your requests, score speech-to-text on it, or compare two scores."""
    from jarvis import speechtest  # ruff: ignore[import-outside-top-level] - imports jarvis.audio
    from jarvis.audio import AudioError  # ruff: ignore[import-outside-top-level] - loads numpy and ONNX Runtime, which most commands never need

    if args.action == "compare":
        return _compare_scores(args.scores)
    cfg = _load()
    folder = args.dir or cfg.data_dir / "speech-test"
    audio = cfg.audio.model_copy(
        update={
            key: value
            for key, value in (
                ("transcribe_engine", args.engine),
                ("transcribe_model", args.model),
                ("vocabulary_hint", False if args.no_hint else None),
            )
            if value is not None
        }
    )
    cfg = cfg.model_copy(update={"audio": audio})
    try:
        jarvis = build(cfg, via="speech-test")
    except SetupError as exc:
        print(f"jarvis: {exc}", file=sys.stderr)
        return 2
    jarvis.ops.refresh()  # your artists, for the cards and the hint
    if args.action == "record" and _listening():
        print("jarvis: Jarvis is listening; `jarvis pause` it first, then record", file=sys.stderr)
        return 1
    heard_with = f"{audio.transcribe_engine or 'default'} {audio.transcribe_model}".strip()
    artists = jarvis.ops.artist_names()
    if args.action == "record" and not artists:
        print("No artists from Spotify, so every card is an open request.")
    try:
        # ValueError: a --model path TOML can't hold
        request, _answer = _transcribers(cfg, jarvis.hint, "voxtype-speech-test.toml")
        if args.action == "record":
            return _record_set(cfg, folder, request, args, (artists, heard_with))
        print(f"Scoring {folder} with {heard_with} …")
        results = speechtest.score(folder, request, print)
    except (AudioError, speechtest.FolderError, OSError, ValueError) as exc:
        print(f"jarvis: {exc}", file=sys.stderr)
        return 1
    print(speechtest.report(results))
    if args.out is not None:
        write_atomic(args.out, json.dumps(results, ensure_ascii=False, indent=1) + "\n", 0o600)
        print(f"Saved each request's result to {args.out}.")
    return 0


def _record_set(
    cfg: config.Config,
    folder: Path,
    request: _Transcriber,
    args: argparse.Namespace,
    about: tuple[list[str], str],
) -> int:
    """Record `args.count` cards into `folder`; return the exit code.

    `about` holds the owner's artists, for the cards, and the transcriber that hears first.
    """
    artists, heard_with = about
    from jarvis import speechtest  # ruff: ignore[import-outside-top-level] - imports jarvis.audio

    rng = random.Random()  # ruff: ignore[suspicious-non-cryptographic-random-usage] - card order
    to_say = speechtest.cards(artists, args.count, rng)
    print(f"Recording into {folder}. Say each card in your own words, as you would to Jarvis,")
    print("without the wake word.")
    listen, stop = _listener(cfg)
    try:
        saved = speechtest.record(
            folder,
            to_say,
            listen,
            request,
            ask=input,
            show=print,
            music=args.music,
            heard_with=heard_with,
        )
    finally:
        stop()
    print(f"Saved {saved} requests; `jarvis speech-test score` scores the set.")
    return 0


COMMANDS: dict[str, Callable[[argparse.Namespace], int]] = {
    "help": cmd_help,
    "start": cmd_start,
    "stop": cmd_stop,
    "restart": cmd_restart,
    "logs": cmd_logs,
    "setup": cmd_setup,
    "spotify-login": cmd_spotify_login,
    "ask": cmd_ask,
    "chat": cmd_chat,
    "run": cmd_run,
    "doctor": cmd_doctor,
    "status": cmd_status,
    "pause": cmd_pause,
    "resume": cmd_resume,
    "hush": cmd_hush,
    "confirm": cmd_confirm,
    "unload": cmd_unload,
    "pc-audio": cmd_pc_audio,
    "consent": cmd_consent,
    "tools": cmd_tools,
    "speed": cmd_speed,
    "wake": cmd_wake,
    "model": cmd_model,
    "speech-test": cmd_speech_test,
}


def main(argv: Sequence[str] | None = None) -> int:
    """Run the `jarvis` command and return its exit code."""
    args = build_parser().parse_args(argv)
    if args.command is None:
        return cmd_overview(args)
    return COMMANDS[args.command](args)
