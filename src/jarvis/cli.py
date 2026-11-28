"""Command-line entry point for `jarvis`."""

import argparse
import contextlib
import io
import json
import logging
import math
import os
import random
import re
import secrets
import shlex
import shutil
import signal
import subprocess
import sys
import termios
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO, cast
from urllib.parse import urlsplit

from jarvis import (
    __version__,
    acp,
    acpstore,
    cliout,
    config,
    control,
    handoff,
    models,
    modes,
    paths,
    pluginsync,
    ramguard,
    services,
    skills,
    t3push,
    update,
    voices,
)
from jarvis.activity import VOICE_VIA, AuditTail
from jarvis.agent import Declined, Hooks, NoReason
from jarvis.app import (
    SetupError,
    build,
    chat_memory_file,
    chat_pictures,
    echoes_hint,
    offered_registry,
    spotify_operations,
    strata_model,
    system_prompt,
    voice_busy,
    with_remote_metadata,
)
from jarvis.audit import AuditLog
from jarvis.chat import clipboard
from jarvis.chat.archive import ChatArchive
from jarvis.chat.history import InputHistory
from jarvis.confirmkey import ConfirmKey
from jarvis.files import write_atomic
from jarvis.model import NOT_PRIVATE, ModelStatus, tls_context
from jarvis.operations import OperationError
from jarvis.paths import spotify_token_file, strata_key_file
from jarvis.pc.hearing import Listening, PcHearing
from jarvis.pc.screenshot import folders as screenshot_folders
from jarvis.pc.workspace import workspace_path, workspace_problem
from jarvis.policy import OpClass, is_yes
from jarvis.recording import OutputRecorder
from jarvis.schedule import JobStore
from jarvis.secretfile import (
    MAX_KEY_INPUT,
    ensure_api_key,
    read_api_key,
    read_private,
    remote_key_input,
)
from jarvis.services import JARVIS_UNIT, STRATA_UNIT, parse_systemd_time, reads_pictures
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
    from jarvis.pictures import Picture
    from jarvis.voice import VoiceLoop

log = logging.getLogger(__name__)

WAKE_DIR = "wake"
WAKE_SUFFIX = ".onnx"
VAD_MODEL = "vad/silero_vad.onnx"
AZURE_PYTHON = "azure-wake/venv/bin/python"
AZURE_CHECK_S = 30.0
SYSTEMCTL_S = 120.0  # a start or stop; Strata takes up to 45 s to stop (TimeoutStopSec)
SYSTEMCTL_QUERY_S = 10.0  # is-active, show
KEY_INPUT_LIMIT = 4096
_TTY = Path("/dev/tty")
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
            ("acp", "talk to Jarvis from a coding app, over ACP on stdio"),
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
            ("update", "check for updates, apply them, or manage automatic updates"),
        ),
    ),
    (
        "Settings",
        (
            ("wake", "show or switch the wake word model (livekit or azure)"),
            ("consent", "show or set whether Jarvis asks before acting"),
            ("tools", "list the tools, or switch one on, off or to asking first"),
            ("speed", "show or set how fast Jarvis speaks (1.0 is normal)"),
            ("voice", "list the voices Jarvis can speak with, or switch to one"),
            ("pc-audio", "let Jarvis take requests from PC audio (Discord by default)"),
            ("plugins", "list, install, uninstall and update skill plugins"),
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
    "acp": (
        "A coding app starts this and talks to it on standard input and output, one JSON "
        "message a line; never run it by hand. Each session is a chat conversation of its own "
        "with the one Jarvis agent, answered in text. A question about an action is asked in "
        "the reply and your next message is the answer: a clear yes runs it, anything else "
        "cancels it. In T3 Code: Settings, Providers, Add provider, Local ACP command, with "
        "this program and the argument acp. The app lists Jarvis's two modes as models, each "
        "by its model's name and what it is for, with a thinking effort to choose."
    ),
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
    "plugins": (
        "With no argument, lists the marketplaces set under [plugins] in the config and what "
        "each holds. `install` enables a plugin (downloading its marketplace first), `uninstall` "
        "disables it, and `update` moves what is enabled to the newest commit of its "
        "repository. Nothing is downloaded while no plugin is enabled. Jarvis uses only a "
        "plugin's skills; its hooks, commands, agents, MCP servers and scripts are never run."
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


def _setup_role_arguments(setup: argparse.ArgumentParser) -> None:
    role = setup.add_mutually_exclusive_group()
    role.add_argument(
        "--remote", metavar="URL", help="use Strata on your own machine on the network"
    )
    role.add_argument("--backend", action="store_true", help="set up only the Strata model service")
    role.add_argument(
        "--local", action="store_true", help="serve Strata and run the voice service here"
    )
    setup.add_argument("--listen", metavar="IP", help="backend bind address (default: 127.0.0.1)")
    setup.add_argument(
        "--ssh-target", metavar="[USER@]HOST", help="SSH destination for the printed client command"
    )
    setup.add_argument(
        "--tls-ca",
        type=Path,
        metavar="PEM",
        help="remote HTTPS CA or self-signed server certificate",
    )
    setup.add_argument(
        "--allow-plain-http",
        action="store_true",
        help="allow cleartext HTTP to a private IP address",
    )


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
    for each in (parser, *made.values()):  # before or after the command, as an agent writes it
        each.add_argument(
            "--agent",
            "--agent-format",
            dest="agent",
            action="store_true",
            default=argparse.SUPPRESS,
            help=f"answer in one line of JSON for an agent (or set {cliout.AGENT_ENV}=1)",
        )

    setup = made["setup"]
    _setup_role_arguments(setup)
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

    _setup_options(made)

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

    made["acp"].add_argument(
        "--coding",
        action="store_true",
        help="a new session starts in coding mode (the coding prompt, limits and tools, and the "
        "coding model when it has one of its own) rather than the normal mode",
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
    _settings_options(made)
    tool = made["tools"]
    tool.add_argument("name", nargs="?", help="the tool, as `jarvis tools` lists it")
    tool.add_argument(
        "setting",
        nargs="?",
        choices=[*TOOL_WORDS, "default"],
        help="on: runs; ask: asks first; key: needs the confirm key; off: not offered; "
        "default: as Jarvis ships it",
    )
    made["plugins"].add_argument(
        "action",
        nargs="?",
        choices=["list", "install", "uninstall", "update"],
        default="list",
        help="what to do",
    )
    made["plugins"].add_argument(
        "target", nargs="?", metavar="PLUGIN@MARKETPLACE", help="install and uninstall: the plugin"
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


def _settings_options(made: Mapping[str, argparse.ArgumentParser]) -> None:
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
    made["voice"].add_argument(
        "name", nargs="?", help="the voice's name, as `jarvis voice` lists it"
    )


def _setup_options(made: Mapping[str, argparse.ArgumentParser]) -> None:
    setup = made["setup"]
    setup.add_argument("--has-run-config", action="store_true", help=argparse.SUPPRESS)
    setup.add_argument("--update-branch", metavar="NAME", help="release branch for jarvis update")
    setup.add_argument(
        "--strata-vram-reserve-mib",
        type=int,
        metavar="MIB",
        help="VRAM to leave free for other programs",
    )

    made["doctor"].add_argument(
        "--no-service", action="store_true", help="check the installation without a running service"
    )
    made["doctor"].add_argument(
        "--essential", action="store_true", help="report Spotify login problems as notes"
    )
    made["doctor"].add_argument(
        "--wait", action="store_true", help="wait for the model server to start"
    )
    update_args = made["update"]
    change = update_args.add_mutually_exclusive_group()
    change.add_argument("--apply", action="store_true", help="install the available update")
    change.add_argument("--auto", choices=("on", "off"), help="enable or disable the daily timer")
    update_args.add_argument(
        "--quiet", action="store_true", help="print only failures and updates applied"
    )


def _load() -> config.Config:
    try:
        return config.load()
    except config.ConfigError as exc:
        print(f"jarvis: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


def _setup_mode(args: argparse.Namespace, current: config.StrataConfig) -> str:
    if args.remote is not None:
        return "remote"
    if args.backend:
        return "backend"
    return "local" if args.local else current.mode


def _setup_role(args: argparse.Namespace, current: config.Config) -> dict[str, object]:
    mode = _setup_mode(args, current.strata)
    if mode != "backend" and (args.listen is not None or args.ssh_target is not None):
        msg = "--listen and --ssh-target require backend mode"
        raise config.ConfigError(msg)
    if mode != "remote" and (args.tls_ca is not None or args.allow_plain_http):
        msg = "--tls-ca and --allow-plain-http require remote mode"
        raise config.ConfigError(msg)
    updates: dict[str, object] = {"mode": mode}
    if mode != "remote":
        updates |= {
            "url": "",
            "tls_ca": Path(),
            "allow_plain_http": False,
            "remote_pictures": False,
        }
    if mode != "backend":
        updates["listen"] = "127.0.0.1"
    if args.listen is not None:
        updates["listen"] = args.listen
    if mode == "remote":
        updates.update(_remote_setup_settings(args, current.strata))
    return updates


def _remote_setup_settings(
    args: argparse.Namespace, current: config.StrataConfig
) -> dict[str, object]:
    updates: dict[str, object] = {}
    if args.remote is not None:
        updates["url"] = args.remote
        if args.remote != current.url:
            updates |= {"tls_ca": Path(), "allow_plain_http": False}
    if args.tls_ca is not None:
        updates["tls_ca"] = args.tls_ca.expanduser().absolute()
    if args.allow_plain_http:
        updates["allow_plain_http"] = True
    return updates


def cmd_setup(args: argparse.Namespace) -> int:
    """Write the config, key and units for the selected role."""
    current = _load()
    if args.has_run_config:
        return int(current.strata.mode == "local" and current.strata.run_config == Path())
    try:
        role = _setup_role(args, current)
    except config.ConfigError as exc:
        print(f"jarvis: {exc}", file=sys.stderr)
        return 2
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
            "vram_reserve_mib": args.strata_vram_reserve_mib,
        }.items()
        if value is not None
    }
    audio = current.audio.model_copy(update={"source": args.mic} if args.mic is not None else {})
    updated = current.model_copy(
        update={
            "spotify": spotify,
            "strata": current.strata.model_copy(update=strata_updates | role),
            "audio": audio,
            "model": current.model.model_copy(update={"base_url": ""})
            if role["mode"] != "local"
            else current.model,
            "update": current.update.model_copy(update={"branch": args.update_branch})
            if args.update_branch is not None
            else current.update,
            **({"data_dir": args.data_dir} if args.data_dir is not None else {}),
        }
    )
    try:
        updated = config.Config.model_validate(updated.model_dump())
    except ValueError as exc:
        print(f"jarvis: setup value refused: {exc}", file=sys.stderr)
        return 2
    if updated.strata.mode != "local":
        return _setup_split(args, updated)
    return _setup_local(args, updated)


def _setup_local(args: argparse.Namespace, updated: config.Config) -> int:
    if args.coding_run_config is not None and updated.strata.run_config != Path():
        coding = modes.effective(updated, "coding")
        try:  # a coding run config that cannot be served is refused before anything is saved
            services.strata_profile(coding)
            services.due_expert_seed(coding)
        except config.ConfigError as exc:
            print(f"jarvis: {exc}", file=sys.stderr)
            return 2
    config.save(updated)
    ensure_api_key(strata_key_file())
    _setup_workspace(updated)
    if updated.strata.run_config == Path():
        print(f"config: {paths.config_file()}")
        print(
            "next: run `strata launch` for the model, then "
            "`jarvis setup --run-config <the strata-<model>.json it writes>`"
        )
        return 2 if args.enable else 0
    try:
        services.strata_profile(
            modes.effective(updated, "coding")
        )  # a change of the usual one can break it too
        with modes.locked():  # not in the middle of a switch of modes, which writes them too
            served, profile = _write_mode_profiles(updated)
            units = services.write_units(served, strata_key_file())
    except config.ConfigError as exc:
        print(f"jarvis: {exc}", file=sys.stderr)
        return 2
    print(f"config: {paths.config_file()}\nstrata profile: {profile}")
    print("units: " + ", ".join(str(u) for u in units))
    result = _setup_units(STRATA_UNIT, JARVIS_UNIT, enable=args.enable)
    _setup_spotify(updated, failed=result)
    return result


def _write_mode_profiles(cfg: config.Config) -> tuple[config.Config, Path]:
    """Write the Strata profile for the mode Jarvis is in; return that mode's config and the file.

    The other mode's learned expert profile is started too, so the doctor finds it ready before
    the first switch. When it cannot be, this says why and goes on.
    """
    current = modes.current()
    served = modes.effective(cfg, current)  # a setup in coding mode keeps it
    profile = services.write_strata_profile(served)
    for mode in modes.MODES:
        if mode != current:
            _seed_other_mode(cfg, mode)
    return served, profile


def _seed_other_mode(cfg: config.Config, mode: modes.Mode) -> None:
    try:
        services.seed_expert_profile(modes.effective(cfg, mode))
    except config.ConfigError as exc:
        print(f"jarvis: {mode} mode's expert profile was not started: {exc}", file=sys.stderr)


def _setup_workspace(cfg: config.Config) -> None:
    """Make Jarvis's workspace folder if it is missing, and say if it cannot be used."""
    folder = workspace_path(cfg.files, Path.home())
    problem = workspace_problem(folder, create=True)
    if problem:
        print(
            f"jarvis: files.workspace {folder} {problem}; fix that or point it elsewhere",
            file=sys.stderr,
        )
    else:
        print(f"workspace: {folder}")


def _workspace_check(cfg: config.Config) -> tuple[bool, str, str]:
    """The doctor's check of Jarvis's workspace: a folder of the owner's alone, or not made yet."""
    folder = workspace_path(cfg.files, Path.home())
    problem = workspace_problem(folder, create=False)
    if problem:
        return (
            False,
            f"workspace {folder} {problem}",
            "fix the folder, or set files.workspace to another one",
        )
    if not folder.is_dir():
        return True, f"workspace {folder} (made when Jarvis starts)", ""
    return True, f"workspace {folder}", ""


def _setup_units(*units: str, enable: bool) -> int:
    commands = [["daemon-reload"]]
    if enable:
        # A unit stopped by its start limit refuses to start until its failed state is cleared.
        commands.extend((["reset-failed", *units], ["enable", "--now", *units]))
    for args in commands:
        try:
            done = subprocess.run(
                ["systemctl", "--user", *args],
                check=False,
                timeout=SYSTEMCTL_S,
                capture_output=args[0] == "reset-failed",
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            print(f"jarvis: could not reach the systemd user manager: {exc}", file=sys.stderr)
            return 1
        if done.returncode and args[0] != "reset-failed":
            print(
                f"jarvis: systemd could not {args[0]} user units; check jarvis logs",
                file=sys.stderr,
            )
            return 1
    return 0


def _setup_spotify(updated: config.Config, *, failed: int) -> None:
    if not failed and updated.spotify.client_id and read_private(spotify_token_file()) is None:
        print("next: jarvis spotify-login")


def _typed_key() -> str:
    """Read one line from the terminal with echo off, and put echo back whatever happens.

    Raises:
        ConfigError: the terminal does not take its echo switched off.
    """
    with _TTY.open(encoding="utf-8") as terminal, _TTY.open("w", encoding="utf-8") as prompt:
        original: list[Any] | None = None
        try:
            original = termios.tcgetattr(terminal)
            hidden = list(original)
            hidden[3] &= ~termios.ECHO
            termios.tcsetattr(terminal, termios.TCSAFLUSH, hidden)
            print("Strata API key: ", file=prompt, end="", flush=True)
            return terminal.readline(KEY_INPUT_LIMIT + 2)
        except termios.error as exc:
            msg = "could not read the key at the terminal; supply it on stdin instead"
            raise config.ConfigError(msg) from exc
        finally:
            if original is not None:
                with contextlib.suppress(termios.error):
                    termios.tcsetattr(terminal, termios.TCSAFLUSH, original)
            print(file=prompt, flush=True)


def _read_remote_key() -> str:
    entered = _typed_key() if sys.stdin.isatty() else sys.stdin.read(MAX_KEY_INPUT + 1)
    key = remote_key_input(entered)
    if key is None:
        msg = "supply the Strata API key on stdin or enter it at the terminal prompt"
        raise config.ConfigError(msg)
    return key


_PLAIN_BACKEND = (
    "warning: the backend listens beyond this machine over plain HTTP, so its key and the "
    "conversations cross the network unencrypted. Keep strata.listen on 127.0.0.1 and use an SSH "
    "tunnel, or put an HTTPS proxy in front of it."
)


def _client_command(cfg: config.Config, target: str | None) -> str:
    ssh_destination = r"(?:[a-zA-Z0-9_][a-zA-Z0-9_.-]*@)?[a-zA-Z0-9][a-zA-Z0-9_.-]*"
    if target is not None and not re.fullmatch(ssh_destination, target):
        msg = "--ssh-target must be an OpenSSH host or user@host"
        raise config.ConfigError(msg)
    destination = shlex.quote(target) if target else "<ssh-target>"
    bind = cfg.strata.listen
    host = f"[{bind}]" if ":" in bind else bind
    key_command = shlex.quote(f"cat -- {shlex.quote(str(services.key_file(cfg)))}")
    return (
        f"ssh -fNT -o ExitOnForwardFailure=yes -L 127.0.0.1:18080:{host}:{cfg.strata.port}"
        f" -- {destination} && ssh -T -- {destination} {key_command} | "
        "jarvis setup --remote http://127.0.0.1:18080 --enable"
    )


def _setup_split(args: argparse.Namespace, cfg: config.Config) -> int:
    try:
        if cfg.strata.mode == "backend":
            return _setup_backend(args, cfg)
        return _setup_remote(args, cfg)
    except config.ConfigError as exc:
        print(f"jarvis: {exc}", file=sys.stderr)
        return 2
    except OSError, ValueError:
        print(
            "jarvis: setup could not read or write its config, TLS CA, key or unit", file=sys.stderr
        )
        return 2


def _setup_remote(args: argparse.Namespace, cfg: config.Config) -> int:
    if cfg.strata.tls_ca != Path():
        tls_context(cfg.strata.tls_ca)
    if not config.private_endpoint(cfg.strata.host()):
        msg = f"strata.url's {NOT_PRIVATE}"
        raise config.ConfigError(msg)
    # An explicit endpoint always needs a key. A routine setup reuses private state.
    key = _read_remote_key() if args.remote is not None else services.read_key(cfg)
    if not key:
        msg = "no remote key file; run jarvis setup --remote <url>"
        raise config.ConfigError(msg)
    with modes.locked():
        unit = services.write_voice_unit()
        if args.remote is not None:
            services.write_remote_key(cfg, key)
        config.save(cfg)
    _setup_workspace(cfg)
    print(f"config: {paths.config_file()}\nvoice unit: {unit}")
    active = args.enable and _unit_state(JARVIS_UNIT) == "active"
    code = _setup_units(JARVIS_UNIT, enable=args.enable)
    if code == 0 and active:
        code = _systemctl("restart", JARVIS_UNIT)
    _setup_spotify(cfg, failed=code)
    return code


def _setup_backend(args: argparse.Namespace, cfg: config.Config) -> int:
    command = _client_command(cfg, args.ssh_target)
    key_file = services.key_file(cfg)
    with modes.locked():
        if (services.unit_dir() / JARVIS_UNIT).exists():
            msg = "backend setup needs an account without an installed Jarvis voice unit"
            raise config.ConfigError(msg)
        services.strata_profile(cfg)
        services.strata_unit(cfg, key_file)  # validate both before writing anything
        ensure_api_key(key_file)
        profile = services.write_strata_profile(cfg)
        unit = services.write_strata_unit(cfg, key_file)
        config.save(cfg)
    print(f"config: {paths.config_file()}\nstrata profile: {profile}\nbackend unit: {unit}")
    if cfg.strata.serves_beyond_loopback():
        print(_PLAIN_BACKEND)
    target_note = (
        " (replace <ssh-target> with the backend's SSH destination)" if not args.ssh_target else ""
    )
    print(f"On the Jarvis client{target_note}:\n{command}")
    active = args.enable and _unit_state(STRATA_UNIT) == "active"
    code = _setup_units(STRATA_UNIT, enable=args.enable)
    return _systemctl("restart", STRATA_UNIT) if code == 0 and active else code


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


def _terminal_confirm_always(question: str) -> bool | Declined:
    """Require an answer from an interactive terminal, even when ordinary consent is bypassed."""
    if not sys.stdin.isatty():
        return Declined(NoReason.NO_ANSWER)
    return _terminal_confirm(question)


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
    always = _terminal_confirm_always if args.yes else None
    reply = jarvis.agent.handle(
        " ".join(args.text),
        Hooks(confirm=confirm, confirm_always=always, waking=lambda: print("(waking up)")),
    )
    print(reply)
    return 0


def _chat_status(cfg: config.Config) -> str:
    voice = _unit_state(JARVIS_UNIT)
    if cfg.strata.mode == "remote":
        return f"voice {voice} · {_probe_strata(cfg).report(cfg.model.model)}"
    model = _strata(cfg).status() if _unit_state(STRATA_UNIT) == "active" else None
    line = f"voice {voice} · model {model or 'server not running'}"
    answer = control.send("status", timeout=1.0)
    if model == "loaded" and answer is not None and answer.get("paused") is True:
        line += " · paused: it unloads when no request is running"
    return line


def _chat_details(cfg: config.Config) -> list[tuple[str, str]]:
    """Return what /status adds to the services: versions, the model and where things live."""
    mode = modes.current()
    if cfg.strata.mode == "remote":
        cfg = with_remote_metadata(cfg, _probe_strata(cfg))
    served = modes.effective(cfg, mode)
    chat = served.chat
    return [
        ("jarvis", __version__),
        ("mode", f"{mode} (/model switches)"),
        ("model", f"{services.model_name(served)}, {served.strata.context:,} tokens of context"),
        ("replies", f"up to {chat.reply_tokens:,} tokens, {chat.max_steps} model calls"),
        (
            "memory",
            f"{chat.memory_detail}, "
            + (
                "as much as the window leaves"
                if chat.memory_tokens is None
                else f"up to {chat.memory_tokens:,} tokens"
            ),
        ),
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
    heading: str  # the tab /config shows it under
    about: str
    risky: str = ""  # what turning it on means, said before it is turned on
    unit: str = ""  # "seconds": shown as 30 s, 10 min, 4 h
    zero: str = ""  # the word /config shows for 0, where 0 means something


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
        "Jarvis also hears requests spoken in the PC's own audio, such as a call or a video. "
        "Whatever they ask to be done always needs your yes.",
        "Jarvis will take requests from voices in the PC's audio, such as a call or a video; "
        "each of their actions still needs your yes.",
    ),
    _Shared(
        "audio",
        "pc_audio_recall_s",
        "Recall PC audio",
        (0, 10, 30, 60),
        "Listening",
        "Keep the last seconds of PC audio in memory so Jarvis can hear what was just said "
        "(needs PC audio on); none keeps nothing.",
        "Jarvis will keep the last seconds of the PC's audio in memory.",
        "seconds",
        "none",
    ),
    _Shared(
        "browser",
        "idle_close_s",
        "Close the browser after",
        (30.0, 60.0, 120.0, 300.0),
        "Chat",
        "How long the headless browser stays open after its last use, in the chat and by "
        "voice; the chat takes it when it opens again.",
        unit="seconds",
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
        "Jarvis acts without your yes, even after reading a web page or a file. "
        "A tool needing the confirm key still waits; an off tool stays off.",
        "Jarvis will act without asking you first, even when a web page or a file could have "
        "steered it.",
    ),
    _Shared(
        "coding",
        "back_after_s",
        "Leave coding mode after",
        (0.0, 600.0, 1800.0, 3600.0, 14400.0),
        "Mode",
        "After coding mode goes unused this long, Jarvis goes back to normal mode. never: it "
        "stays until you switch (/model normal).",
        unit="seconds",
        zero="never",
    ),
    _Shared(
        "coding.chat",
        "auto",
        "Switch mode by itself in the chat",
        (False, True),
        "Mode",
        "The model moves programming work to coding mode and back. Off: only /model and "
        "`jarvis model` switch. The chat takes it when it next opens.",
    ),
    _Shared(
        "coding.voice",
        "auto",
        "Switch mode by itself for speech",
        (False, True),
        "Mode",
        "The same for spoken requests. With a coding model of its own, a switch loads another "
        "model, so the request waits in silence for the load.",
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
            item.unit,
            item.zero,
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


APPLIES_WHEN_IDLE = (
    "Settings saved. They apply when Jarvis is idle: the voice service restarts then, "
    "even if this window is closed."
)
PENDING_CHECK_S = 5.0  # how often the voice service looks for a restart that waits for it
RESTART_WAIT_S = 30.0  # how long it waits to be stopped by a restart systemd accepted


def _apply_from_chat() -> str:
    """Restart the voice service so it takes what the chat's settings changed; never cut a request.

    A restart ends a spoken request in flight. It is made while no voice request can begin
    (`modes.voice_idle`); with one under way, a note is left (`paths.restart_pending_file`) and
    the voice service restarts itself once it is idle (`_restart_when_pending`), whether this
    window is still open or not, and later changes ride on the same restart. The answer is the
    line that says so, else "".

    Raises:
        RestartError: the restart failed, so the voice service runs on the old settings.
    """
    note = paths.restart_pending_file()
    with modes.voice_idle() as idle:
        if idle:
            note.unlink(missing_ok=True)
            if not _restart_if_running():
                raise RestartError(RESTART_FAILED)
            return ""
    write_atomic(note, "settings changed\n", 0o600)
    return APPLIES_WHEN_IDLE


def _watch_for_restart_note(loop: VoiceLoop, stop: threading.Event) -> None:
    """Start the thread that restarts the voice service for settings the chat changed."""
    threading.Thread(
        target=_restart_when_pending, args=(loop, stop), name="jarvis-restart-pending", daemon=True
    ).start()


def _restart_when_pending(loop: VoiceLoop, stop: threading.Event) -> None:
    """Restart the voice service for settings the chat changed, once no request is under way.

    A look is one `stat` of the note every `PENDING_CHECK_S`. The restart is asked of systemd
    without waiting for it, while no voice request can begin; once systemd accepted it the note
    goes, and the lock is kept until this process is stopped, so no request begins in the
    moments before. A restart that was refused leaves the note for the next look. Whatever goes
    wrong is logged and the thread goes on.
    """
    note = paths.restart_pending_file()
    while not stop.wait(PENDING_CHECK_S):
        try:
            if note.exists() and str(loop.state) == "idle":
                _restart_now(note, stop)
        except Exception:
            log.exception("could not restart for the settings the chat changed")


def _restart_now(note: Path, stop: threading.Event) -> None:
    with modes.voice_idle() as idle:
        if not idle:
            return
        log.info("restarting for the settings the chat changed")
        if _restart_if_running(no_block=True):
            note.unlink(missing_ok=True)
            stop.wait(RESTART_WAIT_S)  # the unit stops this process: no request begins meanwhile
        else:
            log.warning("the voice service did not restart: %s", RESTART_FAILED)


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
    return screenshot_folders(home, env)


def _chat_prompt(chat: config.ChatConfig) -> str:
    """Render the chat's instructions for these settings, in the mode Jarvis is in now."""
    mode = modes.current()
    cfg = modes.effective(config.load().model_copy(update={"chat": chat}), mode)
    registry = offered_registry(
        cfg,
        spotify_operations(cfg),
        "chat",
        coding=mode == "coding",
        via="chat",
        can_switch=True,
    )
    return system_prompt(cfg, "chat", coding=mode == "coding", registry=registry)


def _claim_stdio() -> tuple[TextIO, TextIO]:
    """Return the protocol's input and output, and leave nothing else on those two descriptors.

    The protocol keeps private copies of standard input and output. Descriptor 0 then reads
    nothing and descriptor 1 is the error output, so a program a tool starts can neither take
    protocol lines nor write into the protocol.
    """
    source = os.fdopen(os.dup(0), "r", encoding="utf-8", errors="replace", newline="\n")
    out = os.fdopen(os.dup(1), "w", encoding="utf-8", newline="\n")
    sys.stdout.flush()
    nothing = os.open(os.devnull, os.O_RDONLY)
    os.dup2(nothing, 0)
    os.close(nothing)
    os.dup2(2, 1)
    return source, out


class _CodingFirst:
    """Puts the model server on the coding model once, before a `jarvis acp --coding` request."""

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self._ready: config.Config | None = None

    def ready(self, hooks: Hooks) -> tuple[config.Config | None, str]:
        """Return the config ready to use and any line from the first switch."""
        with self._lock:
            if hooks.cancelled() or self._ready is not None:
                return self._ready, ""
            cfg = config.load()
            if not modes.own_model(cfg):
                self._ready = cfg
                return cfg, ""
            hooks.waking()
            if hooks.cancelled():
                return None, ""
            # The same switch as the chat's /model coding, for the owner; nothing is said aloud.
            line = switch_model(
                cfg, "coding", lambda _line: None, "acp", only_if=lambda: not hooks.cancelled()
            )
            if hooks.cancelled():
                return None, ""
            if line.startswith(("Now in", "Already in")):
                self._ready = cfg
            return self._ready, line


def _acp_models(cfg: config.Config, *, coding_first: bool) -> list[acp.ModelChoice]:
    """Return Jarvis's modes as a coding app lists them: model, what it is for, starting effort.

    The title is the model's short name (`modes.model_titles`), which an app's model menu shows
    whole; the description is the mode's purpose.
    """
    titles = modes.model_titles(cfg)
    choices = [
        acp.ModelChoice(
            mode, titles[mode], modes.PURPOSE[mode], modes.effective(cfg, mode).chat.thinking
        )
        for mode in modes.MODES
    ]
    return sorted(choices, key=lambda choice: choice.id != "coding") if coding_first else choices


def _model_choices() -> dict[str, tuple[str, str]]:
    """Return each mode's model name and purpose for the chat's `/model` menu.

    Empty, so the menu offers the modes alone, while the config cannot be read.
    """
    try:
        cfg = config.load()
    except config.ConfigError, OSError, ValueError, RecursionError:
        return {}  # typing a command must not end the chat, whatever the file holds
    return {mode: (modes.list_name(cfg, mode), modes.PURPOSE[mode]) for mode in modes.MODES}


def _at_effort(section: config.ChatConfig, level: str) -> config.ChatConfig:
    """Return `section` thinking at `level`, its ceiling raised to it when lower (as `/effort`)."""
    effort = cast("config.Effort", level)
    ceiling = section.thinking_max
    if config.EFFORTS.index(ceiling) < config.EFFORTS.index(effort):
        ceiling = effort
    return section.model_copy(update={"thinking": effort, "thinking_max": ceiling})


class _NormalSession:
    """A `jarvis acp` session's agent in the normal mode, which follows the owner's mode."""

    def __init__(self, cfg: config.Config, session_id: str) -> None:
        super().__init__()
        self._section = cfg.chat
        self._agent = build(cfg, via="acp", session=session_id, units=StrataUnits()).agent

    def handle(self, text: str, hooks: Hooks, *, pictures: Sequence[Picture] = ()) -> str:
        """Answer `text`."""
        return self._agent.handle(text, hooks, pictures=pictures)

    def set_effort(self, level: str) -> None:
        """Think at `level` from the next request on."""
        self._agent.set_thinking(_at_effort(self._section, level))

    def set_folder(self, folder: str) -> None:
        """Tell the model the session's project folder."""
        self._agent.set_folder(folder)

    def save(self, path: Path) -> None:
        """Keep the conversation in `path`."""
        self._agent.memory.save(path)

    def load(self, path: Path) -> bool:
        """Go on from the conversation kept in `path`."""
        return self._agent.memory.load(path)

    def history(self) -> list[tuple[str, str]]:
        """Return the remembered turns as (request, reply)."""
        return [(turn.request, turn.reply) for turn in self._agent.memory.turns]

    def forget(self) -> None:
        """Start the conversation afresh (`/clear`)."""
        self._agent.forget()


class _CodingSession:
    """An agent whose first request waits for the coding model, and says how that went."""

    def __init__(self, cfg: config.Config, session_id: str, first: _CodingFirst) -> None:
        super().__init__()
        self._config: config.Config = cfg
        self._session_id = session_id
        self._agent = build(cfg, via="acp", session=session_id, mode="coding").agent
        self._first = first
        self._effort = ""  # the effort the session chose, kept across a new agent
        self._folder = ""  # the session's project folder, kept across a new agent the same way

    def set_effort(self, level: str) -> None:
        """Think at `level` from the next request on."""
        self._effort = level
        self._agent.set_thinking(_at_effort(modes.effective(self._config, "coding").chat, level))

    def set_folder(self, folder: str) -> None:
        """Tell the model the session's project folder."""
        self._folder = folder
        self._agent.set_folder(folder)

    def save(self, path: Path) -> None:
        """Keep the conversation in `path`."""
        self._agent.memory.save(path)

    def load(self, path: Path) -> bool:
        """Go on from the conversation kept in `path`."""
        return self._agent.memory.load(path)

    def history(self) -> list[tuple[str, str]]:
        """Return the remembered turns as (request, reply)."""
        return [(turn.request, turn.reply) for turn in self._agent.memory.turns]

    def forget(self) -> None:
        """Start the conversation afresh (`/clear`)."""
        self._agent.forget()

    def handle(self, text: str, hooks: Hooks, *, pictures: Sequence[Picture] = ()) -> str:
        """Answer `text` on the coding model; a switch that failed is the answer instead."""
        cfg, line = self._first.ready(hooks)
        if cfg is None or hooks.cancelled():
            return line
        if cfg != self._config:
            try:
                agent = build(cfg, via="acp", session=self._session_id, mode="coding").agent
            except SetupError as exc:
                return str(exc)
            agent.memory.adopt(self._agent.memory)  # a reopened session's talk is kept
            self._agent, self._config = agent, cfg
            if self._effort:
                self.set_effort(self._effort)
            if self._folder:
                self._agent.set_folder(self._folder)
        if line and hooks.draft is not None:
            hooks.draft(line)
        if hooks.cancelled():
            return ""
        return self._agent.handle(text, hooks, pictures=pictures)


def cmd_acp(args: argparse.Namespace) -> int:
    """Speak the Agent Client Protocol on standard input and output, for a coding app."""
    cfg = _load()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    first = _CodingFirst()
    default: modes.Mode = "coding" if args.coding else "normal"

    def make(session_id: str, model: str = "") -> acp.AgentSession:
        try:
            if (model or default) == "coding":
                return _CodingSession(cfg, session_id, first)
            return _NormalSession(cfg, session_id)
        except SetupError as exc:
            raise acp.StartError(str(exc)) from exc

    # Also made when nothing is pushed: it tells a forged message from a real one.
    pusher = t3push.T3Push(cfg.acp.t3_command, cfg.acp.t3_provider, enabled=cfg.acp.push_to_t3)
    source, out = _claim_stdio()
    return acp.serve(
        source,
        out,
        make,
        names=(cfg.persona.name,),
        version=__version__,
        follow=control.subscribe,
        pusher=pusher,
        record=AuditLog(paths.audit_file(), tags={"via": "acp"}).record,
        models=_acp_models(cfg, coding_first=args.coding),
        store=_acp_store(cfg.chat.archive_keep),
    )


def _acp_store(keep: int) -> Path | None:
    """Return the folder T3 Code's sessions are kept in; None when `chat.archive_keep` is 0.

    With 0, the sessions an earlier setting kept are deleted, as the chat's archive is.
    """
    store = paths.acp_sessions_dir()
    if keep:
        return store
    try:
        gone = acpstore.forget_kept(store)
    except OSError as exc:
        log.warning("the kept T3 Code sessions could not be deleted: %s", exc)
    else:
        if gone:
            log.info("deleted %d kept T3 Code sessions: chat.archive_keep is 0", gone)
    return None


def _export_folder(cfg: config.Config) -> Path:
    """Where the chat saves an exported conversation: the workspace, else ~/Documents or home."""
    folder = workspace_path(cfg.files, Path.home())
    if not workspace_problem(folder, create=True):
        return folder
    documents = Path.home() / "Documents"
    return documents if documents.is_dir() else Path.home()


def cmd_chat(_args: argparse.Namespace) -> int:
    """Open the chat window: typed requests, and everything Jarvis does, live."""
    from jarvis.chat.app import ChatApp, Services

    cfg = _load()
    session_id = secrets.token_hex(8)
    agent = None
    followed = None
    todos = questions = None
    unavailable = ""
    pictures = reads_pictures(cfg)
    try:
        built = build(cfg, via="chat", session=session_id, units=StrataUnits())
        agent, followed = built.agent, built.modes
        todos, questions = built.todos, built.questions
        if cfg.strata.mode == "remote":
            pictures = built.pictures
    except SetupError as exc:
        unavailable = str(exc)
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
        model_choices=_model_choices,
        shared=_shared_settings,
        set_shared=_set_shared,
        apply=_apply_from_chat,
        history=InputHistory(paths.state_dir() / "chat-history", cfg.chat.history_size),
        archive=_chat_archive(cfg.chat.archive_keep),
        export_dir=_export_folder(cfg),
        paste=clipboard.read,
        pictures=pictures,
        screenshots=_screenshot_folders(Path.home(), os.environ),
        shown=chat_pictures(),
        twelve_hour=cfg.clock.twelve_hour,
        announcements=control.subscribe,
        todos=todos,
        questions=questions,
    )
    window = ChatApp(
        AuditTail(paths.audit_file()),
        agent,
        unavailable=unavailable,
        name=cfg.persona.name,
        settings=cfg.chat,
        services=services,
        session_id=session_id,
    )
    audit = AuditLog(paths.audit_file(), tags={"via": "chat", "session": session_id})

    def take(words: str) -> handoff.State:
        # Written first: the request it starts follows it in the log, marked as spoken.
        audit.record("chat_handoff_received", text=words)
        state = window.take_handoff(words)
        audit.record("chat_handoff_placed", state=state.value)
        return state

    listener = handoff.ChatListener(
        session_id, take, refused=lambda why: audit.record("chat_handoff_refused", reason=why)
    )
    try:
        listener.start()  # the voice service hands spoken instructions over here
    except OSError as exc:
        log.warning("the chat cannot take spoken hand-offs: %s", exc)
    try:
        window.run()
    finally:
        listener.close()
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


def _speaker(cfg: config.Config, recording: OutputRecorder | None = None) -> Speaker:
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
    return Speaker(
        synthesizer, own_wake=own_wake, record_output=None if recording is None else recording.feed
    )


def _serve_controls(handler: control.Controls, audit: AuditLog) -> control.ControlServer:
    """Start the control socket; a run without one still listens, so a failure is only logged."""
    controls = control.ControlServer(handler, audit=audit)
    try:
        controls.start()
    except OSError as exc:
        log.warning("no control socket (status, pause and resume will not reach this run): %s", exc)
    return controls


def _load_client() -> config.Config:
    """Load the config for a command that needs the whole client.

    Raises:
        SystemExit: this machine is a backend only, which has no client.
    """
    cfg = _load()
    if cfg.strata.mode == "backend":
        print(
            "jarvis: backend mode only serves the model. Run this command on the Jarvis client.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return cfg


def _remote_status(
    cfg: config.Config, model: StrataModel
) -> Callable[[], dict[str, object]] | None:
    """What the control server reports about a remote model (a probe, never a generation)."""
    if cfg.strata.mode != "remote":
        return None
    return lambda: _probe_fields(model.probe(), model.config.model)


def _load_for_run() -> config.Config:
    """Load the config for the voice service, after dropping the restart note it makes true.

    What a note stood for is saved in the config already; one written after the drop is kept
    for the watcher.
    """
    paths.restart_pending_file().unlink(missing_ok=True)
    return _load_client()


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
    cfg = _load_for_run()
    # The hint is the agent's, built after the transcribers that `pc_audio_listen` needs; it is
    # first asked for once a request is heard.
    request, answer = _transcribers(cfg, lambda: jarvis.hint())  # ruff: ignore[unnecessary-lambda] - `jarvis` is built below
    mic = PausableMic(Microphone(cfg.audio.source, stall_s=MIC_STALL_S))
    listening = _pc_listening(cfg, answer, paused=lambda: mic.paused)
    recording = OutputRecorder(paths.state_dir() / "recordings")
    try:
        jarvis = build(
            cfg, via=VOICE_VIA, units=StrataUnits(), listening=listening, recording=recording
        )
    except SetupError as exc:
        print(f"jarvis: {exc}", file=sys.stderr)
        return 2
    speaker = _speaker(cfg, recording)
    pc_audio = AppAudio(cfg.audio.pc_audio_apps) if cfg.audio.listen_to_pc else None
    stop_requested, refresh_stop = threading.Event(), threading.Event()

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
    if jarvis.ops is not None:
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
    jobs = JobStore(paths.schedule_file())
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
        due=jobs.take_due,
        schedule_ready=jobs.may_be_due,
        watching=jarvis.watch,
    )
    handler = control.Controls(
        loop,
        mic,
        jarvis.model,
        jarvis.audit,
        keys=keys,
        prime=jarvis.agent.prime,
        remote_status=_remote_status(cfg, jarvis.model),
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
        loop,
        stop_requested,
        _serve_controls(handler, jarvis.audit),
        voxtype,
        wakes,
        recording=recording,
    )


MODE_CHECK_S = 60.0  # how often the voice service looks whether coding mode idled too long


def _back_when_idle(switcher: modes.Switcher, loop: VoiceLoop, stop: threading.Event) -> None:
    """Return to the normal mode once coding mode went unused too long, between requests.

    The thread sleeps between looks, and a look reads one small file.
    """
    while not stop.wait(MODE_CHECK_S):
        if str(loop.state) == "idle":
            switcher.back_if_idle()


def _run_until_stopped(  # ruff: ignore[too-many-arguments] - the live voice service's resources
    loop: VoiceLoop,
    stop_requested: threading.Event,
    controls: control.ControlServer,
    voxtype: VoxtypeWatch,
    wakes: list[WakeWord | FallbackWake],
    *,
    recording: OutputRecorder | None = None,
) -> int:
    """Run `loop` with Voxtype's status followed, then end what the run started."""
    from jarvis.audio import FallbackWake  # ruff: ignore[import-outside-top-level] - loads numpy and ONNX Runtime, which most commands never need

    try:
        _watch_for_restart_note(loop, stop_requested)
        voxtype.start()
        loop.run(stop_requested.is_set)
    except Exception:
        if stop_requested.is_set():
            return 0
        raise
    finally:
        controls.close()
        if recording is not None:
            recording.close()
        voxtype.stop()
        for wake in wakes:  # an Azure helper ends with its input; end it now, not at exit
            if isinstance(wake, FallbackWake):
                wake.close()
    return 0


def _model_files(cfg: config.Config) -> list[tuple[str, Path]]:
    """Return the (role, file) pairs the doctor can check against a known release."""
    from jarvis.audio import KOKORO_PREFIX, voice_files  # ruff: ignore[import-outside-top-level] - loads numpy and ONNX Runtime, which most commands never need
    from jarvis.sing.engines import singer_files  # ruff: ignore[import-outside-top-level] - imports jarvis.audio, as above

    found = [*(("wake", path) for path in wake_files(cfg)), ("vad", cfg.data_dir / VAD_MODEL)]
    if cfg.audio.voice.startswith(KOKORO_PREFIX):
        model, voices = voice_files(cfg.data_dir, cfg.audio.voice)
        found += [("kokoro-model", model), ("kokoro-voices", voices)]
    found += singer_files(cfg)
    if cfg.strata.mode == "local":
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
    if (
        cfg.strata.mode == "local"
        and services.fills_the_card(run)
        and not cfg.strata.vram_reserve_mib
    ):
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
    for tool, package, said in (
        ("grim", "grim", "screenshots can't be taken"),
        ("wl-copy", "wl-clipboard", "screenshots can't be copied to the clipboard"),
    ):
        if not shutil.which(tool):
            notes.append(f"note tool {tool} missing: {said} (`pacman -S {package}`)")
    if cfg.chat.browser or cfg.agent.browser:  # the headless browser is part of Jarvis
        program = shutil.which(browser.PROGRAM) is not None
        checks.append(
            (
                program,
                f"tool {browser.PROGRAM}",
                "install it as the README's Set up says (install.sh)",
            )
        )
        if program:
            checks.append(
                (
                    browser.chrome_found(Path.home()),
                    f"Chrome for {browser.PROGRAM}",
                    f"run `{browser.PROGRAM} install` (it downloads one into your home folder)",
                )
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


def _ram_report(
    cfg: config.Config, *, essential: bool
) -> tuple[list[tuple[bool, str, str]], list[str]]:
    """Return the doctor's line on whether the model fits in the memory available now.

    A server that runs holds its model, so its memory counts as free (`ramguard.held_by_unit`).
    The update's health check only notes it: an update is not undone because a game is running.
    """
    served = modes.effective(cfg, modes.current()) if cfg.strata.mode == "local" else cfg
    verdict = ramguard.check(
        ramguard.served_model(served),
        cfg.strata,
        credit=ramguard.held_by_unit(STRATA_UNIT),
    )
    if not verdict.known:
        return [], []
    what = f"memory for the model ({verdict.summary})"
    if essential and verdict.short:
        return [], [
            f"note {what}: close a game or other heavy program, or set strata.allow_low_ram"
        ]
    fix = "close a game or other heavy program, or set strata.allow_low_ram = true"
    return [(not verdict.short, what, fix)], []


def _memlock_notes(cfg: config.Config) -> list[str]:
    """Note that the model cannot be locked in RAM when the user manager's limit keeps it out."""
    limit = _user_memlock_limit()
    if limit in {"", "infinity"}:
        return []
    for each, where in ((cfg, ""), (modes.effective(cfg, "coding"), " in coding mode")):
        if not each.strata.keep_in_ram:
            continue
        if limit.isdigit() and int(limit) >= _model_bytes(each):
            continue  # the whole model fits under the limit
        note = (
            f"note the model can't be locked in RAM{where}: your user services may lock {limit} "
            "bytes at most (it still stays out of swap); see "
            "docs/configuration.md#keeping-the-model-in-ram to raise it"
        )
        return [note]
    return []


def _greedy_notes(cfg: config.Config) -> list[str]:
    """Warn of a mode whose requests Strata would decode greedily.

    Strata fills in the model card's sampling only for a request that asks for some sampling;
    one that asks for none stays greedy while its engine drafts with MTP (`--mtp`), unless the
    run config's own `sampling` block sets it. A mode whose temperature is the model's own
    (`"model"`, sent as none) on such a run config is greedy, which can repeat itself endlessly.
    """
    if cfg.strata.mode != "local":
        return []
    notes: list[str] = []
    keys: tuple[tuple[modes.Mode, str], ...] = (
        ("normal", "model.temperature"),
        ("coding", "coding.temperature"),
    )
    for mode, key in keys:
        served = modes.effective(cfg, mode)
        if served.model.temperature is not None:
            continue
        try:
            run = services.read_run_config(served)
        except services.StrataSetupError:
            continue  # the run config's own check says what is wrong with it
        if "--mtp" not in run.args or run.data.get("sampling"):
            continue
        name = "the normal mode" if mode == "normal" else "coding mode"
        notes.append(
            f"warning: {name} is decoded greedily: its run config {run.path} drafts with MTP and "
            f'{key} is "model", so Strata uses none of the model card\'s sampling; set {key} to '
            "the temperature the model's card gives (or a `sampling` block in the run config)"
        )
    return notes


def essential_checks(cfg: config.Config) -> tuple[list[tuple[bool, str, str]], list[str]]:
    """Return the local checks shared by doctor and the update health check."""
    from jarvis.audio import voice_files  # ruff: ignore[import-outside-top-level] - loads numpy and ONNX Runtime, which most commands never need
    from jarvis.sing.engines import doctor_checks  # ruff: ignore[import-outside-top-level] - imports jarvis.audio, as above

    config_check = (
        paths.config_file().is_file(),
        f"config {paths.config_file()}",
        "run jarvis setup",
    )
    key_check = (_has_key(cfg), "Strata API key", _setup_fix(cfg))
    if cfg.strata.mode == "backend":  # only the model service lives here: check just that
        unit = services.unit_dir() / STRATA_UNIT
        unit_check = (unit.is_file(), f"backend unit {unit}", "run jarvis setup --backend")
        return [config_check, key_check, unit_check, *_model_checks(cfg)], []
    wake_checks, wake_notes = _wake_checks(cfg)
    tool_checks, tool_notes = _tool_checks(cfg)
    sing_checks, sing_notes = doctor_checks(cfg)
    files = [cfg.data_dir / VAD_MODEL, *voice_files(cfg.data_dir, cfg.audio.voice)]
    local = cfg.strata.mode == "local"
    return [
        config_check,
        key_check,
        *wake_checks,
        *((f.is_file(), f"file {f}", FETCH_MODELS) for f in files),
        *_whisper_model(cfg),
        *(_model_url(cfg) if local else []),
        *(_model_checks(cfg) if local else []),
        *(_coding_checks(cfg) if local else []),
        _workspace_check(cfg),
        *tool_checks,
        *sing_checks,
        *_source_checks(cfg),
        *_azure_checks(cfg),
        *_lock_checks(cfg),
    ], [*wake_notes, *tool_notes, *sing_notes]


def _has_key(cfg: config.Config) -> bool:
    try:
        return bool(services.read_key(cfg))
    except OSError, ValueError:
        return False


def _setup_fix(cfg: config.Config) -> str:
    return {
        "local": "run jarvis setup",
        "remote": "run jarvis setup --remote <url>",
        "backend": "run jarvis setup --backend",
    }[cfg.strata.mode]


def strata_status(cfg: config.Config) -> str | None:
    """Read Strata's state without generating tokens or loading its model."""
    if cfg.strata.mode != "local":
        probe = _probe_strata(cfg)
        return (probe.state or "unknown") if probe.reachable else None
    return _strata(cfg).status() if read_api_key(strata_key_file()) else None


def _split_model_check(cfg: config.Config) -> tuple[bool, str, str]:
    """The reachability, model name and latency of a Strata that runs on another machine."""
    probe = _probe_strata(cfg)
    if cfg.strata.mode == "remote":
        return (
            probe.reachable and probe.state != "failed",
            probe.report(cfg.model.model),
            _remote_fix(cfg),
        )
    return (
        probe.reachable and probe.state != "failed",
        probe.report(services.model_name(cfg), "Strata backend"),
        "check the key, then run jarvis start on the backend",
    )


def _remote_fix(cfg: config.Config) -> str:
    """Say what to try when the remote model server does not answer."""
    fix = "check the backend, tunnel or TLS settings and the key from jarvis setup --remote"
    parts = urlsplit(cfg.strata.url)
    if parts.hostname not in {"127.0.0.1", "localhost", "::1"}:
        return fix
    return (
        f"{fix}. An SSH tunnel ends at a reboot or logout; start it again with `ssh -fNT -o "
        f"ExitOnForwardFailure=yes -L 127.0.0.1:{parts.port or 80}:<backend listen address>:"
        "<backend port> -- <ssh-target>` (jarvis setup --backend on the backend prints it)"
    )


def print_checks(checks: Sequence[tuple[bool, str, str]]) -> int:
    """Print the checks and return their failure count."""
    for ok, what, fix in checks:
        print(cliout.check(what, fix, ok=ok))
    return sum(not ok for ok, _, _ in checks)


@dataclass(frozen=True, slots=True)
class DoctorReport:
    """What `jarvis doctor` found: its checks, its notes, and how many releases it verified."""

    checks: list[tuple[bool, str, str]]
    notes: list[str]
    verified: list[str]  # "name: release, sha256 verified", one per file it matched

    @property
    def failed(self) -> int:
        """How many checks failed."""
        return sum(not ok for ok, _, _ in self.checks)

    @property
    def summary(self) -> str:
        """The closing line: how many checks ran and which failed."""
        return cliout.summary(self.checks, extra=len(self.verified))


def _t3_notes(cfg: config.Config) -> list[str]:
    """Say how to add Jarvis to T3 Code when T3 Code is set up on this PC, and what is off."""
    if not t3push.t3_home().is_dir():
        return []
    program = shutil.which("jarvis") or "jarvis"
    where = "Settings, Providers, Add provider"
    head = (
        f"note T3 Code is set up here: to talk to Jarvis in it, add a Local ACP command ({where})"
    )
    notes = [
        (
            f"{head} with the executable {program} and the argument acp; "
            "it lists the normal and coding modes as models"
        )
    ]
    pusher = t3push.T3Push(cfg.acp.t3_command, cfg.acp.t3_provider)
    if cfg.acp.push_to_t3 and (problem := pusher.provider_problem()):
        notes.append(f"note {problem}")
    return notes


def _remote_notes(cfg: config.Config, *, no_service: bool) -> list[str]:
    """Notes for a client whose model runs on another machine."""
    notes: list[str] = []
    if cfg.strata.plain_lan():
        notes.append("warning: unencrypted HTTP sends the model's key and requests in cleartext")
    if not no_service and _unit_state(STRATA_UNIT) in {"active", "activating"}:
        notes.append(
            f"note {STRATA_UNIT} still runs on this PC and holds RAM and VRAM this remote client "
            f"does not use; free them with: systemctl --user disable --now {STRATA_UNIT}"
        )
    return notes


def doctor_report(args: argparse.Namespace) -> DoctorReport:
    """Run every check `jarvis doctor` runs; print nothing."""
    cfg = _load()
    checks, notes = essential_checks(cfg)
    if cfg.strata.mode == "backend":
        return _backend_report(cfg, checks, wait=args.wait, no_service=args.no_service)
    if cfg.spotify.client_id:
        login = _spotify_login(cfg)
        if not login[0] and args.essential:
            notes.append(f"note {login[1]}: {login[2]}")
        else:
            checks.append(login)
        notes.extend(_spotify_scope_notes(cfg))
    else:
        notes.append(
            "note Spotify is off; set a client ID and run jarvis spotify-login to connect it"
        )
    if not args.no_service:
        if cfg.strata.mode == "remote":
            if args.wait:
                update.wait_for(lambda: strata_status(cfg))
            checks.append(_split_model_check(cfg))
        else:
            state = update.wait_for(lambda: strata_status(cfg)) if args.wait else strata_status(cfg)
            checks.append(
                (
                    state is not None and state != "failed",
                    f"Strata server ({state or 'unreachable'})",
                    "jarvis start (it clears a start limit first)",
                )
            )
    release_lines, unreadable = _release_lines(cfg)
    checks.extend(unreadable)
    if cfg.strata.mode == "local":
        ram_checks, ram_notes = _ram_report(cfg, essential=args.essential)
        checks.extend(ram_checks)
        notes.extend([*ram_notes, *_memlock_notes(cfg), *_greedy_notes(cfg)])
    else:
        notes.extend(_remote_notes(cfg, no_service=args.no_service))
    notes.extend([*_t3_notes(cfg), *update.doctor_notes(cfg, launch_for_update)])
    notes.extend(skills.doctor_notes())
    notes.extend(pluginsync.doctor_notes(cfg, launch_for_update))
    verified = [line.removeprefix("ok").strip() for line in release_lines if line.startswith("ok ")]
    notes.extend(line for line in release_lines if not line.startswith("ok "))
    return DoctorReport(checks, notes, verified)


def _backend_report(
    cfg: config.Config, checks: list[tuple[bool, str, str]], *, wait: bool, no_service: bool
) -> DoctorReport:
    """The doctor on a backend-only machine: its unit, key, model files and server, nothing else."""
    if not no_service:
        if wait:
            update.wait_for(lambda: strata_status(cfg))
        checks.append(_split_model_check(cfg))
    notes = [_PLAIN_BACKEND] if cfg.strata.serves_beyond_loopback() else []
    ram_checks, ram_notes = _ram_report(cfg, essential=False)
    return DoctorReport([*checks, *ram_checks], [*notes, *ram_notes, *_memlock_notes(cfg)], [])


def cmd_doctor(args: argparse.Namespace) -> int:
    """Print one line per check with its fix, then the notes; never loads the model."""
    report = doctor_report(args)
    print_checks(report.checks)
    for line in report.verified:
        print(cliout.check(line, ok=True))
    for line in report.notes:
        print(cliout.note(line))
    print(report.summary)
    return int(bool(report.failed))


FETCH_MODELS = "uv run tools/get_models.py (or see docs/models.md)"  # the fix for a missing model


def launch_for_update(  # ruff: ignore[too-many-arguments] - the launcher's own keyword options
    args: tuple[str, ...],
    *,
    cwd: Path,
    timeout: float,
    env: Mapping[str, str],
    show: bool,
    abort_when: Callable[[], bool] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Start one program `jarvis update` needs (git, uv, systemctl, the updated Jarvis).

    Raises:
        update.UpdateError: it ran past `timeout` or `abort_when` said yes, and it was stopped.
    """
    if abort_when is not None:
        return _launch_watched(args, cwd=cwd, timeout=timeout, env=env, abort_when=abort_when)
    try:
        return subprocess.run(
            args,
            cwd=cwd,
            check=False,
            capture_output=not show,
            text=True,
            env=dict(env),
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        msg = f"{Path(args[0]).name} ran past {timeout:g} s and was stopped"
        raise update.UpdateError(msg) from exc


def _update_plugins(cfg: config.Config, audit: AuditLog) -> None:
    """Bring the plugins up to date as part of `jarvis update --apply`."""
    pluginsync.update_all(cfg, launch_for_update, audit)


def _launch_watched(
    args: tuple[str, ...],
    *,
    cwd: Path,
    timeout: float,
    env: Mapping[str, str],
    abort_when: Callable[[], bool],
) -> subprocess.CompletedProcess[str]:
    """Run a program, asking `abort_when` twice a second and stopping it once it says yes.

    Raises:
        update.UpdateError: it ran past `timeout` or `abort_when` said yes, and it was stopped.
    """
    deadline = time.monotonic() + timeout
    with subprocess.Popen(
        args,
        cwd=cwd,
        env=dict(env),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ) as proc:
        while True:
            try:
                out, err = proc.communicate(timeout=0.5)
            except subprocess.TimeoutExpired:
                reason = "was stopped" if abort_when() else ""
                if not reason and time.monotonic() > deadline:
                    reason = f"ran past {timeout:g} s and was stopped"
                if reason:
                    proc.kill()
                    proc.communicate()
                    msg = f"{Path(args[0]).name} {reason}"
                    raise update.UpdateError(msg) from None
            else:
                return subprocess.CompletedProcess(args, proc.returncode, out, err)


def finish_update() -> int:
    """Run the health work in the checkout's newly synced environment."""
    cfg = _load()
    if sys.argv[1] == "ready":
        if "strata" in sys.argv[2:]:
            state = update.wait_for(lambda: strata_status(cfg))
            if state is None or state == "failed":
                print(f"jarvis: model server is {state or 'unreachable'}", file=sys.stderr)
                return 1
        if "voice" in sys.argv[2:]:
            reply = update.wait_for(lambda: control.send("status", timeout=1))
            if not reply or not reply.get("ok"):
                print("jarvis: voice service did not open its control socket", file=sys.stderr)
                return 1
        return 0
    if sys.argv[1] != "health":
        return 2
    match cfg.strata.mode:
        case "remote":
            services.write_voice_unit()
        case "backend":
            services.write_strata_profile(cfg)
            services.write_strata_unit(cfg, services.key_file(cfg))
        case _:
            served, _profile = _write_mode_profiles(cfg)
            services.write_units(served, paths.strata_key_file())
    checks, _notes = essential_checks(cfg)
    return int(bool(print_checks([check for check in checks if not check[0]])))


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
        FETCH_MODELS,
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
    return strata_model(cfg, services.read_key(cfg) or "")


def _probe_strata(cfg: config.Config) -> ModelStatus:
    try:
        key = services.read_key(cfg)
    except OSError, ValueError:
        return ModelStatus(reachable=False, latency_ms=0, issue="key could not be read")
    if not key:
        return ModelStatus(reachable=False, latency_ms=0, issue="key missing for this endpoint")
    return strata_model(cfg, key).probe()


def _probe_fields(probe: ModelStatus, name: str) -> dict[str, object]:
    return {
        "model": (probe.state or "unknown") if probe.reachable else "unreachable",
        "model_name": probe.name or name,
        "model_latency_ms": probe.latency_ms,
        "model_reachable": probe.reachable,
        "model_issue": probe.issue,
    }


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
    return modes.Switcher(
        cfg, StrataUnits(), _strata, audit, busy=voice_busy, ram_reason=ramguard.switch_reason
    )


def model_report(cfg: config.Config) -> list[str]:
    """Return what `jarvis model` and the chat's /model show: the mode and each mode's model."""
    state = modes.read_state()
    if cfg.strata.mode == "remote":
        probe = _probe_strata(cfg)
        served = with_remote_metadata(cfg, probe)
        return [
            f"mode: {state.mode}",
            probe.report(cfg.model.model),
            f"context: {served.strata.context:,} tokens in either mode",
            "Use model normal or model coding to change client settings without changing Strata.",
        ]
    if cfg.strata.mode == "backend":
        return [_probe_strata(cfg).report(services.model_name(cfg), "Strata backend")]
    running = _unit_state(STRATA_UNIT) == "active"
    loaded = (_strata(cfg).status() if running else None) or "server not running"
    lines = [f"mode: {state.mode} (model {loaded})"]
    for mode in modes.MODES:
        served = modes.effective(cfg, mode)
        mark = "*" if mode == state.mode else " "
        name = services.model_title(served) or services.model_name(served)
        shown = f"{name}, {served.strata.context:,} tokens of context"
        lines.extend((f"{mark} {mode}: {shown}", f"    {modes.PURPOSE[mode]}"))
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


def switch_model(
    cfg: config.Config,
    to: str,
    say: Callable[[str], None],
    source: str,
    *,
    only_if: Callable[[], bool] = lambda: True,
) -> str:
    """Switch to mode `to` for the owner if `only_if` still allows it under the mode lock."""
    if to not in modes.MODES:
        return f"No mode {to}: it is normal or coding."
    wanted: modes.Mode = "coding" if to == "coding" else "normal"
    done = _switcher(cfg, source).switch(wanted, "owner", say, only_if=only_if)
    if done.error:
        return done.error
    if done.mode != wanted:
        return ""  # the request was stopped before the switch
    name = modes.model_name(cfg, done.mode)
    if cfg.strata.mode == "remote":
        return (
            f"Now in {done.mode} mode on this client."
            if done.changed
            else (f"Already in {done.mode} mode on this client. The remote model stays the same.")
        )
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
    cfg = _load()
    if cfg.strata.mode != "local":
        return _split_status(cfg, as_json=args.json)
    reply = control.send("status")
    running = reply is not None
    if reply is not None and reply.get("ok"):
        shown = {key: value for key, value in reply.items() if key != "ok"}
    else:
        strata = _strata(cfg)
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


def _split_status(cfg: config.Config, *, as_json: bool) -> int:
    remote = cfg.strata.mode == "remote"
    reply = control.send("status") if remote else None
    unit = JARVIS_UNIT if remote else STRATA_UNIT
    running = _unit_state(unit) in {"active", "activating"}
    if reply and reply.get("ok") and "model_reachable" in reply:
        shown = {key: value for key, value in reply.items() if key != "ok"}
    else:
        probe = _probe_strata(cfg)
        name = cfg.model.model if remote else services.model_name(cfg)
        shown = _probe_fields(probe, name)
        shown["state"] = str(reply.get("state", "unavailable")) if reply else "unavailable"
        shown["paused"] = bool(reply and reply.get("paused"))
    shown["running"] = running
    if as_json:
        print(json.dumps(shown))
        return 0
    label = "remote model server" if remote else "Strata backend"
    prefix = f"jarvis: {shown['state'] if running else 'not running'}, " if remote else ""
    detail = (
        f"reachable, {shown['model_name']} ({shown['model']}, {shown['model_latency_ms']} ms)"
        if shown["model_reachable"]
        else f"{shown['model_issue'] or 'unreachable'} ({shown['model_latency_ms']} ms)"
    )
    print(f"{prefix}{label} {detail}")
    return 0 if shown["model_reachable"] else 1


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
    cfg = _load()
    if cfg.strata.mode == "remote":
        print("The remote model is managed on the backend. Run jarvis unload there.")
        return 1
    ok = _strata(cfg).unload()
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


_RUNNING = {"active", "activating", "reloading"}


def _no_room(cfg: config.Config, where: str) -> bool:
    """Print why the model server must not load its model now, and return True when it must not.

    The memory the running server holds counts as free: a restart gives it back first. A load
    that would leave less than `strata.min_free_ram_mib` is refused, as the unit's own check
    refuses it too; `strata.allow_low_ram` lets it through.
    """
    why = ramguard.reload_reason(modes.effective(cfg, modes.current()), where)
    if why:
        print(f"jarvis: {why}", file=sys.stderr)
    return bool(why)


def _not_started(cfg: config.Config, where: str) -> bool:
    """Say so and return True when `systemctl` reported a start and the model server is not running.

    A unit whose condition fails is skipped without failing, so `systemctl start` says 0 with the
    server stopped: the memory check of its unit, a model file that is not there. When the load
    would not fit now, that reason is printed, as for a refusal up front.
    """
    if _unit_state(STRATA_UNIT) in _RUNNING:
        return False
    print(
        "jarvis: the model server did not start; see jarvis doctor and "
        "journalctl --user -u jarvis-strata",
        file=sys.stderr,
    )
    _no_room(cfg, where)
    return True


def cmd_start(_args: argparse.Namespace) -> int:
    """Start both services, unless loading the model would take the memory a program needs."""
    cfg = _load()
    if cfg.strata.mode != "local":
        return _split_lifecycle("start", cfg)
    if _unit_state(STRATA_UNIT) not in _RUNNING and _no_room(cfg, "start"):
        print("jarvis: nothing was started", file=sys.stderr)
        return 1
    _reset_failed(STRATA_UNIT, JARVIS_UNIT)
    code = _systemctl("start", STRATA_UNIT, JARVIS_UNIT)
    print(f"jarvis: {_unit_state(JARVIS_UNIT)}, model server: {_unit_state(STRATA_UNIT)}")
    return code or int(_not_started(cfg, "start"))


def cmd_stop(_args: argparse.Namespace) -> int:
    """Stop the voice service, then the model server (which frees the model)."""
    cfg = _load()
    if cfg.strata.mode != "local":
        return _split_lifecycle("stop", cfg)
    code = _systemctl("stop", JARVIS_UNIT, STRATA_UNIT)
    print(f"jarvis: {_unit_state(JARVIS_UNIT)}, model server: {_unit_state(STRATA_UNIT)}")
    return code


def _split_lifecycle(action: str, cfg: config.Config) -> int:
    remote = cfg.strata.mode == "remote"
    unit = JARVIS_UNIT if remote else STRATA_UNIT
    loads = not remote and (action == "restart" or _unit_state(unit) not in _RUNNING)
    if action != "stop":
        if loads and _no_room(cfg, action):
            print("jarvis: nothing was changed", file=sys.stderr)
            return 1
        try:
            if _prepare_split_unit(cfg, action):
                return 1
        except config.ConfigError, OSError, ValueError:
            print(
                f"jarvis: could not prepare {unit}. Check jarvis setup and jarvis doctor.",
                file=sys.stderr,
            )
            return 1
        _reset_failed(unit)
    code = _systemctl(action, unit)
    label = "jarvis" if remote else "Strata backend"
    print(f"{label}: {_unit_state(unit)}")
    return code or int(action != "stop" and not remote and _not_started(cfg, action))


def _prepare_split_unit(cfg: config.Config, action: str) -> int:
    if cfg.strata.mode == "remote":
        voice = services.unit_dir() / JARVIS_UNIT
        if voice.is_file() and STRATA_UNIT in voice.read_text(encoding="utf-8"):
            services.write_voice_unit()
    elif action == "restart":
        with modes.locked():
            services.write_strata_profile(cfg)
            services.write_strata_unit(cfg, services.key_file(cfg))
    else:
        return 0
    return _systemctl("daemon-reload")


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


def _refresh_strata() -> None:
    """Write Strata's profile and unit from the config, so a setting changed there reaches it.

    Files that already hold the same text keep their time, so an unchanged Strata is not
    restarted. Only `jarvis setup` writes the voice unit, which names the environment Jarvis
    runs from, and before setup has written Strata's unit nothing is written. A config that
    cannot be read or built, or a file that cannot be written, is said and Jarvis restarts anyway.
    """
    try:
        cfg = config.load()
        if (
            cfg.strata.mode != "local"
            or cfg.strata.run_config == Path()
            or not (services.unit_dir() / STRATA_UNIT).is_file()
        ):
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
    try:
        cfg = config.load()
    except config.ConfigError as exc:
        print(f"jarvis: the model server's settings were left as they were: {exc}")
        _reset_failed(JARVIS_UNIT)
        return _systemctl("restart", JARVIS_UNIT)
    if cfg.strata.mode != "local":
        return _split_lifecycle("restart", cfg)
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
        changed = started is not None and services.strata_changed_since(started)
        if (started is None or changed) and _no_room(cfg, "restart"):
            print("jarvis: nothing was restarted", file=sys.stderr)
            return 1
        cut = changed and not free
        code = _systemctl("restart" if changed else "start", STRATA_UNIT)
        if not code and _not_started(cfg, "restart"):
            print("jarvis: the voice service was not restarted", file=sys.stderr)
            return 1
    code = code or _systemctl("restart", JARVIS_UNIT)
    print(f"jarvis: {_unit_state(JARVIS_UNIT)}" + (", model server restarted" if changed else ""))
    if cut:
        print("jarvis: a request that was being answered was cut off by the model server's restart")
    return code


def cmd_logs(_args: argparse.Namespace) -> int:
    """Follow the local services' journals."""
    units = {
        "local": ["-u", JARVIS_UNIT, "-u", STRATA_UNIT],
        "remote": ["-u", JARVIS_UNIT],
        "backend": ["-u", STRATA_UNIT],
    }[_load().strata.mode]
    cmd = [
        "journalctl",
        "--user",
        *units,
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


def _restart_if_running(*, no_block: bool = False) -> bool:
    """Restart the voice service if it runs; return False when that restart failed.

    With `no_block` the call returns once systemd has accepted the restart, which is what the
    voice service itself needs: the unit's own SIGTERM cannot then read as a failure.
    """
    if _unit_state(JARVIS_UNIT) != "active":
        return True
    # A restart counts against the start limit (5 in 15 min): clear it first, so a few
    # config changes in a row do not leave the voice service stopped.
    _systemctl("reset-failed", JARVIS_UNIT)
    if _systemctl(*(["--no-block"] if no_block else []), "restart", JARVIS_UNIT) != 0:
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


def cmd_voice(args: argparse.Namespace) -> int:
    """List the voices with what each sounds like, or switch to one and restart the voice."""
    cfg = _load()
    if args.name is not None:
        chosen = voices.find(args.name)
        if chosen is None:
            print(
                f"jarvis: no voice is called {args.name!r}; `jarvis voice` lists them",
                file=sys.stderr,
            )
            return 2
        audio = cfg.audio.model_copy(update={"voice": chosen.setting})
        cfg = cfg.model_copy(update={"audio": audio})
        config.save(cfg)
        _restart_if_running()
        print(f"voice: {chosen.id} ({chosen.accent} {chosen.gender}, {chosen.about})")
        return 0
    current = cfg.audio.voice
    print(f"voice: {current.removeprefix(voices.KOKORO)}")
    for v in voices.VOICES:
        mark = "*" if v.setting == current else " "
        print(f"{mark} {v.id:<12} {v.accent:<8} {v.gender:<6} {v.grade:<3} {v.about}")
    print("Grades are Kokoro's own; `jarvis voice <name>` switches.")
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


def _next_tool_setting(name: str, by: int = 1) -> ToolSetting:
    """Move one tool on (`by` 1) or back (`by` -1) round its settings, for the chat.

    The settings are on, asks first, needs the key, off, never below how the tool ships.

    Raises:
        ConfigError: as `set_tool`.
    """
    found = next((s for s in _tool_settings(config.load()) if s.name == name), None)
    if found is None:
        msg = f"no tool is called {name!r}"
        raise config.ConfigError(msg)
    moved = found.following() if by > 0 else found.preceding()
    return set_tool(name, moved, source="chat", restart=False)


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
    cfg = _load()
    if cfg.strata.mode != "local":
        return _split_status(cfg, as_json=False)
    voice, server = _unit_state(JARVIS_UNIT), _unit_state(STRATA_UNIT)
    state = _strata(cfg).status() if server == "active" else None
    shown = (_UNIT_SHOWN.get(unit, unit) for unit in (voice, server))
    print("jarvis: {}   model server: {}".format(*shown), end="")
    print(f"   model: {state}" if state else "")
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
    if jarvis.ops is not None:
        jarvis.ops.refresh()  # your artists, for the cards and the hint
    if args.action == "record" and _listening():
        print("jarvis: Jarvis is listening; `jarvis pause` it first, then record", file=sys.stderr)
        return 1
    heard_with = f"{audio.transcribe_engine or 'default'} {audio.transcribe_model}".strip()
    artists = jarvis.ops.artist_names() if jarvis.ops is not None else []
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
    "acp": cmd_acp,
    "chat": cmd_chat,
    "run": cmd_run,
    "doctor": cmd_doctor,
    "update": lambda args: update.command(args, launch_for_update, _update_plugins),
    "status": cmd_status,
    "pause": cmd_pause,
    "resume": cmd_resume,
    "hush": cmd_hush,
    "confirm": cmd_confirm,
    "unload": cmd_unload,
    "pc-audio": cmd_pc_audio,
    "consent": cmd_consent,
    "tools": cmd_tools,
    "plugins": lambda args: pluginsync.command(args, launch_for_update),
    "speed": cmd_speed,
    "voice": cmd_voice,
    "wake": cmd_wake,
    "model": cmd_model,
    "speech-test": cmd_speech_test,
}


def main(argv: Sequence[str] | None = None) -> int:
    """Run the `jarvis` command and return its exit code."""
    words = sys.argv[1:] if argv is None else list(argv)
    if cliout.agent_mode(asked=any(word in AGENT_FLAGS for word in words)):
        return _agent_main(words)
    args = build_parser().parse_args(words)
    if args.command is None:
        return cmd_overview(args)
    return COMMANDS[args.command](args)


AGENT_FLAGS = frozenset({"--agent", "--agent-format"})


def _agent_main(words: list[str]) -> int:
    """`main` for an agent: even a mistyped command line is answered in one line of JSON."""
    out, err = io.StringIO(), io.StringIO()
    args: argparse.Namespace | None = None
    code = 2
    with cliout.captured(out, err):
        try:
            args = build_parser().parse_args(words)
        except SystemExit as exc:  # --help, or arguments argparse refused
            code = exc.code if isinstance(exc.code, int) else 2
    if args is not None:
        return answer_agent(args)
    command = next((word for word in words if word in COMMANDS), None)
    cliout.emit(
        {
            "command": command,
            "ok": code == 0,
            "exit": code,
            "output": _lines(out.getvalue()),
            "errors": _lines(err.getvalue()),
        }
    )
    return code


# Commands that wait on a person or follow a live stream: they have no one-line answer.
INTERACTIVE = frozenset({"acp", "chat", "logs", "run", "ask", "speech-test", "spotify-login"})


def _lines(text: str) -> list[str]:
    return [cliout.strip(line) for line in text.splitlines() if line.strip()]


def _agent_speed(_args: argparse.Namespace) -> dict[str, object]:
    return {"speech_speed": config.load().audio.speech_speed}


def _agent_voice(_args: argparse.Namespace) -> dict[str, object]:
    current = config.load().audio.voice
    return {
        "voice": current,
        "voices": [
            {
                "id": v.id,
                "accent": v.accent,
                "gender": v.gender,
                "grade": v.grade,
                "about": v.about,
                "current": v.setting == current,
            }
            for v in voices.VOICES
        ],
    }


def _agent_consent(_args: argparse.Namespace) -> dict[str, object]:
    bypass = config.load().boundaries.bypass_consent
    return {"consent": "bypassed" if bypass else "asks", "said": CONSENT_SAID[bypass]}


def _tool_entry(s: ToolSetting) -> dict[str, object]:
    return {
        "name": s.name,
        "setting": s.label,
        "state": s.state.value,
        "shipped": s.shipped.value,
        "changed": s.state is not s.shipped,
        "description": s.description or s.summary,
    }


def _agent_tools(args: argparse.Namespace) -> dict[str, object]:
    rows = _tool_settings(config.load())
    if args.name is not None:
        return {"tool": next((_tool_entry(s) for s in rows if s.name == args.name), None)}
    return {"tools": [_tool_entry(s) for s in rows]}


# Commands whose answer has fields of its own; any other command's answer carries its lines.
def _agent_plugins(_args: argparse.Namespace) -> dict[str, object]:
    return {"marketplaces": pluginsync.inventory(config.load(), launch_for_update)}


AGENT_FIELDS: dict[str, Callable[[argparse.Namespace], dict[str, object]]] = {
    "plugins": _agent_plugins,
    "speed": _agent_speed,
    "voice": _agent_voice,
    "consent": _agent_consent,
    "tools": _agent_tools,
}


def answer_agent(args: argparse.Namespace) -> int:
    """Run the command and write its answer as one line of JSON (`jarvis.cliout`)."""
    name = args.command or "status"
    if name in INTERACTIVE:
        said = f"{name} is interactive: run it in a terminal, without --agent"
        cliout.emit({"command": name, "ok": False, "exit": 2, "errors": [said]})
        return 2
    out, err = io.StringIO(), io.StringIO()
    answer: dict[str, object] = {"command": name}
    with cliout.captured(out, err):
        try:
            code = _run_for_agent(name, args, answer)
        except SystemExit as exc:  # a command that gives up the way argparse does
            code = exc.code if isinstance(exc.code, int) else 1
            if isinstance(exc.code, str):
                print(exc.code, file=sys.stderr)
        except Exception:  # ruff: ignore[blind-except] - the agent still gets its one line, and the traceback
            code = 1
            traceback.print_exc()
    answer |= {"ok": code == 0, "exit": code}
    printed, errors = _lines(out.getvalue()), _lines(err.getvalue())
    if name == "status" and printed:
        answer |= json.loads(printed[-1])
    elif name not in AGENT_FIELDS and name != "doctor":
        answer["output"] = printed
    if errors or (name not in AGENT_FIELDS and name not in {"doctor", "status"}):
        answer["errors"] = errors
    cliout.emit(answer)
    return code


def _run_for_agent(name: str, args: argparse.Namespace, answer: dict[str, object]) -> int:
    """Run command `name`, adding its own fields to `answer`; return its exit code."""
    if name == "doctor":
        report = doctor_report(args)
        answer |= _doctor_fields(report)
        return int(bool(report.failed))
    if name == "status":
        args.json = True
        return cmd_status(args)
    code = COMMANDS[name](args)
    if code == 0 and name in AGENT_FIELDS:
        answer |= AGENT_FIELDS[name](args)
    return code


def _doctor_fields(report: DoctorReport) -> dict[str, object]:
    checks = [
        {"status": "ok" if ok else "fail", "what": what, "fix": "" if ok else fix}
        for ok, what, fix in report.checks
    ]
    checks += [{"status": "ok", "what": line, "fix": ""} for line in report.verified]
    notes = [cliout.strip(cliout.note(line))[len("note ") :] for line in report.notes]
    return {"checks": checks, "notes": notes, "summary": report.summary}
