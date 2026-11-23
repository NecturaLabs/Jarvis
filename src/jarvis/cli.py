"""Command-line entry point for `jarvis`."""

import argparse
import json
import logging
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from jarvis import __version__, config, control, models, paths, services
from jarvis.activity import AuditTail
from jarvis.agent import Declined, Hooks, NoReason
from jarvis.app import SetupError, build, chat_memory_file, chat_pictures, echoes_hint, strata_model
from jarvis.audit import AuditLog
from jarvis.chat import clipboard
from jarvis.chat.history import InputHistory
from jarvis.confirmkey import ConfirmKey
from jarvis.files import write_atomic
from jarvis.paths import spotify_token_file, strata_key_file
from jarvis.policy import is_yes
from jarvis.secretfile import ensure_api_key, read_api_key, read_private
from jarvis.services import JARVIS_UNIT, STRATA_UNIT
from jarvis.spotify.auth import SpotifyAuthError, TokenStore, login, redirect_uri
from jarvis.sysinfo import Sampler

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    import numpy as np
    from numpy.typing import NDArray

    from jarvis.audio import FallbackWake, Speaker, VoxtypeWatch, WakeWord
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

    return WakeWord(wake_files(cfg), cfg.audio.wake_threshold, cfg.audio.wake_thresholds)


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
        "--model-file", type=Path, default=None, metavar="PATH", help="the model file Strata serves"
    )
    setup.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=None,
        metavar="PATH",
        help="the model's tokenizer folder",
    )
    setup.add_argument(
        "--mic",
        default=None,
        metavar="NODE",
        help="PipeWire source node name (empty: the default source)",
    )
    setup.add_argument("--enable", action="store_true", help="enable and start the user units")

    made["spotify-login"].add_argument(
        "--no-browser",
        action="store_true",
        help="print the login link to open yourself instead of opening a browser; it still "
        "waits for the login",
    )

    ask = made["ask"]
    ask.add_argument("text", nargs="+", metavar="words", help="the request, as you would say it")
    ask.add_argument(
        "--yes",
        action="store_true",
        help="answer yes to Jarvis's questions (default: ask here); it cannot press the "
        "confirm key a strong action waits for",
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
            "model_file": args.model_file,
            "tokenizer_dir": args.tokenizer_dir,
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
    config.save(updated)
    ensure_api_key(strata_key_file())
    try:
        profile = services.write_strata_profile(updated)
        units = services.write_units(updated, strata_key_file())
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
    try:
        answer = input(f"{question} [yes/no] ")
    except EOFError:
        return Declined(NoReason.NO_ANSWER)  # no terminal to answer on

    return is_yes(answer)


def cmd_ask(args: argparse.Namespace) -> int:
    """Answer one typed request and print the reply."""
    try:
        jarvis = build(_load(), via="ask")
    except SetupError as exc:
        print(f"jarvis: {exc}", file=sys.stderr)
        return 2
    confirm: Callable[[str], bool | Declined] = (
        (lambda q: print(q, "-> yes") is None) if args.yes else _terminal_confirm
    )
    reply = jarvis.agent.handle(
        " ".join(args.text), Hooks(confirm=confirm, waking=lambda: print("(waking up)"))
    )
    print(reply)
    return 0


def _chat_status(cfg: config.Config) -> str:
    voice = _unit_state(JARVIS_UNIT)
    model = _strata(cfg).status() if _unit_state(STRATA_UNIT) == "active" else None
    return f"voice {voice} · model {model or 'server not running'}"


def _chat_details(cfg: config.Config) -> list[tuple[str, str]]:
    """Return what /status adds to the services: versions, the model and where things live."""
    return [
        ("jarvis", __version__),
        ("model", f"{cfg.strata.model_id}, {cfg.strata.context:,} tokens of context"),
        ("replies", f"up to {cfg.chat.reply_tokens:,} tokens, {cfg.chat.max_steps} model calls"),
        ("config", str(paths.config_file())),
        ("conversation", str(chat_memory_file())),
        ("audit log", str(paths.audit_file())),
    ]


def _save_chat_settings(chat: config.ChatConfig) -> None:
    """Write the chat's settings into the config file as it is now on disk."""
    current = config.load()
    config.save(current.model_copy(update={"chat": chat}))


def _screenshot_folders(home: Path, env: Mapping[str, str]) -> tuple[Path, ...]:
    """Return where Omarchy saves screenshots (its setting first), then the Pictures folder."""
    chosen = [env.get(name, "") for name in ("OMASNAP_SCREENSHOT_DIR", "OMARCHY_SCREENSHOT_DIR")]
    folders = [Path(value).expanduser() for value in chosen if value]
    folders += [home / "Pictures" / "Screenshots", home / "Pictures"]
    return tuple(dict.fromkeys(folders))


def cmd_chat(_args: argparse.Namespace) -> int:
    """Open the chat window: typed requests, and everything Jarvis does, live."""
    from jarvis.chat.app import ChatApp, Services

    cfg = _load()
    session_id = secrets.token_hex(8)
    agent = None
    unavailable = ""
    try:
        agent = build(cfg, via="chat", session=session_id).agent
    except SetupError as exc:
        unavailable = str(exc)
    documents = Path.home() / "Documents"
    services = Services(
        status=lambda: _chat_status(cfg),
        details=lambda: _chat_details(cfg),
        stats=Sampler().sample,
        save_settings=_save_chat_settings,
        history=InputHistory(paths.state_dir() / "chat-history", cfg.chat.history_size),
        export_dir=documents if documents.is_dir() else Path.home(),
        paste=clipboard.read,
        pictures=cfg.strata.pictures,
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


def _transcribers(cfg: config.Config, hint: Callable[[], str]) -> tuple[_Transcriber, _Transcriber]:
    """Return the request transcriber (with the vocabulary hint) and the yes/no one (without).

    Both run Voxtype with Jarvis's own speed settings, written here; the owner's Voxtype config
    is never touched.
    """
    from jarvis.audio import transcribe, voxtype_settings  # ruff: ignore[import-outside-top-level] - loads numpy and ONNX Runtime, which most commands never need

    audio = cfg.audio
    engine, model = audio.transcribe_engine, audio.transcribe_model
    settings = paths.config_dir() / "voxtype.toml"
    write_atomic(
        settings,
        voxtype_settings(audio.transcribe_threads, short_window=audio.transcribe_short_window),
    )

    def request(samples: NDArray[np.int16]) -> str:
        primed = hint() if audio.vocabulary_hint else ""
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


def cmd_run(_args: argparse.Namespace) -> int:
    """Run the voice loop until stopped."""
    from jarvis.audio import (  # ruff: ignore[import-outside-top-level] - loads numpy and ONNX Runtime, which most commands never need
        MIC_STALL_S,
        AppAudio,
        Microphone,
        PausableMic,
        Vad,
        VoxtypeWatch,
        screen_locked,
    )
    from jarvis.voice import PcListener, VoiceLoop  # ruff: ignore[import-outside-top-level] - imports jarvis.audio, which loads numpy and ONNX Runtime

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = _load()
    try:
        jarvis = build(cfg)
    except SetupError as exc:
        print(f"jarvis: {exc}", file=sys.stderr)
        return 2
    data = cfg.data_dir
    mic = PausableMic(Microphone(cfg.audio.source, stall_s=MIC_STALL_S))
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

    threading.Thread(target=stopper, name="jarvis-stopper", daemon=True).start()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    # Off the request path: the Spotify login and library (for the hint and matching) are kept
    # warm, and the model is loaded now, so the first request waits for neither.
    jarvis.ops.start_refresh(refresh_stop)
    threading.Thread(target=jarvis.model.warm, name="jarvis-model-warm", daemon=True).start()
    request, answer = _transcribers(cfg, jarvis.hint)
    mic.start()
    wakes = [wake_word(cfg), *([wake_word(cfg)] if pc_audio is not None else [])]

    def after_sleep() -> None:
        handler.after_sleep()  # made below, once the loop it controls exists

    keys = ConfirmKey(audit=jarvis.audit)  # pressed through the socket, waited on by the loop

    voxtype = VoxtypeWatch()  # started with the loop, so nothing before can leave it running
    loop = VoiceLoop(
        jarvis.agent,
        mic,
        wakes[0],
        Vad(data / VAD_MODEL),
        speaker,
        cfg.audio,
        request,
        prewarm=lambda: None if jarvis.model.is_loaded() else jarvis.model.load(),
        pc=None
        if pc_audio is None
        else PcListener(
            pc_audio,
            wakes[-1],
            Vad(data / VAD_MODEL),
        ),
        transcribe_answer=answer,
        names=(cfg.persona.name,),
        dictating=voxtype.dictating,
        meeting=voxtype.meeting,
        audit=jarvis.audit,
        paused=lambda: mic.paused,
        until_resumed=mic.wait_resumed,
        reopen=mic.recover,
        locked=_never if cfg.audio.listen_when_locked else screen_locked,
        after_sleep=after_sleep,
        keys=keys,
    )
    handler = control.Controls(loop, mic, jarvis.model, jarvis.audit, keys=keys)
    jarvis.audit.record("service_started", bypass_consent=cfg.boundaries.bypass_consent)
    return _run_until_stopped(
        loop, stop_requested, _serve_controls(handler, jarvis.audit), voxtype, wakes
    )


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
    if cfg.strata.pictures:
        found.append(("vision", cfg.strata.vision_file))
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


def _whisper_model(cfg: config.Config) -> list[tuple[bool, str, str]]:
    """Return the check that Voxtype has downloaded the Whisper model Jarvis asks it for.

    Other engines keep their models in their own forms, so Voxtype itself reports on those.
    """
    from jarvis import audio as audio_module  # ruff: ignore[import-outside-top-level] - loads numpy and ONNX Runtime, which most commands never need

    audio = cfg.audio
    name = audio.transcribe_model
    if audio.transcribe_engine not in {"", "whisper"} or not name:
        return []
    fix = "download it with `voxtype setup model`"
    installed = None if name.startswith("/") else audio_module.installed_models("whisper")
    if installed is not None:
        found = name in installed
        said = "installed" if found else "not installed"
        return [(found, f"speech model {name} (Voxtype reports it {said})", fix)]
    model = (
        Path(name)
        if name.startswith("/")
        else paths.xdg_data_home() / "voxtype" / "models" / f"ggml-{name}.bin"
    )
    return [(model.is_file(), f"speech model {model}", fix)]


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


def cmd_doctor(_args: argparse.Namespace) -> int:
    """Print one line per check with its fix; never loads the model.

    Model files found on disk are also hashed and named by the release they match; a file that
    matches no known release is only a note.
    """
    cfg = _load()
    from jarvis.audio import voice_files  # ruff: ignore[import-outside-top-level] - loads numpy and ONNX Runtime, which most commands never need

    wake_checks, wake_notes = _wake_checks(cfg)
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
        (
            cfg.strata.model_file.is_file(),
            f"model {cfg.strata.model_file}",
            "mount its drive, or set --model-file",
        ),
        *(
            (shutil.which(t) is not None, f"tool {t}", f"install {t}")
            for t in ("pw-record", "pw-play", "voxtype")
        ),
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
    for line in [*wake_notes, *release_lines]:
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


def cmd_restart(_args: argparse.Namespace) -> int:
    """Restart the voice service; restart Strata too when its settings changed since it started.

    A running Strata is otherwise left alone, so the model stays loaded.
    """
    _reset_failed(STRATA_UNIT, JARVIS_UNIT)
    started = _started_at(STRATA_UNIT)
    changed = started is not None and _strata_changed_since(started)
    code = _systemctl("restart" if changed else "start", STRATA_UNIT)
    code = code or _systemctl("restart", JARVIS_UNIT)
    print(f"jarvis: {_unit_state(JARVIS_UNIT)}" + (", model server restarted" if changed else ""))
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


def _restart_if_running() -> None:
    if _unit_state(JARVIS_UNIT) == "active":
        # A restart counts against the start limit (5 in 15 min): clear it first, so a few
        # config changes in a row do not leave the voice service stopped.
        _systemctl("reset-failed", JARVIS_UNIT)
        _systemctl("restart", JARVIS_UNIT)


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
    "wake": cmd_wake,
}


def main(argv: Sequence[str] | None = None) -> int:
    """Run the `jarvis` command and return its exit code."""
    args = build_parser().parse_args(argv)
    if args.command is None:
        return cmd_overview(args)
    return COMMANDS[args.command](args)
