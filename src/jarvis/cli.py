"""Command-line entry point for `jarvis`."""

import argparse
import logging
import re
import shutil
import signal
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from jarvis import __version__, config, models, paths
from jarvis.agent import Hooks
from jarvis.app import SetupError, build, echoes_hint
from jarvis.model import StrataModel
from jarvis.paths import spotify_token_file, strata_key_file
from jarvis.secretfile import ensure_api_key, read_api_key, read_private

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import numpy as np
    from numpy.typing import NDArray

    from jarvis.audio import FallbackWake, WakeWord

log = logging.getLogger(__name__)

WAKE_MODEL = "wake/hey_jarvis.onnx"
VAD_MODEL = "vad/silero_vad.onnx"
AZURE_PYTHON = "azure-wake/venv/bin/python"
AZURE_CHECK_S = 30.0
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


def wake_word(cfg: config.Config) -> WakeWord | FallbackWake:
    """Return the wake word detector `audio.wake_engine` names.

    An Azure model that cannot start (no model file, no helper Python) falls back to LiveKit's,
    so a switch never leaves Jarvis deaf; `jarvis doctor` says why.
    """
    from jarvis.audio import (  # ruff: ignore[import-outside-top-level]
        AudioError,
        AzureWake,
        FallbackWake,
        WakeWord,
        azure_wake_command,
    )

    def livekit() -> WakeWord:
        return WakeWord(cfg.data_dir / WAKE_MODEL, cfg.audio.wake_threshold)

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


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser for the `jarvis` command."""
    parser = argparse.ArgumentParser(prog="jarvis", description="Local voice agent for Omarchy.")
    parser.add_argument("--version", action="version", version=f"jarvis {__version__}")
    sub = parser.add_subparsers(dest="command")

    setup = sub.add_parser("setup", help="write the config, the Strata profile and the user units")
    setup.add_argument("--spotify-client-id", default=None)
    setup.add_argument(
        "--data-dir", type=Path, default=None, help="where wake, VAD and voice models live"
    )
    setup.add_argument(
        "--strata-home", type=Path, default=None, help="Strata's data folder (STRATA_DATA_DIR)"
    )
    setup.add_argument(
        "--strata-components-parent", type=Path, default=None, help="XDG_DATA_HOME for Strata"
    )
    setup.add_argument("--model-file", type=Path, default=None)
    setup.add_argument("--tokenizer-dir", type=Path, default=None)
    setup.add_argument(
        "--mic", default=None, help="PipeWire source node name (empty: the default source)"
    )
    setup.add_argument("--enable", action="store_true", help="enable and start the user units")

    login = sub.add_parser("spotify-login", help="connect Jarvis to your Spotify account (PKCE)")
    login.add_argument("--no-browser", action="store_true", help="only print the login URL")

    ask = sub.add_parser("ask", help="send one typed request, as if spoken")
    ask.add_argument("text", nargs="+")
    ask.add_argument(
        "--yes", action="store_true", help="answer yes to confirmations (default: ask here)"
    )

    sub.add_parser("chat", help="type requests and watch what Jarvis does, live")
    sub.add_parser("run", help="run the voice loop (what jarvis.service runs)")
    sub.add_parser("start", help="start Jarvis (the voice service and its model server)")
    sub.add_parser("stop", help="stop Jarvis and free the model")
    sub.add_parser("restart", help="restart the voice service (after a config change)")
    sub.add_parser("logs", help="follow Jarvis's live log (Ctrl+C to leave)")
    sub.add_parser("doctor", help="check every part without loading the model")
    sub.add_parser("status", help="show whether the model is loaded")
    sub.add_parser("unload", help="free the model's memory now")
    wake = sub.add_parser("wake", help="show or switch the wake word model (livekit or azure)")
    wake.add_argument(
        "engine", nargs="?", choices=["livekit", "azure"], help="omit to show the setting"
    )
    wake.add_argument("--model", metavar="FILE", help="the Azure custom keyword .table file")
    pc = sub.add_parser(
        "pc-audio", help="let Jarvis take requests from PC audio (Discord by default)"
    )
    pc.add_argument("state", nargs="?", choices=["on", "off"], help="omit to show the setting")
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
    from jarvis import services  # ruff: ignore[import-outside-top-level]

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
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)
    if args.enable:
        subprocess.run(
            ["systemctl", "--user", "enable", "--now", services.STRATA_UNIT, services.JARVIS_UNIT],
            check=False,
        )
    if not updated.spotify.client_id:
        print("next: jarvis setup --spotify-client-id <your app's client ID>")
    elif read_private(spotify_token_file()) is None:
        print("next: jarvis spotify-login")
    return 0


def cmd_spotify_login(args: argparse.Namespace) -> int:
    """Run the PKCE login and store the tokens."""
    from jarvis.spotify.auth import SpotifyAuthError, login, redirect_uri  # ruff: ignore[import-outside-top-level]

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


def _terminal_confirm(question: str) -> bool:
    try:
        answer = input(f"{question} [yes/no] ")
    except EOFError:
        return False
    from jarvis.policy import is_yes  # ruff: ignore[import-outside-top-level]

    return is_yes(answer)


def cmd_ask(args: argparse.Namespace) -> int:
    """Answer one typed request and print the reply."""
    try:
        jarvis = build(_load(), via="ask")
    except SetupError as exc:
        print(f"jarvis: {exc}", file=sys.stderr)
        return 2
    confirm: Callable[[str], bool] = (
        (lambda q: print(q, "-> yes") is None) if args.yes else _terminal_confirm
    )
    reply = jarvis.agent.handle(
        " ".join(args.text), Hooks(confirm=confirm, waking=lambda: print("(waking up)"))
    )
    print(reply)
    return 0


def _chat_status(cfg: config.Config) -> str:
    from jarvis.services import JARVIS_UNIT, STRATA_UNIT  # ruff: ignore[import-outside-top-level]

    voice = _unit_state(JARVIS_UNIT)
    model = _strata(cfg).status() if _unit_state(STRATA_UNIT) == "active" else None
    return f"voice {voice} · model {model or 'server not running'}"


def _chat_details(cfg: config.Config) -> list[tuple[str, str]]:
    """Return what /status adds to the services: versions, the model and where things live."""
    from jarvis.app import chat_memory_file  # ruff: ignore[import-outside-top-level]

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


def cmd_chat(_args: argparse.Namespace) -> int:
    """Open the chat window: typed requests, and everything Jarvis does, live."""
    import secrets  # ruff: ignore[import-outside-top-level]

    from jarvis.activity import AuditTail  # ruff: ignore[import-outside-top-level]
    from jarvis.chat.app import ChatApp, Services  # ruff: ignore[import-outside-top-level]
    from jarvis.chat.history import InputHistory  # ruff: ignore[import-outside-top-level]
    from jarvis.sysinfo import Sampler  # ruff: ignore[import-outside-top-level]

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
    from jarvis.audio import transcribe, voxtype_settings  # ruff: ignore[import-outside-top-level]
    from jarvis.files import write_atomic  # ruff: ignore[import-outside-top-level]

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


def cmd_run(_args: argparse.Namespace) -> int:
    """Run the voice loop until stopped."""
    from jarvis.audio import (  # ruff: ignore[import-outside-top-level]
        AppAudio,
        FallbackWake,
        Microphone,
        OwnVoiceCheck,
        Speaker,
        Vad,
        WakeWord,
        dictation_active,
        make_synthesizer,
    )
    from jarvis.voice import PcListener, VoiceLoop  # ruff: ignore[import-outside-top-level]

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = _load()
    try:
        jarvis = build(cfg)
    except SetupError as exc:
        print(f"jarvis: {exc}", file=sys.stderr)
        return 2
    data = cfg.data_dir
    mic = Microphone(cfg.audio.source)
    speaker = Speaker(
        make_synthesizer(data, cfg.audio.voice, cfg.audio.speech_speed),
        own_wake=OwnVoiceCheck(
            WakeWord(data / WAKE_MODEL, cfg.audio.wake_threshold),
            WAKE_NAME if cfg.audio.wake_engine == "azure" else None,
        )
        if cfg.audio.barge_in
        else None,
    )
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
        dictating=dictation_active,
    )
    try:
        loop.run(stop_requested.is_set)
    except Exception:
        if stop_requested.is_set():
            return 0
        raise
    finally:
        for wake in wakes:  # an Azure helper ends with its input; end it now, not at exit
            if isinstance(wake, FallbackWake):
                wake.close()
    return 0


def _model_files(cfg: config.Config) -> list[tuple[str, Path]]:
    """Return the (role, file) pairs the doctor can check against a known release."""
    from jarvis.audio import KOKORO_PREFIX, voice_files  # ruff: ignore[import-outside-top-level]

    found = [("wake", cfg.data_dir / WAKE_MODEL), ("vad", cfg.data_dir / VAD_MODEL)]
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
            lines.append(f"ok  {path.name}: {release}, sha256 verified")
    return lines, failures


def _whisper_model(cfg: config.Config) -> list[tuple[bool, str, str]]:
    """Return the check that Voxtype has downloaded the Whisper model Jarvis asks it for.

    Other engines keep their models in their own forms, so Voxtype itself reports on those.
    """
    audio = cfg.audio
    name = audio.transcribe_model
    if audio.transcribe_engine not in {"", "whisper"} or not name:
        return []
    model = (
        Path(name)
        if name.startswith("/")
        else paths.xdg_data_home() / "voxtype" / "models" / f"ggml-{name}.bin"
    )
    return [(model.is_file(), f"speech model {model}", "download it with `voxtype setup model`")]


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
    from jarvis.spotify.auth import TokenStore  # ruff: ignore[import-outside-top-level]

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
    from jarvis.audio import voice_files  # ruff: ignore[import-outside-top-level]

    files = [
        cfg.data_dir / WAKE_MODEL,
        cfg.data_dir / VAD_MODEL,
        *voice_files(cfg.data_dir, cfg.audio.voice),
    ]
    state = _strata(cfg).status() if read_api_key(strata_key_file()) else None
    release_lines, unreadable = _release_lines(cfg)
    checks: list[tuple[bool, str, str]] = [
        (paths.config_file().exists(), f"config {paths.config_file()}", "run jarvis setup"),
        (bool(cfg.spotify.client_id), "Spotify client ID", "jarvis setup --spotify-client-id <id>"),
        _spotify_login(cfg),
        *((f.exists(), f"file {f}", "see docs/models.md") for f in files),
        *unreadable,
        *_whisper_model(cfg),
        *_model_url(cfg),
        (cfg.strata.model_file.is_file(), f"model {cfg.strata.model_file}", "set --model-file"),
        *(
            (shutil.which(t) is not None, f"tool {t}", f"install {t}")
            for t in ("pw-record", "pw-play", "voxtype")
        ),
        *_azure_checks(cfg),
        (
            state is not None,
            f"Strata server ({state or 'unreachable'})",
            "systemctl --user start jarvis-strata",
        ),
    ]
    for ok, what, fix in checks:
        print(f"{'ok ' if ok else 'FAIL'} {what}" + ("" if ok else f"  -> {fix}"))
    for line in release_lines:
        print(line)
    return 0 if all(ok for ok, _, _ in checks) else 1


def _azure_checks(cfg: config.Config) -> list[tuple[bool, str, str]]:
    """Return the checks for the Azure wake word when `audio.wake_engine` is "azure".

    The last one runs the helper itself, with no network, to try the model as Jarvis would.
    """
    from jarvis.audio import azure_env, azure_wake_command  # ruff: ignore[import-outside-top-level]

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
    local = cfg.model.model_copy(update={"base_url": cfg.model_base_url()})
    return StrataModel(local, read_api_key(strata_key_file()) or "")


def cmd_status(_args: argparse.Namespace) -> int:
    """Print the model state."""
    state = _strata(_load()).status()
    print(f"model: {state or 'server unreachable'}")
    return 0 if state else 1


def cmd_unload(_args: argparse.Namespace) -> int:
    """Free the model now."""
    ok = _strata(_load()).unload()
    print("model unloaded" if ok else "could not unload (busy or unreachable)")
    return 0 if ok else 1


def _systemctl(*args: str) -> int:
    return subprocess.run(["systemctl", "--user", *args], check=False).returncode


def _unit_state(unit: str) -> str:
    done = subprocess.run(
        ["systemctl", "--user", "is-active", unit], capture_output=True, text=True, check=False
    )
    return done.stdout.strip() or "unknown"


def cmd_start(_args: argparse.Namespace) -> int:
    """Start both services."""
    from jarvis.services import JARVIS_UNIT, STRATA_UNIT  # ruff: ignore[import-outside-top-level]

    code = _systemctl("start", STRATA_UNIT, JARVIS_UNIT)
    print(f"jarvis: {_unit_state(JARVIS_UNIT)}, model server: {_unit_state(STRATA_UNIT)}")
    return code


def cmd_stop(_args: argparse.Namespace) -> int:
    """Stop the voice service, then the model server (which frees the model)."""
    from jarvis.services import JARVIS_UNIT, STRATA_UNIT  # ruff: ignore[import-outside-top-level]

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
    done = subprocess.run(
        ["systemctl", "--user", *show],
        capture_output=True,
        text=True,
        check=False,
    )
    if _unit_state(unit) != "active":
        return None
    return parse_systemd_time(done.stdout)


def _strata_changed_since(started: float) -> bool:
    """Return whether Strata's profile or unit was rewritten after it started."""
    from jarvis import services  # ruff: ignore[import-outside-top-level]

    files = (services.strata_profile_file(), services.unit_dir() / services.STRATA_UNIT)
    return any(path.exists() and path.stat().st_mtime > started for path in files)


def cmd_restart(_args: argparse.Namespace) -> int:
    """Restart the voice service; restart Strata too when its settings changed since it started.

    A running Strata is otherwise left alone, so the model stays loaded.
    """
    from jarvis.services import JARVIS_UNIT, STRATA_UNIT  # ruff: ignore[import-outside-top-level]

    started = _started_at(STRATA_UNIT)
    changed = started is not None and _strata_changed_since(started)
    code = _systemctl("restart" if changed else "start", STRATA_UNIT)
    code = code or _systemctl("restart", JARVIS_UNIT)
    print(f"jarvis: {_unit_state(JARVIS_UNIT)}" + (", model server restarted" if changed else ""))
    return code


def cmd_logs(_args: argparse.Namespace) -> int:
    """Follow both services' journals."""
    from jarvis.services import JARVIS_UNIT, STRATA_UNIT  # ruff: ignore[import-outside-top-level]

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
    from jarvis.services import JARVIS_UNIT  # ruff: ignore[import-outside-top-level]

    if _unit_state(JARVIS_UNIT) == "active":
        _systemctl("restart", JARVIS_UNIT)


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
    if not update:
        print(f"wake word: {_wake_summary(cfg.audio)}")
        return 0
    new = cfg.model_copy(update={"audio": cfg.audio.model_copy(update=update)})
    config.save(config.Config.model_validate(new.model_dump()))
    print(f"wake word: {_wake_summary(new.audio)}")
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
    return f"LiveKit hey_jarvis (threshold {audio.wake_threshold})"


def cmd_overview(args: argparse.Namespace) -> int:
    """Show what is running and the model state, then follow the live log."""
    from jarvis.services import JARVIS_UNIT, STRATA_UNIT  # ruff: ignore[import-outside-top-level]

    state = _strata(_load()).status() if _unit_state(STRATA_UNIT) == "active" else None
    print(f"jarvis: {_unit_state(JARVIS_UNIT)}   model server: {_unit_state(STRATA_UNIT)}", end="")
    print(f"   model: {state}" if state else "")
    cfg = _load()
    print(f"pc audio: {_pc_audio_summary(cfg.audio)}")
    ok, login, fix = _spotify_login(cfg)
    print(login if ok else f"{login}: run {fix}")
    print(
        "commands: jarvis start | stop | restart | status | unload | doctor | logs | chat"
        " | ask <text> | pc-audio [on|off] | wake [livekit|azure]"
    )
    if _unit_state(JARVIS_UNIT) != "active":
        print("Jarvis is not running; start it with: jarvis start")
        return 0
    if not sys.stdout.isatty():
        return 0
    print("live log (Ctrl+C to leave; Jarvis keeps running):")
    return cmd_logs(args)


COMMANDS: dict[str, Callable[[argparse.Namespace], int]] = {
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
    "unload": cmd_unload,
    "pc-audio": cmd_pc_audio,
    "wake": cmd_wake,
}


def main(argv: Sequence[str] | None = None) -> int:
    """Run the `jarvis` command and return its exit code."""
    args = build_parser().parse_args(argv)
    if args.command is None:
        return cmd_overview(args)
    return COMMANDS[args.command](args)
