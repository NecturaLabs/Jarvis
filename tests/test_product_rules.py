"""The product rules of AGENTS.md, enforced by scanning the source tree.

Each rule is a scanner function that takes one file's source and returns findings, so it is tested
on small snippets (it flags what it should and passes what it should), and one test per rule runs it
over `src/jarvis` and expects nothing. A finding names the rule and `file:line`. To allow something
new, add it to the rule's allow-list below with a one-line reason, and say why in your pull request.
"""

import ast
import json
import re
import subprocess
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import override
from urllib.parse import urlsplit

import pytest

SOURCE = Path(__file__).resolve().parent.parent / "src" / "jarvis"

# Files that may import httpx: the only places that talk to a server.
HTTPX_FILES = {
    "model.py": "the client of the local Strata server",
    "spotify/api.py": "the Spotify Web API client",
    "spotify/auth.py": "the Spotify OAuth token exchange",
    "web/guard.py": "the client that reaches only public addresses, for every outside web service",
    "web/searxng.py": "the client of the local SearXNG search service (loopback only)",
    "web/openmeteo.py": "the Open-Meteo weather and geocoding client (docs/adr/0007)",
    "web/arctic.py": "the Arctic Shift Reddit archive client (docs/adr/0007)",
    "web/page.py": "reads a web page a search returned, through web/guard.py's client",
}

# Files that may import httpcore, httpx's connection layer: only the guard that checks every
# connection's destination builds on it directly.
HTTPCORE_FILES = {
    "web/guard.py": "pins each connection to an address it checked is public",
}

# Files that may import subprocess: the only places that start another program.
SUBPROCESS_FILES = {
    "cli.py": "systemctl, journalctl, the browser, and `jarvis doctor`'s no-network check",
    "audio.py": "pw-record, pw-play, pw-dump, `voxtype transcribe`, `voxtype status` and "
    "`voxtype info models` (the commands AGENTS.md lists, `status` also with `--follow "
    "--extended`), `wpctl` and `playerctl` with the fixed arguments of `pc_volume` and "
    "`pc_media`, and the "
    "Azure keyword helper, only inside `unshare --net` (ADR 0010)",
    "chat/clipboard.py": "`wl-paste` with a fixed argument list, for Ctrl+V in the chat (the "
    "owner's request, 2026-11-21)",
    "pc/shell.py": "`pc_shell`, the opt-in `strong` shell ADR 0002 allows (#19): "
    "`systemd-run --user --scope` with fixed limits and `setpriv --no-new-privs`, and "
    "`systemctl --user stop` of that scope",
    "web/browser.py": "`agent-browser`, headless, with fixed options and only the commands its "
    "action policy allows, every connection through web/proxy.py (ADR 0016)",
}

# Files that may build a path with a `voxtype` component other than the config directory.
VOXTYPE_PATH_FILES = {
    "cli.py": "`jarvis doctor` and the read-ahead at the wake word read (never write) Voxtype's "
    "downloaded model in its data dir",
}

# Hosts a string literal may point at; everything else is a hosted service Jarvis must not call.
ALLOWED_HOSTS = {
    "127.0.0.1": "loopback: the local Strata server and the OAuth callback",
    "localhost": "loopback",
    "::1": "loopback",
    "accounts.spotify.com": "Spotify's OAuth endpoints, the one outside service a feature needs",
    "api.spotify.com": "Spotify's Web API, the one outside service a feature needs",
    "api.open-meteo.com": "Open-Meteo's free forecast API, for weather (docs/adr/0007)",
    "geocoding-api.open-meteo.com": "Open-Meteo's place lookup, for weather and time abroad",
    "arctic-shift.photon-reddit.com": "the Reddit archive, since Reddit refuses automated clients",
}

# Words an operation or tool name may not contain: one agent does all the work.
FORBIDDEN_NAME_WORDS = ("delegate", "subagent", "spawn", "agent", "assistant", "handoff")


@dataclass(frozen=True, slots=True)
class Finding:
    """One place a rule was broken."""

    path: str
    line: int
    detail: str

    @override
    def __str__(self) -> str:
        """Return `file:line: detail`."""
        return f"src/jarvis/{self.path}:{self.line}: {self.detail}"


type Scanner = Callable[[str, str], list[Finding]]


def _docstrings(tree: ast.AST) -> set[int]:
    """Return the ids of the string nodes that are docstrings."""
    found: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            first = node.body[0] if node.body else None
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                found.add(id(first.value))
    return found


def _strings(tree: ast.AST) -> Iterator[tuple[int, str]]:
    """Yield the line and text of every string literal that is not a docstring."""
    skip = _docstrings(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in skip:
            yield node.lineno, node.value


def _imports(tree: ast.AST, module: str) -> Iterator[int]:
    """Yield the line of every import of `module` or one of its submodules."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names = [node.module]
        else:
            continue
        if any(name == module or name.startswith(f"{module}.") for name in names):
            yield node.lineno


def _import_scanner(module: str, allowed: dict[str, str], hint: str) -> Scanner:
    def scan(path: str, source: str) -> list[Finding]:
        if path in allowed:
            return []
        return [
            Finding(path, line, f"imports {module}; {hint} Only {', '.join(sorted(allowed))} may.")
            for line in _imports(ast.parse(source), module)
        ]

    return scan


scan_httpx = _import_scanner(
    "httpx",
    HTTPX_FILES,
    "Jarvis talks to a server only through its named clients (model, Spotify, web).",
)
scan_httpcore = _import_scanner(
    "httpcore",
    HTTPCORE_FILES,
    "Below httpx, only the guard that checks every connection may open one.",
)
scan_subprocess = _import_scanner(
    "subprocess",
    SUBPROCESS_FILES,
    "Starting a program is a typed, policy-checked operation (AGENTS.md, Boundaries).",
)


def scan_urls(path: str, source: str) -> list[Finding]:
    """Flag a string literal holding a web address whose host is not allowed."""
    findings: list[Finding] = []
    for line, text in _strings(ast.parse(source)):
        for match in re.finditer(r"(?:https?|wss?)://[^\s\"'<>]+", text):
            host = urlsplit(match.group()).hostname
            if host is not None and host not in ALLOWED_HOSTS:
                findings.append(
                    Finding(
                        path,
                        line,
                        f"names {host}; Jarvis is local and free, with no hosted model or paid "
                        f"API. Only {', '.join(sorted(ALLOWED_HOSTS))} may appear.",
                    )
                )
    return findings


def _path_parts(node: ast.expr) -> list[ast.expr]:
    """Return the operands of a chain of `/` joins, left to right."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        return [*_path_parts(node.left), *_path_parts(node.right)]
    return [node]


# Calls that build a path out of their arguments.
_PATH_BUILDERS = frozenset({"Path", "PurePath", "PosixPath", "PurePosixPath", "join", "joinpath"})


def _is_voxtype_component(node: ast.expr) -> bool:
    """Return whether a string constant has a path component that is exactly `voxtype`."""
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and "voxtype" in node.value.replace("\\", "/").split("/")
    )


def _builds_voxtype_path(node: ast.AST) -> bool:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        return any(_is_voxtype_component(part) for part in _path_parts(node))
    if isinstance(node, ast.Call):
        callee = node.func
        name = callee.attr if isinstance(callee, ast.Attribute) else getattr(callee, "id", "")
        return name in _PATH_BUILDERS and any(_is_voxtype_component(arg) for arg in node.args)
    return False


def scan_voxtype_config(path: str, source: str) -> list[Finding]:
    """Flag a path into Voxtype's config, spelled as text or built from its parts.

    A built path counts when any component is exactly `voxtype`: an operand of `/`, or an argument
    of `Path(...)`, `os.path.join(...)` or `joinpath(...)`. Jarvis's own `voxtype.toml` is not one.
    """
    tree = ast.parse(source)
    message = (
        "points into ~/.config/voxtype or builds a path with a `voxtype` component; Jarvis never "
        "edits the owner's Voxtype config and drives it only through the commands AGENTS.md "
        "allows (Product rules)."
    )
    lines = {line for line, text in _strings(tree) if ".config/voxtype" in text.replace("\\", "/")}
    if path not in VOXTYPE_PATH_FILES:
        lines |= {
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.BinOp | ast.Call) and _builds_voxtype_path(node)
        }
    return [Finding(path, line, message) for line in sorted(lines)]


def _operation_names(tree: ast.AST) -> Iterator[tuple[int, ast.expr | None]]:
    """Yield the line and name argument of every `Operation(...)` call."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        callee = node.func
        if getattr(callee, "id", getattr(callee, "attr", "")) != "Operation":
            continue
        keyword = next((k.value for k in node.keywords if k.arg == "name"), None)
        yield node.lineno, node.args[0] if node.args else keyword


def scan_operation_names(path: str, source: str) -> list[Finding]:
    """Flag an operation (tool) name that suggests a second agent, or one the scan cannot read."""
    findings: list[Finding] = []
    for line, name in _operation_names(ast.parse(source)):
        if not (isinstance(name, ast.Constant) and isinstance(name.value, str)):
            findings.append(
                Finding(
                    path,
                    line,
                    "has an Operation whose name is not a string literal; write the name out so "
                    "the product-rule scan can read it.",
                )
            )
            continue
        word = next((w for w in FORBIDDEN_NAME_WORDS if w in name.value.lower()), None)
        if word is not None:
            findings.append(
                Finding(
                    path,
                    line,
                    f"names the tool {name.value!r}, which contains {word!r}; one agent does "
                    "all the work, so Jarvis has no subagent, delegation or second-agent tool.",
                )
            )
    return findings


def _scan_tree(scan: Scanner) -> list[Finding]:
    files = sorted(SOURCE.rglob("*.py"))
    assert files, f"no source files under {SOURCE}"
    return [
        finding
        for file in files
        for finding in scan(file.relative_to(SOURCE).as_posix(), file.read_text(encoding="utf-8"))
    ]


def _report(findings: list[Finding]) -> str:
    return "product rule broken (AGENTS.md, Product rules):\n" + "\n".join(map(str, findings))


# --- each scanner, on snippets --------------------------------------------------------------------


@pytest.mark.parametrize(
    "source",
    [
        "import httpx",
        "import httpx.exceptions",
        "from httpx import Client",
        "import os\nfrom httpx import Client as C",
    ],
)
def test_httpx_outside_the_clients_is_flagged(source: str) -> None:
    findings = scan_httpx("voice.py", source)
    assert [f.line for f in findings] == [source.count("\n") + 1]
    assert "imports httpx" in str(findings[0])
    assert str(findings[0]).startswith("src/jarvis/voice.py:")


def test_httpx_in_a_client_or_not_at_all_passes() -> None:
    assert scan_httpx("model.py", "import httpx") == []
    assert scan_httpx("spotify/api.py", "import httpx") == []
    assert scan_httpx("voice.py", "import httpxy\nx = 'httpx'") == []


def test_httpcore_outside_the_guard_is_flagged() -> None:
    findings = scan_httpcore("web/page.py", "import httpcore")
    assert [f.line for f in findings] == [1]
    assert "imports httpcore" in str(findings[0])
    assert scan_httpcore("web/guard.py", "import httpcore") == []


def test_subprocess_outside_the_launchers_is_flagged() -> None:
    findings = scan_subprocess("agent.py", "import subprocess")
    assert [f.line for f in findings] == [1]
    assert "imports subprocess" in str(findings[0])
    assert scan_subprocess("agent.py", "from subprocess import run")


def test_subprocess_in_a_launcher_passes() -> None:
    for path in SUBPROCESS_FILES:
        assert scan_subprocess(path, "import subprocess") == []


@pytest.mark.parametrize(
    "literal",
    [
        "https://api.openai.com/v1",
        "http://example.com:8080/x",
        "https://accounts.spotify.com.evil.example/authorize",
        "see https://hosted.example/ for more",
        "wss://api.hosted.example/v1/stream",
        "ws://hosted.example:9000/",
        "connect to wss://hosted.example/x now",
    ],
)
def test_a_hosted_address_is_flagged(literal: str) -> None:
    findings = scan_urls("config.py", f"URL = {literal!r}")
    assert [f.line for f in findings] == [1]
    assert "no hosted model or paid API" in str(findings[0])


def test_an_address_in_an_f_string_is_flagged_by_its_literal_part() -> None:
    assert scan_urls("config.py", 'URL = f"https://hosted.example/{path}"')


@pytest.mark.parametrize(
    "literal",
    [
        "http://127.0.0.1:8080/v1",
        "http://localhost:8080",
        "http://[::1]:8080/",
        "https://accounts.spotify.com/authorize",
        "https://api.spotify.com/v1",
        "https://",
        "ws://127.0.0.1:8080/stream",
        "wss://localhost/stream",
        "ws://",
    ],
)
def test_a_local_or_spotify_address_passes(literal: str) -> None:
    assert scan_urls("config.py", f"URL = {literal!r}") == []


def test_an_address_in_a_docstring_is_not_a_backend() -> None:
    assert scan_urls("config.py", '"""See https://docs.example/ for details."""') == []
    assert scan_urls("config.py", 'def f():\n    """Docs: https://docs.example/."""') == []


@pytest.mark.parametrize(
    "source",
    [
        'P = "~/.config/voxtype/config.toml"',
        'P = f"{home}/.config/voxtype/config.toml"',
        'P = Path.home() / ".config" / "voxtype" / "config.toml"',
        'P = xdg_config_home() / "voxtype"',
        'P = paths.xdg_config_home() / "voxtype" / "config.toml"',
        # Built paths: a component that is exactly "voxtype", however the path is assembled.
        'P = Path(tmp) / "voxtype"',
        'P = base / "voxtype" / "config.toml"',
        'P = base / "voxtype/config.toml"',
        'P = os.path.join(base, "voxtype")',
        'P = os.path.join(base, "voxtype", "config.toml")',
        'P = os.path.join(base, ".config/voxtype")',
        'P = Path(base, "voxtype")',
        'P = Path("voxtype")',
        'P = Path(base).joinpath("voxtype", "config.toml")',
        'P = PurePosixPath("/etc/voxtype")',
        'P = join(base, "voxtype")',
    ],
)
def test_a_path_into_the_owners_voxtype_config_is_flagged(source: str) -> None:
    findings = scan_voxtype_config("audio.py", source)
    assert findings
    assert {f.line for f in findings} == {1}
    assert "~/.config/voxtype" in str(findings[0])


def test_a_voxtype_path_is_reported_on_its_own_line() -> None:
    source = 'a = 1\nP = Path.home() / ".config" / "voxtype" / "config.toml"\nQ = base / "voxtype"'
    assert [f.line for f in scan_voxtype_config("audio.py", source)] == [2, 3]


def test_a_voxtype_data_path_is_allowed_only_where_listed() -> None:
    source = 'P = xdg_data_home() / "voxtype" / "models"'
    assert scan_voxtype_config("cli.py", source) == []
    assert scan_voxtype_config("audio.py", source)


@pytest.mark.parametrize(
    "source",
    [
        'P = config_dir() / "voxtype.toml"',
        'P = xdg_config_home() / "jarvis"',
        'P = Path(base, "voxtype.toml")',
        'P = os.path.join(base, "not-voxtype")',
        'P = base / "voxtype-notes"',
        'CMD = ["voxtype", "transcribe"]',
        'CMD = ("voxtype", "record", "start")',
        'run(["voxtype", "record", "stop", "--wait", "--json"])',
        'PROGRAM = "voxtype"',
    ],
)
def test_jarvis_own_voxtype_files_pass(source: str) -> None:
    assert scan_voxtype_config("audio.py", source) == []


@pytest.mark.parametrize(
    "name",
    [
        "delegate_task",
        "spawn_helper",
        "run_subagent",
        "agent_call",
        "Ask_Agent",
        "do_Delegate",
        "ask_assistant",
        "Assistant_call",
        "handoff_to_model",
        "do_Handoff",
    ],
)
def test_a_tool_that_suggests_a_second_agent_is_flagged(name: str) -> None:
    for source in (f'Operation("{name}", "d", {{}}, run)', f'Operation(name="{name}")'):
        findings = scan_operation_names("ops.py", source)
        assert len(findings) == 1
        assert "one agent does all the work" in str(findings[0])


def test_an_operation_name_the_scan_cannot_read_is_flagged() -> None:
    findings = scan_operation_names("ops.py", "Operation(NAME, 'd', {}, run)")
    assert "not a string literal" in str(findings[0])
    assert scan_operation_names("ops.py", "Operation(f'x_{y}')")


def test_ordinary_operation_names_pass() -> None:
    for name in ("spotify_play", "no_action", "set_volume"):
        assert scan_operation_names("ops.py", f'Operation("{name}", "d", {{}}, run)') == []
    assert scan_operation_names("ops.py", "x = Other('delegate')") == []


# --- the real tree --------------------------------------------------------------------------------


def test_the_allow_lists_name_real_files() -> None:
    for path in (*HTTPX_FILES, *HTTPCORE_FILES, *SUBPROCESS_FILES, *VOXTYPE_PATH_FILES):
        assert (SOURCE / path).is_file(), f"allow-list entry src/jarvis/{path} no longer exists"
    reasons = [
        *HTTPX_FILES.values(),
        *HTTPCORE_FILES.values(),
        *SUBPROCESS_FILES.values(),
        *VOXTYPE_PATH_FILES.values(),
        *ALLOWED_HOSTS.values(),
    ]
    assert all(reasons), "every allow-list entry needs a one-line reason"


def test_the_tree_imports_httpx_only_in_the_clients() -> None:
    findings = _scan_tree(scan_httpx)
    assert not findings, _report(findings)


def test_the_tree_imports_httpcore_only_in_the_guard() -> None:
    findings = _scan_tree(scan_httpcore)
    assert not findings, _report(findings)


def test_the_tree_starts_programs_only_from_the_launchers() -> None:
    findings = _scan_tree(scan_subprocess)
    assert not findings, _report(findings)


def test_the_tree_names_no_hosted_service() -> None:
    findings = _scan_tree(scan_urls)
    assert not findings, _report(findings)


def test_the_tree_never_points_into_the_owners_voxtype_config() -> None:
    findings = _scan_tree(scan_voxtype_config)
    assert not findings, _report(findings)


def test_the_tree_has_no_second_agent_tool() -> None:
    findings = _scan_tree(scan_operation_names)
    assert not findings, _report(findings)


# --- the lint table -------------------------------------------------------------------------------

REPOSITORY = Path(__file__).resolve().parent.parent

# What `ruff.toml` must ban. Written out here (not read from the table) so deleting an entry fails.
BANNED_MODULES = (
    "openai",
    "anthropic",
    "google.generativeai",
    "google.genai",
    "litellm",
    "cohere",
    "mistralai",
    "groq",
    "together",
    "replicate",
    "huggingface_hub",
    "urllib.request",
    "http.client",
    "elevenlabs",
    "deepgram",
    "assemblyai",
    "google.cloud.speech",
    "google.cloud.texttospeech",
    "azure.cognitiveservices.speech",
)
BANNED_CALLS = (
    "os.system",
    "os.popen",
    "asyncio.create_subprocess_exec",
    "asyncio.create_subprocess_shell",
    *(f"os.{name}" for name in ("posix_spawn", "posix_spawnp")),
    *(f"os.{kind}{suffix}" for kind in ("exec", "spawn") for suffix in ("v", "ve", "vp", "vpe")),
    *(f"os.{kind}{suffix}" for kind in ("exec", "spawn") for suffix in ("l", "le", "lp", "lpe")),
)


def test_ruff_bans_every_hosted_sdk_and_unlisted_way_to_reach_out(tmp_path: Path) -> None:
    lines = ["import asyncio", "import os"]
    lines += [f"import {module}" for module in BANNED_MODULES]
    lines += [f"_ = {call}" for call in BANNED_CALLS]
    probe = tmp_path / "probe.py"
    probe.write_text("\n".join(lines) + "\n", encoding="utf-8")

    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "--config",
            str(REPOSITORY / "ruff.toml"),
            "--select",
            "TID251",
            "--no-cache",
            "--output-format",
            "json",
            str(probe),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    reported = [d["message"] for d in json.loads(done.stdout) if d["code"] == "TID251"]
    for name in (*BANNED_MODULES, *BANNED_CALLS):
        assert any(m.startswith(f"`{name}` is banned") for m in reported), f"{name} is not banned"
    assert all("Product rule" in m for m in reported), "every ban names the rule it enforces"


AZURE_HELPER = "src/jarvis/azure_wake_helper.py"
TID251_EXCEPTIONS = {"tests/test_spotify_auth.py"}


def test_only_the_listed_files_may_skip_the_banned_imports() -> None:
    import tomllib  # ruff: ignore[import-outside-top-level] - only this test reads ruff.toml

    ignores = tomllib.loads((REPOSITORY / "ruff.toml").read_text(encoding="utf-8"))["lint"][
        "extend-per-file-ignores"
    ]
    skipping = {path for path, rules in ignores.items() if "TID251" in rules}

    assert skipping == TID251_EXCEPTIONS


def test_only_the_azure_helper_imports_the_speech_sdk_and_jarvis_never_imports_the_helper() -> None:
    sdk = "azure.cognitiveservices.speech"
    for path in sorted(SOURCE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        } | {node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        relative = path.relative_to(REPOSITORY).as_posix()
        if relative == AZURE_HELPER:
            assert sdk in names
            assert not any(name.startswith("jarvis") for name in names)
        else:
            assert not any(name.startswith("azure") for name in names), relative
            assert "jarvis.azure_wake_helper" not in names or relative.startswith("tests/"), (
                relative
            )


def test_only_the_speech_sdk_import_in_its_helper_skips_the_ban_inline() -> None:
    skipping: list[str] = []
    for path in sorted(SOURCE.rglob("*.py")):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "ignore[" in line and "banned-api" in line:
                skipping.append(
                    f"{path.relative_to(REPOSITORY).as_posix()}:{number}:{line.strip()}"
                )

    assert len(skipping) == 1, skipping
    assert skipping[0].startswith(AZURE_HELPER)
    assert "import azure.cognitiveservices.speech as sdk" in skipping[0]
