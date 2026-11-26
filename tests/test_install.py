"""Drive the piped installer with a private home and fake prerequisites."""

import os
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

_INSTALL = Path(__file__).resolve().parents[1] / "install.sh"
_BASH = shutil.which("bash")
_STUB = """#!/bin/bash
set -eu
name=${0##*/}
printf '%s %s\\n' "$name" "$*" >> "$INSTALL_CALLS"
case $name:$1 in
  git:check-ref-format) ;;
  git:clone)
    mkdir -p -- "${@: -1}/.git"
    ;;
  git:-C)
    case $3 in
      rev-parse) printf '%s\\n' "$2" ;;
      remote) echo https://github.com/NecturaLabs/Jarvis.git ;;
      symbolic-ref) echo main ;;
      status) ;;
      config) printf '%s\\n' refs/heads/main ;;
      *) exit 90 ;;
    esac
    ;;
  systemctl:--user) ;;
  uv:sync)
    mkdir -p .venv/bin
    ln -sf "$INSTALL_BIN/jarvis" .venv/bin/jarvis
    ;;
  jarvis:setup)
    if [[ ${2:-} == --has-run-config ]]; then
      [[ -f "$INSTALL_CALLS.configured" ]]
    else
      touch "$INSTALL_CALLS.configured"
      mkdir -p "$XDG_CONFIG_HOME/jarvis"
      touch "$XDG_CONFIG_HOME/jarvis/config.toml"
    fi
    ;;
  uv:run)
    if [[ $* == *chrome_found* ]]; then [[ -f "$INSTALL_CALLS.chrome" ]]; fi
    ;;
  agent-browser:install) ;;
  jarvis:doctor|jarvis:update) ;;
  sudo:*|pacman:*|curl:*) exit 91 ;;
  *) exit 90 ;;
esac
"""

type Install = Callable[..., subprocess.CompletedProcess[str]]


@pytest.fixture
def install(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Install:
    """Test piped bash with no terminal and a PATH that contains only safe tools."""
    folder = tmp_path / "bin"
    folder.mkdir()
    stub = folder / "stub"
    stub.write_text(_STUB, encoding="utf-8")
    stub.chmod(0o755)
    for name in ("git", "uv", "systemctl", "jarvis", "sudo", "pacman", "curl", "agent-browser"):
        (folder / name).symlink_to(stub)
    for name in (
        "python3.14",
        "omarchy-version",
        "voxtype",
        "strata",
        "pw-play",
        "pw-record",
        "pw-dump",
        "wpctl",
    ):
        (folder / name).symlink_to("/usr/bin/true")
    for name in ("mkdir", "touch", "ln"):
        (folder / name).symlink_to(Path("/usr/bin") / name)
    monkeypatch.setenv("PATH", str(folder))
    monkeypatch.setenv("INSTALL_BIN", str(folder))
    monkeypatch.setenv("INSTALL_CALLS", str(tmp_path / "calls"))
    monkeypatch.delenv("STRATA_DATA_DIR", raising=False)

    def run(*args: str) -> subprocess.CompletedProcess[str]:
        assert _BASH is not None
        return subprocess.run(
            [_BASH, "-s", "--", *args],
            input=_INSTALL.read_text(encoding="utf-8"),
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=False,
            start_new_session=True,
            timeout=10,
        )

    return run


def _strata_config() -> Path:
    folder = Path(os.environ["XDG_DATA_HOME"]) / "strata" / "install"
    folder.mkdir(parents=True)
    run = folder / "strata-example.json"
    run.write_text("{}", encoding="utf-8")
    return run


def test_fresh_and_repeat_install_use_one_checkout(tmp_path: Path, install: Install) -> None:
    run = _strata_config()
    for _ in range(2):
        done = install("--non-interactive")
        assert done.returncode == 0, done.stderr
    calls = (tmp_path / "calls").read_text(encoding="utf-8")
    assert calls.count("git clone ") == 1
    assert calls.count("uv run --no-sync tools/get_models.py") == 2
    assert calls.count("jarvis doctor --no-service") == 2
    assert calls.count(f"--run-config {run}") == 1
    assert "jarvis update --apply" in calls
    assert "--enable" not in calls


def test_enable_is_explicit_and_download_precedes_it(tmp_path: Path, install: Install) -> None:
    run = _strata_config()
    done = install("--enable", "--non-interactive", "--run-config", str(run))
    assert done.returncode == 0, done.stderr
    calls = (tmp_path / "calls").read_text(encoding="utf-8")
    assert calls.index("uv run --no-sync tools/get_models.py") < calls.index(
        "jarvis setup --enable"
    )
    assert calls.rstrip().endswith("jarvis doctor --wait --essential")


def test_missing_prerequisite_prints_command_and_never_runs_sudo(
    tmp_path: Path, install: Install
) -> None:
    (tmp_path / "bin" / "pw-play").unlink()
    done = install("--non-interactive")
    assert done.returncode != 0
    assert "pw-play" in done.stderr
    assert "sudo pacman -S pipewire" in done.stderr
    assert not (tmp_path / "calls").exists()


def test_a_missing_browser_stops_the_install_with_how_to_get_it(
    tmp_path: Path, install: Install
) -> None:
    (tmp_path / "bin" / "agent-browser").unlink()
    done = install("--non-interactive")
    assert done.returncode != 0
    assert "agent-browser" in done.stderr
    assert not (tmp_path / "calls").exists()


def test_the_browser_gets_its_chrome_when_it_has_none(tmp_path: Path, install: Install) -> None:
    run = _strata_config()
    done = install("--non-interactive", "--run-config", str(run))
    assert done.returncode == 0, done.stderr
    calls = (tmp_path / "calls").read_text(encoding="utf-8")
    assert calls.index("agent-browser install") < calls.index("jarvis doctor")
    assert "--with-deps" not in calls  # that one runs sudo


def test_a_browser_with_its_chrome_is_left_as_it_is(tmp_path: Path, install: Install) -> None:
    run = _strata_config()
    (tmp_path / "calls.chrome").touch()
    done = install("--non-interactive", "--run-config", str(run))
    assert done.returncode == 0, done.stderr
    assert "agent-browser install" not in (tmp_path / "calls").read_text(encoding="utf-8")


def test_pipe_without_tty_explains_missing_answer(tmp_path: Path, install: Install) -> None:
    done = install()
    assert done.returncode != 0
    assert "--run-config" in done.stderr
    assert "git clone " not in (tmp_path / "calls").read_text(encoding="utf-8")
    assert not (Path(os.environ["XDG_STATE_HOME"]) / "jarvis").exists()


def test_dry_run_leaves_home_and_tools_untouched(tmp_path: Path, install: Install) -> None:
    _strata_config()
    done = install("--dry-run", "--enable", "--non-interactive")
    assert done.returncode == 0, done.stderr
    assert all(
        command in done.stdout
        for command in (
            "git clone",
            "uv sync",
            "uv run --no-sync tools/get_models.py",
            "jarvis setup --enable",
            "jarvis doctor",
        )
    )
    assert not (tmp_path / "calls").exists()
    assert not (Path(os.environ["XDG_STATE_HOME"]) / "jarvis").exists()


@pytest.mark.usefixtures("install")
def test_truncated_script_executes_nothing(tmp_path: Path) -> None:
    assert _BASH is not None
    script = _INSTALL.read_text(encoding="utf-8").rstrip().removesuffix(")")
    done = subprocess.run(
        [_BASH],
        input=script,
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )
    assert done.returncode != 0
    assert not (tmp_path / "calls").exists()
