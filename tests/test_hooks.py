"""The opt-in git hooks in tools/hooks and tools/install-hooks.sh, driven in throwaway repositories.

A fake `uv` first on PATH records its arguments, so no real check or test run starts. Each repo
lives in `tmp_path` with HOME and the global git config pointed there, so nothing of the real
repository or the maintainer's git configuration is read or changed.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict, cast

import pytest

if TYPE_CHECKING:
    from collections.abc import Mapping

TOOLS = Path(__file__).resolve().parent.parent / "tools"
HOOKS = TOOLS / "hooks"
PRE_COMMIT = HOOKS / "pre-commit"
COMMIT_MSG = HOOKS / "commit-msg"

_FAKE_UV = """#!{python}
import json, os, sys
from pathlib import Path

record = {{
    "args": sys.argv[1:],
    "pipewire_remote": os.environ.get("PIPEWIRE_REMOTE"),
    "pulse_server": os.environ.get("PULSE_SERVER"),
    "git_vars": sorted(name for name in os.environ if name.startswith("GIT_")),
}}
with Path(os.environ["FAKE_UV_LOG"]).open("a", encoding="utf-8") as log:
    log.write(json.dumps(record) + "\\n")
sys.exit(1 if os.environ.get("FAKE_UV_FAIL") in sys.argv[1:] else 0)
"""

_PROBE = """#!/usr/bin/env bash
env | grep '^GIT_' | cut -d= -f1 >"${PROBE_LOG}"
"""

_GIT_NAME = "Hook Test"
_GIT_EMAIL = "hook-test@example.invalid"


class UvCall(TypedDict):
    """What the fake `uv` records about one invocation."""

    args: list[str]
    pipewire_remote: str | None
    pulse_server: str | None
    git_vars: list[str]


class Sandbox:
    """A throwaway repository plus the environment the hooks run in."""

    def __init__(self, root: Path) -> None:
        super().__init__()
        self.root = root
        self.repo = root / "repo"
        self.repo.mkdir()
        self.uv_log = root / "uv.log"
        bin_dir = root / "bin"
        bin_dir.mkdir()
        uv = bin_dir / "uv"
        uv.write_text(_FAKE_UV.format(python=sys.executable), encoding="utf-8")
        uv.chmod(0o700)
        gitconfig = root / "gitconfig"
        gitconfig.write_text("", encoding="utf-8")
        self.env: dict[str, str] = {
            name: value
            for name, value in os.environ.items()
            if not name.startswith(("GIT_", "FAKE_UV_", "PIPEWIRE_", "PULSE_"))
        }
        self.env |= {
            "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
            "HOME": str(root),
            "GIT_CONFIG_GLOBAL": str(gitconfig),
            "GIT_CONFIG_NOSYSTEM": "1",
            "FAKE_UV_LOG": str(self.uv_log),
        }
        self.git("init", "-q", "-b", "main")
        self.git("config", "--local", "user.name", _GIT_NAME)
        self.git("config", "--local", "user.email", _GIT_EMAIL)
        self.git("config", "--local", "commit.gpgsign", "false")

    def git(
        self, *args: str, env: Mapping[str, str] | None = None, check: bool = True
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "-C", str(self.repo), *args],
            env=self.env | dict(env or {}),
            capture_output=True,
            text=True,
            check=check,
        )

    def stage(self, *names: str) -> None:
        for name in names:
            path = self.repo / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("x\n", encoding="utf-8")
            self.git("add", "--", name)

    def run(
        self, script: Path, *args: str, env: Mapping[str, str] | None = None
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(script), *args],
            cwd=self.repo,
            env=self.env | dict(env or {}),
            capture_output=True,
            text=True,
            check=False,
        )

    def uv_calls(self) -> list[UvCall]:
        if not self.uv_log.exists():
            return []
        return [
            cast("UvCall", json.loads(line))
            for line in self.uv_log.read_text(encoding="utf-8").splitlines()
        ]


@pytest.fixture
def sandbox(tmp_path: Path) -> Sandbox:
    return Sandbox(tmp_path)


def _uv_commands(box: Sandbox) -> list[list[str]]:
    return [call["args"] for call in box.uv_calls()]


# --- pre-commit ---------------------------------------------------------------------------------


def test_pre_commit_skips_when_nothing_relevant_is_staged(sandbox: Sandbox) -> None:
    sandbox.stage("art/logo.png", "data/notes.txt", "Makefile")

    done = sandbox.run(PRE_COMMIT)

    assert done.returncode == 0, done.stderr
    assert "skipped" in done.stderr
    assert sandbox.uv_calls() == []


def test_pre_commit_skips_when_nothing_is_staged(sandbox: Sandbox) -> None:
    done = sandbox.run(PRE_COMMIT)

    assert done.returncode == 0, done.stderr
    assert sandbox.uv_calls() == []


@pytest.mark.parametrize(
    "name",
    [
        "src/jarvis/new.py",
        "src/jarvis/new.pyi",
        "tools/run.sh",
        "pyproject.toml",
        "docs/plan.md",
        ".github/workflows/ci.yml",
        "uv.lock",
        "tools/hooks/extra-hook",
        "docs/with space.md",
    ],
)
def test_pre_commit_runs_the_checks_for_a_relevant_file(sandbox: Sandbox, name: str) -> None:
    sandbox.stage(name)

    done = sandbox.run(PRE_COMMIT)

    assert done.returncode == 0, done.stderr
    assert _uv_commands(sandbox) == [
        ["run", ".lint/check.sh"],
        ["run", "pytest", "-q", "-m", "not local"],
    ]


def test_pre_commit_runs_the_checks_for_a_deleted_python_file(sandbox: Sandbox) -> None:
    sandbox.stage("src/old.py")
    sandbox.git("commit", "-q", "-m", "chore: add old", "--no-verify")
    sandbox.git("rm", "-q", "--", "src/old.py")

    done = sandbox.run(PRE_COMMIT)

    assert done.returncode == 0, done.stderr
    assert len(sandbox.uv_calls()) == 2


def test_pre_commit_runs_the_tests_silently(sandbox: Sandbox) -> None:
    sandbox.stage("a.py")

    sandbox.run(PRE_COMMIT)

    pytest_call = sandbox.uv_calls()[1]
    assert pytest_call["pipewire_remote"] == "/nonexistent"
    assert pytest_call["pulse_server"] == "unix:/nonexistent"


def test_pre_commit_reads_a_file_name_with_a_newline_as_one_name(sandbox: Sandbox) -> None:
    sandbox.stage("odd\nname.py")

    done = sandbox.run(PRE_COMMIT)

    assert done.returncode == 0, done.stderr
    assert len(sandbox.uv_calls()) == 2


def test_pre_commit_does_not_split_a_file_name_at_a_newline(sandbox: Sandbox) -> None:
    # Read line by line, "evil.py" would match; the one staged file is "evil.py\nnotes.txt".
    sandbox.stage("evil.py\nnotes.txt")

    done = sandbox.run(PRE_COMMIT)

    assert done.returncode == 0, done.stderr
    assert "skipped" in done.stderr
    assert sandbox.uv_calls() == []


def test_pre_commit_refuses_when_the_checks_fail(sandbox: Sandbox) -> None:
    sandbox.stage("a.py")

    done = sandbox.run(PRE_COMMIT, env={"FAKE_UV_FAIL": ".lint/check.sh"})

    assert done.returncode == 1
    assert "the checks found problems" in done.stderr
    assert _uv_commands(sandbox) == [["run", ".lint/check.sh"]]


def test_pre_commit_refuses_when_a_test_fails(sandbox: Sandbox) -> None:
    sandbox.stage("a.py")

    done = sandbox.run(PRE_COMMIT, env={"FAKE_UV_FAIL": "pytest"})

    assert done.returncode == 1
    assert "a test failed" in done.stderr
    assert len(sandbox.uv_calls()) == 2


def _hooks_path(sandbox: Sandbox, path: Path) -> None:
    sandbox.git("config", "--local", "core.hooksPath", str(path))


def test_git_exports_repository_variables_to_a_hook(sandbox: Sandbox, tmp_path: Path) -> None:
    """The premise of the next test: without the hook's clean-up, git's variables reach `uv`."""
    probe_dir = tmp_path / "probe-hooks"
    probe_dir.mkdir()
    probe = probe_dir / "pre-commit"
    probe.write_text(_PROBE, encoding="utf-8")
    probe.chmod(0o700)
    _hooks_path(sandbox, probe_dir)
    sandbox.stage("a.py")

    sandbox.git(
        "commit", "-q", "-m", "chore: probe", env={"PROBE_LOG": str(tmp_path / "probe.log")}
    )

    exported = (tmp_path / "probe.log").read_text(encoding="utf-8").split()
    assert "GIT_INDEX_FILE" in exported


def test_pre_commit_unsets_the_variables_git_exports(sandbox: Sandbox) -> None:
    _hooks_path(sandbox, HOOKS)
    sandbox.stage("a.py")

    done = sandbox.git("commit", "-q", "-m", "chore: add a", "-s", check=False)

    assert done.returncode == 0, done.stderr
    local_vars = set(sandbox.git("rev-parse", "--local-env-vars").stdout.split())
    assert {"GIT_DIR", "GIT_INDEX_FILE"} <= local_vars
    calls = sandbox.uv_calls()
    assert len(calls) == 2
    for call in calls:
        # GIT_DIR, GIT_INDEX_FILE, GIT_PREFIX and the rest of the repository context are gone; the
        # test's own GIT_CONFIG_* and git's author and editor settings are not that context.
        assert local_vars.isdisjoint(call["git_vars"])
        assert "GIT_CONFIG_GLOBAL" in call["git_vars"]


def test_a_failing_check_stops_a_real_commit(sandbox: Sandbox) -> None:
    _hooks_path(sandbox, HOOKS)
    sandbox.stage("a.py")

    done = sandbox.git(
        "commit", "-q", "-m", "chore: add a", env={"FAKE_UV_FAIL": ".lint/check.sh"}, check=False
    )

    assert done.returncode != 0
    assert sandbox.git("rev-parse", "--verify", "HEAD", check=False).returncode != 0


# --- commit-msg ---------------------------------------------------------------------------------

_SIGNED = "\n\nSigned-off-by: A Person <a@example.invalid>\n"


def _message(sandbox: Sandbox, text: str) -> Path:
    path = sandbox.root / "COMMIT_EDITMSG"
    path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "subject",
    [
        "feat: add a thing",
        "fix(cli): handle an empty argument",
        "feat(spotify)!: drop the old login",
        "refactor!: rename the module",
        "perf(audio): reuse the buffer",
        "docs: explain the hooks",
        "test(policy): cover a refused case",
        "build: bump a pin",
        "ci: cache the tools",
        "chore(deps): update ruff",
        "style: format",
        "revert: undo the change",
        "Merge branch 'dev' into feature/x",
        'Revert "feat: add a thing"',
        "fixup! feat: add a thing",
        "squash! fix(cli): handle an empty argument",
    ],
)
def test_commit_msg_accepts(sandbox: Sandbox, subject: str) -> None:
    done = sandbox.run(COMMIT_MSG, str(_message(sandbox, subject + _SIGNED)))

    assert done.returncode == 0, done.stderr
    assert not done.stderr


@pytest.mark.parametrize(
    "subject",
    [
        "",
        "add a thing",
        "Feat: add a thing",
        "feature: add a thing",
        "feat add a thing",
        "feat:add a thing",
        "feat: ",
        "feat(): add a thing",
        "feat(a b): add a thing",
        "feat(cli) : add a thing",
        "wip",
        "Update README.md",
        "merge: not a merge",
    ],
)
def test_commit_msg_refuses(sandbox: Sandbox, subject: str) -> None:
    done = sandbox.run(COMMIT_MSG, str(_message(sandbox, subject + _SIGNED)))

    assert done.returncode == 1
    assert "not a Conventional Commit" in done.stderr
    assert "feat, fix, perf, refactor, docs, test, build, ci, chore, style, revert" in done.stderr


def test_commit_msg_reads_the_first_line_that_is_not_a_comment(sandbox: Sandbox) -> None:
    text = "# Please enter the commit message\n\nfix: a real subject\n\nbody\n" + _SIGNED.lstrip(
        "\n"
    )

    done = sandbox.run(COMMIT_MSG, str(_message(sandbox, text)))

    assert done.returncode == 0, done.stderr


def test_commit_msg_warns_without_a_sign_off_but_accepts(sandbox: Sandbox) -> None:
    done = sandbox.run(COMMIT_MSG, str(_message(sandbox, "fix: a thing\n\nBody text.\n")))

    assert done.returncode == 0
    assert "warning, no Signed-off-by" in done.stderr


def test_commit_msg_does_not_warn_with_a_sign_off(sandbox: Sandbox) -> None:
    done = sandbox.run(COMMIT_MSG, str(_message(sandbox, "fix: a thing\n\nBody text." + _SIGNED)))

    assert done.returncode == 0
    assert "Signed-off-by" not in done.stderr


def test_commit_msg_leaves_the_message_untouched(sandbox: Sandbox) -> None:
    text = "fix: a thing\n\nBody text.\n"
    path = _message(sandbox, text)

    sandbox.run(COMMIT_MSG, str(path))

    assert path.read_text(encoding="utf-8") == text


def test_commit_msg_refuses_a_real_commit(sandbox: Sandbox) -> None:
    _hooks_path(sandbox, HOOKS)
    sandbox.stage("art/logo.png")

    done = sandbox.git("commit", "-q", "-m", "added a logo", check=False)

    assert done.returncode != 0
    assert "not a Conventional Commit" in done.stderr
    assert sandbox.git("rev-parse", "--verify", "HEAD", check=False).returncode != 0


# --- install-hooks.sh ---------------------------------------------------------------------------


@pytest.fixture
def installable(sandbox: Sandbox) -> Sandbox:
    """The sandbox repository with its own copy of tools/, as a checkout has it."""
    shutil.copytree(TOOLS, sandbox.repo / "tools", ignore=shutil.ignore_patterns("__pycache__"))
    return sandbox


def _install(box: Sandbox, *args: str) -> subprocess.CompletedProcess[str]:
    return box.run(box.repo / "tools" / "install-hooks.sh", *args)


def _hooks_path_setting(box: Sandbox) -> str:
    return box.git("config", "--local", "--get", "core.hooksPath", check=False).stdout.strip()


def test_install_sets_the_hooks_path_for_this_repository(installable: Sandbox) -> None:
    done = _install(installable)

    assert done.returncode == 0, done.stderr
    assert "hooks installed: core.hooksPath = tools/hooks" in done.stdout
    assert _hooks_path_setting(installable) == "tools/hooks"
    assert not (installable.root / "gitconfig").read_text(encoding="utf-8")
    for hook in ("pre-commit", "commit-msg"):
        assert os.access(installable.repo / "tools" / "hooks" / hook, os.X_OK)


def test_installed_hooks_run_on_a_commit(installable: Sandbox) -> None:
    _install(installable)
    installable.stage("a.py")

    done = installable.git("commit", "-q", "-m", "not conventional", check=False)

    assert done.returncode != 0
    assert "not a Conventional Commit" in done.stderr


def test_install_twice_changes_nothing(installable: Sandbox) -> None:
    _install(installable)

    done = _install(installable)

    assert done.returncode == 0
    assert "already installed" in done.stdout
    assert _hooks_path_setting(installable) == "tools/hooks"


def test_uninstall_unsets_the_hooks_path(installable: Sandbox) -> None:
    _install(installable)

    done = _install(installable, "--uninstall")

    assert done.returncode == 0, done.stderr
    assert "hooks uninstalled" in done.stdout
    assert not _hooks_path_setting(installable)


def test_uninstall_without_hooks_says_so(installable: Sandbox) -> None:
    done = _install(installable, "--uninstall")

    assert done.returncode == 0
    assert "not installed" in done.stdout


def test_install_refuses_another_hooks_path(installable: Sandbox) -> None:
    installable.git("config", "--local", "core.hooksPath", ".githooks")

    done = _install(installable)

    assert done.returncode == 1
    assert ".githooks" in done.stderr
    assert "already set" in done.stderr
    assert _hooks_path_setting(installable) == ".githooks"


def test_install_refuses_a_global_hooks_path(installable: Sandbox) -> None:
    gitconfig = installable.root / "gitconfig"
    gitconfig.write_text("[core]\n\thooksPath = /somewhere/hooks\n", encoding="utf-8")

    done = _install(installable)

    assert done.returncode == 1
    assert "/somewhere/hooks" in done.stderr
    assert not _hooks_path_setting(installable)


def test_uninstall_leaves_another_hooks_path_alone(installable: Sandbox) -> None:
    installable.git("config", "--local", "core.hooksPath", ".githooks")

    done = _install(installable, "--uninstall")

    assert done.returncode == 1
    assert ".githooks" in done.stderr
    assert _hooks_path_setting(installable) == ".githooks"


def test_install_rejects_an_unknown_argument(installable: Sandbox) -> None:
    done = _install(installable, "--frobnicate")

    assert done.returncode == 2
    assert "usage:" in done.stderr
    assert not _hooks_path_setting(installable)
