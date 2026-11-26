"""Exercise Git transactions using a local fixture remote and fake services."""

import json
import subprocess
from typing import TYPE_CHECKING

import pytest
from strata_fixture import engine_run_config

from jarvis import cli, config, modes, paths, ramguard, update

if TYPE_CHECKING:
    from pathlib import Path


def _git(folder: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "-C", str(folder), *args], check=True, capture_output=True, text=True, timeout=10
    )
    return done.stdout.strip()


def _commit(folder: Path, value: str) -> str:
    (folder / "marker").write_text(value, encoding="utf-8")
    _git(folder, "add", "marker")
    _git(
        folder,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.com",
        "commit",
        "-m",
        value,
    )
    return _git(folder, "rev-parse", "HEAD")


@pytest.fixture
def checkout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, str]:
    remote = tmp_path / "remote"
    remote.mkdir()
    _git(remote, "init", "--initial-branch=main")
    previous = _commit(remote, "before")
    local = tmp_path / "checkout"
    _git(tmp_path, "clone", str(remote), str(local))
    venv = local / ".venv" / "bin"
    venv.mkdir(parents=True)
    (local / ".git" / "info" / "exclude").write_text(".venv/\n", encoding="utf-8")
    monkeypatch.setattr(update.sys, "prefix", str(venv.parent))
    config.save(config.Config())
    return remote, local, previous


@pytest.fixture
def called(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, ...]]:
    calls: list[tuple[str, ...]] = []
    original = update._run  # pyright: ignore[reportPrivateUsage] - keep real Git within the fixture

    def fake(
        args: tuple[str, ...], *, cwd: Path, timeout: float, show: bool = False
    ) -> update.Finished:
        if args[0] == "git":
            return original(args, cwd=cwd, timeout=timeout, show=show)
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(update, "_run", fake)

    def which(command: str) -> str:
        return f"/usr/bin/{command}"

    def inactive(_folder: Path, _unit: str) -> str:
        return "inactive"

    monkeypatch.setattr(update.shutil, "which", which)
    monkeypatch.setattr(update, "_unit_state", inactive)
    return calls


def _results() -> list[str]:
    return [
        json.loads(line)["result"]
        for line in paths.audit_file().read_text(encoding="utf-8").splitlines()
    ]


def test_preview_then_apply(
    checkout: tuple[Path, Path, str],
    called: list[tuple[str, ...]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    remote, local, previous = checkout
    target = _commit(remote, "new release")
    assert cli.main(["update"]) == 0
    assert "new release" in capsys.readouterr().out
    assert _git(local, "rev-parse", "HEAD") == previous
    assert not called

    assert cli.main(["update", "--apply"]) == 0
    assert _git(local, "rev-parse", "HEAD") == target
    assert any(args[1:3] == ("sync", "--locked") for args in called)
    assert _results() == ["available", "applied"]


@pytest.mark.parametrize("reason", ["dirty", "branch", "diverged", "ahead"])
def test_refuses_unsafe_checkout(
    checkout: tuple[Path, Path, str], called: list[tuple[str, ...]], reason: str
) -> None:
    remote, local, _previous = checkout
    if reason != "ahead":
        _commit(remote, "new upstream")
    match reason:
        case "dirty":
            (local / "new file").touch()
        case "branch":
            _git(local, "switch", "-c", "other")
        case _:
            _commit(local, "local commit")
    before = _git(local, "rev-parse", "HEAD")

    assert cli.main(["update", "--apply"]) != 0

    assert _git(local, "rev-parse", "HEAD") == before
    assert not called
    assert _results() == ["refused"]


def test_request_in_flight_defers_apply(
    checkout: tuple[Path, Path, str], called: list[tuple[str, ...]]
) -> None:
    remote, local, previous = checkout
    _commit(remote, "new upstream")
    with modes.Answering():
        assert cli.main(["update", "--apply"]) == 75
    assert _git(local, "rev-parse", "HEAD") == previous
    assert not called
    assert _results() == ["busy"]


def test_spoken_request_holds_off_update_through_playback(
    checkout: tuple[Path, Path, str],
    called: list[tuple[str, ...]],
) -> None:
    remote, local, previous = checkout
    _commit(remote, "new upstream")
    with modes.voice_request() as admitted:
        assert admitted
        assert cli.main(["update", "--apply"]) == 75
    assert _git(local, "rev-parse", "HEAD") == previous
    assert not called


@pytest.mark.parametrize("failed_phase", ["sync", "health", "start"])
def test_failure_restores_checkout_and_locked_environment(
    checkout: tuple[Path, Path, str],
    called: list[tuple[str, ...]],
    monkeypatch: pytest.MonkeyPatch,
    failed_phase: str,
) -> None:
    remote, local, previous = checkout
    _commit(remote, "new upstream")
    original = update._run  # pyright: ignore[reportPrivateUsage] - extends the fixture stub
    attempted = False

    def fail_once(
        args: tuple[str, ...], *, cwd: Path, timeout: float, show: bool = False
    ) -> update.Finished:
        nonlocal attempted
        done = original(args, cwd=cwd, timeout=timeout, show=show)
        if not attempted and failed_phase in args:
            attempted = True
            return subprocess.CompletedProcess(args, 1, "", "")
        return done

    monkeypatch.setattr(update, "_run", fail_once)

    def active(_folder: Path, _unit: str) -> str:
        return "active"

    if failed_phase == "start":
        monkeypatch.setattr(update, "_unit_state", active)

    assert cli.main(["update", "--apply"]) != 0

    assert attempted
    assert _git(local, "rev-parse", "HEAD") == previous
    assert sum(args[1:3] == ("sync", "--locked") for args in called) == 2
    assert _results() == ["rolled back"]


def test_timer_is_opt_in_and_can_be_removed(
    checkout: tuple[Path, Path, str],
    called: list[tuple[str, ...]],
) -> None:
    _remote, local, _previous = checkout
    unit_dir = paths.xdg_config_home() / "systemd" / "user"
    assert not config.load().update.auto
    assert not unit_dir.exists()

    assert cli.main(["update", "--auto", "on"]) == 0

    assert config.load().update.auto
    timer = unit_dir / "jarvis-update.timer"
    service = unit_dir / "jarvis-update.service"
    assert "OnCalendar=daily" in timer.read_text(encoding="utf-8")
    assert "RandomizedDelaySec=" in timer.read_text(encoding="utf-8")
    text = service.read_text(encoding="utf-8")
    assert str(local / ".venv" / "bin" / "jarvis") in text
    assert "update --apply --quiet" in text
    assert any(args[1:] == ("--user", "enable", "--now", "jarvis-update.timer") for args in called)

    assert cli.main(["update", "--auto", "off"]) == 0

    assert not config.load().update.auto
    assert not timer.exists()
    assert not service.exists()
    assert _results() == ["auto on", "auto off"]


@pytest.mark.parametrize("profile_newer", [True, False])
def test_apply_preserves_running_strata_unless_its_profile_needs_reloading(
    checkout: tuple[Path, Path, str],
    called: list[tuple[str, ...]],
    monkeypatch: pytest.MonkeyPatch,
    *,
    profile_newer: bool,
) -> None:
    remote, local, _previous = checkout
    target = _commit(remote, "release")
    profile = paths.config_dir() / "strata-profile.json"
    profile.parent.mkdir(parents=True, exist_ok=True)
    profile.write_text("existing profile", encoding="utf-8")

    def active(_folder: Path, _unit: str) -> str:
        return "active"

    original = update._run  # pyright: ignore[reportPrivateUsage] - extends the service stub

    def start_time(
        args: tuple[str, ...], *, cwd: Path, timeout: float, show: bool = False
    ) -> update.Finished:
        done = original(args, cwd=cwd, timeout=timeout, show=show)
        if "ActiveEnterTimestamp" in args:
            stamp = "Thu 1970-01-01" if profile_newer else "Thu 2099-01-01"
            return subprocess.CompletedProcess(args, 0, f"{stamp} 00:00:01.000000 UTC\n", "")
        return done

    monkeypatch.setattr(update, "_run", start_time)
    monkeypatch.setattr(update, "_unit_state", active)

    assert cli.main(["update", "--apply"]) == 0
    assert _git(local, "rev-parse", "HEAD") == target
    restarted = ("systemctl", "--user", "restart", "jarvis-strata.service") in called
    assert restarted == profile_newer


def test_rollback_refuses_to_move_a_branch_checked_out_mid_update(
    checkout: tuple[Path, Path, str], called: list[tuple[str, ...]], monkeypatch: pytest.MonkeyPatch
) -> None:
    remote, local, _previous = checkout
    target = _commit(remote, "release")
    original = update._run  # pyright: ignore[reportPrivateUsage] - extends the service stub

    def change_branch(
        args: tuple[str, ...], *, cwd: Path, timeout: float, show: bool = False
    ) -> update.Finished:
        done = original(args, cwd=cwd, timeout=timeout, show=show)
        if "health" in args:
            _git(local, "switch", "-c", "personal")
            return subprocess.CompletedProcess(args, 1, "", "")
        return done

    monkeypatch.setattr(update, "_run", change_branch)

    assert cli.main(["update", "--apply"]) != 0
    assert _git(local, "branch", "--show-current") == "personal"
    assert _git(local, "rev-parse", "HEAD") == target
    assert _results() == ["rollback failed"]
    assert sum(args[1:3] == ("sync", "--locked") for args in called) == 1


def test_doctor_uses_the_cached_branch_and_reports_the_last_attempt(
    checkout: tuple[Path, Path, str],
    called: list[tuple[str, ...]],
) -> None:
    remote, _local, _previous = checkout
    target = _commit(remote, "new release")
    assert cli.main(["update"]) == 0

    available, automatic, result = update.doctor_notes(config.load(), cli.launch_for_update)

    assert target[:12] in available
    assert automatic.endswith("off")
    assert "available" in result
    assert not called


def test_apply_does_not_restart_a_stale_strata_into_too_little_memory(
    checkout: tuple[Path, Path, str],
    called: list[tuple[str, ...]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    remote, local, _previous = checkout
    target = _commit(remote, "release")
    run = engine_run_config(tmp_path)
    model = tmp_path / "models" / "q4.gguf"
    with model.open("wb") as file:
        file.truncate(39 * 1024**3)
    config.save(config.Config(strata=config.StrataConfig(run_config=run, vram_reserve_mib=12000)))
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemAvailable:   17825792 kB\n", encoding="utf-8")  # 17 GiB
    monkeypatch.setattr(ramguard, "MEMINFO", meminfo)
    monkeypatch.setattr(ramguard, "CGROUP_ROOT", tmp_path / "cgroup")
    profile = paths.config_dir() / "strata-profile.json"
    profile.parent.mkdir(parents=True, exist_ok=True)
    profile.write_text("existing profile", encoding="utf-8")

    def active(_folder: Path, _unit: str) -> str:
        return "active"

    original = update._run  # pyright: ignore[reportPrivateUsage] - extends the service stub

    def stale(
        args: tuple[str, ...], *, cwd: Path, timeout: float, show: bool = False
    ) -> update.Finished:
        done = original(args, cwd=cwd, timeout=timeout, show=show)
        if "ActiveEnterTimestamp" in args:
            return subprocess.CompletedProcess(args, 0, "Thu 1970-01-01 00:00:01.000000 UTC\n", "")
        return done

    monkeypatch.setattr(update, "_run", stale)
    monkeypatch.setattr(update, "_unit_state", active)

    assert cli.main(["update", "--apply"]) == 0

    assert _git(local, "rev-parse", "HEAD") == target
    assert ("systemctl", "--user", "restart", "jarvis-strata.service") not in called
    assert "was not restarted" in capsys.readouterr().err
