"""What Jarvis says about itself, and reading its own configuration files."""

import json
import os
from pathlib import Path

import pytest

from jarvis import __version__, paths
from jarvis.about import HIDDEN, AboutOps, facts, hide_secrets, jarvis_files, section_of
from jarvis.agent import Limits
from jarvis.config import BoundariesConfig, Config, StrataConfig
from jarvis.operations import OperationError, validate_args
from jarvis.policy import Origin

LIMITS = Limits(tool_text_chars=3000)


def _run_config(tmp_path: Path, name: str = "some-model-q4") -> Path:
    path = tmp_path / f"strata-{name}.json"
    data = {
        "exe": "component:strata-engine",
        "args": ["--native", str(tmp_path / "weights.gguf"), "--max-context", "32768"],
        "model_name": name,
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _ops(config: Config, files: dict[str, Path]) -> AboutOps:
    return AboutOps(config, "chat", LIMITS, files=lambda: files)


def test_jarvis_names_its_version_and_the_model_its_run_config_names(tmp_path: Path) -> None:
    config = Config(strata=StrataConfig(run_config=_run_config(tmp_path)))

    said = AboutOps(config, "voice", LIMITS).about({})

    assert said.origin is Origin.TOOL_STATUS
    assert __version__ in said.text
    assert "some-model-q4" in said.text
    assert "weights.gguf" in said.text
    assert str(paths.config_file()) in said.text


def test_with_no_usable_run_config_it_says_there_is_no_model() -> None:
    assert "Model: none usable" in facts(Config(), "voice", LIMITS)


@pytest.mark.parametrize(
    ("bypass", "said"), [(True, "Asking before acting: off"), (False, "Asking before acting: on")]
)
def test_it_says_whether_it_asks_before_acting(*, bypass: bool, said: str) -> None:
    config = Config(
        boundaries=BoundariesConfig(bypass_consent=bypass, classes={"write_file": "deny"})
    )

    text = facts(config, "chat", LIMITS)

    assert said in text
    assert "write_file deny" in text


def test_a_config_file_is_read_as_file_text(tmp_path: Path) -> None:
    config_file = tmp_path / "config.toml"
    config_file.write_text('[audio]\nvoice = "kokoro:bm_george"\n', encoding="utf-8")

    read = _ops(Config(), {"config": config_file}).read({"file": "config"})

    assert read.origin is Origin.FILE  # outside text: an action after it needs a yes
    assert 'voice = "kokoro:bm_george"' in read.text
    assert str(config_file) in read.text


@pytest.mark.parametrize(
    "line",
    [
        'api_key = "abc123"',
        "  token = abc123",
        '"refresh_token": "abc123",',
        "apiKey: abc123",
        "STRATA_API_KEY=abc123",
        "Environment=STRATA_API_KEY=abc123",
        'Environment="SPOTIFY_SECRET=abc123"',
        "password=abc123",
        '"--api-key",\n  "abc123",',
        "--api-key=abc123",
    ],
)
def test_a_value_under_a_secrets_name_is_hidden(line: str) -> None:
    shown = hide_secrets(f"[x]\n{line}\nother = 1\n")

    assert "abc123" not in shown
    assert HIDDEN in shown
    assert "other = 1" in shown


@pytest.mark.parametrize(
    "line",
    [
        "memory_tokens = 9100",
        "reply_tokens = 2048",
        'azure_keyword_model = "/models/kw.table"',
        "EnvironmentFile=/state/strata.env",
        "Environment=PIPEWIRE_REMOTE=/nonexistent",
        "reasoning_budget_tokens = 512",
        '"--max-context",\n  "32768",',
    ],
)
def test_ordinary_settings_are_shown_as_they_are(line: str) -> None:
    assert hide_secrets(line) == line


def test_the_state_folder_is_never_among_the_files_it_reads(tmp_path: Path) -> None:
    config = Config(strata=StrataConfig(run_config=_run_config(tmp_path)))

    for path in jarvis_files(config).values():
        assert not path.is_relative_to(paths.state_dir())


def test_the_file_comes_from_a_fixed_list_and_never_from_a_path() -> None:
    op = next(
        op
        for op in AboutOps(Config(), "chat", LIMITS).operations()
        if op.name == "read_jarvis_file"
    )

    with pytest.raises(ValueError, match="file"):
        validate_args(op, {"file": "/etc/passwd"})
    with pytest.raises(ValueError, match="path"):
        validate_args(op, {"file": "config", "path": "/etc/passwd"})


def test_neither_operation_changes_anything() -> None:
    assert all(not op.side_effect for op in AboutOps(Config(), "chat", LIMITS).operations())


def test_one_section_comes_with_its_own_tables(tmp_path: Path) -> None:
    text = (
        "[audio]\nvoice = 1\n\n[files]\nread_roots = []\n\n[files.extra]\nx = 2\n\n"
        "[shell]\non = 3\n"
    )
    config_file = tmp_path / "config.toml"
    config_file.write_text(text, encoding="utf-8")

    read = _ops(Config(), {"config": config_file}).read({"file": "config", "section": "[files]"})

    assert "read_roots" in read.text
    assert "x = 2" in read.text
    assert "voice" not in read.text
    assert "on = 3" not in read.text


def test_a_section_the_file_lacks_is_refused(tmp_path: Path) -> None:
    config_file = tmp_path / "config.toml"
    config_file.write_text("[audio]\n", encoding="utf-8")

    with pytest.raises(OperationError, match="no \\[weather\\] section"):
        _ops(Config(), {"config": config_file}).read({"file": "config", "section": "weather"})


def test_a_long_file_is_cut_to_the_channels_room(tmp_path: Path) -> None:
    config_file = tmp_path / "config.toml"
    config_file.write_text("[a]\n" + "x = 1\n" * 2000, encoding="utf-8")

    read = _ops(Config(), {"config": config_file}).read({"file": "config"})

    assert len(read.text) <= LIMITS.tool_text_chars
    assert "ask for one section" in read.text


@pytest.mark.parametrize("missing", ["unset", "absent"])
def test_a_file_that_is_not_set_up_or_not_there_is_refused(tmp_path: Path, missing: str) -> None:
    path = Path() if missing == "unset" else tmp_path / "nothing.json"

    with pytest.raises(OperationError, match="not set up" if missing == "unset" else "cannot read"):
        _ops(Config(), {"strata_run_config": path}).read({"file": "strata_run_config"})


def test_section_of_finds_a_unit_files_section() -> None:
    unit = "[Unit]\nDescription=x\n\n[Service]\nExecStart=/bin/true\n"

    assert section_of(unit, "service") == "[Service]\nExecStart=/bin/true"


def test_a_secret_in_a_file_read_is_hidden(tmp_path: Path) -> None:
    unit = tmp_path / "jarvis-strata.service"
    unit.write_text("[Service]\nEnvironment=STRATA_API_KEY=abc123\nExecStart=/bin/strata\n")

    read = _ops(Config(), {"strata_unit": unit}).read({"file": "strata_unit"})

    assert "abc123" not in read.text
    assert "ExecStart=/bin/strata" in read.text


def test_a_file_the_files_boundary_keeps_private_is_refused_through_a_link(tmp_path: Path) -> None:
    secret = tmp_path / "private" / "notes.toml"
    secret.parent.mkdir()
    secret.write_text("x = 1\n", encoding="utf-8")
    link = tmp_path / "config.toml"
    link.symlink_to(secret)
    ops = AboutOps(
        Config(),
        "chat",
        LIMITS,
        refused=lambda path: path.is_relative_to(tmp_path / "private"),
        files=lambda: {"config": link},
    )

    with pytest.raises(OperationError, match="kept private"):
        ops.read({"file": "config"})


def test_something_that_is_not_a_file_is_refused(tmp_path: Path) -> None:
    pipe = tmp_path / "config.toml"
    os.mkfifo(pipe)

    with pytest.raises(OperationError, match="not a file"):
        _ops(Config(), {"config": pipe}).read({"file": "config"})
