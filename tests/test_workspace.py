"""Jarvis's workspace folder (ADR 0025): where it is, who may use it, and when it asks first."""

import json
import os
import stat
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest
from test_agent import Clock, Counter, ScriptedModel

from jarvis import cli, paths
from jarvis.about import facts
from jarvis.agent import Agent, Hooks, Limits
from jarvis.app import files_boundary, no_action_operation, system_prompt
from jarvis.audit import AuditLog
from jarvis.config import DEFAULT_WORKSPACE, Config, ConfigError, FilesConfig, ShellConfig
from jarvis.operations import UNASKED_ARG, Operation, Registry
from jarvis.pc.files import MADE_FILE, NEW_FILE, Boundary, FileOps, FileRefusalError
from jarvis.pc.shell import ShellOps
from jarvis.pc.workspace import Ledger, workspace_path, workspace_problem
from jarvis.policy import OpClass, Origin, Verdict, decide

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "me"
    (home / "Documents").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / ".config"))
    monkeypatch.setenv("XDG_STATE_HOME", str(home / ".local" / "state"))
    return home


def _files(**fields: object) -> FilesConfig:
    return FilesConfig.model_validate(fields)


def _ops(home: Path, config: FilesConfig | None = None, *, made: Ledger | None = None) -> FileOps:
    boundary = Boundary.of(config or _files(), home, protected=(paths.state_dir(),))
    return FileOps(boundary, max_chars=2000, made=made)


def _write_op(ops: FileOps) -> Operation:
    return Registry(ops.operations()).get("write_file") or pytest.fail("no write_file")


def _run(ops: FileOps, **args: object) -> str:
    op = _write_op(ops)
    bound = ops.bind_write({"mode": "create", **args})
    return op.run(bound).text


# --- the setting ------------------------------------------------------------------------------


def test_the_workspace_is_in_documents_by_default_and_in_both_roots(home: Path) -> None:
    boundary = Boundary.of(_files(), home)

    assert Config().files.workspace == DEFAULT_WORKSPACE == "~/Documents/Jarvis"
    assert boundary.workspace == home / "Documents" / "Jarvis"
    assert boundary.workspace in boundary.read_roots
    assert boundary.workspace in boundary.write_roots


def test_a_custom_workspace_is_in_both_roots_whatever_the_roots_say(home: Path) -> None:
    config = _files(workspace="/data/jarvis-work", read_roots=["~/Projects"], write_roots=[])
    boundary = Boundary.of(config, home)

    assert boundary.workspace is not None
    assert str(boundary.workspace) == "/data/jarvis-work"
    assert boundary.workspace in boundary.read_roots
    assert boundary.workspace in boundary.write_roots


@pytest.mark.parametrize(
    "value",
    [
        "",
        "relative/folder",
        "~",
        "/",
        "~/Documents/../Jarvis",
        "~/.local/state/jarvis",
        "~/.local/state/jarvis/work",
        "~/.local/state",
        "~/.config/jarvis",
        "~/.config/jarvis/work",
        "/proc/work",
        "/sys",
        "~/.jarvis-work",
        "~/.cache/jarvis",
        "~/.local/share/jarvis-work",
        "~/.config/anything",
    ],
)
def test_a_workspace_that_is_home_protected_or_odd_is_refused_by_the_config(
    home: Path, value: str
) -> None:
    del home
    with pytest.raises(ValueError, match=r"workspace|root"):
        _files(workspace=value)


def test_a_workspace_inside_a_folder_the_caller_protects_is_refused(home: Path) -> None:
    secret = home / "Vault"

    with pytest.raises(ConfigError, match="workspace"):
        Boundary.of(_files(workspace="~/Vault/work"), home, protected=(secret,))
    with pytest.raises(ConfigError, match="workspace"):
        Boundary.of(_files(workspace="~/Vault"), home, protected=(secret,))


# --- the folder on disk -----------------------------------------------------------------------


def test_a_missing_workspace_is_made_private(home: Path) -> None:
    folder = workspace_path(_files(), home)
    old = os.umask(0)
    try:
        assert not workspace_problem(folder, create=True)
    finally:
        os.umask(old)

    assert folder.is_dir()
    assert stat.S_IMODE(folder.stat().st_mode) == 0o700


def test_checking_does_not_make_the_folder(home: Path) -> None:
    folder = workspace_path(_files(), home)

    assert not workspace_problem(folder, create=False)
    assert not folder.exists()


def test_a_folder_the_owner_made_is_used_as_it_is(home: Path) -> None:
    folder = home / "Documents" / "Jarvis"
    folder.mkdir()
    folder.chmod(0o755)

    assert not workspace_problem(folder, create=True)
    assert stat.S_IMODE(folder.stat().st_mode) == 0o755


def test_a_workspace_that_is_a_link_is_refused(home: Path) -> None:
    real = home / "elsewhere"
    real.mkdir()
    folder = home / "Documents" / "Jarvis"
    folder.symlink_to(real)

    assert "link" in workspace_problem(folder, create=True)


def test_a_workspace_that_is_a_file_is_refused(home: Path) -> None:
    folder = home / "Documents" / "Jarvis"
    folder.write_text("hello\n", encoding="utf-8")

    assert "not a folder" in workspace_problem(folder, create=True)


@pytest.mark.parametrize("mode", [0o770, 0o707, 0o777])
def test_a_workspace_others_can_write_to_is_refused(home: Path, mode: int) -> None:
    folder = home / "Documents" / "Jarvis"
    folder.mkdir()
    folder.chmod(mode)

    assert "written by other users" in workspace_problem(folder, create=True)
    assert stat.S_IMODE(folder.stat().st_mode) == mode


def test_a_workspace_of_another_user_is_refused(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = home / "Documents" / "Jarvis"
    folder.mkdir()
    monkeypatch.setattr(os, "geteuid", lambda: folder.stat().st_uid + 1)

    assert "another user" in workspace_problem(folder, create=True)


def test_the_files_boundary_makes_the_workspace_and_drops_an_unsafe_one(home: Path) -> None:
    config = Config()
    first = files_boundary(config, home)

    assert first.workspace == home / "Documents" / "Jarvis"
    assert first.workspace is not None
    assert first.workspace.is_dir()

    first.workspace.chmod(0o777)
    assert files_boundary(config, home).workspace is None


def test_a_link_in_the_way_does_not_lead_the_workspace_out(home: Path) -> None:
    elsewhere = home / "disk"
    elsewhere.mkdir()
    (home / "Documents").rmdir()
    (home / "Documents").symlink_to(elsewhere)

    boundary = files_boundary(Config(), home)

    assert boundary.workspace == elsewhere / "Jarvis"


# --- where things go --------------------------------------------------------------------------


def test_a_bare_name_and_a_relative_path_are_written_in_the_workspace(home: Path) -> None:
    ops = _ops(home)
    workspace = home / "Documents" / "Jarvis"

    _run(ops, path="notes.md", content="one")
    _run(ops, path="drafts/two.md", content="two")

    assert (workspace / "notes.md").read_text(encoding="utf-8") == "one\n"
    assert (workspace / "drafts" / "two.md").read_text(encoding="utf-8") == "two\n"
    assert not (home / "notes.md").exists()


def test_home_and_absolute_paths_are_unchanged(home: Path) -> None:
    ops = _ops(home)

    _run(ops, path="~/Documents/here.md", content="a")
    _run(ops, path=str(home / "Documents" / "there.md"), content="b")

    assert (home / "Documents" / "here.md").is_file()
    assert (home / "Documents" / "there.md").is_file()
    assert not (home / "Documents" / "Jarvis" / "here.md").exists()


def test_a_relative_read_finds_the_workspace_first_then_home(home: Path) -> None:
    ops = _ops(home)
    (home / "Documents" / "Jarvis").mkdir()
    (home / "Documents" / "Jarvis" / "both.md").write_text("in workspace\n", encoding="utf-8")
    (home / "both.md").write_text("in home\n", encoding="utf-8")
    (home / "only-home.md").write_text("only home\n", encoding="utf-8")
    read = Registry(ops.operations()).get("read_file") or pytest.fail("no read_file")

    assert "in workspace" in read.run({"path": "both.md"}).text
    assert "only home" in read.run({"path": "only-home.md"}).text


def test_the_workspace_is_writable_even_when_no_root_covers_it(home: Path) -> None:
    ops = _ops(home, _files(write_roots=[], workspace="~/Elsewhere/Jarvis"))

    _run(ops, path="a.md", content="x")

    assert (home / "Elsewhere" / "Jarvis" / "a.md").is_file()


def test_the_folder_itself_is_not_a_file_to_write(home: Path) -> None:
    ops = _ops(home)

    with pytest.raises(FileRefusalError):
        ops.bind_write({"path": "~/Documents/Jarvis", "content": "x"})


def test_a_hidden_name_is_allowed_in_the_workspace_only(home: Path) -> None:
    ops = _ops(home)

    _run(ops, path=".scratch/todo.md", content="x")

    assert (home / "Documents" / "Jarvis" / ".scratch" / "todo.md").is_file()
    with pytest.raises(FileRefusalError, match="hidden"):
        ops.bind_write({"path": "~/Documents/.scratch.md", "content": "x"})


@pytest.mark.parametrize(
    "path",
    ["api-secret.txt", ".env", "keys/id_ed25519", "my password list.txt", "site/.git/hooks/x"],
)
def test_secret_names_and_version_control_folders_are_refused_inside_the_workspace(
    home: Path, path: str
) -> None:
    ops = _ops(home)

    with pytest.raises(FileRefusalError):
        ops.bind_write({"path": path, "content": "x"})


def test_the_owners_deny_list_still_applies_in_the_workspace(home: Path) -> None:
    ops = _ops(home, _files(deny=["private*"]))

    with pytest.raises(FileRefusalError, match="private"):
        ops.bind_write({"path": "private-notes.md", "content": "x"})


# --- when it asks -----------------------------------------------------------------------------


def _agent(ops: FileOps, replies: list[dict[str, Any]], **overrides: str) -> tuple[Agent, AuditLog]:
    audit = AuditLog(None)
    counter = Counter()
    registry = Registry(
        [
            *ops.operations(),
            counter.op("page", origin=Origin.WEB, side_effect=False),
            no_action_operation(),
        ]
    )
    agent = Agent(
        ScriptedModel(replies),
        registry,
        "system",
        audit,
        dict(overrides),
        clock=Clock(),
        limits=Limits(),
    )
    return agent, audit


def _write(
    path: str, content: str = "x", mode: str = "create", call_id: str = "c1"
) -> dict[str, Any]:
    args = {"path": path, "content": content, "mode": mode}
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": "write_file", "arguments": json.dumps(args)},
            }
        ],
    }


def _page() -> dict[str, Any]:
    call = _write("x")
    call["tool_calls"][0]["function"] = {"name": "page", "arguments": "{}"}
    return call


DONE = {"role": "assistant", "content": "Done."}


def _asked(agent: Agent, text: str = "save it") -> list[str]:
    asked: list[str] = []
    agent.handle(text, Hooks(confirm=lambda q: asked.append(q) or False))
    return asked


def _verdict(audit: AuditLog) -> tuple[str, str]:
    call = next(e for e in audit.events if e["event"] == "tool_call" and e["op"] == "write_file")
    return str(call["verdict"]), str(call["reason"])


def test_a_new_file_in_the_workspace_is_written_without_asking(home: Path) -> None:
    agent, audit = _agent(_ops(home), [_write("notes.md", "hello"), DONE])

    assert _asked(agent) == []

    assert (home / "Documents" / "Jarvis" / "notes.md").read_text(encoding="utf-8") == "hello\n"
    assert _verdict(audit) == ("run", "class safe: a new file in the workspace")


def test_a_new_file_outside_the_workspace_still_asks(home: Path) -> None:
    agent, audit = _agent(_ops(home), [_write("~/Documents/notes.md"), DONE])

    assert len(_asked(agent)) == 1

    assert not (home / "Documents" / "notes.md").exists()
    assert _verdict(audit)[0] == "confirm"


def test_a_new_file_in_the_workspace_asks_after_outside_text(home: Path) -> None:
    agent, audit = _agent(_ops(home), [_page(), _write("notes.md"), DONE])

    assert len(_asked(agent)) == 1

    assert not (home / "Documents" / "Jarvis" / "notes.md").exists()
    assert _verdict(audit) == ("confirm", "side effect while untrusted text is in the conversation")


def test_a_new_file_in_the_workspace_asks_when_the_owner_set_the_class(home: Path) -> None:
    agent, _ = _agent(_ops(home), [_write("notes.md"), DONE], write_file="confirm")

    assert len(_asked(agent)) == 1


def test_a_file_the_owner_put_in_the_workspace_asks_before_it_is_replaced(home: Path) -> None:
    workspace = home / "Documents" / "Jarvis"
    workspace.mkdir()
    (workspace / "mine.md").write_text("owner's words\n", encoding="utf-8")
    ledger = Ledger(home / "ledger.json")
    agent, audit = _agent(_ops(home, made=ledger), [_write("mine.md", "new", "replace"), DONE])

    assert len(_asked(agent)) == 1

    assert (workspace / "mine.md").read_text(encoding="utf-8") == "owner's words\n"
    assert _verdict(audit)[0] == "confirm"


def test_a_file_the_owner_put_in_the_workspace_asks_before_it_is_added_to(home: Path) -> None:
    workspace = home / "Documents" / "Jarvis"
    workspace.mkdir()
    (workspace / "mine.md").write_text("owner's words\n", encoding="utf-8")
    agent, _ = _agent(
        _ops(home, made=Ledger(home / "ledger.json")), [_write("mine.md", "more", "append"), DONE]
    )

    assert len(_asked(agent)) == 1


def test_a_file_jarvis_made_is_changed_without_asking_until_someone_swaps_it(home: Path) -> None:
    workspace = home / "Documents" / "Jarvis"
    ledger = Ledger(home / "ledger.json")
    ops = _ops(home, made=ledger)
    first, _ = _agent(ops, [_write("a.md", "one"), DONE])
    assert _asked(first) == []

    second, audit = _agent(ops, [_write("a.md", "two", "replace"), DONE])
    assert _asked(second) == []
    assert (workspace / "a.md").read_text(encoding="utf-8") == "two\n"
    assert _verdict(audit) == ("run", "class safe: a file Jarvis made in the workspace")

    third, _ = _agent(ops, [_write("a.md", "three", "append"), DONE])
    assert _asked(third) == []
    assert (workspace / "a.md").read_text(encoding="utf-8") == "two\nthree\n"

    # The owner saves their own version over it with an editor that renames a new file in.
    (workspace / "a.md.new").write_text("owner's version\n", encoding="utf-8")
    (workspace / "a.md.new").replace(workspace / "a.md")
    fourth, _ = _agent(ops, [_write("a.md", "four", "replace"), DONE])
    assert len(_asked(fourth)) == 1
    assert (workspace / "a.md").read_text(encoding="utf-8") == "owner's version\n"


def test_replacing_the_owners_file_after_a_yes_does_not_make_it_jarvis_s(home: Path) -> None:
    workspace = home / "Documents" / "Jarvis"
    workspace.mkdir()
    (workspace / "mine.md").write_text("owner's words\n", encoding="utf-8")
    ops = _ops(home, made=Ledger(home / "ledger.json"))
    yes, _ = _agent(ops, [_write("mine.md", "new", "replace"), DONE])
    yes.handle("replace it", Hooks(confirm=lambda _q: True))
    assert (workspace / "mine.md").read_text(encoding="utf-8") == "new\n"

    again, _ = _agent(ops, [_write("mine.md", "newer", "replace"), DONE])

    assert len(_asked(again)) == 1


def test_without_a_ledger_only_new_files_are_unasked(home: Path) -> None:
    ops = _ops(home)
    first, _ = _agent(ops, [_write("a.md", "one"), DONE])
    assert _asked(first) == []

    second, _ = _agent(ops, [_write("a.md", "two", "replace"), DONE])

    assert len(_asked(second)) == 1


def test_the_unasked_reason_is_given_only_for_a_confirm_without_an_override() -> None:
    op = Counter().op("write_file", cls=OpClass.CONFIRM)

    lowered = decide(op, "write_file", {}, untrusted_in_context=False, unasked="a new file")
    assert (lowered.verdict, lowered.reason) == (Verdict.RUN, "class safe: a new file")
    assert decide(op, "write_file", {}, untrusted_in_context=False).verdict is Verdict.CONFIRM
    assert (
        decide(op, "write_file", {}, untrusted_in_context=True, unasked="a new file").verdict
        is Verdict.CONFIRM
    )
    assert (
        decide(
            op, "write_file", {"write_file": "confirm"}, untrusted_in_context=False, unasked="x"
        ).verdict
        is Verdict.CONFIRM
    )
    assert (
        decide(
            op, "write_file", {"write_file": "deny"}, untrusted_in_context=False, unasked="x"
        ).verdict
        is Verdict.DENY
    )


def test_the_ledger_survives_a_restart_and_a_damaged_file_reads_as_empty(home: Path) -> None:
    file = home / "ledger.json"
    made = home / "a.md"
    made.write_text("x", encoding="utf-8")
    info = made.stat()
    Ledger(file).record(made, info)

    assert Ledger(file).made(made, info)
    assert stat.S_IMODE(file.stat().st_mode) == 0o600
    file.write_text("{not json", encoding="utf-8")
    assert not Ledger(file).made(made, info)
    file.write_text('{"files": {"/x": "wrong"}}', encoding="utf-8")
    assert not Ledger(file).made(made, info)


# --- what the model and the owner are told ----------------------------------------------------


@pytest.mark.usefixtures("home")
def test_the_prompt_and_the_facts_tell_the_model_where_the_workspace_is() -> None:
    custom = Config(files=_files(workspace="~/Work/Jarvis"))

    assert "Bare file names save in your workspace, ~/Documents/Jarvis" in system_prompt(Config())
    assert "your workspace, ~/Work/Jarvis" in system_prompt(custom, "chat")
    assert "workspace ~/Work/Jarvis is always readable and writable" in facts(
        custom, "voice", Limits()
    )


def test_the_prompt_stays_silent_about_a_workspace_that_cannot_be_used(home: Path) -> None:
    folder = home / "Documents" / "Jarvis"
    folder.mkdir()
    folder.chmod(0o777)

    assert "in your workspace" not in system_prompt(Config())


def test_doctor_checks_the_workspace(home: Path) -> None:
    cfg = Config()
    ok, what, _ = cli._workspace_check(cfg)  # pyright: ignore[reportPrivateUsage] - the doctor's own check
    assert ok
    assert "made when Jarvis starts" in what

    folder = home / "Documents" / "Jarvis"
    folder.mkdir()
    folder.chmod(0o700)
    ok, what, _ = cli._workspace_check(cfg)  # pyright: ignore[reportPrivateUsage] - same
    assert (ok, what) == (True, f"workspace {folder}")

    folder.chmod(0o777)
    ok, what, fix = cli._workspace_check(cfg)  # pyright: ignore[reportPrivateUsage] - same
    assert not ok
    assert "written by other users" in what
    assert "files.workspace" in fix


def test_setup_makes_the_workspace_and_says_where(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cli._setup_workspace(Config())  # pyright: ignore[reportPrivateUsage] - the setup step itself

    assert (home / "Documents" / "Jarvis").is_dir()
    assert f"workspace: {home / 'Documents' / 'Jarvis'}" in capsys.readouterr().out


def test_a_shell_command_starts_in_the_workspace_with_home_unchanged(home: Path) -> None:
    folder = home / "Documents" / "Jarvis"
    folder.mkdir()
    ops = ShellOps(ShellConfig(enabled=True), home, max_chars=1000, workspace=folder)
    argv = ops.command_line("pwd", "jarvis-shell-1")

    assert f"--working-directory={folder}" in argv
    assert f"HOME={home}" in argv
    plain = ShellOps(ShellConfig(enabled=True), home, max_chars=1000).command_line("pwd", "u")
    assert f"--working-directory={home}" in plain


def test_the_chat_exports_into_the_workspace_else_where_it_did_before(home: Path) -> None:
    cfg = Config()

    assert cli._export_folder(cfg) == home / "Documents" / "Jarvis"  # pyright: ignore[reportPrivateUsage] - the chat's own folder choice
    (home / "Documents" / "Jarvis").chmod(0o777)
    assert cli._export_folder(cfg) == home / "Documents"  # pyright: ignore[reportPrivateUsage] - same


def test_a_file_under_a_reused_inode_is_not_jarvis_s(home: Path) -> None:
    # The owner deleted Jarvis's file and made their own; the filesystem may hand the same inode
    # back. Same device and inode, another change time: Jarvis asks.
    workspace = home / "Documents" / "Jarvis"
    ledger = Ledger(home / "ledger.json")
    ops = _ops(home, made=ledger)
    first, _ = _agent(ops, [_write("a.md", "one"), DONE])
    assert _asked(first) == []
    file = workspace / "a.md"
    before = file.stat()
    file.write_text("the owner's own\n", encoding="utf-8")  # same inode, new change time
    after = file.stat()
    assert (after.st_dev, after.st_ino) == (before.st_dev, before.st_ino)
    assert after.st_ctime_ns != before.st_ctime_ns
    assert not ledger.made(file, after)

    again, _ = _agent(ops, [_write("a.md", "two", "replace"), DONE])

    assert len(_asked(again)) == 1
    assert file.read_text(encoding="utf-8") == "the owner's own\n"


def test_a_file_swapped_after_the_call_was_bound_is_not_written(home: Path) -> None:
    workspace = home / "Documents" / "Jarvis"
    ops = _ops(home, made=Ledger(home / "ledger.json"))
    _run(ops, path="a.md", content="one")
    bound = ops.bind_write(
        {"path": "a.md", "content": "two", "mode": "append", UNASKED_ARG: MADE_FILE}
    )
    (workspace / "a.md").write_text("the owner's own\n", encoding="utf-8")

    with pytest.raises(FileRefusalError, match="changed"):
        _write_op(ops).run(bound)

    assert (workspace / "a.md").read_text(encoding="utf-8") == "the owner's own\n"


# --- review fixes: floor, bound verdict, re-judging, hidden names, links ---


def test_the_unasked_lowering_respects_the_operations_floor() -> None:
    floored = Operation(
        "write_file",
        "x",
        {"type": "object", "properties": {}},
        lambda _a: pytest.fail("not run"),
        default_class=OpClass.CONFIRM,
        lowest_class=OpClass.CONFIRM,
    )

    decision = decide(floored, "write_file", {}, untrusted_in_context=False, unasked=NEW_FILE)

    assert decision.verdict is Verdict.CONFIRM
    assert not decision.unasked


def test_the_decision_carries_the_reason_the_operation_is_held_to() -> None:
    op = Counter().op("write_file", cls=OpClass.CONFIRM)

    assert decide(op, "write_file", {}, untrusted_in_context=False, unasked="x").unasked == "x"
    assert not decide(op, "write_file", {}, untrusted_in_context=True, unasked="x").unasked


def test_a_new_file_verdict_is_bound_to_the_write_and_a_file_that_appeared_is_not_touched(
    home: Path,
) -> None:
    workspace = home / "Documents" / "Jarvis"
    workspace.mkdir()
    ops = _ops(home, made=Ledger(home / "ledger.json"))
    write = _write_op(ops)
    for mode in ("create", "replace", "append"):
        name = f"{mode}.md"
        args = {"path": name, "content": "ai", "mode": mode, UNASKED_ARG: NEW_FILE}
        bound = ops.bind_write(args)  # the file is not there yet
        (workspace / name).write_text("the owner's\n", encoding="utf-8")  # then it appears

        with pytest.raises(FileRefusalError, match=r"already exists|changed"):
            write.run(bound)
        with pytest.raises(FileRefusalError, match=r"appeared|already exists"):
            ops.bind_write(args)

        assert (workspace / name).read_text(encoding="utf-8") == "the owner's\n"


def test_a_new_file_verdict_makes_the_file_even_when_asked_to_append(home: Path) -> None:
    ops = _ops(home)
    bound = ops.bind_write(
        {"path": "n.md", "content": "hi", "mode": "append", UNASKED_ARG: NEW_FILE}
    )

    assert "Created" in _write_op(ops).run(bound).text


def test_a_made_file_verdict_is_held_to_the_file_judged(home: Path) -> None:
    ops = _ops(home, made=Ledger(home / "ledger.json"))
    _run(ops, path="a.md", content="one")
    link = home / "Documents" / "Jarvis" / "second-name.md"
    bound = ops.bind_write(
        {"path": "a.md", "content": "x", "mode": "append", UNASKED_ARG: MADE_FILE}
    )
    os.link(home / "Documents" / "Jarvis" / "a.md", link)  # a second name since the judgement

    with pytest.raises(FileRefusalError, match="changed"):
        _write_op(ops).run(bound)


def test_a_workspace_made_writable_by_others_after_start_stops_unasked_writes(home: Path) -> None:
    ops = _ops(home)
    workspace = home / "Documents" / "Jarvis"
    workspace.mkdir()
    workspace.chmod(0o700)
    first, _ = _agent(ops, [_write("a.md"), DONE])
    assert _asked(first) == []
    workspace.chmod(0o770)

    second, _ = _agent(ops, [_write("b.md"), DONE])
    assert len(_asked(second)) == 1
    with pytest.raises(FileRefusalError, match="written by other users"):
        _write_op(ops).run(ops.bind_write({"path": "b.md", "content": "x"}))

    assert not (workspace / "b.md").exists()


def test_a_workspace_swapped_for_a_link_after_start_is_not_written_to(home: Path) -> None:
    ops = _ops(home)
    workspace = home / "Documents" / "Jarvis"
    workspace.mkdir()
    elsewhere = home / "Documents" / "Elsewhere"
    elsewhere.mkdir()
    workspace.rmdir()
    workspace.symlink_to(elsewhere)

    agent, _ = _agent(ops, [_write("a.md"), DONE])

    assert len(_asked(agent)) == 1
    assert not (elsewhere / "a.md").exists()


def test_a_workspace_that_went_missing_is_made_private_again(home: Path) -> None:
    ops = _ops(home)
    workspace = home / "Documents" / "Jarvis"
    old = os.umask(0)
    try:
        agent, _ = _agent(ops, [_write("sub/a.md"), DONE])
        assert _asked(agent) == []
    finally:
        os.umask(old)

    assert stat.S_IMODE(workspace.stat().st_mode) == 0o700
    assert (workspace / "sub" / "a.md").is_file()


def test_a_workspace_below_a_folder_others_can_write_to_is_refused(tmp_path: Path) -> None:
    shared = tmp_path / "shared"
    shared.mkdir()
    shared.chmod(0o777)
    folder = shared / "Jarvis"

    assert "folder above it" in workspace_problem(folder, create=True)

    shared.chmod(0o1777)  # sticky, like /tmp: others cannot rename what is not theirs
    assert not workspace_problem(folder, create=True)


@pytest.mark.parametrize("name", [".envrc", ".vscode/tasks.json", "src/.mise.toml", ".mine.md"])
def test_a_hidden_name_in_the_workspace_is_written_only_after_a_yes(home: Path, name: str) -> None:
    agent, audit = _agent(_ops(home), [_write(name), DONE])

    assert len(_asked(agent)) == 1

    assert _verdict(audit)[0] == "confirm"
    assert not (home / "Documents" / "Jarvis" / name).exists()
    yes, _ = _agent(_ops(home), [_write(name), DONE])
    yes.handle("save it", Hooks(confirm=lambda _q: True))
    assert (home / "Documents" / "Jarvis" / name).is_file()


def test_a_link_in_the_workspace_to_an_owner_file_is_not_unasked(home: Path) -> None:
    owner = home / "Documents" / "diary.md"
    owner.write_text("dear diary\n", encoding="utf-8")
    workspace = home / "Documents" / "Jarvis"
    workspace.mkdir()
    (workspace / "link.md").symlink_to(owner)
    agent, audit = _agent(
        _ops(home, made=Ledger(home / "ledger.json")), [_write("link.md", "x", "append"), DONE]
    )

    assert len(_asked(agent)) == 1

    assert _verdict(audit)[0] == "confirm"
    assert owner.read_text(encoding="utf-8") == "dear diary\n"


def test_a_hard_link_to_an_owner_file_outside_home_is_not_unasked(
    home: Path, tmp_path: Path
) -> None:
    workspace = tmp_path / "work" / "jarvis"
    owner = tmp_path / "owner.md"
    owner.write_text("owner's\n", encoding="utf-8")
    workspace.mkdir(parents=True)
    workspace.chmod(0o700)
    os.link(owner, workspace / "linked.md")
    ledger = Ledger(home / "ledger.json")
    ops = _ops(home, _files(workspace=str(workspace)), made=ledger)
    agent, _ = _agent(ops, [_write("linked.md", "x", "append"), DONE])

    assert len(_asked(agent)) == 1

    assert owner.read_text(encoding="utf-8") == "owner's\n"


def test_a_file_jarvis_made_that_got_a_second_name_asks(home: Path, tmp_path: Path) -> None:
    ops = _ops(home, made=Ledger(home / "ledger.json"))
    _run(ops, path="a.md", content="one")
    os.link(home / "Documents" / "Jarvis" / "a.md", tmp_path / "other-name.md")
    agent, _ = _agent(ops, [_write("a.md", "two", "append"), DONE])

    assert len(_asked(agent)) == 1


def test_dots_in_a_relative_write_cannot_climb_out_unasked(home: Path) -> None:
    agent, audit = _agent(_ops(home), [_write("../outside.md"), DONE])

    assert len(_asked(agent)) == 1

    assert _verdict(audit)[0] == "confirm"
    assert not (home / "Documents" / "outside.md").exists()


def test_the_files_boundary_keeps_the_config_folder_shut(home: Path) -> None:
    config_dir = paths.config_dir()
    (config_dir / "skills").mkdir(parents=True)
    boundary = files_boundary(Config(), home)

    for raw in ("~/.config/jarvis/config.toml", "~/.config/jarvis/skills/mine.md"):
        with pytest.raises(FileRefusalError):
            boundary.writable(raw)
        with pytest.raises(FileRefusalError):
            boundary.readable(raw)


def test_the_facts_say_when_the_workspace_is_not_in_use(home: Path) -> None:
    assert "always readable and writable" in facts(Config(), "voice", Limits())
    folder = home / "Documents" / "Jarvis"
    folder.mkdir()
    folder.chmod(0o777)

    said = facts(Config(), "voice", Limits())

    assert "always readable and writable" not in said
    assert "not in use" in said


def test_a_shell_question_says_where_the_command_runs_and_a_missing_workspace_means_home(
    home: Path,
) -> None:
    folder = home / "Documents" / "Jarvis"
    ops = ShellOps(ShellConfig(enabled=True), home, max_chars=1000, workspace=folder)
    op = Registry(ops.operations()).get("pc_shell") or pytest.fail("no pc_shell")

    assert "in your home folder: " in op.spoken_summary({"command": "ls"})
    assert f"--working-directory={home}" in ops.command_line("ls", "u")
    folder.mkdir()
    assert f"in {folder}: " in op.spoken_summary({"command": "ls"})
    assert f"--working-directory={folder}" in ops.command_line("ls", "u")


def test_a_file_changed_inside_a_coarse_change_time_tick_is_not_jarvis_s(home: Path) -> None:
    # A filesystem that keeps the change time to a second or two shows the same one before and
    # after a quick rewrite; the modification time and the size still tell.
    made = home / "a.md"
    made.write_text("one\n", encoding="utf-8")
    real = made.stat()
    ledger = Ledger(home / "ledger.json")
    ledger.record(made, real)

    def seen(**changed: int) -> os.stat_result:
        fields = {
            "st_dev": real.st_dev,
            "st_ino": real.st_ino,
            "st_ctime_ns": real.st_ctime_ns,
            "st_mtime_ns": real.st_mtime_ns,
            "st_size": real.st_size,
        } | changed
        return cast("os.stat_result", SimpleNamespace(**fields))

    assert ledger.made(made, seen())
    assert not ledger.made(made, seen(st_size=real.st_size + 1))
    assert not ledger.made(made, seen(st_mtime_ns=real.st_mtime_ns + 1))
    assert not ledger.made(made, seen(st_ino=real.st_ino + 1))
    assert not ledger.made(made, seen(st_ctime_ns=real.st_ctime_ns + 1))
