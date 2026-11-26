# SPDX-FileCopyrightText: NecturaLabs and the Jarvis contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""The affected-test selector reads imports and package data references without importing tests."""

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "tools" / "affected_tests.py"


def _select(root: Path, files: dict[str, str], *changes: str) -> tuple[list[str], str]:
    for name, source in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(SCRIPT), *changes],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.splitlines(), result.stderr


def test_selects_direct_importers_and_changed_tests(tmp_path: Path) -> None:
    module = "src/jarvis/pc/watch.py"
    changed_test = "tests/test_changed.py"
    files = {
        module: "",
        changed_test: "",
        "tests/test_import.py": "import os, jarvis.pc.watch as watcher\n",
        "tests/test_from_module.py": "from jarvis.pc.watch import ScreenWatch\n",
        "tests/test_from_package.py": "from jarvis.pc import (\n    watch as watcher,\n)\n",
        "tests/test_nested.py": "def test_local_import():\n    from jarvis.pc import watch\n",
        "tests/test_unrelated.py": "import jarvis.pc.watchdog\n# import jarvis.pc.watch\n",
    }

    selected, stderr = _select(tmp_path, files, module, changed_test, module)

    assert selected == [
        changed_test,
        "tests/test_from_module.py",
        "tests/test_from_package.py",
        "tests/test_import.py",
        "tests/test_nested.py",
    ]
    assert not stderr


@pytest.mark.parametrize("helper", ["test_windows", "windows_fixture"])
def test_selects_tests_that_import_changed_helpers(tmp_path: Path, helper: str) -> None:
    helper_path = f"tests/{helper}.py"
    files = {
        helper_path: "",
        "tests/__init__.py": "",
        "tests/screenshot_fixture.py": f"from {helper} import FakeDesktop\n",
        "tests/test_direct.py": f"from . import {helper}\n",
        "tests/test_qualified.py": f"from tests import {helper}\n",
        "tests/test_relative.py": f"from .{helper} import FakeDesktop\n",
        "tests/test_screenshot.py": "from screenshot_fixture import Capture\n",
        "tests/test_unrelated.py": f"import {helper}_extra\n",
    }

    selected, _ = _select(tmp_path, files, helper_path)

    expected = [
        "tests/test_direct.py",
        "tests/test_qualified.py",
        "tests/test_relative.py",
        "tests/test_screenshot.py",
    ]
    if helper.startswith("test_"):
        expected.append(helper_path)
    assert selected == sorted(expected)


def test_selects_tests_that_import_nested_helpers(tmp_path: Path) -> None:
    helper = "tests/helpers/desktop.py"
    files = {
        helper: "",
        "tests/helpers/__init__.py": "",
        "tests/test_desktop.py": "from helpers.desktop import FakeDesktop\n",
    }

    selected, _ = _select(tmp_path, files, helper)

    assert selected == ["tests/test_desktop.py"]


def test_selects_tests_that_mention_packaged_data(tmp_path: Path) -> None:
    prompt = "src/jarvis/prompts/core.md"
    files = {
        prompt: "",
        "tests/test_filename.py": 'name = "prompts/core.md"\n',
        "tests/test_package.py": "from jarvis import prompts\n",
        "tests/test_resource.py": 'package = "jarvis.prompts"\n',
        "tests/test_unrelated.py": 'package = "jarvis.prompts_extra"\n',
        "tests/test_unrelated_data.py": 'names = ["extra-core.md", "core.md.bak"]\n',
    }

    selected, _ = _select(tmp_path, files, prompt)

    assert selected == ["tests/test_filename.py", "tests/test_package.py", "tests/test_resource.py"]


def test_ignores_docs_tooling_and_missing_files(tmp_path: Path) -> None:
    changes = [
        "docs/plan.md",
        "CHANGELOG.md",
        "README.md",
        ".github/workflows/ci.yml",
        "tools/run.py",
    ]
    files = dict.fromkeys(changes, "")
    files["tests/test_tool.py"] = "import run\n"
    files["tests/test_gone.py"] = "import jarvis.gone\n"

    selected, stderr = _select(
        tmp_path, files, *changes, "src/jarvis/gone.py", "tests/test_lost.py"
    )

    assert selected == []
    assert not stderr


def test_ci_only_paths_are_reported_without_selecting_tests(tmp_path: Path) -> None:
    changes = ["uv.lock", "tests/conftest.py", "pyproject.toml"]
    files = dict.fromkeys(changes, "")
    files["tests/test_only.py"] = "import conftest\n"

    selected, stderr = _select(tmp_path, files, *changes)

    assert selected == []
    assert "CI" in stderr
    assert all(path in stderr for path in changes)


def test_a_packaged_file_selects_the_tests_of_the_modules_that_load_its_folder(
    tmp_path: Path,
) -> None:
    """Regression: a changed prompt selected no test, the pinned prompt digests included."""
    files = {
        "src/jarvis/prompts/core.md": "Be brief.\n",
        "src/jarvis/app.py": 'prompts = files("jarvis").joinpath("prompts")\n',
        "src/jarvis/skills.py": 'folder = files("jarvis").joinpath("prompts/skills")\n',
        "src/jarvis/unrelated.py": 'name = "promptsy"\n',
        "tests/test_router.py": "from jarvis.app import build\n",
        "tests/test_skills.py": "import jarvis.skills\n",
        "tests/test_unrelated.py": "import jarvis.unrelated\n",
    }

    selected, _ = _select(tmp_path, files, "src/jarvis/prompts/core.md")

    assert selected == ["tests/test_router.py", "tests/test_skills.py"]
