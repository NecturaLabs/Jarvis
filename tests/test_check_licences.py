"""tools/check_licences.py: dependency licences against the GPL-3.0-or-later allow-list."""

from pathlib import Path

import check_licences as cl
import pytest

ROOT = Path(__file__).resolve().parent.parent


def dist(
    name: str = "pkg",
    *,
    expression: str | None = None,
    licence: str | None = None,
    classifiers: tuple[str, ...] = (),
) -> cl.Dist:
    return cl.Dist(
        name=name,
        version="1.0",
        license_expression=expression,
        license=licence,
        classifiers=classifiers,
        metadata_path=f"site/{name}-1.0.dist-info/METADATA",
    )


def accepts(expression: str) -> bool:
    node = cl.parse_expression(expression)
    assert node is not None, expression
    return cl.passes(node, cl.is_allowed)


@pytest.mark.parametrize(
    "expression",
    [
        "MIT",
        "mit",
        "Apache-2.0 OR BSD-2-Clause",
        "BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0",
        "GPL-2.0-only OR MIT",
        "(GPL-2.0-only OR MIT) AND Apache-2.0",
        "Apache-2.0 WITH LLVM-exception",
        "GPL-3.0-or-later",
        "GPL-2.0+",
        "LGPL-2.1-only",
        "LGPL-3.0-or-later",
    ],
)
def test_allowed_expressions_pass(expression: str) -> None:
    assert accepts(expression)


@pytest.mark.parametrize(
    "expression",
    [
        "GPL-2.0-only",
        "MIT AND GPL-2.0-only",
        "GPL-2.0-only OR AGPL-3.0-only",
        "(MIT OR Apache-2.0) AND SSPL-1.0",
        "LicenseRef-Proprietary",
        "CC-BY-SA-4.0",
        "AGPL-3.0-or-later",
        "(MIT OR GPL-2.0-only) AND SSPL-1.0",
    ],
)
def test_other_expressions_are_refused(expression: str) -> None:
    assert not accepts(expression)


def test_and_binds_tighter_than_or() -> None:
    assert accepts("MIT OR GPL-2.0-only AND SSPL-1.0")
    assert not accepts("GPL-2.0-only OR SSPL-1.0 AND MIT")


@pytest.mark.parametrize(
    "text", ["", "MIT License", "Apache 2.0", "MIT AND", "(MIT", "MIT)", "AND MIT", "MIT, BSD"]
)
def test_text_that_is_not_an_expression_does_not_parse(text: str) -> None:
    assert cl.parse_expression(text) is None


def test_license_expression_wins_over_the_other_fields() -> None:
    resolved = cl.resolve(
        dist(
            expression="MIT",
            licence="GPL-2.0-only",
            classifiers=("License :: OSI Approved :: MIT License",),
        )
    )
    assert resolved.how == "License-Expression"
    assert resolved.node == cl.Term("MIT")


def test_an_invalid_license_expression_is_a_problem() -> None:
    resolved = cl.resolve(dist(expression="MIT License"))
    assert resolved.node is None
    assert resolved.problem is not None
    assert "not valid SPDX" in resolved.problem


def test_a_short_spdx_license_field_is_used() -> None:
    assert cl.resolve(dist(licence="Apache-2.0")).how == "License"
    assert cl.resolve(dist(licence="MIT")).how == "License"


@pytest.mark.parametrize(
    ("licence", "expected"),
    [
        ("BSD", "BSD-3-Clause"),
        ("MIT License", "MIT"),
        ("Apache 2.0", "Apache-2.0"),
        ("Apache License\n Version 2.0, January 2004\n ...", "Apache-2.0"),
    ],
)
def test_free_text_license_falls_back_to_the_classifiers(licence: str, expected: str) -> None:
    classifiers = {
        "BSD-3-Clause": "License :: OSI Approved :: BSD License",
        "MIT": "License :: OSI Approved :: MIT License",
        "Apache-2.0": "License :: OSI Approved :: Apache Software License",
    }
    resolved = cl.resolve(
        dist(licence=licence, classifiers=("License :: OSI Approved", classifiers[expected]))
    )
    assert resolved.how == "classifiers"
    assert resolved.node == cl.Term(expected)


def test_several_classifiers_must_all_be_allowed() -> None:
    both = (
        "License :: OSI Approved :: MIT License",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    )
    assert cl.resolve(dist(classifiers=both)).node == cl.Op(
        "AND", (cl.Term("MIT"), cl.Term("GPL-3.0-only"))
    )


def test_an_unmapped_classifier_is_a_problem_not_a_guess() -> None:
    resolved = cl.resolve(dist(classifiers=("License :: Other/Proprietary License",)))
    assert resolved.node is None
    assert resolved.problem is not None
    assert "Other/Proprietary" in resolved.problem


def test_no_licence_metadata_is_a_problem() -> None:
    resolved = cl.resolve(
        dist(classifiers=("License :: OSI Approved", "Programming Language :: Python"))
    )
    assert resolved.node is None
    assert resolved.problem is not None


def test_an_allowed_distribution_has_no_finding() -> None:
    assert cl.check_distributions([dist(expression="MIT")], "jarvis") == []


def test_a_disallowed_licence_is_reported_with_the_metadata_path() -> None:
    [finding] = cl.check_distributions([dist("copyleft", expression="GPL-2.0-only")], "jarvis")
    assert finding.startswith("site/copyleft-1.0.dist-info/METADATA:1: copyleft 1.0: ")
    assert "GPL-2.0-only" in finding
    assert "not allowed" in finding


def test_an_or_licence_with_one_allowed_branch_passes_and_an_and_one_does_not() -> None:
    either = dist("either", expression="GPL-2.0-only OR MIT")
    both = dist("both", expression="GPL-2.0-only AND MIT")
    findings = cl.check_distributions([either, both], "jarvis")
    assert len(findings) == 1
    assert "both" in findings[0]


def test_a_distribution_with_no_usable_metadata_is_reported() -> None:
    [finding] = cl.check_distributions([dist("mystery")], "jarvis")
    assert "mystery 1.0" in finding
    assert "OVERRIDES" in finding


def test_the_project_itself_is_skipped() -> None:
    assert cl.check_distributions([dist("Jarvis", expression="GPL-2.0-only")], "jarvis") == []


def test_names_are_compared_normalised() -> None:
    [finding] = cl.check_distributions([dist("Foo_Bar.baz", expression="GPL-2.0-only")], "jarvis")
    assert "Foo_Bar.baz" in finding
    assert (
        cl.check_distributions([dist("My_Project", expression="GPL-2.0-only")], "my-project") == []
    )


def test_an_override_supplies_the_licence_when_the_metadata_has_none() -> None:
    assert "protobuf" in cl.OVERRIDES
    assert (
        cl.check_distributions([dist("protobuf", licence="3-Clause BSD License")], "jarvis") == []
    )


def test_an_override_the_metadata_has_outgrown_is_reported() -> None:
    [finding] = cl.check_distributions([dist("protobuf", expression="BSD-3-Clause")], "jarvis")
    assert "no longer needed" in finding


def test_an_override_does_not_excuse_a_disallowed_licence_in_the_metadata() -> None:
    findings = cl.check_distributions([dist("protobuf", expression="GPL-2.0-only")], "jarvis")
    assert any("not allowed" in f for f in findings)


def test_an_exception_widens_only_its_own_distribution() -> None:
    assert (
        cl.check_distributions([dist("reuse", expression="Apache-2.0 AND CC-BY-SA-4.0")], "jarvis")
        == []
    )
    [finding] = cl.check_distributions(
        [dist("other", expression="Apache-2.0 AND CC-BY-SA-4.0")], "jarvis"
    )
    assert "CC-BY-SA-4.0" in finding


LOCK = """\
version = 1

[[package]]
name = "jarvis"
version = "0.1.0"
source = { editable = "." }

[[package]]
name = "good"
version = "1.0"
source = { registry = "https://pypi.org/simple" }
"""


def test_a_lock_of_pypi_and_the_project_is_clean() -> None:
    assert cl.check_lock(LOCK, "jarvis") == []


def test_a_virtual_project_source_is_accepted() -> None:
    assert cl.check_lock(LOCK.replace('editable = "."', 'virtual = "."'), "jarvis") == []


@pytest.mark.parametrize(
    "source",
    [
        '{ git = "https://example.invalid/x.git?rev=abc#abc" }',
        '{ url = "https://example.invalid/x.tar.gz" }',
        '{ path = "../x" }',
        '{ registry = "https://example.invalid/simple" }',
        '{ registry = "https://pypi.org/simple/" }',
        '{ editable = "../x" }',
    ],
)
def test_a_package_from_anywhere_else_is_reported_at_its_line(source: str) -> None:
    lock = LOCK + f'\n[[package]]\nname = "bad"\nversion = "1"\nsource = {source}\n'
    [finding] = cl.check_lock(lock, "jarvis")
    assert finding.startswith("uv.lock:14: package bad has source ")


def test_a_package_without_a_source_is_reported() -> None:
    [finding] = cl.check_lock(LOCK + '\n[[package]]\nname = "bare"\nversion = "1"\n', "jarvis")
    assert "package bare has no source" in finding


def test_only_the_project_may_use_the_project_source() -> None:
    lock = LOCK + '\n[[package]]\nname = "sneaky"\nversion = "1"\nsource = { editable = "." }\n'
    [finding] = cl.check_lock(lock, "jarvis")
    assert "sneaky" in finding


def test_the_real_environment_and_lock_have_no_findings() -> None:
    project = cl.project_name(ROOT)
    assert cl.check_distributions(cl.installed(), project) == []
    assert cl.check_lock((ROOT / "uv.lock").read_text(encoding="utf-8"), project) == []


def test_main_exits_zero_on_the_real_tree(capsys: pytest.CaptureFixture[str]) -> None:
    assert cl.main([]) == 0
    assert not capsys.readouterr().out


def test_main_prints_findings_and_exits_one(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cl, "installed", lambda: [dist("copyleft", expression="GPL-2.0-only")])
    assert cl.main([]) == 1
    assert "copyleft 1.0" in capsys.readouterr().out


def test_main_rejects_arguments(capsys: pytest.CaptureFixture[str]) -> None:
    assert cl.main(["--bogus"]) == 2
    assert "usage" in capsys.readouterr().err
