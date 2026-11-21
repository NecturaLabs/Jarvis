"""The table of known model files and the hashing that checks a file against it."""

import hashlib
import re
from typing import TYPE_CHECKING

from jarvis import models

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _entry(role: str, sha256: str) -> models.KnownModel:
    return models.KnownModel(
        role=role, project="Example", version="1.2.3", sha256=sha256, licence="MIT"
    )


def test_every_known_release_is_well_formed_and_unique() -> None:
    table = models.KNOWN_MODELS

    assert table
    assert all(SHA256.match(m.sha256) for m in table)
    assert all(m.project and m.version and m.licence for m in table)
    assert len({m.sha256 for m in table}) == len(table)
    assert len({(m.role, m.version) for m in table}) == len(table)


def test_the_wake_word_release_is_pinned() -> None:
    wake = [m for m in models.KNOWN_MODELS if m.role == "wake"]

    assert [(m.project, m.version, m.licence) for m in wake] == [
        ("LiveKit hey_jarvis", "2.1.3", "Apache-2.0")
    ]
    assert wake[0].sha256 == "8b554c50809888ac14f67fa21acc3bd85e58f5678ee4f2be5d0041f9a2e62843"


def test_the_digest_matches_hashlib(tmp_path: Path) -> None:
    path = tmp_path / "f.bin"
    path.write_bytes(b"some model bytes")

    assert models.sha256_file(path) == hashlib.sha256(b"some model bytes").hexdigest()


def test_a_file_is_hashed_in_chunks(tmp_path: Path) -> None:
    data = bytes(range(256)) * 41
    path = tmp_path / "f.bin"
    path.write_bytes(data)

    assert models.sha256_file(path, chunk_size=7) == hashlib.sha256(data).hexdigest()


def test_a_known_digest_is_identified_by_role(tmp_path: Path) -> None:
    path = tmp_path / "wake.onnx"
    path.write_bytes(b"wake")
    digest = hashlib.sha256(b"wake").hexdigest()
    table = (_entry("wake", digest), _entry("vad", "0" * 64))

    result = models.verify("wake", path, table)

    assert result.digest == digest
    assert result.known == table[0]


def test_an_unknown_digest_has_no_release(tmp_path: Path) -> None:
    path = tmp_path / "wake.onnx"
    path.write_bytes(b"something else")

    result = models.verify("wake", path, (_entry("wake", "1" * 64),))

    assert result.known is None
    assert result.digest == hashlib.sha256(b"something else").hexdigest()


def test_a_digest_known_for_another_role_is_not_a_match(tmp_path: Path) -> None:
    path = tmp_path / "x.onnx"
    path.write_bytes(b"x")
    digest = hashlib.sha256(b"x").hexdigest()

    assert models.verify("vad", path, (_entry("wake", digest),)).known is None


def test_the_default_table_is_read_when_called(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "x.onnx"
    path.write_bytes(b"x")
    patched = (_entry("wake", hashlib.sha256(b"x").hexdigest()),)
    monkeypatch.setattr(models, "KNOWN_MODELS", patched)

    assert models.verify("wake", path).known == patched[0]
