"""The model files Jarvis loads and the upstream releases their sha256 digests are pinned to.

`jarvis doctor` hashes each file it locates and looks the digest up here. A digest is listed only
when it was confirmed against the upstream release itself; a file that matches nothing is not an
error (a newer release, a different build), only a note.
"""

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

CHUNK_SIZE = 1 << 20


@dataclass(frozen=True, slots=True)
class KnownModel:
    """One file of one upstream release."""

    role: str  # which file it is: wake, vad, kokoro-model, kokoro-voices
    project: str
    version: str  # the release's version or tag
    sha256: str
    licence: str  # SPDX identifier


@dataclass(frozen=True, slots=True)
class Verdict:
    """The digest of a local file and the release it matches, if any."""

    digest: str
    known: KnownModel | None


KNOWN_MODELS: tuple[KnownModel, ...] = (
    KnownModel(
        role="wake",
        project="LiveKit hey_jarvis",
        version="2.1.3",
        sha256="8b554c50809888ac14f67fa21acc3bd85e58f5678ee4f2be5d0041f9a2e62843",
        licence="Apache-2.0",
    ),
    KnownModel(
        role="vad",
        project="Silero VAD",
        version="6.2.3",
        sha256="1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3",
        licence="MIT",
    ),
    KnownModel(
        role="kokoro-model",
        project="Kokoro ONNX (kokoro-onnx release kokoro-v1.0.onnx)",
        version="model-files-v1.1",
        sha256="beb0d1848dee9a49da392cc3df26958d46cfa35d321edf434f52949153f0df3a",
        licence="Apache-2.0",
    ),
    KnownModel(
        role="kokoro-voices",
        project="Kokoro ONNX (kokoro-onnx release voices-v1.0.bin)",
        version="model-files-v1.1",
        sha256="bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d",
        licence="Apache-2.0",
    ),
)


def sha256_file(path: Path, chunk_size: int = CHUNK_SIZE) -> str:
    """Return the file's sha256 as lowercase hex, reading it in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify(role: str, path: Path, table: Sequence[KnownModel] | None = None) -> Verdict:
    """Hash `path` and return the release of that role it matches (None: not a known release)."""
    digest = sha256_file(path)
    known = next(
        (
            m
            for m in (KNOWN_MODELS if table is None else table)
            if m.role == role and m.sha256 == digest
        ),
        None,
    )
    return Verdict(digest, known)
