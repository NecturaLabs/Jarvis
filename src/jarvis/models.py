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

    role: str  # which file it is: wake, vad, kokoro-model, kokoro-voices, vision, soulx-*
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
    KnownModel(
        role="vision",
        project="Qwen3.6-35B-A3B picture reader (unsloth/Qwen3.6-35B-A3B-MTP-GGUF mmproj-F16.gguf)",
        version="5bc3e238d916f48a861bac2f8a1990a0e9b7e98d",
        sha256="71f3cbc1f7cc0f30d09d41cfa924c0060827ebc33bf15ace7e86661e856f0160",
        licence="Apache-2.0",
    ),
    KnownModel(
        role="vision",
        project=(
            "Qwen3.8-Flash-Next picture reader "
            "(ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF mmproj-Qwen3.8-Flash-Next-BF16.gguf)"
        ),
        version="ed59f92082b1e93c0e96d60a8b11aab089b52f09",
        sha256="b1a82259702816a5330d7bd7607cd9676b11780e79ff7348c21103ff3ce49bd0",
        # The quantiser's card says apache-2.0; the weights are Qwen's, under its own licence.
        licence="LicenseRef-Qwen-Community-License-1.0",
    ),
    KnownModel(
        role="vision",
        project="Inkling 2 Mini picture reader (strata-engine/Inkling-2-Mini-Strata)",
        version="04996a1e2c45009b6a04f2af126cb6bdb2ba3587",
        sha256="6589eff47f9c20bc3a032faa3d449f10bf4f04a84cc6b2ee8a44d51d2b6fc677",
        licence="Apache-2.0",
    ),
    KnownModel(
        role="soulx-model",
        project="SoulX-Singer (audio-cpp/SoulX-Singer-GGUF soulx-singer-q8_0.gguf)",
        version="ded36295f789661e590e4bf6ad49a25470cf830d",
        sha256="c146c09427f88202db37a71869cdd498e6c0e92fa2cd08065bd5c2f735eb82fd",
        licence="Apache-2.0",
    ),
    KnownModel(
        role="soulx-binary",
        project="audio.cpp audiocpp_cli (audio-v0.9.6-bin-ubuntu-x64-vulkan-portable.tar.gz)",
        version="v0.9.6",
        sha256="36c4a51f9b4019c7d22d5687ad4106d83e51c59a98d12bb3cb3fb8082ec3e539",
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
