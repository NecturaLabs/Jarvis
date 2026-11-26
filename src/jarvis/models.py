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


@dataclass(frozen=True, slots=True)
class RunnerFile:
    """One file of SoulX-Singer's runner bundle: its name in the bundle, its size and sha256."""

    name: str
    size: int
    sha256: str


# SoulX-Singer's model, `soulx-singer-q8_0.gguf` (the `soulx-model` release above), in bytes.
SOULX_MODEL_BYTES = 1_173_681_632
# Every file `tools/get_singer.py` takes from the audio.cpp v0.9.6 release
# audio-v0.9.6-bin-ubuntu-x64-vulkan-portable.tar.gz (the `soulx-binary` release above): the runner,
# its libraries and its licence, each read from the archive itself. Jarvis starts the runner only
# while all of them are in place as these files (`jarvis.sing.soulx.verify`).
SOULX_RUNNER: tuple[RunnerFile, ...] = (
    RunnerFile(
        "LICENSE", 10_276, "8f0b762d062e1ec79a3e73189b29474051c0e957973f43146aefeb03f2e11581"
    ),
    RunnerFile(
        "audiocpp_cli",
        61_448_664,
        "36c4a51f9b4019c7d22d5687ad4106d83e51c59a98d12bb3cb3fb8082ec3e539",
    ),
    RunnerFile(
        "libggml-base.so",
        975_744,
        "24c3392b3d61ece3829b0889e98220f59ec83a6550661b9b524386880c12e7e8",
    ),
    RunnerFile(
        "libggml-base.so.0",
        975_744,
        "24c3392b3d61ece3829b0889e98220f59ec83a6550661b9b524386880c12e7e8",
    ),
    RunnerFile(
        "libggml-base.so.0.12.0",
        975_744,
        "24c3392b3d61ece3829b0889e98220f59ec83a6550661b9b524386880c12e7e8",
    ),
    RunnerFile(
        "libggml-cpu-alderlake.so",
        1_219_248,
        "46e3646b4c6985cb810751cbdc489745b2112ffa88ebca0a411cfad9bff5e754",
    ),
    RunnerFile(
        "libggml-cpu-cannonlake.so",
        1_462_600,
        "3ed77fb1d797dd0c0d4c737f89af1d64401cd584c9003e88da416d3c98e41818",
    ),
    RunnerFile(
        "libggml-cpu-cascadelake.so",
        1_462_600,
        "f5b3d1b6202d4b5fc2ad4619fe5d2376c486240cb84cc7fd33d34cd58d266e65",
    ),
    RunnerFile(
        "libggml-cpu-cooperlake.so",
        1_462_760,
        "e23fd91802c33d574cdeb1fa121db46a0cb51f5c9cf227747aa342f75458f41a",
    ),
    RunnerFile(
        "libggml-cpu-haswell.so",
        1_219_248,
        "000c3e7664a0b4b80d15d93163c38de45e369bb14b27e8643483a1d620e00e5b",
    ),
    RunnerFile(
        "libggml-cpu-icelake.so",
        1_458_504,
        "6404a7e272868eb26cc9d4ee6e774341208cb8a61f095e8848f50b8b77e29537",
    ),
    RunnerFile(
        "libggml-cpu-ivybridge.so",
        1_219_248,
        "b66b113044cce04edb4d8672de31bbedb1b49352e44226fa583d6ab0c145812e",
    ),
    RunnerFile(
        "libggml-cpu-piledriver.so",
        1_219_248,
        "925305498dba809b401d34cb2547417f8a060dee122779c4910bf2bf84f3c425",
    ),
    RunnerFile(
        "libggml-cpu-sandybridge.so",
        1_219_248,
        "28751df2dd0f8ee6f2aad5a51ce235d3fd909493b467cec714e7bf94c872ab52",
    ),
    RunnerFile(
        "libggml-cpu-sapphirerapids.so",
        1_728_880,
        "c4ab584a306c1bcca3e2ef3aaa2bb461f26deca6767005ee7d6d9ad074059033",
    ),
    RunnerFile(
        "libggml-cpu-skylakex.so",
        1_466_696,
        "f76e89b29dcbeab3f7178281a70d952d96a6faea384e1ff74df1b7af84f76ed0",
    ),
    RunnerFile(
        "libggml-cpu-sse42.so",
        1_219_248,
        "32948736d6f514fae6547ccd6ad28d8343301151f88d7eedec910347f0625c9f",
    ),
    RunnerFile(
        "libggml-cpu-x64.so",
        1_219_248,
        "7d393d49551cd0f84a9cbc438d5c7bd7e25d9c8f7a4031273f176ccfaf0bddd3",
    ),
    RunnerFile(
        "libggml-cpu-zen4.so",
        1_462_760,
        "354304031925cca21592bcba3dedf29de742ee2cfce60a2628224d74ed8ec93e",
    ),
    RunnerFile(
        "libggml-vulkan.so",
        58_745_304,
        "aa659976c21136668ab65363b61bcc85cd7b97848c8303f47df2cdd50f64a3c7",
    ),
    RunnerFile(
        "libggml.so", 63_288, "c49d529d031df33c2b015c7e69c28da38cfefff35e762f0822126579b6446e87"
    ),
    RunnerFile(
        "libggml.so.0", 63_288, "c49d529d031df33c2b015c7e69c28da38cfefff35e762f0822126579b6446e87"
    ),
    RunnerFile(
        "libggml.so.0.12.0",
        63_288,
        "c49d529d031df33c2b015c7e69c28da38cfefff35e762f0822126579b6446e87",
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
