"""Strata run configs shaped like the ones `strata launch` writes, for tests that need a model."""

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def touch(path: Path) -> Path:
    """Create an empty file and its folders; return its path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def engine_run_config(tmp_path: Path, model: Path | None = None) -> Path:
    """Write a run config shaped like the one `strata launch` writes for Strata's main engine."""
    model = touch(model or tmp_path / "models" / "q4.gguf")
    pack = tmp_path / "packs" / "q4"
    pack.mkdir(parents=True, exist_ok=True)
    args = [
        "--pack", str(pack), "--native", str(model), "--expert-cache", "auto",
        "--prefill", "auto", "--max-context", "32768", "--kv", "int8", "--vision",
    ]  # fmt: skip
    data = {
        "exe": "component:strata-engine",
        "args": args,
        "args_written": args,
        "args_derived": args,
        "cwd": "/strata/control/0.1.66",
        "tokenizer": str(pack / "tokenizer"),
        "model_name": "flash-q4",
        "log": "/strata/install/strata-q4.log",
        "lib_dirs": [],
        "port": 8099,
        "backend": "hip",
        "gpu": 0,
        "gpus_asked": True,
        "engine_ready_s": 884,
        "vision": {
            "exe": "/strata/engine/0.1.62/strata-vision",
            "mmproj": str(touch(tmp_path / "models" / "mmproj.gguf")),
            "model": str(model),
            "gpu": False,
            "max_tokens": 300,
            "threads": 8,
        },
    }
    path = tmp_path / "install" / "strata-q4.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def inkling_run_config(tmp_path: Path) -> Path:
    """Write a run config shaped like the one `strata launch` writes for Inkling's own engine."""
    model = touch(tmp_path / "models" / "inkling.strata")
    reader = touch(tmp_path / "models" / "inkling-vision.strata")
    data = {
        "exe": "component:strata-inkling",
        "args": [
            "--native",
            str(model),
            "--max-context",
            "32768",
            "--vision",
            str(reader),
            "--vram-experts-gb",
            "12.28",
            "--ram-experts-gb",
            "0.00",
        ],
        "tokenizer": str(tmp_path / "packs" / "inkling" / "tokenizer"),
        "model_name": "inkling",
        "arch": "inkling2_loop",
        "drafting": "dspark",
        "parallel": 2,
        "backend": "hip",
    }
    path = tmp_path / "install" / "strata-inkling.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def install_engine(components_parent: Path) -> Path:
    """Create an installed Strata engine folder with its picture reader; return the reader."""
    return touch(components_parent / "strata" / "components" / "engine" / "0.1.0" / "strata-vision")
