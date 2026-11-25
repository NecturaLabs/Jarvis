---
name: add-dependency-or-voice
description: Checklist for adding or bumping a Python dependency, a model file or a voice in Jarvis - licence against GPL-3.0-or-later, the lock, the advisories check, where the file lives, the pinned sha256 and the docs row. Use before `uv add`, before pinning a new model or voice in models.py, and when Dependabot proposes a major version.
---

# Add a dependency, model or voice

Jarvis is GPL-3.0-or-later ([ADR 0006](../../../docs/adr/0006-gpl-licence.md)) and runs locally and free
([ADR 0005](../../../docs/adr/0005-local-only.md)). A dependency that calls a hosted model or a paid API is refused
outright; `ruff.toml` bans the usual SDKs.

## Python package

1. Check its licence (package metadata and its LICENSE file) is compatible with GPL-3.0-or-later. Not compatible or
   unclear: stop and ask the maintainer.
2. `uv add <name>==<version>` (or `--dev`); commit `pyproject.toml` and `uv.lock` together. Every source must be PyPI.
3. Run `tools/check.sh`: its "dependency licences and sources" and "known vulnerabilities" steps must pass (the
   second audits the locked requirements, as CI does, and needs the network).
4. Add a row to the dependency table in `docs/models.md` with the version and licence, and say what it pulls in.

## Model file or voice

1. Check the weights' licence on the model card; record where you checked.
2. Pin it in `src/jarvis/models.py` with its sha256 and licence, so `jarvis doctor` verifies it.
3. Never commit the file (`.gitignore` refuses `*.onnx`, `*.gguf`, `ggml-*.bin` and `*.wav`); downloads go outside
   the repository.
4. Add or update its section in `docs/models.md`: where it goes under `data_dir`, its cost and its licence.

Done when: `tools/check.sh` passes, `uv run reuse lint` is clean and the docs row names the licence.
