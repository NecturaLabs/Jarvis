#!/usr/bin/env python3
"""Run `livekit-wakeword` without its two gated Hugging Face test sets (ESC-50 and GTZAN).

Both need approved access, and the trainer's `setup` loads every registered source, even
without `--eval`, so it stops with `GatedRepoError` until access is granted (#37). They are
test-only: training draws on the train pool alone, so the model is trained without them, as it
would be with them; only the test set's false alarms on music and everyday sounds are not
counted, so a score from an ungated run is not comparable with a full one.

This relies on private modules of `livekit-wakeword-bench` 2.1.3 (`livekit_wakeword_bench.sources`
and its `SOURCES` table) and on the trainer's `livekit.wakeword._cli`. It exits 2 with a message
when the bench is another version or either name is gone from the table, instead of quietly
keeping a gated source. Run it with the trainer's Python, in place of the `livekit-wakeword`
command; the installed package is not changed.

    <trainer venv>/bin/python tools/wake_ungated.py setup --wakeword "jarvis" --plan build --eval
"""

import importlib
import importlib.metadata
import sys

GATED = ("esc50", "gtzan")
BENCH = "livekit-wakeword-bench"
CHECKED_VERSION = "2.1.3"


def refusal() -> str | None:
    """Return why the gated sources cannot be dropped safely, or None when they can."""
    try:
        version = importlib.metadata.version(BENCH)
    except importlib.metadata.PackageNotFoundError:
        return f"{BENCH} is not installed; this wrapper was written for {CHECKED_VERSION}"
    if version != CHECKED_VERSION:
        return (
            f"{BENCH} is {version}; this wrapper was checked only against {CHECKED_VERSION}. "
            "Check its private SOURCES table and update the wrapper, or use the plain command."
        )
    sources = importlib.import_module("livekit_wakeword_bench.sources").SOURCES
    missing = [name for name in GATED if name not in sources]
    if missing:
        return (
            f"{BENCH} {version} has no source named {', '.join(missing)} to drop "
            f"(it has {', '.join(sorted(sources))}): "
            "a gated set may now be loaded under another name"
        )
    return None


def main() -> int:
    """Drop the gated sources, then hand the arguments to the trainer's own command line."""
    why = refusal()
    if why is not None:
        sys.stderr.write(f"wake_ungated.py: {why}\n")
        return 2
    sources = importlib.import_module("livekit_wakeword_bench.sources").SOURCES
    for name in GATED:
        del sources[name]
    return int(importlib.import_module("livekit.wakeword._cli").main() or 0)


if __name__ == "__main__":
    sys.exit(main())
