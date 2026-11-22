#!/usr/bin/env python3
"""Run `livekit-wakeword` without its two gated Hugging Face test sets (ESC-50 and GTZAN).

Both need approved access, and the trainer's `setup` loads every registered source, even
without `--eval`, so it stops with `GatedRepoError` until access is granted (#37). They are
test-only: training never draws on them, so a model trained this way is the same; only the test
set's false alarms on music and everyday sounds are not counted. Run it with the trainer's
Python, in place of the `livekit-wakeword` command; the installed package is not changed.

    <trainer venv>/bin/python tools/wake_ungated.py setup --wakeword "jarvis" --plan build --eval
"""

import importlib
import sys

GATED = ("esc50", "gtzan")


def main() -> int:
    """Drop the gated sources, then hand the arguments to the trainer's own command line."""
    sources = importlib.import_module("livekit_wakeword_bench.sources").SOURCES
    for name in GATED:
        sources.pop(name, None)
    return int(importlib.import_module("livekit.wakeword._cli").main() or 0)


if __name__ == "__main__":
    sys.exit(main())
