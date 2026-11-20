"""Command-line entry point for `jarvis`."""

import argparse
from typing import TYPE_CHECKING

from jarvis import __version__

if TYPE_CHECKING:
    from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser for the `jarvis` command."""
    parser = argparse.ArgumentParser(prog="jarvis", description="Local voice agent for Omarchy.")
    parser.add_argument("--version", action="version", version=f"jarvis {__version__}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the `jarvis` command and return its exit code."""
    build_parser().parse_args(argv)
    return 0
