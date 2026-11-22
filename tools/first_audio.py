#!/usr/bin/env python3
"""Measure how long the first sentence of a reply waits for the model, streamed or not (#12).

It sends typed requests to the model server Jarvis already uses, through the real agent and its
sentence splitter, and notes two times per request, from the moment it is sent:

- first: the first sentence is handed to the speaker. This is when streaming lets the voice start.
- done: the model has written the whole reply. Without streaming the voice could not start before.

`done - first` is what streaming saves on that reply. The speaker here is a stand-in that only
reads the clock: nothing is synthesized or played, so add the voice's own time to both (Kokoro's
first sentence: a median 0.24 s, `docs/models.md`). The model comes from the config
(`strata.*` and `model.*`); a server that is not running, or whose model is not loaded, is a
refusal: this never starts Strata, loads the model or writes the audit log, and the agent it
builds has no tools, so a request can do nothing on the PC.

    uv run tools/first_audio.py
    uv run tools/first_audio.py --runs 5 --prompt "Explain how tides work in four sentences."
"""

import argparse
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from jarvis import app, config
from jarvis.agent import Agent, Hooks
from jarvis.audit import AuditLog
from jarvis.model import ChatModel
from jarvis.operations import Registry
from jarvis.paths import strata_key_file
from jarvis.secretfile import read_api_key

DEFAULT_PROMPTS = (
    "Tell me about owls in four or five sentences.",
    "Explain how a rainbow forms, in a few sentences.",
    "Describe how tides work, in a few sentences.",
)
DEFAULT_RUNS = 3
P90 = 0.9

if TYPE_CHECKING:
    import httpx

type Connect = Callable[[config.Config], ChatModel]


class UnavailableError(Exception):
    """The model cannot be measured now: no key, server down or model not loaded."""


@dataclass(frozen=True, slots=True)
class Sample:
    """One request: when its first sentence reached the speaker, and when the model was done."""

    prompt: str
    first_s: float | None  # None: nothing was spoken
    done_s: float
    sentences: int


def time_request(agent: Agent, text: str, clock: Callable[[], float]) -> Sample:
    """Send one request and return when its first sentence was handed to the speaker."""
    started = clock()
    handed: list[float] = []

    def speak(_sentence: str) -> None:
        handed.append(clock() - started)

    agent.handle(text, Hooks(confirm=lambda _question: False, speak=speak))
    return Sample(text, handed[0] if handed else None, clock() - started, len(handed))


def percentile(values: Sequence[float], fraction: float) -> float:
    """Return the value at `fraction` of the sorted values (nearest rank, never below the first)."""
    ordered = sorted(values)
    return ordered[max(0, min(len(ordered) - 1, round(fraction * len(ordered) + 0.5) - 1))]


def _row(label: str, values: Sequence[float]) -> str:
    return (
        f"{label:<16}median {statistics.median(values):6.2f} s"
        f"   p90 {percentile(values, P90):6.2f} s   max {max(values):6.2f} s"
    )


def report(samples: Sequence[Sample]) -> str:
    """Return the table of per-request times and the summary, as text."""
    lines = [f"{'first':>8} {'done':>8} {'saved':>8}  sentences  prompt"]
    spoken = [s for s in samples if s.first_s is not None]
    for sample in samples:
        if sample.first_s is None:
            lines.append(
                f"{'-':>8} {sample.done_s:8.2f} {'-':>8}  {sample.sentences:9}  {sample.prompt}"
            )
        else:
            saved = sample.done_s - sample.first_s
            lines.append(
                f"{sample.first_s:8.2f} {sample.done_s:8.2f} {saved:8.2f}"
                f"  {sample.sentences:9}  {sample.prompt}"
            )
    lines.append("")
    if spoken:
        firsts = [s.first_s for s in spoken if s.first_s is not None]
        lines += [
            _row("first sentence", firsts),
            _row("reply written", [s.done_s for s in spoken]),
            _row("saved", [s.done_s - f for s, f in zip(spoken, firsts, strict=True)]),
        ]
    if len(spoken) < len(samples):
        lines.append(f"{len(samples) - len(spoken)} request(s) spoke nothing: not counted above.")
    lines.append("Times stop at the hand-over to the speaker; the voice adds its own.")
    return "\n".join(lines)


def connect(cfg: config.Config, transport: httpx.BaseTransport | None = None) -> ChatModel:
    """Return the configured model client, only when the server is up with its model loaded.

    Only the server's status is read to find out: nothing is generated or loaded. `transport`
    replaces the network in tests.

    Raises:
        UnavailableError: no API key, no answer from the server, or the model is not loaded.
    """
    key = read_api_key(strata_key_file())
    if not key:
        msg = f"no Strata API key at {strata_key_file()}; run `jarvis setup`"
        raise UnavailableError(msg)
    model = replace(app.strata_model(cfg, key), transport=transport)
    state = model.status()
    if state is None:
        msg = f"no answer from the model server at {cfg.model_base_url()}; start it first"
        raise UnavailableError(msg)
    if state != "loaded":
        msg = f"the model is {state}, not loaded; this measures a loaded model and loads nothing"
        raise UnavailableError(msg)
    return model


def measure(
    model: ChatModel,
    cfg: config.Config,
    prompts: Sequence[str],
    runs: int,
    clock: Callable[[], float] = time.monotonic,
) -> list[Sample]:
    """Return one sample per prompt and run, each from a fresh agent with only `no_action`."""
    samples: list[Sample] = []
    for _ in range(runs):
        for prompt in prompts:
            agent = Agent(
                model,
                Registry([app.no_action_operation()]),
                app.system_prompt(cfg),
                AuditLog(None),
                {},
                names=(cfg.persona.name,),
            )
            samples.append(time_request(agent, prompt, clock))
    return samples


def _runs(text: str) -> int:
    number = int(text)
    if number < 1:
        msg = "must be at least 1"
        raise argparse.ArgumentTypeError(msg)
    return number


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    parser = argparse.ArgumentParser(
        prog="first_audio.py",
        description="Time the first sentence of a streamed reply against the whole reply.",
    )
    parser.add_argument("--runs", type=_runs, default=DEFAULT_RUNS, help="rounds (default 3)")
    parser.add_argument(
        "--prompt",
        action="append",
        default=[],
        metavar="TEXT",
        help="a request to send (repeatable; default: three general-knowledge questions)",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    connector: Connect = connect,
    clock: Callable[[], float] = time.monotonic,
) -> int:
    """Run the measurement and return the exit status: 0 measured, 2 cannot measure."""
    args = build_parser().parse_args(argv)
    try:
        cfg = config.load()
        model = connector(cfg)
    except (config.ConfigError, UnavailableError) as exc:
        print(f"first_audio: {exc}", file=sys.stderr)
        return 2
    prompts = args.prompt or list(DEFAULT_PROMPTS)
    print(report(measure(model, cfg, prompts, args.runs, clock)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
