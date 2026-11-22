#!/usr/bin/env python3
"""Measure how long the first sentence of a reply waits for the model, streamed or not (#12).

It sends typed requests to the model server Jarvis already uses, through the real agent and its
sentence splitter, and notes two times per request, from the moment it is sent:

- first: the first sentence is handed to the speaker. This is when streaming lets the voice start.
- done: the model has written the whole reply. Without streaming the voice could not start before.

`done - first` (saved) is what streaming saves on that reply; `done` is the streamed run's own
end, not a separate run without streaming. The speaker here is a stand-in that only reads the
clock: nothing is synthesized or played, so add the voice's own time to both (Kokoro's first
sentence: a median 0.24 s, `docs/models.md`). Only a model already loaded is measured (the warm
case); the cold load is not.

The agent offers only `no_action`, so its prompt is far smaller than real Jarvis's (about 2,900
tokens with every tool, `docs/models.md`): the absolute `first` and `done` are lower than in real
use, and only `saved` carries over. The model comes from the config (`strata.*` and `model.*`); a
server that is not running, or whose model is not loaded, is a refusal: this never starts Strata,
loads the model or writes the audit log, and a request can do nothing on the PC.

A request the model failed (an error, the step or time cap, a reply cut at the token limit) ends
in a fixed apology, not a reply: it is marked failed and left out of the medians.

Exit status: 0 measured, 2 refused (cannot measure, or the model went away mid-run), 3 measured
but at least one request failed.

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
from jarvis.agent import TOO_LONG, TOO_MANY_STEPS, Agent, Hooks
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
EXIT_REFUSED = 2
EXIT_FAILED = 3

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
    failed: str | None = None  # why the model failed; what was spoken was an apology, not a reply


def _failure(reply: str, audit: AuditLog) -> str | None:
    """Return why the request failed, or None when the model answered it.

    The agent speaks a fixed apology when the model fails, and the audit log is the only place
    that says which failure it was; a step or time cap has no event, only its fixed reply.
    """
    events = {str(event["event"]) for event in audit.events}
    if "model_error" in events:
        return "the model did not answer"
    if reply == TOO_LONG:
        return "ran out of time"
    if reply == TOO_MANY_STEPS:
        return "ran out of steps"
    if "reply_limit" in events:
        return "reply cut at the token limit"
    return None


def time_request(agent: Agent, text: str, clock: Callable[[], float], audit: AuditLog) -> Sample:
    """Send one request and return when its first sentence was handed to the speaker.

    `audit` is the log `agent` writes to: it tells a reply from a failure's spoken apology.
    """
    started = clock()
    handed: list[float] = []

    def speak(_sentence: str) -> None:
        handed.append(clock() - started)

    reply = agent.handle(text, Hooks(confirm=lambda _question: False, speak=speak))
    return Sample(
        text, handed[0] if handed else None, clock() - started, len(handed), _failure(reply, audit)
    )


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
    counted = [s for s in samples if s.failed is None and s.first_s is not None]
    for sample in samples:
        if sample.failed is not None:
            lines.append(
                f"{'-':>8} {'-':>8} {'-':>8}  {sample.sentences:9}  {sample.prompt}"
                f"  failed ({sample.failed})"
            )
        elif sample.first_s is None:
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
    if counted:
        firsts = [s.first_s for s in counted if s.first_s is not None]
        lines += [
            _row("first sentence", firsts),
            _row("reply written", [s.done_s for s in counted]),
            _row("saved", [s.done_s - f for s, f in zip(counted, firsts, strict=True)]),
        ]
    failed = sum(s.failed is not None for s in samples)
    silent = sum(s.failed is None and s.first_s is None for s in samples)
    if failed:
        lines.append(f"{failed} request(s) failed: shown above, not counted.")
    if silent:
        lines.append(f"{silent} request(s) spoke nothing: not counted above.")
    lines += [
        "Times stop at the hand-over to the speaker; the voice adds its own.",
        "Only the model loaded is measured, and `done` is this streamed run's own end.",
        "This agent offers only no_action, so its prompt is far smaller than Jarvis's (about 2,900",
        "tokens with every tool): `first` and `done` are lower than in real use, and",
        "only `saved` carries over.",
    ]
    return "\n".join(lines)


def connect(cfg: config.Config, transport: httpx.BaseTransport | None = None) -> ChatModel:
    """Return the configured model client, only when the server is up with its model loaded.

    Only the server's status is read to find out: nothing is generated or loaded. `transport`
    replaces the network in tests.

    Raises:
        UnavailableError: no API key or one that cannot be read safely, no answer from the
            server, or the model is not loaded.
    """
    try:
        key = read_api_key(strata_key_file())
    except OSError as exc:  # PermissionError: the file is not private; the text has no key
        msg = f"cannot use the Strata API key file: {exc}"
        raise UnavailableError(msg) from exc
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
    """Return one sample per prompt and run, each from a fresh agent with only `no_action`.

    Raises:
        UnavailableError: the model is no longer loaded before a sample (nothing is sent then,
            since the server might load it for a request).
    """
    samples: list[Sample] = []
    for _ in range(runs):
        for prompt in prompts:
            if model.is_loaded() is not True:
                msg = "the model is no longer loaded (or the server stopped answering); run stopped"
                raise UnavailableError(msg)
            audit = AuditLog(None)
            agent = Agent(
                model,
                Registry([app.no_action_operation()]),
                app.system_prompt(cfg),
                audit,
                {},
                clock=clock,
                names=(cfg.persona.name,),
            )
            samples.append(time_request(agent, prompt, clock, audit))
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
    """Run the measurement and return the exit status: 0 measured, 2 refused, 3 a request failed."""
    args = build_parser().parse_args(argv)
    prompts = args.prompt or list(DEFAULT_PROMPTS)
    try:
        cfg = config.load()
        model = connector(cfg)
        samples = measure(model, cfg, prompts, args.runs, clock)
    except (config.ConfigError, UnavailableError) as exc:
        print(f"first_audio: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    print(report(samples))
    return EXIT_FAILED if any(s.failed is not None for s in samples) else 0


if __name__ == "__main__":
    sys.exit(main())
