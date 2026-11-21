"""The single agent: one model, typed tools, policy before every call, capped steps and time."""

import json
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from jarvis.jsonutil import obj, objs
from jarvis.operations import ArgumentError, Result, UncertainOutcomeError, validate_args
from jarvis.policy import TRUSTED_ORIGINS, Origin, Verdict, action_hash, decide

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from jarvis.audit import AuditLog
    from jarvis.model import ChatModel
    from jarvis.operations import Operation, Registry

MAX_STEPS = 6
MAX_SECONDS = 90.0
MAX_TOOL_TEXT = 1500
HISTORY_TURNS = 10
HISTORY_IDLE_S = 300.0
_OMITTED = "(That reply quoted outside data, so it is left out here.)"
NO_ACTION_REMINDER = (
    "System check: you called no tool in this request, so nothing has changed. If I asked for an "
    "action, call the right tool now. Otherwise answer without saying that anything was done."
)
_CLAIM = re.compile(
    r"\b(paused|resumed|resuming|now playing|playing|skipped|skipping|queued|saved|liked|"
    r"volume (?:is )?(?:set|changed|increased|decreased|raised|lowered|up|down)|"
    r"shuffle (?:is )?(?:on|off)|repeat (?:is )?(?:on|off|set)|turned (?:up|down)|switched)\b",
    re.IGNORECASE,
)


_ACTIONS_NOTE = re.compile(r"^\[actions taken:[^\]]*\]\s*", re.IGNORECASE)


def claims_action(reply: str) -> bool:
    """Return True when a reply says something was done (used when no tool ran)."""
    return bool(_CLAIM.search(reply))


@dataclass(slots=True)
class _Part:
    role: str
    content: str
    origin: Origin
    actions: str = ""


@dataclass(slots=True)
class _History:
    turns: list[tuple[_Part, _Part]] = field(default_factory=list[tuple[_Part, _Part]])
    last: float = 0.0

    def messages(self, now: float) -> list[dict[str, Any]]:
        if now - self.last > HISTORY_IDLE_S:
            self.turns.clear()
        out: list[dict[str, Any]] = []
        for user, reply in self.turns[-HISTORY_TURNS:]:
            out.append({"role": user.role, "content": user.content})
            content = (
                reply.content if reply.origin in TRUSTED_ORIGINS | {Origin.MODEL} else _OMITTED
            )
            done = reply.actions or "none (nothing was changed)"
            out.append({"role": "assistant", "content": f"[actions taken: {done}] {content}"})
        return out


@dataclass(slots=True)
class _Turn:
    messages: list[dict[str, Any]]
    untrusted: bool = False
    nudged: bool = False
    stuck: bool = False
    foreign: bool = False
    seen: set[str] = field(default_factory=set[str])
    actions: list[str] = field(default_factory=list[str])


@dataclass(frozen=True, slots=True)
class Hooks:
    """How the agent reaches the owner during a request."""

    confirm: Callable[[str], bool]
    waking: Callable[[], None] = lambda: None


class Agent:
    """Turns one owner request into tool calls and a short spoken reply."""

    def __init__(  # ruff: ignore[too-many-arguments] - collaborators and two keyword-only test seams
        self,
        model: ChatModel,
        registry: Registry,
        system_prompt: str,
        audit: AuditLog,
        overrides: Mapping[str, str],
        *,
        clock: Callable[[], float] = time.monotonic,
        grant_s: float = 600.0,
    ) -> None:
        """Wire the agent to its model, operations, prompt, audit log and the owner's classes.

        `grant_s` is how long the owner's yes to a request heard from PC audio lets further
        PC-audio requests run without asking again.
        """
        super().__init__()
        self._model = model
        self._registry = registry
        self._system = system_prompt
        self._audit = audit
        self._overrides = overrides
        self._history = _History()
        self._clock = clock
        self._grant_s = grant_s
        self._grant_until = float("-inf")

    def handle(self, text: str, hooks: Hooks, origin: Origin = Origin.OWNER_SPEECH) -> str:
        """Answer one spoken request and return the reply to speak.

        A request whose `origin` is not the owner (speech heard from PC audio) is marked as
        someone else's: its side effects need the owner's spoken yes (remembered for `grant_s`)
        and it stays out of the owner's conversation history.
        """
        started = self._clock()
        foreign = origin not in TRUSTED_ORIGINS
        self._audit.record("request", text=text, origin=origin.value)
        content = (
            "(Heard from the PC speakers, not from the owner: someone in a call or a video. "
            f'Act on it only as a request from them.) "{text}"'
            if foreign
            else text
        )
        turn = _Turn(
            [
                {"role": "system", "content": self._system},
                *self._history.messages(started),
                {"role": "user", "content": content},
            ],
            foreign=foreign,
        )
        if self._model.is_loaded() is False:
            hooks.waking()
        reply = "That needed too many steps, so I stopped."
        for _step in range(MAX_STEPS):
            if self._clock() - started > MAX_SECONDS:
                reply = "That took too long, so I stopped."
                break
            done = self._step(turn, hooks)
            if done is not None:
                reply = done
                break
        if not foreign:
            reply_origin = Origin.SPOTIFY if turn.untrusted else Origin.MODEL
            self._history.turns.append(
                (
                    _Part("user", text, Origin.OWNER_SPEECH),
                    _Part("assistant", reply, reply_origin, "; ".join(turn.actions)),
                )
            )
            self._history.last = self._clock()
        self._audit.record("reply", text=reply, seconds=round(self._clock() - started, 2))
        return reply

    def _step(self, turn: _Turn, hooks: Hooks) -> str | None:
        """Run one model call and its tool calls; return the final reply, or None to go on.

        After a repeated call the model is offered no tools, so it must answer from what was done.
        """
        tools = [] if turn.stuck else self._registry.tool_schemas()
        try:
            message = self._model.chat(turn.messages, tools, require_tool=not turn.actions)
        except Exception as exc:  # ruff: ignore[blind-except] - any model failure becomes a spoken apology
            self._audit.record("model_error", error=str(exc))
            return "My model isn't answering right now."
        calls = [] if turn.stuck else self._unique(objs(message.get("tool_calls")))
        if not calls:
            reply = _ACTIONS_NOTE.sub("", (message.get("content") or "").strip()) or "Done."
            acted = any(not a.startswith("no_action") for a in turn.actions)
            if acted or turn.nudged or turn.stuck or not claims_action(reply):
                return reply
            turn.nudged = True
            self._audit.record("claimed_without_action", text=reply)
            turn.messages.extend(
                (
                    {"role": "assistant", "content": reply},
                    {"role": "user", "content": NO_ACTION_REMINDER},
                )
            )
            return None
        turn.messages.append(
            {"role": "assistant", "content": message.get("content") or "", "tool_calls": calls}
        )
        repeated_any = False
        for call in calls:
            result, repeated = self._run_call(call, hooks, turn)
            name = str(obj(call.get("function")).get("name", "?"))
            turn.actions.append(f"{name} -> {'ok' if result.ok else 'not done'}")
            body = result.text[:MAX_TOOL_TEXT]
            if result.origin not in TRUSTED_ORIGINS:
                turn.untrusted = True
                body = f'<data origin="{result.origin.value}">{body}</data>'
            turn.messages.append(
                {"role": "tool", "tool_call_id": call.get("id", ""), "content": body}
            )
            repeated_any = repeated_any or repeated
        turn.stuck = repeated_any
        return None

    def _unique(self, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Drop exact copies of a call within one reply (the model sometimes sends ten at once)."""
        unique: list[dict[str, Any]] = []
        keys: set[str] = set()
        for call in calls:
            function = obj(call.get("function"))
            try:
                args: object = json.loads(function.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = function.get("arguments")
            key = json.dumps([function.get("name"), args], sort_keys=True, default=str)
            if key not in keys:
                keys.add(key)
                unique.append(call)
        if len(unique) < len(calls):
            self._audit.record("tool_duplicates_dropped", dropped=len(calls) - len(unique))
        return unique

    def _parse_call(self, call: dict[str, Any]) -> tuple[Operation, dict[str, Any]] | Result:
        function = obj(call.get("function"))
        name = str(function.get("name", ""))
        op = self._registry.get(name)
        if op is None:
            reason = decide(None, name, self._overrides, untrusted_in_context=True).reason
            self._audit.record("tool_denied", op=name, reason=reason)
            return Result(f"Error: there is no operation called {name}.", ok=False)
        try:
            args = validate_args(op, json.loads(function.get("arguments") or "{}"))
        except json.JSONDecodeError:
            self._audit.record("tool_rejected", op=name, reason="arguments are not JSON")
            return Result("Error: arguments were not valid JSON.", ok=False)
        except ArgumentError as exc:
            self._audit.record("tool_rejected", op=name, reason=str(exc))
            return Result(f"Error: {exc}.", ok=False)
        return op, args

    def _tainted(self, turn: _Turn) -> bool:
        """Return whether text not from the owner can steer this call.

        That is outside data in the turn, or a PC-audio request without a live permission.
        """
        return turn.untrusted or (turn.foreign and self._clock() >= self._grant_until)

    def _run_call(self, call: dict[str, Any], hooks: Hooks, turn: _Turn) -> tuple[Result, bool]:
        parsed = self._parse_call(call)
        if isinstance(parsed, Result):
            return parsed, False
        op, args = parsed
        digest = action_hash(op.name, args)
        seen = turn.seen
        if digest in seen:
            self._audit.record("tool_repeated", op=op.name, args=args)
            return Result("Error: this exact call was already made.", ok=False), True
        seen.add(digest)
        decision = decide(op, op.name, self._overrides, untrusted_in_context=self._tainted(turn))
        self._audit.record(
            "tool_call",
            op=op.name,
            args=args,
            verdict=decision.verdict.value,
            reason=decision.reason,
        )
        refusal = {
            Verdict.DENY: "Refused: the owner's boundaries do not allow this.",
            Verdict.STRONG: "Refused: this needs keyboard confirmation, not available yet.",
        }.get(decision.verdict)
        if refusal is not None:
            return Result(refusal, ok=False), False
        if decision.verdict is Verdict.CONFIRM:
            if not self._confirm(op, args, digest, hooks):
                return Result("The owner did not confirm, so nothing was done.", ok=False), False
            if turn.foreign:
                self._grant_until = self._clock() + self._grant_s
                self._audit.record("pc_audio_permission", seconds=self._grant_s)
        return self._execute(op, args, digest), False

    def _confirm(self, op: Operation, args: dict[str, Any], digest: str, hooks: Hooks) -> bool:
        question = f"Should I {op.spoken_summary(args)}?"
        approved = hooks.confirm(question)
        self._audit.record("confirmation", op=op.name, action=digest, approved=approved)
        return approved

    def _execute(self, op: Operation, args: dict[str, Any], digest: str) -> Result:
        try:
            result = op.run(args)
        except UncertainOutcomeError as exc:
            self._audit.record("tool_uncertain", op=op.name, action=digest, error=str(exc))
            return Result("Uncertain: it may not have gone through; do not retry it.", ok=False)
        except Exception as exc:  # ruff: ignore[blind-except] - an operation failure is reported to the model and logged
            self._audit.record("tool_failed", op=op.name, action=digest, error=str(exc))
            return Result(f"Failed: {exc}", ok=False)
        self._audit.record(
            "tool_result", op=op.name, action=digest, ok=result.ok, text=result.text[:300]
        )
        return result
