"""Model-backed coordinator behind the existing Coordinator seam.

Fail-closed by construction:

* every model call must first be admitted by the transactional global daily
  budget (``StateStore.consume_model_call``); when the budget is exhausted or
  unreadable the coordinator NEVER calls the model and answers with the
  deterministic fallback proposal instead, so anonymous callers cannot amplify
  Vertex cost;
* any transport error raises ``PolicyDenied`` (the service persists
  NEEDS_OPERATOR — progress is never fabricated on a model failure);
* model output is untrusted: it is parsed mechanically into a bounded choice
  (action + opaque summary token), every infrastructure field of the proposal
  is derived locally, and the assembled proposal still passes through
  ``authorize_intent`` unchanged.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any, Protocol

from control_room.agents.agent import Coordinator
from control_room.domain import Action, Task
from control_room.policy import PolicyDenied
from control_room.state import MAX_MODEL_CALLS_PER_DAY_BOUND, StateStore

DEFAULT_MODEL_CALLS_PER_DAY = 50
MAX_MODEL_OUTPUT_CHARS = 8_192
_ALLOWED_OUTPUT_KEYS = frozenset({"action", "summary_ref"})
_ACTION_ACTORS: dict[Action, str] = {
    Action.RECORD_PROGRESS: "worker-1",
    Action.VERIFY_TASK: "verifier-1",
}
_SUMMARY_REF = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}").fullmatch
_FENCE = re.compile(r"```[a-zA-Z0-9]*\n(.*)\n```", re.DOTALL)


class ModelTransport(Protocol):
    def generate(self, prompt: str) -> str: ...


def _prompt(task: Task, cycle_key: str) -> str:
    checks = ", ".join(task.required_checks)
    return (
        "You coordinate exactly one bounded cycle of a synthetic demo task. "
        "Reply with ONLY a JSON object with exactly these two keys and no other "
        'text: {"action": "record_progress" or "verify_task", "summary_ref": '
        "<snake_case token, 1-64 chars, [a-z0-9_-]>}. Deterministic policy code "
        "is the sole authority; never claim work passed verification.\n"
        f"Task id: {task.task_id}\n"
        f"Task title: {task.title}\n"
        f"Task state: {task.state.value}\n"
        f"Required checks: {checks}\n"
        f"Completed tool actions: {task.tool_actions} of 3\n"
        f"Cycle key: {cycle_key}"
    )


def _parse_untrusted_choice(raw: object) -> tuple[Action, str]:
    if not isinstance(raw, str) or not raw.strip() or len(raw) > MAX_MODEL_OUTPUT_CHARS:
        raise PolicyDenied("model_output_invalid", "model output is not bounded text")
    text = raw.strip()
    fenced = _FENCE.fullmatch(text)
    if fenced is not None:
        text = fenced.group(1).strip()
    try:
        value = json.loads(text)
    except ValueError as exc:
        raise PolicyDenied("model_output_invalid", "model output is not JSON") from exc
    if not isinstance(value, dict) or set(value) != _ALLOWED_OUTPUT_KEYS:
        raise PolicyDenied(
            "model_output_invalid", "model output must contain exactly action and summary_ref"
        )
    action_value = value["action"]
    if not isinstance(action_value, str):
        raise PolicyDenied("model_output_invalid", "model action must be a string")
    try:
        action = Action(action_value)
    except ValueError as exc:
        raise PolicyDenied("model_output_invalid", "model action is not recognized") from exc
    if action not in _ACTION_ACTORS:
        raise PolicyDenied("model_output_invalid", "model action is not permitted")
    summary_ref = value["summary_ref"]
    if not isinstance(summary_ref, str) or _SUMMARY_REF(summary_ref) is None:
        raise PolicyDenied(
            "model_output_invalid", "model summary_ref must be a bounded opaque token"
        )
    return action, summary_ref


class GeminiCoordinator:
    """Budget-gated, fail-closed coordinator producing untrusted proposals."""

    def __init__(
        self,
        transport: ModelTransport,
        *,
        consume_model_call: Callable[[str], bool],
        fallback: Coordinator,
    ) -> None:
        self._transport = transport
        self._consume_model_call = consume_model_call
        self._fallback = fallback

    def propose(self, task: Task, cycle_key: str, now: datetime) -> dict[str, Any]:
        day = now.astimezone(UTC).strftime("%Y-%m-%d")
        try:
            admitted = self._consume_model_call(day)
        except Exception:
            # Budget state unknown: never call the model on an unverifiable bound.
            admitted = False
        if not admitted:
            proposal = self._fallback.propose(task, cycle_key, now)
            if not isinstance(proposal, dict):
                raise PolicyDenied("fallback_invalid", "fallback proposal is not a mapping")
            return proposal
        try:
            raw = self._transport.generate(_prompt(task, cycle_key))
        except Exception as exc:
            raise PolicyDenied(
                "model_unavailable", "model call failed; cycle fails closed"
            ) from exc
        action, summary_ref = _parse_untrusted_choice(raw)
        return {
            "task_id": task.task_id,
            "cycle_key": cycle_key,
            "actor_id": _ACTION_ACTORS[action],
            "action": action,
            "expected_version": task.version,
            "issued_at": now,
            "payload": {"summary_ref": summary_ref},
        }


class AdkModelTransport:
    """Real Vertex transport through the ADK root agent.

    Construction performs no network activity; each ``generate`` runs one
    bounded request (temperature 0, max_output_tokens 512, thinking disabled —
    see ``root_agent.generate_content_config``) in a fresh in-memory session so
    no per-call state accumulates.
    """

    _APP_NAME = "control-room"
    _USER_ID = "coordinator"

    def generate(self, prompt: str) -> str:
        from google.adk.runners import InMemoryRunner
        from google.adk.sessions import InMemorySessionService
        from google.genai import types

        from control_room.agents.agent import root_agent

        runner = InMemoryRunner(agent=root_agent, app_name=self._APP_NAME)
        session_service = runner.session_service
        if not isinstance(session_service, InMemorySessionService):
            raise TypeError("InMemoryRunner did not provide an in-memory session service")
        session = session_service.create_session_sync(
            app_name=self._APP_NAME, user_id=self._USER_ID
        )
        chunks: list[str] = []
        for event in runner.run(
            user_id=self._USER_ID,
            session_id=session.id,
            new_message=types.Content(role="user", parts=[types.Part(text=prompt)]),
        ):
            content = getattr(event, "content", None)
            for part in getattr(content, "parts", None) or []:
                text = getattr(part, "text", None)
                if isinstance(text, str):
                    chunks.append(text)
        return "".join(chunks)


def model_calls_per_day(environ: Mapping[str, str]) -> int:
    raw = environ.get("CONTROL_ROOM_MODEL_CALLS_PER_DAY", "").strip()
    if not raw:
        return DEFAULT_MODEL_CALLS_PER_DAY
    try:
        limit = int(raw)
    except ValueError as exc:
        raise ValueError("CONTROL_ROOM_MODEL_CALLS_PER_DAY must be an integer") from exc
    if limit < 1 or limit > MAX_MODEL_CALLS_PER_DAY_BOUND:
        raise ValueError(
            "CONTROL_ROOM_MODEL_CALLS_PER_DAY must be between 1 and "
            f"{MAX_MODEL_CALLS_PER_DAY_BOUND}"
        )
    return limit


def build_gemini_coordinator(
    environ: Mapping[str, str],
    *,
    store: StateStore,
    transport_factory: Callable[[], ModelTransport] | None = None,
) -> GeminiCoordinator:
    from control_room.service import DeterministicCoordinator

    limit = model_calls_per_day(environ)
    transport = transport_factory() if transport_factory is not None else AdkModelTransport()
    return GeminiCoordinator(
        transport,
        consume_model_call=lambda day: store.consume_model_call(day=day, limit=limit),
        fallback=DeterministicCoordinator(),
    )
