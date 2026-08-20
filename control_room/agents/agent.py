from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any, Protocol

from google.adk.agents.llm_agent import Agent
from google.genai import types

from control_room.domain import ActorRole, Intent, Task
from control_room.policy import PolicyDenied, authorize_intent

GEMINI_MODEL = "gemini-3.5-flash"
GEMINI_TEMPERATURE = 0.0
GEMINI_MAX_OUTPUT_TOKENS = 512


class Coordinator(Protocol):
    def propose(self, task: Task, cycle_key: str, now: datetime) -> Any: ...


def propose_authorized_intent(
    coordinator: Coordinator,
    task: Task,
    cycle_key: str,
    actor_role: ActorRole | Callable[[str], ActorRole],
    *,
    now: datetime,
) -> Intent:
    proposed = coordinator.propose(task, cycle_key, now)
    if isinstance(actor_role, ActorRole):
        resolved_role = actor_role
    else:
        proposed_actor = proposed.get("actor_id") if isinstance(proposed, dict) else None
        if isinstance(proposed_actor, str) and proposed_actor.strip():
            try:
                resolved_role = actor_role(proposed_actor)
            except KeyError as exc:
                raise PolicyDenied("unknown_actor", "actor is not registered") from exc
        else:
            resolved_role = ActorRole.WORKER
    return authorize_intent(proposed, task, resolved_role, now=now, cycle_key=cycle_key)


root_agent = Agent(
    name="control_room_coordinator",
    model=GEMINI_MODEL,
    description="Coordinates one bounded synthetic Control Room cycle.",
    instruction=(
        "Propose exactly one typed action. Deterministic policy code is the sole authority; "
        "never claim that work passed verification."
    ),
    generate_content_config=types.GenerateContentConfig(
        temperature=GEMINI_TEMPERATURE,
        max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    ),
)
