from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from pydantic import ValidationError

from control_room.domain import (
    Action,
    ActorRole,
    Evidence,
    Intent,
    Task,
    TaskState,
    VerificationStatus,
)


class PolicyDenied(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.target_state = TaskState.NEEDS_OPERATOR


ROLE_ACTIONS: dict[ActorRole, Action] = {
    ActorRole.WORKER: Action.RECORD_PROGRESS,
    ActorRole.VERIFIER: Action.VERIFY_TASK,
}


def authorize_intent(
    raw_intent: Intent | dict[str, Any],
    task: Task,
    actor_role: ActorRole,
    *,
    now: datetime | None = None,
    cycle_key: str | None = None,
) -> Intent:
    try:
        candidate = raw_intent.model_dump() if isinstance(raw_intent, Intent) else raw_intent
        intent = Intent.model_validate(candidate)
    except (ValidationError, ValueError, TypeError) as exc:
        raise PolicyDenied("malformed", "model output is not a valid intent") from exc

    current_time = now if now is not None else datetime.now(UTC)
    if cycle_key is not None and intent.cycle_key != cycle_key:
        raise PolicyDenied(
            "cycle_key_mismatch",
            "proposed cycle key does not match caller idempotency key",
        )
    age = current_time - intent.issued_at
    if age >= timedelta(seconds=300) or age < timedelta(0):
        raise PolicyDenied("stale_intent", "intent is stale or issued in the future")
    if intent.task_id != task.task_id:
        raise PolicyDenied("task_mismatch", "intent targets a different task")
    if ROLE_ACTIONS.get(actor_role) is not intent.action:
        raise PolicyDenied("actor_action_mismatch", "actor role cannot perform action")
    if intent.expected_version != task.version:
        raise PolicyDenied("stale_version", "intent version does not match stored task")
    steps = intent.payload.get("steps")
    if isinstance(steps, list) and len(steps) > 1:
        raise PolicyDenied("excess_steps", "a cycle admits at most one action")
    if task.tool_actions >= 3 or task.failed_actions >= 1:
        raise PolicyDenied("budget_exhausted", "task action budget is exhausted")
    if task.state in {TaskState.PASSED, TaskState.FAILED}:
        raise PolicyDenied("terminal_task", "terminal task is immutable")
    return intent


def require_pass_evidence(
    task: Task, evidence: tuple[Evidence, ...], *, verifier_id: str
) -> None:
    if task.worker_actor_id is None:
        raise PolicyDenied("missing_worker", "task has no worker identity")
    if verifier_id == task.worker_actor_id:
        raise PolicyDenied("verifier_not_distinct", "verifier must be distinct from worker")

    verified_checks: set[str] = set()
    for item in evidence:
        if (
            item.task_id != task.task_id
            or item.actor_id != verifier_id
            or item.kind != "verification"
        ):
            continue
        if item.typed_verification_status() is not VerificationStatus.VERIFIED:
            raise ValueError("verification evidence does not represent a verified result")
        verified_checks.add(item.check_id)
    missing = set(task.required_checks) - verified_checks
    if missing:
        names = ", ".join(sorted(missing))
        raise PolicyDenied(
            "missing_evidence",
            f"missing evidence: verification records for required checks: {names}",
        )
