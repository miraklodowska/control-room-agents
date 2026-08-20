from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime
from hashlib import sha256
from typing import Any

from control_room.agents.agent import Coordinator, propose_authorized_intent
from control_room.domain import (
    Action,
    ActionResult,
    ActorRole,
    Evidence,
    Intent,
    Origin,
    Task,
    TaskState,
)
from control_room.policy import PolicyDenied
from control_room.state import CycleOutcome, StateStore
from control_room.telemetry import TelemetrySink

MAX_CALLER_IDEMPOTENCY_KEY_LENGTH = 256


def opaque_cycle_key(caller_idempotency_key: str) -> str:
    if (
        not caller_idempotency_key.strip()
        or len(caller_idempotency_key) > MAX_CALLER_IDEMPOTENCY_KEY_LENGTH
    ):
        raise ValueError("idempotency key must be nonblank and at most 256 characters")
    return sha256(caller_idempotency_key.encode("utf-8")).hexdigest()


class DeterministicCoordinator:
    """Provider-free Phase 0 coordinator injected through the same model seam."""

    def propose(self, task: Task, cycle_key: str, now: datetime) -> dict[str, Any]:
        return {
            "task_id": task.task_id,
            "cycle_key": cycle_key,
            "actor_id": "worker-1",
            "action": Action.RECORD_PROGRESS,
            "expected_version": task.version,
            "issued_at": now,
            "payload": {"summary_ref": "synthetic-progress"},
        }


def _canonical_operation(kind: str, task_id: str, cycle_key: str) -> str:
    return json.dumps(
        {"kind": kind, "task_id": task_id, "cycle_key": cycle_key},
        sort_keys=True,
        separators=(",", ":"),
    )


class ControlRoomService:
    def __init__(
        self,
        store: StateStore,
        *,
        now: Callable[[], datetime],
        id_factory: Callable[[], str],
        coordinator: Coordinator | None = None,
        telemetry: TelemetrySink | None = None,
    ) -> None:
        self.store = store
        self._now = now
        self._id_factory = id_factory
        self._coordinator = coordinator or DeterministicCoordinator()
        self.telemetry = telemetry or TelemetrySink()
        self.store.initialize_actors(
            {
                "coordinator-1": ActorRole.COORDINATOR,
                "worker-1": ActorRole.WORKER,
                "verifier-1": ActorRole.VERIFIER,
            }
        )

    def create_task(self, title: str, required_checks: tuple[str, ...]) -> Task:
        task = Task.new(
            f"syn_{self._id_factory()}", title, required_checks, Origin.LOCAL_API
        )
        return self.store.create_task(task)

    def run_cycle(self, task_id: str, caller_idempotency_key: str) -> ActionResult:
        cycle_key = opaque_cycle_key(caller_idempotency_key)
        operation = _canonical_operation("normal", task_id, cycle_key)

        def transact(task: Task) -> ActionResult:
            now = self._now()
            try:
                intent = propose_authorized_intent(
                    self._coordinator,
                    task,
                    cycle_key,
                    self.store.get_actor_role,
                    now=now,
                )
            except PolicyDenied as exc:
                return self.store.fail_cycle(
                    task_id,
                    cycle_key,
                    canonical_operation=operation,
                    expected_version=task.version,
                    failure_code=exc.code,
                )
            except Exception:
                return self.store.fail_cycle(
                    task_id,
                    cycle_key,
                    canonical_operation=operation,
                    expected_version=task.version,
                    failure_code="coordinator_exception",
                )

            def execute(_admitted: Task) -> CycleOutcome:
                item = Evidence(
                    evidence_id=f"ev_{task_id}_{cycle_key}",
                    task_id=task_id,
                    check_id="progress",
                    actor_id=intent.actor_id,
                    kind="progress",
                    reference={"cycle_key": cycle_key, "status": "recorded"},
                    created_at=now,
                )
                return CycleOutcome(
                    target_state=TaskState.RUNNING,
                    evidence=(item,),
                    baton={"last_cycle_key": cycle_key, "next_step": "verify"},
                )

            result = self.store.apply_cycle(intent, execute, canonical_operation=operation)
            self.telemetry.emit(
                "cycle.completed",
                task_id=task_id,
                actor_id=intent.actor_id,
                action=intent.action.value,
                outcome=result.task.state.value,
                version=result.task.version,
            )
            return result

        return self.store.run_cycle_transaction(
            task_id,
            cycle_key,
            canonical_operation=operation,
            transact=transact,
        )

    def trip_demo_breaker(
        self, task_id: str, caller_idempotency_key: str
    ) -> ActionResult:
        cycle_key = opaque_cycle_key(caller_idempotency_key)
        operation = _canonical_operation("breaker", task_id, cycle_key)

        def transact(task: Task) -> ActionResult:
            if task.origin is not Origin.LOCAL_API or not task.task_id.startswith("syn_"):
                raise PermissionError("breaker requires stored local synthetic provenance")
            intent = Intent(
                task_id=task_id,
                cycle_key=cycle_key,
                actor_id="worker-1",
                action=Action.RECORD_PROGRESS,
                expected_version=task.version,
                issued_at=self._now(),
                payload={"failure_code": "demo_missing_evidence"},
            )
            result = self.store.apply_cycle(
                intent,
                lambda _: CycleOutcome.failed("demo_missing_evidence"),
                canonical_operation=operation,
            )
            self.telemetry.emit(
                "breaker.completed",
                task_id=task_id,
                actor_id=intent.actor_id,
                action=intent.action.value,
                outcome=result.task.state.value,
                version=result.task.version,
                failure_code=result.failure_code,
            )
            return result

        return self.store.run_cycle_transaction(
            task_id,
            cycle_key,
            canonical_operation=operation,
            transact=transact,
        )
