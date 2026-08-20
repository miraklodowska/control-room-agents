from __future__ import annotations

import json
import re
import secrets
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from threading import RLock, local
from time import sleep
from typing import Any, Protocol, TypeVar, cast

from pydantic import BaseModel, ConfigDict, field_validator

from control_room.domain import (
    Action,
    ActionResult,
    ActorRole,
    Evidence,
    Intent,
    Origin,
    Task,
    TaskState,
    baton_metadata_only,
    canonical_json,
    opaque_metadata_token,
    transition_task,
)
from control_room.policy import ROLE_ACTIONS, PolicyDenied, require_pass_evidence


class VersionConflict(RuntimeError):
    pass


class IdempotencyConflict(RuntimeError):
    pass


class EvidenceConflict(RuntimeError):
    pass


class BudgetExhausted(RuntimeError):
    pass


class ActionFailed(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class CycleOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    target_state: TaskState = TaskState.RUNNING
    evidence: tuple[Evidence, ...] = ()
    baton: dict[str, Any] | None = None
    failure_code: str | None = None

    _baton_metadata_only = field_validator("baton")(baton_metadata_only)
    _failure_code_is_opaque = field_validator("failure_code")(opaque_metadata_token)

    @classmethod
    def failed(cls, code: str) -> CycleOutcome:
        return cls(target_state=TaskState.NEEDS_OPERATOR, failure_code=code)


@dataclass(frozen=True)
class CycleRecord:
    canonical_operation: str
    canonical_intent: str | None
    result: ActionResult


@dataclass
class MemoryBacking:
    tasks: dict[str, Task] = field(default_factory=dict)
    evidence: dict[str, list[Evidence]] = field(default_factory=dict)
    evidence_ids: set[str] = field(default_factory=set)
    cycles: dict[tuple[str, str], CycleRecord] = field(default_factory=dict)
    batons: dict[str, dict[str, Any]] = field(default_factory=dict)
    actors: dict[str, ActorRole] = field(default_factory=dict)
    registry_version: int = 0
    cycle_count: int = 0
    active_cycle: str | None = None
    model_calls: dict[str, int] = field(default_factory=dict)
    lock: RLock = field(default_factory=RLock)


class StateStore(Protocol):
    @property
    def backend_name(self) -> str: ...

    def create_task(self, task: Task) -> Task: ...

    def get_task(self, task_id: str) -> Task: ...

    def get_task_status(
        self, task_id: str
    ) -> tuple[Task, dict[str, Any] | None]: ...

    def list_evidence(self, task_id: str) -> tuple[Evidence, ...]: ...

    def apply_cycle(
        self,
        intent: Intent,
        execute: Callable[[Task], CycleOutcome],
        *,
        canonical_operation: str | None = None,
    ) -> ActionResult: ...

    def run_cycle_transaction(
        self,
        task_id: str,
        cycle_key: str,
        *,
        canonical_operation: str,
        canonical_intent: str | None = None,
        transact: Callable[[Task], ActionResult],
    ) -> ActionResult: ...

    def initialize_actors(self, actors: dict[str, ActorRole]) -> int: ...

    def append_evidence(self, evidence: Evidence, *, expected_version: int) -> Task: ...

    def fail_cycle(
        self,
        task_id: str,
        cycle_key: str,
        *,
        canonical_operation: str,
        expected_version: int,
        failure_code: str,
    ) -> ActionResult: ...

    def get_cycle_result(
        self,
        task_id: str,
        cycle_key: str,
        *,
        canonical_operation: str | None = None,
    ) -> ActionResult | None: ...

    def set_baton(
        self, task_id: str, baton: dict[str, Any], *, expected_version: int
    ) -> Task: ...

    def get_baton(self, task_id: str) -> dict[str, Any] | None: ...

    def register_actor(
        self, actor_id: str, role: ActorRole | str, *, expected_version: int
    ) -> int: ...

    def get_registry_version(self) -> int: ...

    def get_actor_role(self, actor_id: str) -> ActorRole: ...

    def list_actors(self) -> dict[str, ActorRole]: ...

    def consume_model_call(self, *, day: str, limit: int) -> bool: ...


class MemoryStateStore:
    def __init__(self, backing: MemoryBacking | None = None) -> None:
        self._backing = backing if backing is not None else MemoryBacking()

    @property
    def backend_name(self) -> str:
        return "memory"

    def create_task(self, task: Task) -> Task:
        with self._backing.lock:
            if task.task_id in self._backing.tasks:
                raise ValueError(f"task already exists: {task.task_id}")
            if len(self._backing.tasks) >= MAX_FIRESTORE_TASKS:
                raise ValueError("global task count bound exceeded")
            self._backing.tasks[task.task_id] = task
            self._backing.evidence[task.task_id] = []
            return task

    def get_task(self, task_id: str) -> Task:
        with self._backing.lock:
            try:
                return self._backing.tasks[task_id]
            except KeyError:
                raise KeyError(f"unknown task: {task_id}") from None

    def get_task_status(
        self, task_id: str
    ) -> tuple[Task, dict[str, Any] | None]:
        with self._backing.lock:
            try:
                task = self._backing.tasks[task_id]
            except KeyError:
                raise KeyError(f"unknown task: {task_id}") from None
            baton = self._backing.batons.get(task_id)
            return task, None if baton is None else deepcopy(baton)

    def get_status(self, task_id: str) -> tuple[Task, dict[str, Any] | None]:
        return self.get_task_status(task_id)

    def list_evidence(self, task_id: str) -> tuple[Evidence, ...]:
        with self._backing.lock:
            if task_id not in self._backing.tasks:
                raise KeyError(f"unknown task: {task_id}")
            return deepcopy(tuple(self._backing.evidence[task_id]))

    def run_cycle_transaction(
        self,
        task_id: str,
        cycle_key: str,
        *,
        canonical_operation: str,
        canonical_intent: str | None = None,
        transact: Callable[[Task], ActionResult],
    ) -> ActionResult:
        """Serialize one bounded service cycle across stores sharing this backing."""
        with self._backing.lock:
            previous = self._get_cycle_result_locked(
                task_id, cycle_key, canonical_operation=canonical_operation
            )
            if previous is not None:
                return previous
            task = self.get_task(task_id)
            result = transact(task)
            persisted = self._get_cycle_result_locked(
                task_id, cycle_key, canonical_operation=canonical_operation
            )
            if persisted is None or persisted != result:
                raise RuntimeError("cycle transaction did not persist its result")
            return persisted

    def append_evidence(self, evidence: Evidence, *, expected_version: int) -> Task:
        with self._backing.lock:
            task = self.get_task(evidence.task_id)
            self._require_mutable(task)
            self._require_version(task, expected_version)
            validated = Evidence.model_validate(evidence.model_dump())
            if evidence.evidence_id in self._backing.evidence_ids:
                raise EvidenceConflict(f"evidence is append-only: {evidence.evidence_id}")
            updated = task.model_copy(update={"version": task.version + 1})
            self._backing.evidence[evidence.task_id].append(deepcopy(validated))
            self._backing.evidence_ids.add(evidence.evidence_id)
            self._backing.tasks[evidence.task_id] = updated
            return updated

    def apply_cycle(
        self,
        intent: Intent,
        execute: Callable[[Task], CycleOutcome],
        *,
        canonical_operation: str | None = None,
    ) -> ActionResult:
        intended_operation = canonical_operation or "apply_cycle"
        validated_intent = Intent.model_validate(intent.model_dump(warnings="none"))
        canonical_intent = canonical_json(validated_intent)
        key = (validated_intent.task_id, validated_intent.cycle_key)
        with self._backing.lock:
            previous = self._backing.cycles.get(key)
            if previous is not None:
                if previous.canonical_operation != intended_operation:
                    raise IdempotencyConflict(
                        f"cycle key reused with different operation: {validated_intent.cycle_key}"
                    )
                if previous.canonical_intent != canonical_intent:
                    raise IdempotencyConflict(
                        f"cycle key reused with different intent: {validated_intent.cycle_key}"
                    )
                return deepcopy(previous.result)

            task = self.get_task(validated_intent.task_id)
            self._require_version(task, validated_intent.expected_version)
            if task.state in {TaskState.PASSED, TaskState.FAILED}:
                raise ValueError(f"terminal task {task.state} is immutable")
            if task.tool_actions >= 3 or task.failed_actions >= 1:
                raise BudgetExhausted("task action budget is exhausted")
            try:
                actor_role = self.get_actor_role(validated_intent.actor_id)
            except KeyError as exc:
                raise PolicyDenied("unknown_actor", "actor is not registered") from exc
            if ROLE_ACTIONS.get(actor_role) is not validated_intent.action:
                raise PolicyDenied(
                    "actor_action_mismatch", "registered actor role cannot perform action"
                )

            admitted_state = (
                TaskState.RUNNING if task.state is TaskState.QUEUED else task.state
            )
            admitted = task.model_copy(
                update={
                    "state": admitted_state,
                    "version": task.version + 1,
                    "tool_actions": task.tool_actions + 1,
                    "worker_actor_id": (
                        task.worker_actor_id or validated_intent.actor_id
                        if actor_role is ActorRole.WORKER
                        else task.worker_actor_id
                    ),
                }
            )
            pending_ids: set[str] = set()
            validated_evidence: list[Evidence] = []
            try:
                try:
                    outcome = execute(admitted)
                except ActionFailed as exc:
                    failure_code = (
                        exc.code if isinstance(exc.code, str) else "invalid_outcome"
                    )
                    outcome = CycleOutcome.failed(failure_code)
                except Exception:
                    outcome = CycleOutcome.failed("action_exception")

                if not isinstance(outcome, CycleOutcome):
                    raise TypeError("action returned an invalid outcome")
                outcome = CycleOutcome.model_validate(outcome.model_dump(warnings="none"))
                if outcome.failure_code is not None:
                    if outcome.evidence or outcome.baton is not None:
                        raise ValueError("failed outcome cannot persist evidence or baton")
                    updated = admitted.model_copy(
                        update={
                            "state": TaskState.NEEDS_OPERATOR,
                            "failed_actions": task.failed_actions + 1,
                        }
                    )
                else:
                    if outcome.target_state is TaskState.FAILED:
                        raise ValueError("FAILED outcome requires a typed failure_code")
                    for item in outcome.evidence:
                        validated_item = Evidence.model_validate(item.model_dump())
                        if validated_item.task_id != task.task_id:
                            raise ValueError("evidence task does not match intent task")
                        if (
                            validated_item.evidence_id in self._backing.evidence_ids
                            or validated_item.evidence_id in pending_ids
                        ):
                            raise EvidenceConflict(
                                f"evidence is append-only: {validated_item.evidence_id}"
                            )
                        pending_ids.add(validated_item.evidence_id)
                        validated_evidence.append(validated_item)
                    if outcome.target_state is TaskState.PASSED:
                        role = self.get_actor_role(validated_intent.actor_id)
                        if (
                            role is not ActorRole.VERIFIER
                            or validated_intent.action is not Action.VERIFY_TASK
                        ):
                            raise PolicyDenied(
                                "pass_requires_verifier",
                                "only a registered verifier can authorize PASS",
                            )
                        require_pass_evidence(
                            admitted,
                            tuple(validated_evidence),
                            verifier_id=validated_intent.actor_id,
                        )
                    updated = transition_task(admitted, outcome.target_state)
                stored_evidence = deepcopy(tuple(validated_evidence))
                result = ActionResult(
                    task=updated,
                    evidence=stored_evidence,
                    failure_code=outcome.failure_code,
                )
            except Exception as exc:
                failure_code = (
                    exc.code
                    if isinstance(exc, PolicyDenied) and isinstance(exc.code, str)
                    else "invalid_outcome"
                )
                outcome = CycleOutcome.failed(failure_code)
                pending_ids.clear()
                validated_evidence.clear()
                updated = admitted.model_copy(
                    update={
                        "state": TaskState.NEEDS_OPERATOR,
                        "failed_actions": task.failed_actions + 1,
                    }
                )
                stored_evidence = deepcopy(tuple(validated_evidence))
                result = ActionResult(
                    task=updated,
                    evidence=stored_evidence,
                    failure_code=outcome.failure_code,
                )
            self._backing.tasks[task.task_id] = updated
            self._backing.evidence[task.task_id].extend(stored_evidence)
            self._backing.evidence_ids.update(pending_ids)
            if outcome.failure_code is None and outcome.baton is not None:
                self._backing.batons[task.task_id] = deepcopy(outcome.baton)
            self._backing.cycles[key] = CycleRecord(
                intended_operation, canonical_intent, deepcopy(result)
            )
            return deepcopy(result)

    def fail_cycle(
        self,
        task_id: str,
        cycle_key: str,
        *,
        canonical_operation: str,
        expected_version: int,
        failure_code: str,
    ) -> ActionResult:
        key = (task_id, cycle_key)
        with self._backing.lock:
            previous = self._backing.cycles.get(key)
            if previous is not None:
                if previous.canonical_operation != canonical_operation:
                    raise IdempotencyConflict(
                        f"cycle key reused with different operation: {cycle_key}"
                    )
                return deepcopy(previous.result)
            task = self.get_task(task_id)
            self._require_version(task, expected_version)
            if task.state in {TaskState.PASSED, TaskState.FAILED}:
                raise ValueError(f"terminal task {task.state} is immutable")
            updated = task.model_copy(
                update={
                    "state": TaskState.NEEDS_OPERATOR,
                    "version": task.version + 1,
                }
            )
            result = ActionResult(task=updated, failure_code=failure_code)
            self._backing.tasks[task_id] = updated
            self._backing.cycles[key] = CycleRecord(
                canonical_operation, None, deepcopy(result)
            )
            return deepcopy(result)

    def get_cycle_result(
        self,
        task_id: str,
        cycle_key: str,
        *,
        canonical_operation: str | None = None,
    ) -> ActionResult | None:
        with self._backing.lock:
            return self._get_cycle_result_locked(
                task_id, cycle_key, canonical_operation=canonical_operation
            )

    def _get_cycle_result_locked(
        self,
        task_id: str,
        cycle_key: str,
        *,
        canonical_operation: str | None = None,
    ) -> ActionResult | None:
        record = self._backing.cycles.get((task_id, cycle_key))
        if (
            record is not None
            and canonical_operation is not None
            and record.canonical_operation != canonical_operation
        ):
            raise IdempotencyConflict(
                f"cycle key reused with different operation: {cycle_key}"
            )
        return None if record is None else deepcopy(record.result)

    def set_baton(
        self, task_id: str, baton: dict[str, Any], *, expected_version: int
    ) -> Task:
        validated_baton = baton_metadata_only(baton)
        with self._backing.lock:
            task = self.get_task(task_id)
            self._require_mutable(task)
            self._require_version(task, expected_version)
            updated = task.model_copy(update={"version": task.version + 1})
            self._backing.batons[task_id] = deepcopy(validated_baton)
            self._backing.tasks[task_id] = updated
            return updated

    def get_baton(self, task_id: str) -> dict[str, Any] | None:
        with self._backing.lock:
            baton = self._backing.batons.get(task_id)
            return None if baton is None else deepcopy(baton)

    def register_actor(
        self, actor_id: str, role: ActorRole | str, *, expected_version: int
    ) -> int:
        with self._backing.lock:
            if self._backing.registry_version != expected_version:
                raise VersionConflict(
                    f"expected registry version {expected_version}, "
                    f"found {self._backing.registry_version}"
                )
            parsed_role = ActorRole(role)
            existing = self._backing.actors.get(actor_id)
            if existing is not None and existing is not parsed_role:
                raise ValueError(f"actor role is immutable: {actor_id}")
            if existing is parsed_role:
                return self._backing.registry_version
            self._backing.actors[actor_id] = parsed_role
            self._backing.registry_version += 1
            return self._backing.registry_version

    def initialize_actors(self, actors: dict[str, ActorRole]) -> int:
        with self._backing.lock:
            parsed = {actor_id: ActorRole(role) for actor_id, role in actors.items()}
            for actor_id, role in parsed.items():
                existing = self._backing.actors.get(actor_id)
                if existing is not None and existing is not role:
                    raise ValueError(f"actor role is immutable: {actor_id}")
            for actor_id, role in parsed.items():
                if actor_id not in self._backing.actors:
                    self._backing.actors[actor_id] = role
                    self._backing.registry_version += 1
            return self._backing.registry_version

    def get_registry_version(self) -> int:
        with self._backing.lock:
            return self._backing.registry_version

    def get_actor_role(self, actor_id: str) -> ActorRole:
        with self._backing.lock:
            try:
                return self._backing.actors[actor_id]
            except KeyError:
                raise KeyError(f"unknown actor: {actor_id}") from None

    def list_actors(self) -> dict[str, ActorRole]:
        with self._backing.lock:
            return dict(sorted(self._backing.actors.items()))

    def consume_model_call(self, *, day: str, limit: int) -> bool:
        _validate_model_budget_args(day, limit)
        with self._backing.lock:
            used = self._backing.model_calls.get(day, 0)
            if used >= limit:
                return False
            self._backing.model_calls = {day: used + 1}
            return True

    @staticmethod
    def _require_version(task: Task, expected_version: int) -> None:
        if task.version != expected_version:
            raise VersionConflict(
                f"expected version {expected_version}, found {task.version}"
            )

    @staticmethod
    def _require_mutable(task: Task) -> None:
        if task.state in {TaskState.PASSED, TaskState.FAILED}:
            raise ValueError(f"terminal task {task.state} is immutable")


T = TypeVar("T")
TransactionRunner = Callable[[Callable[[Any], T]], T]

MAX_FIRESTORE_TASK_BYTES = 256_000
MAX_FIRESTORE_TASKS = 256
MAX_FIRESTORE_EVIDENCE = 24
MAX_FIRESTORE_REQUIRED_CHECKS = 8
MAX_FIRESTORE_ACTORS = 64
MAX_FIRESTORE_CYCLES_PER_TASK = 64
MAX_FIRESTORE_CYCLE_BYTES = 64_000
MAX_FIRESTORE_REGISTRY_BYTES = 64_000
MAX_FIRESTORE_CANONICAL_FIELD_LENGTH = 4_096
MAX_FIRESTORE_ACTOR_ID_LENGTH = 128
MAX_MODEL_CALLS_PER_DAY_BOUND = 1_000
FIRESTORE_CLAIM_LEASE_SECONDS = 120
_MODEL_BUDGET_DAY = re.compile(r"\d{4}-\d{2}-\d{2}").fullmatch
_SYNTHETIC_TITLE = re.compile(r"synthetic_[a-z0-9_-]{1,54}").fullmatch
_SYNTHETIC_CHECKS = frozenset({"check", "progress", "quality", "safety"})
_ACTOR_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}").fullmatch


def _snapshot_dict(document: Any, *, transaction: Any | None = None) -> dict[str, Any] | None:
    snapshot = document.get(transaction=transaction)
    return cast(dict[str, Any] | None, snapshot.to_dict() if snapshot.exists else None)


def _task_backing(value: Mapping[str, Any] | None) -> MemoryBacking:
    if value is None:
        return MemoryBacking()
    if value.get("schema_version") != 2:
        raise ValueError("unsupported Firestore task schema")
    _validate_document_bytes(value, MAX_FIRESTORE_TASK_BYTES, "per-task")
    cycle_count = value.get("cycle_count", 0)
    if (
        type(cycle_count) is not int
        or cycle_count < 0
        or cycle_count > MAX_FIRESTORE_CYCLES_PER_TASK
    ):
        raise ValueError("Firestore per-task cycle count bound exceeded")
    active_cycle = value.get("active_cycle")
    if active_cycle is not None:
        _validate_cycle_key(active_cycle)
    evidence_value = value.get("evidence")
    if not isinstance(evidence_value, list):
        raise ValueError("Firestore per-task evidence must be a list")
    if len(evidence_value) > MAX_FIRESTORE_EVIDENCE:
        raise ValueError("Firestore per-task evidence bound exceeded")
    task = Task.model_validate(value["task"])
    evidence = [Evidence.model_validate(item) for item in evidence_value]
    baton_value = value.get("baton")
    return MemoryBacking(
        tasks={task.task_id: task},
        evidence={task.task_id: evidence},
        evidence_ids={item.evidence_id for item in evidence},
        batons={} if baton_value is None else {task.task_id: deepcopy(baton_value)},
        cycle_count=cycle_count,
        active_cycle=active_cycle,
    )


def _task_document(backing: MemoryBacking, task_id: str) -> dict[str, Any]:
    value = {
        "schema_version": 2,
        "task": backing.tasks[task_id].model_dump(mode="json"),
        "evidence": [item.model_dump(mode="json") for item in backing.evidence[task_id]],
        "baton": deepcopy(backing.batons.get(task_id)),
        "cycle_count": backing.cycle_count,
        "active_cycle": backing.active_cycle,
    }
    if len(backing.evidence[task_id]) > MAX_FIRESTORE_EVIDENCE:
        raise ValueError("Firestore per-task evidence bound exceeded")
    _validate_document_bytes(value, MAX_FIRESTORE_TASK_BYTES, "per-task")
    return value


def _registry_backing(value: Mapping[str, Any] | None) -> MemoryBacking:
    if value is None:
        return MemoryBacking()
    if value.get("schema_version") != 1:
        raise ValueError("unsupported Firestore registry schema")
    _validate_document_bytes(value, MAX_FIRESTORE_REGISTRY_BYTES, "registry")
    actors = cast(dict[str, str], value["actors"])
    if len(actors) > MAX_FIRESTORE_ACTORS:
        raise ValueError("Firestore registry actor bound exceeded")
    for actor_id in actors:
        _validate_actor_id(actor_id)
    return MemoryBacking(
        actors={
            actor_id: ActorRole(role)
            for actor_id, role in actors.items()
        },
        registry_version=cast(int, value["registry_version"]),
    )


def _registry_document(backing: MemoryBacking) -> dict[str, Any]:
    if len(backing.actors) > MAX_FIRESTORE_ACTORS:
        raise ValueError("Firestore registry actor bound exceeded")
    for actor_id in backing.actors:
        _validate_actor_id(actor_id)
    value = {
        "schema_version": 1,
        "actors": {key: value.value for key, value in sorted(backing.actors.items())},
        "registry_version": backing.registry_version,
    }
    _validate_document_bytes(value, MAX_FIRESTORE_REGISTRY_BYTES, "registry")
    return value


def _validate_model_budget_args(day: str, limit: int) -> None:
    if not isinstance(day, str) or _MODEL_BUDGET_DAY(day) is None:
        raise ValueError("model budget day must be a YYYY-MM-DD string")
    if type(limit) is not int or limit < 1 or limit > MAX_MODEL_CALLS_PER_DAY_BOUND:
        raise ValueError("model budget limit must be a bounded positive integer")


def _model_budget_used(value: Mapping[str, Any] | None, day: str) -> int:
    if value is None:
        return 0
    used = value.get("used")
    stored_day = value.get("day")
    if (
        value.get("schema_version") != 1
        or type(used) is not int
        or used < 0
        or used > MAX_MODEL_CALLS_PER_DAY_BOUND
        or not isinstance(stored_day, str)
        or _MODEL_BUDGET_DAY(stored_day) is None
    ):
        raise ValueError("invalid Firestore model budget document")
    return used if stored_day == day else 0


def _validate_actor_id(actor_id: str) -> None:
    if (
        not isinstance(actor_id, str)
        or len(actor_id) > MAX_FIRESTORE_ACTOR_ID_LENGTH
        or _ACTOR_ID(actor_id) is None
    ):
        raise ValueError("Firestore actor ID must be a bounded opaque token")


def _validate_cycle_key(cycle_key: Any) -> None:
    if not isinstance(cycle_key, str) or not cycle_key or len(cycle_key) > 256:
        raise ValueError("Firestore cycle key must be a bounded nonblank string")


def _json_default(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"unsupported document value: {type(value).__name__}")


def _validate_document_bytes(value: Mapping[str, Any], maximum: int, kind: str) -> None:
    encoded_size = len(
        json.dumps(
            value,
            allow_nan=False,
            default=_json_default,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    )
    if encoded_size > maximum:
        raise ValueError(f"Firestore {kind} document size bound exceeded")


def _validate_canonical_field(name: str, value: str | None) -> None:
    if value is None and name == "canonical_intent":
        return
    if (
        not isinstance(value, str)
        or not value
        or len(value) > MAX_FIRESTORE_CANONICAL_FIELD_LENGTH
    ):
        raise ValueError(f"Firestore {name} must be a bounded nonblank string")


def _utc_timestamp(value: Any, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"Firestore cycle {name} must be a UTC timestamp")
    return value.astimezone(UTC)


def _cycle_document(value: Mapping[str, Any]) -> dict[str, Any]:
    document = deepcopy(dict(value))
    _validate_canonical_field("canonical_operation", document.get("canonical_operation"))
    _validate_canonical_field("canonical_intent", document.get("canonical_intent"))
    if document.get("status") not in {"claimed", "finalized"}:
        raise ValueError("unsupported Firestore cycle status")
    if document["status"] == "claimed":
        owner = document.get("owner")
        if not isinstance(owner, str) or not owner or len(owner) > 128:
            raise ValueError("Firestore cycle owner must be a bounded opaque token")
    document["created_at"] = _utc_timestamp(document.get("created_at"), "created_at")
    document["expires_at"] = _utc_timestamp(document.get("expires_at"), "expires_at")
    if document["expires_at"] <= document["created_at"]:
        raise ValueError("Firestore cycle expiry must follow creation")
    _validate_document_bytes(document, MAX_FIRESTORE_CYCLE_BYTES, "cycle")
    return document


class FirestoreStateStore:
    """Durable, bounded StateStore with retry-pure Firestore transactions.

    Each task has its own bounded document. Registry, evidence identifiers, and cycle
    claims use independent documents, so unrelated tasks never share a hot state blob.
    """

    def __init__(
        self,
        client: Any,
        *,
        collection: str = "control_room",
        transaction_runner: TransactionRunner[Any] | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._client = client
        self._collection = client.collection(collection)
        self._registry = self._collection.document("registry")
        self._counters = self._collection.document("counters")
        self._transaction_runner = transaction_runner or self._run_google_transaction
        self._now = now or (lambda: datetime.now(UTC))
        self._claim_local = local()

    @property
    def backend_name(self) -> str:
        return "firestore"

    def _task_ref(self, task_id: str) -> Any:
        return self._collection.document(f"task__{task_id}")

    def _cycle_ref(self, task_id: str, cycle_key: str) -> Any:
        digest = sha256(f"{task_id}\0{cycle_key}".encode()).hexdigest()
        return self._collection.document(f"cycle__{digest}")

    def _evidence_ref(self, evidence_id: str) -> Any:
        digest = sha256(evidence_id.encode()).hexdigest()
        return self._collection.document(f"evidence__{digest}")

    def _run_google_transaction(self, callback: Callable[[Any], T]) -> T:
        from google.cloud import firestore

        transaction = self._client.transaction()
        transactional = firestore.transactional(callback)
        return cast(T, transactional(transaction))

    def _read_task(self, task_id: str) -> MemoryBacking:
        return _task_backing(_snapshot_dict(self._task_ref(task_id)))

    def _mutate_task(
        self, task_id: str, operation: Callable[[MemoryStateStore], T]
    ) -> T:
        task_ref = self._task_ref(task_id)

        def callback(transaction: Any) -> T:
            raw = _snapshot_dict(task_ref, transaction=transaction)
            if raw is not None and raw.get("active_cycle") is not None:
                raise VersionConflict("task has an active cycle")
            backing = _task_backing(raw)
            before_ids = set(backing.evidence_ids)
            result = operation(MemoryStateStore(backing))
            new_ids = backing.evidence_ids - before_ids
            for evidence_id in sorted(new_ids):
                marker = self._evidence_ref(evidence_id)
                if _snapshot_dict(marker, transaction=transaction) is not None:
                    raise EvidenceConflict(f"evidence is append-only: {evidence_id}")
            transaction.set(task_ref, _task_document(backing, task_id))
            for evidence_id in sorted(new_ids):
                transaction.set(self._evidence_ref(evidence_id), {"task_id": task_id})
            return result

        return cast(T, self._transaction_runner(callback))

    @staticmethod
    def _validate_public_task(task: Task) -> None:
        if (
            task.origin is not Origin.LOCAL_API
            or _SYNTHETIC_TITLE(task.title) is None
            or len(task.required_checks) > MAX_FIRESTORE_REQUIRED_CHECKS
            or any(check not in _SYNTHETIC_CHECKS for check in task.required_checks)
        ):
            raise ValueError("Firestore public task fields must be synthetic-only")

    def create_task(self, task: Task) -> Task:
        self._validate_public_task(task)
        task_ref = self._task_ref(task.task_id)

        def callback(transaction: Any) -> Task:
            task_value = _snapshot_dict(task_ref, transaction=transaction)
            counters_value = _snapshot_dict(self._counters, transaction=transaction)
            if counters_value is None:
                created_tasks = 0
            else:
                raw_created_tasks = counters_value.get("created_tasks")
                if (
                    counters_value.get("schema_version") != 1
                    or type(raw_created_tasks) is not int
                    or raw_created_tasks < 0
                    or raw_created_tasks > MAX_FIRESTORE_TASKS
                ):
                    raise ValueError("invalid Firestore task counter")
                created_tasks = raw_created_tasks
            if created_tasks >= MAX_FIRESTORE_TASKS:
                raise ValueError("global task count bound exceeded")

            backing = _task_backing(task_value)
            result = MemoryStateStore(backing).create_task(task)
            task_document = _task_document(backing, task.task_id)
            counter_document = {
                "schema_version": 1,
                "created_tasks": created_tasks + 1,
            }
            transaction.set(task_ref, task_document)
            transaction.set(self._counters, counter_document)
            return result

        return cast(Task, self._transaction_runner(callback))

    def get_task(self, task_id: str) -> Task:
        return MemoryStateStore(self._read_task(task_id)).get_task(task_id)

    def get_task_status(
        self, task_id: str
    ) -> tuple[Task, dict[str, Any] | None]:
        return MemoryStateStore(self._read_task(task_id)).get_task_status(task_id)

    def list_evidence(self, task_id: str) -> tuple[Evidence, ...]:
        return MemoryStateStore(self._read_task(task_id)).list_evidence(task_id)

    def run_cycle_transaction(
        self,
        task_id: str,
        cycle_key: str,
        *,
        canonical_operation: str,
        canonical_intent: str | None = None,
        transact: Callable[[Task], ActionResult],
    ) -> ActionResult:
        cycle_ref = self._cycle_ref(task_id, cycle_key)
        task_ref = self._task_ref(task_id)
        owner = secrets.token_hex(32)
        _validate_cycle_key(cycle_key)
        _validate_canonical_field("canonical_operation", canonical_operation)
        _validate_canonical_field("canonical_intent", canonical_intent)

        def claim(
            transaction: Any,
        ) -> tuple[str, Task | ActionResult | None, str | None]:
            now = _utc_timestamp(self._now(), "created_at")
            expires_at = now + timedelta(seconds=FIRESTORE_CLAIM_LEASE_SECONDS)
            raw_cycle = _snapshot_dict(cycle_ref, transaction=transaction)
            cycle = None if raw_cycle is None else _cycle_document(raw_cycle)
            if cycle is not None:
                if cycle["canonical_operation"] != canonical_operation:
                    raise IdempotencyConflict(
                        f"cycle key reused with different operation: {cycle_key}"
                    )
                stored_intent = cycle.get("canonical_intent")
                if cycle["status"] == "finalized":
                    if (
                        canonical_intent is not None
                        and stored_intent is not None
                        and stored_intent != canonical_intent
                    ):
                        raise IdempotencyConflict(
                            f"cycle key reused with different intent: {cycle_key}"
                        )
                    return (
                        "finalized",
                        ActionResult.model_validate(cycle["result"]),
                        stored_intent,
                    )
                if cast(datetime, cycle["expires_at"]) > now:
                    if (
                        canonical_intent is not None
                        and stored_intent is not None
                        and stored_intent != canonical_intent
                    ):
                        raise IdempotencyConflict(
                            f"cycle key reused with different intent: {cycle_key}"
                        )
                    return "waiting", None, stored_intent
                task_value = _snapshot_dict(task_ref, transaction=transaction)
                backing = _task_backing(task_value)
                task = MemoryStateStore(backing).get_task(task_id)
                claimed_task = dict(cast(dict[str, Any], task_value))
                claimed_task["active_cycle"] = cycle_key
                _validate_document_bytes(
                    claimed_task, MAX_FIRESTORE_TASK_BYTES, "per-task"
                )
                takeover_intent = cast(str | None, stored_intent) or canonical_intent
                transaction.set(task_ref, claimed_task)
                transaction.set(
                    cycle_ref,
                    _cycle_document(
                        {
                            "schema_version": 1,
                            "status": "claimed",
                            "owner": owner,
                            "canonical_operation": canonical_operation,
                            "canonical_intent": takeover_intent,
                            "created_at": now,
                            "expires_at": expires_at,
                        }
                    ),
                )
                return "claimed", task, takeover_intent
            task_value = _snapshot_dict(task_ref, transaction=transaction)
            backing = _task_backing(task_value)
            task = MemoryStateStore(backing).get_task(task_id)
            active_cycle = backing.active_cycle
            if active_cycle is not None:
                active_ref = self._cycle_ref(task_id, active_cycle)
                raw_active = _snapshot_dict(active_ref, transaction=transaction)
                active = None if raw_active is None else _cycle_document(raw_active)
                if (
                    active is not None
                    and active.get("status") == "claimed"
                    and cast(datetime, active["expires_at"]) > now
                ):
                    return "waiting", None, None
                if active is not None and active.get("status") == "claimed":
                    transaction.delete(active_ref)
            if backing.cycle_count >= MAX_FIRESTORE_CYCLES_PER_TASK:
                raise ValueError("Firestore per-task cycle count bound exceeded")
            claimed_task = dict(cast(dict[str, Any], task_value))
            claimed_task["active_cycle"] = cycle_key
            _validate_document_bytes(claimed_task, MAX_FIRESTORE_TASK_BYTES, "per-task")
            transaction.set(task_ref, claimed_task)
            transaction.set(
                cycle_ref,
                _cycle_document(
                    {
                        "schema_version": 1,
                        "status": "claimed",
                        "owner": owner,
                        "canonical_operation": canonical_operation,
                        "canonical_intent": canonical_intent,
                        "created_at": now,
                        "expires_at": expires_at,
                    }
                ),
            )
            return "claimed", task, canonical_intent

        status, value, claimed_intent = cast(
            tuple[str, Task | ActionResult | None, str | None],
            self._transaction_runner(claim),
        )
        if status == "finalized":
            return cast(ActionResult, value)
        for _ in range(100):
            if status != "waiting":
                break
            sleep(0.01)
            status, value, claimed_intent = cast(
                tuple[str, Task | ActionResult | None, str | None],
                self._transaction_runner(claim),
            )
            if status == "finalized":
                return cast(ActionResult, value)
        if status == "waiting":
            raise VersionConflict("cycle is already in progress")

        self._claim_local.value = (
            task_id,
            cycle_key,
            canonical_operation,
            owner,
            claimed_intent,
        )
        try:
            result = transact(cast(Task, value))
        except BaseException:
            try:
                self._abort_claim(task_id, cycle_key, owner)
            except BaseException:
                pass
            raise
        finally:
            del self._claim_local.value
        persisted = self.get_cycle_result(
            task_id, cycle_key, canonical_operation=canonical_operation
        )
        if persisted is None or persisted != result:
            raise RuntimeError("cycle transaction did not persist its result")
        return persisted

    def _abort_claim(self, task_id: str, cycle_key: str, owner: str) -> None:
        task_ref = self._task_ref(task_id)
        cycle_ref = self._cycle_ref(task_id, cycle_key)

        def abort(transaction: Any) -> None:
            cycle = _snapshot_dict(cycle_ref, transaction=transaction)
            task_value = _snapshot_dict(task_ref, transaction=transaction)
            if cycle is None or cycle.get("owner") != owner:
                return
            if task_value is not None and task_value.get("active_cycle") == cycle_key:
                cleared = dict(task_value)
                cleared["active_cycle"] = None
                transaction.set(task_ref, cleared)
            transaction.delete(cycle_ref)

        self._transaction_runner(abort)

    def append_evidence(self, evidence: Evidence, *, expected_version: int) -> Task:
        return self._mutate_task(
            evidence.task_id,
            lambda store: store.append_evidence(evidence, expected_version=expected_version),
        )

    def apply_cycle(
        self,
        intent: Intent,
        execute: Callable[[Task], CycleOutcome],
        *,
        canonical_operation: str | None = None,
    ) -> ActionResult:
        operation = canonical_operation or "apply_cycle"
        validated_intent = Intent.model_validate(intent.model_dump(warnings="none"))
        intended_intent = canonical_json(validated_intent)
        existing = _snapshot_dict(
            self._cycle_ref(validated_intent.task_id, validated_intent.cycle_key)
        )
        if (
            existing is not None
            and existing.get("status") == "finalized"
            and existing.get("canonical_intent") != intended_intent
        ):
            raise IdempotencyConflict(
                f"cycle key reused with different intent: {validated_intent.cycle_key}"
            )
        intent = validated_intent
        claim = getattr(self._claim_local, "value", None)
        if claim is None or claim[:3] != (intent.task_id, intent.cycle_key, operation):
            return self.run_cycle_transaction(
                intent.task_id,
                intent.cycle_key,
                canonical_operation=operation,
                canonical_intent=intended_intent,
                transact=lambda _task: self.apply_cycle(
                    intent, execute, canonical_operation=operation
                ),
            )
        claimed_intent = claim[4]
        if claimed_intent is not None:
            stored_intent = Intent.model_validate_json(claimed_intent)
            if (
                stored_intent.task_id != intent.task_id
                or stored_intent.cycle_key != intent.cycle_key
                or canonical_json(stored_intent) != claimed_intent
            ):
                raise ValueError("Firestore cycle contains an invalid canonical intent")
            intent = stored_intent
            intended_intent = claimed_intent
        self._bind_claim_intent(intent.task_id, intent.cycle_key, operation, intended_intent)
        snapshot = self._read_task(intent.task_id)
        registry = _registry_backing(_snapshot_dict(self._registry))
        snapshot.actors = registry.actors
        snapshot.registry_version = registry.registry_version
        memory = MemoryStateStore(snapshot)
        result = memory.apply_cycle(intent, execute, canonical_operation=operation)
        return self._finalize_claim(intent.task_id, intent.cycle_key, operation, result, snapshot)

    def _bind_claim_intent(
        self,
        task_id: str,
        cycle_key: str,
        canonical_operation: str,
        canonical_intent: str,
    ) -> None:
        claim = getattr(self._claim_local, "value", None)
        if claim is None:
            raise RuntimeError("missing active Firestore cycle claim")
        owner = claim[3]
        cycle_ref = self._cycle_ref(task_id, cycle_key)

        def bind(transaction: Any) -> None:
            raw = _snapshot_dict(cycle_ref, transaction=transaction)
            if raw is None:
                raise VersionConflict("cycle claim was lost")
            cycle = _cycle_document(raw)
            if (
                cycle.get("status") != "claimed"
                or cycle.get("owner") != owner
                or cycle.get("canonical_operation") != canonical_operation
            ):
                raise VersionConflict("cycle claim was lost")
            stored = cycle.get("canonical_intent")
            if stored is not None and stored != canonical_intent:
                raise IdempotencyConflict(f"cycle key reused with different intent: {cycle_key}")
            if stored is None:
                cycle["canonical_intent"] = canonical_intent
                transaction.set(cycle_ref, _cycle_document(cycle))

        self._transaction_runner(bind)

    def fail_cycle(
        self,
        task_id: str,
        cycle_key: str,
        *,
        canonical_operation: str,
        expected_version: int,
        failure_code: str,
    ) -> ActionResult:
        claim = getattr(self._claim_local, "value", None)
        if claim is None or claim[:3] != (task_id, cycle_key, canonical_operation):
            return self.run_cycle_transaction(
                task_id,
                cycle_key,
                canonical_operation=canonical_operation,
                transact=lambda _task: self.fail_cycle(
                    task_id,
                    cycle_key,
                    canonical_operation=canonical_operation,
                    expected_version=expected_version,
                    failure_code=failure_code,
                ),
            )
        snapshot = self._read_task(task_id)
        memory = MemoryStateStore(snapshot)
        result = memory.fail_cycle(
            task_id,
            cycle_key,
            canonical_operation=canonical_operation,
            expected_version=expected_version,
            failure_code=failure_code,
        )
        return self._finalize_claim(task_id, cycle_key, canonical_operation, result, snapshot)

    def _finalize_claim(
        self,
        task_id: str,
        cycle_key: str,
        canonical_operation: str,
        result: ActionResult,
        backing: MemoryBacking,
    ) -> ActionResult:
        claim = getattr(self._claim_local, "value", None)
        if claim is None:
            raise RuntimeError("missing active Firestore cycle claim")
        owner = claim[3]
        task_ref = self._task_ref(task_id)
        cycle_ref = self._cycle_ref(task_id, cycle_key)
        record = backing.cycles.get((task_id, cycle_key))
        backing.active_cycle = None
        backing.cycle_count += 1
        if backing.cycle_count > MAX_FIRESTORE_CYCLES_PER_TASK:
            raise ValueError("Firestore per-task cycle count bound exceeded")
        value = _task_document(backing, task_id)

        def finalize(transaction: Any) -> ActionResult:
            now = _utc_timestamp(self._now(), "expires_at")
            raw_cycle = _snapshot_dict(cycle_ref, transaction=transaction)
            cycle = None if raw_cycle is None else _cycle_document(raw_cycle)
            if cycle is None or cycle.get("owner") != owner or cycle.get("status") != "claimed":
                raise VersionConflict("cycle claim was lost")
            if cast(datetime, cycle["expires_at"]) <= now:
                raise VersionConflict("cycle claim expired")
            current = _task_backing(_snapshot_dict(task_ref, transaction=transaction))
            if current.tasks[task_id].version not in {result.task.version, result.task.version - 1}:
                raise VersionConflict("task changed after cycle claim")
            for item in result.evidence:
                marker = self._evidence_ref(item.evidence_id)
                if _snapshot_dict(marker, transaction=transaction) is not None:
                    raise EvidenceConflict(f"evidence is append-only: {item.evidence_id}")
            transaction.set(task_ref, value)
            for item in result.evidence:
                transaction.set(self._evidence_ref(item.evidence_id), {"task_id": task_id})
            transaction.set(
                cycle_ref,
                _cycle_document(
                    {
                        "schema_version": 1,
                        "status": "finalized",
                        "canonical_operation": canonical_operation,
                        "canonical_intent": (
                            cycle.get("canonical_intent")
                            if record is None
                            else record.canonical_intent
                        ),
                        "created_at": cycle["created_at"],
                        "expires_at": cycle["expires_at"],
                        "result": result.model_dump(mode="json"),
                    }
                ),
            )
            return result

        return cast(ActionResult, self._transaction_runner(finalize))

    def get_cycle_result(
        self,
        task_id: str,
        cycle_key: str,
        *,
        canonical_operation: str | None = None,
    ) -> ActionResult | None:
        value = _snapshot_dict(self._cycle_ref(task_id, cycle_key))
        if value is None:
            return None
        value = _cycle_document(value)
        if canonical_operation is not None and value["canonical_operation"] != canonical_operation:
            raise IdempotencyConflict(f"cycle key reused with different operation: {cycle_key}")
        if value["status"] != "finalized":
            return None
        return ActionResult.model_validate(value["result"])

    def set_baton(
        self, task_id: str, baton: dict[str, Any], *, expected_version: int
    ) -> Task:
        return self._mutate_task(
            task_id,
            lambda store: store.set_baton(task_id, baton, expected_version=expected_version),
        )

    def get_baton(self, task_id: str) -> dict[str, Any] | None:
        return MemoryStateStore(self._read_task(task_id)).get_baton(task_id)

    def register_actor(
        self, actor_id: str, role: ActorRole | str, *, expected_version: int
    ) -> int:
        _validate_actor_id(actor_id)
        return self._mutate_registry(
            lambda store: store.register_actor(actor_id, role, expected_version=expected_version)
        )

    def initialize_actors(self, actors: dict[str, ActorRole]) -> int:
        for actor_id in actors:
            _validate_actor_id(actor_id)
        return self._mutate_registry(lambda store: store.initialize_actors(actors))

    def _mutate_registry(self, operation: Callable[[MemoryStateStore], T]) -> T:
        def callback(transaction: Any) -> T:
            backing = _registry_backing(_snapshot_dict(self._registry, transaction=transaction))
            result = operation(MemoryStateStore(backing))
            transaction.set(self._registry, _registry_document(backing))
            return result

        return cast(T, self._transaction_runner(callback))

    def get_registry_version(self) -> int:
        backing = _registry_backing(_snapshot_dict(self._registry))
        return MemoryStateStore(backing).get_registry_version()

    def get_actor_role(self, actor_id: str) -> ActorRole:
        backing = _registry_backing(_snapshot_dict(self._registry))
        return MemoryStateStore(backing).get_actor_role(actor_id)

    def list_actors(self) -> dict[str, ActorRole]:
        return MemoryStateStore(_registry_backing(_snapshot_dict(self._registry))).list_actors()

    def consume_model_call(self, *, day: str, limit: int) -> bool:
        """Transactionally admit one model call under the global daily bound."""
        _validate_model_budget_args(day, limit)
        budget_ref = self._collection.document("model_budget")

        def callback(transaction: Any) -> bool:
            used = _model_budget_used(
                _snapshot_dict(budget_ref, transaction=transaction), day
            )
            if used >= limit:
                return False
            transaction.set(
                budget_ref,
                {"schema_version": 1, "day": day, "used": used + 1},
            )
            return True

        return cast(bool, self._transaction_runner(callback))


def build_state_store(
    environ: Mapping[str, str],
    *,
    firestore_client_factory: Callable[..., Any] | None = None,
    transaction_runner: TransactionRunner[Any] | None = None,
) -> StateStore:
    configured = environ.get("CONTROL_ROOM_STATE_STORE")
    on_cloud_run = bool(environ.get("K_SERVICE", "").strip())
    if on_cloud_run and configured != "firestore":
        raise ValueError("Cloud Run requires Firestore state configuration")
    if configured is None:
        return MemoryStateStore()
    if configured == "memory":
        return MemoryStateStore()
    if configured != "firestore":
        raise ValueError("CONTROL_ROOM_STATE_STORE must be 'memory' or 'firestore'")

    project = environ.get("GOOGLE_CLOUD_PROJECT", "").strip()
    if not project:
        raise ValueError("GOOGLE_CLOUD_PROJECT is required for Firestore state")
    database = environ.get("CONTROL_ROOM_FIRESTORE_DATABASE", "(default)").strip()
    if not database:
        raise ValueError("CONTROL_ROOM_FIRESTORE_DATABASE must not be blank")
    if firestore_client_factory is None:
        from google.cloud import firestore

        firestore_client_factory = firestore.Client
    client = firestore_client_factory(project=project, database=database)
    return FirestoreStateStore(client, transaction_runner=transaction_runner)
