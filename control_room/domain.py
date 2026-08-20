from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_METADATA_STEPS = 8
MAX_METADATA_TOKEN_LENGTH = 128
METADATA_STATUS_VALUES = frozenset({"ok", "recorded", "verified", "failed"})
INTENT_METADATA_KEYS = frozenset(
    {"failure_code", "status", "steps", "summary_ref", "verification_run_id"}
)
EVIDENCE_METADATA_KEYS = frozenset(
    {
        "artifact_id",
        "cycle_key",
        "failure_code",
        "status",
        "summary_ref",
        "verification_run_ref",
    }
)
EVIDENCE_KIND_VALUES = frozenset({"check_result", "progress", "verification"})
BATON_METADATA_KEYS = frozenset({"last_cycle_key", "next_step", "steps"})
_OPAQUE_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}").fullmatch


class TaskState(StrEnum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    NEEDS_OPERATOR = "NEEDS_OPERATOR"
    PASSED = "PASSED"
    FAILED = "FAILED"


class ActorRole(StrEnum):
    COORDINATOR = "coordinator"
    WORKER = "worker"
    VERIFIER = "verifier"


class Action(StrEnum):
    RECORD_PROGRESS = "record_progress"
    VERIFY_TASK = "verify_task"


class Origin(StrEnum):
    LOCAL_API = "LOCAL_API"
    EXTERNAL = "EXTERNAL"


class VerificationStatus(StrEnum):
    VERIFIED = "verified"
    FAILED = "failed"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _not_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("must not be blank")
    return value


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError("timestamp must be UTC")
    return value


def _metadata_error() -> ValueError:
    return ValueError("reference must be metadata-only and bounded")


def _opaque_metadata_token(value: Any) -> str:
    if (
        type(value) is not str
        or len(value) > MAX_METADATA_TOKEN_LENGTH
        or _OPAQUE_TOKEN(value) is None
    ):
        raise _metadata_error()
    return value


def _field_specific_metadata(value: Any, *, allowed_keys: frozenset[str]) -> Any:
    if type(value) is not dict or len(value) > len(allowed_keys):
        raise _metadata_error()
    for key, child in value.items():
        if type(key) is not str or key not in allowed_keys:
            raise _metadata_error()
        if key == "status":
            if type(child) is not str or child not in METADATA_STATUS_VALUES:
                raise _metadata_error()
        elif key == "steps":
            if type(child) is not list or len(child) > MAX_METADATA_STEPS:
                raise _metadata_error()
            for step in child:
                _opaque_metadata_token(step)
        else:
            _opaque_metadata_token(child)
    return value


def _metadata_only(value: Any) -> Any:
    return _field_specific_metadata(value, allowed_keys=INTENT_METADATA_KEYS)


def _evidence_metadata_only(value: Any) -> Any:
    return _field_specific_metadata(value, allowed_keys=EVIDENCE_METADATA_KEYS)


def baton_metadata_only(value: Any) -> Any:
    if value is None:
        return None
    return _field_specific_metadata(value, allowed_keys=BATON_METADATA_KEYS)


def opaque_metadata_token(value: str | None) -> str | None:
    if value is None:
        return None
    return _opaque_metadata_token(value)


def _evidence_kind(value: Any) -> str:
    if type(value) is not str or value not in EVIDENCE_KIND_VALUES:
        raise ValueError("evidence kind is not recognized for Phase 0")
    return value


class Task(StrictModel):
    task_id: str
    title: str
    required_checks: tuple[str, ...]
    origin: Origin
    state: TaskState = TaskState.QUEUED
    version: int = Field(default=0, ge=0)
    tool_actions: int = Field(default=0, ge=0)
    failed_actions: int = Field(default=0, ge=0)
    worker_actor_id: str | None = None

    _task_id_not_blank = field_validator("task_id")(_not_blank)
    _title_not_blank = field_validator("title")(_not_blank)

    @field_validator("required_checks")
    @classmethod
    def checks_are_present(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or any(not check.strip() for check in value) or len(set(value)) != len(value):
            raise ValueError("required_checks must be unique and non-empty")
        return value

    @classmethod
    def new(
        cls, task_id: str, title: str, required_checks: tuple[str, ...], origin: Origin
    ) -> Task:
        return cls(task_id=task_id, title=title, required_checks=required_checks, origin=origin)


class Intent(StrictModel):
    task_id: str
    cycle_key: str
    actor_id: str
    action: Action
    expected_version: int = Field(ge=0)
    issued_at: datetime
    payload: dict[str, Any]

    _task_id_not_blank = field_validator("task_id")(_not_blank)
    _cycle_key_not_blank = field_validator("cycle_key")(_not_blank)
    _actor_id_not_blank = field_validator("actor_id")(_not_blank)
    _issued_at_utc = field_validator("issued_at")(_utc)
    _payload_metadata_only = field_validator("payload")(_metadata_only)


class Evidence(StrictModel):
    evidence_id: str
    task_id: str
    check_id: str
    actor_id: str
    kind: str
    reference: dict[str, Any]
    created_at: datetime

    _evidence_id_is_opaque = field_validator("evidence_id")(_opaque_metadata_token)
    _task_id_is_opaque = field_validator("task_id")(_opaque_metadata_token)
    _check_id_is_opaque = field_validator("check_id")(_opaque_metadata_token)
    _actor_id_is_opaque = field_validator("actor_id")(_opaque_metadata_token)
    _kind_is_allowed = field_validator("kind")(_evidence_kind)
    _reference_metadata_only = field_validator("reference")(_evidence_metadata_only)
    _created_at_utc = field_validator("created_at")(_utc)

    def typed_verification_status(self) -> VerificationStatus:
        raw_status = self.reference.get("status")
        if type(raw_status) is not str:
            raise ValueError("verification status must be a typed string value")
        try:
            return VerificationStatus(raw_status)
        except ValueError as exc:
            raise ValueError("verification status is not recognized") from exc


class ActionResult(StrictModel):
    task: Task
    evidence: tuple[Evidence, ...] = ()
    failure_code: str | None = None

    _failure_code_is_opaque = field_validator("failure_code")(opaque_metadata_token)


LEGAL_TRANSITIONS: dict[TaskState, frozenset[TaskState]] = {
    TaskState.QUEUED: frozenset(
        {TaskState.RUNNING, TaskState.NEEDS_OPERATOR, TaskState.FAILED}
    ),
    TaskState.RUNNING: frozenset(
        {TaskState.RUNNING, TaskState.NEEDS_OPERATOR, TaskState.PASSED, TaskState.FAILED}
    ),
    TaskState.NEEDS_OPERATOR: frozenset({TaskState.RUNNING, TaskState.FAILED}),
    TaskState.PASSED: frozenset(),
    TaskState.FAILED: frozenset(),
}


def transition_task(task: Task, target: TaskState) -> Task:
    if task.state in {TaskState.PASSED, TaskState.FAILED}:
        raise ValueError(f"terminal task {task.state} is immutable")
    if target not in LEGAL_TRANSITIONS[task.state]:
        raise ValueError(f"illegal transition from {task.state} to {target}")
    return task.model_copy(update={"state": target})


def canonical_json(model: BaseModel) -> str:
    return json.dumps(
        model.model_dump(mode="json"),
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def utc_now() -> datetime:
    return datetime.now(UTC)
