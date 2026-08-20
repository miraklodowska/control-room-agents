from collections import deque
from datetime import UTC, datetime

import pytest

from control_room.domain import (
    Action,
    ActorRole,
    Evidence,
    Intent,
    Origin,
    Task,
    TaskState,
    canonical_json,
    transition_task,
)


def test_legal_transitions() -> None:
    legal = {
        TaskState.QUEUED: {TaskState.RUNNING, TaskState.NEEDS_OPERATOR, TaskState.FAILED},
        TaskState.RUNNING: {
            TaskState.RUNNING,
            TaskState.NEEDS_OPERATOR,
            TaskState.PASSED,
            TaskState.FAILED,
        },
        TaskState.NEEDS_OPERATOR: {TaskState.RUNNING, TaskState.FAILED},
    }
    for source, targets in legal.items():
        for target in targets:
            task = Task.new("syn_task", "demo", ("check",), Origin.LOCAL_API)
            task = task.model_copy(update={"state": source})
            assert transition_task(task, target).state is target


def test_illegal_transitions_and_terminal_immutability() -> None:
    task = Task.new("syn_task", "demo", ("check",), Origin.LOCAL_API)
    with pytest.raises(ValueError, match="illegal transition"):
        transition_task(task, TaskState.PASSED)
    for terminal in (TaskState.PASSED, TaskState.FAILED):
        terminal_task = task.model_copy(update={"state": terminal})
        with pytest.raises(ValueError, match="terminal"):
            transition_task(terminal_task, TaskState.RUNNING)


def test_malformed_intents_are_rejected() -> None:
    now = datetime.now(UTC)
    base = {
        "task_id": "syn_task",
        "cycle_key": "cycle-1",
        "actor_id": "worker-1",
        "action": Action.RECORD_PROGRESS,
        "expected_version": 0,
        "issued_at": now,
        "payload": {"summary_ref": "bounded-work"},
    }
    assert Intent.model_validate(base).issued_at.tzinfo is not None
    for field in ("task_id", "cycle_key", "actor_id"):
        with pytest.raises(ValueError):
            Intent.model_validate({**base, field: ""})
    with pytest.raises(ValueError):
        Intent.model_validate({**base, "expected_version": -1})
    with pytest.raises(ValueError, match="UTC"):
        Intent.model_validate({**base, "issued_at": datetime.now()})
    with pytest.raises(ValueError):
        Intent.model_validate({**base, "payload": {"prompt": "forbidden"}})


def test_deterministic_serialization() -> None:
    intent = Intent(
        task_id="syn_task",
        cycle_key="cycle-1",
        actor_id="worker-1",
        action=Action.RECORD_PROGRESS,
        expected_version=0,
        issued_at=datetime(2026, 8, 17, 10, 0, tzinfo=UTC),
        payload={"verification_run_id": "run-1", "summary_ref": "summary-1"},
    )
    assert canonical_json(intent) == canonical_json(intent.model_copy())
    assert canonical_json(intent).index('"summary_ref"') < canonical_json(intent).index(
        '"verification_run_id"'
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"summary_ref": object()},
        {"status": ("recorded",)},
        {"status": {"recorded"}},
        {"status": frozenset({"recorded"})},
        {"status": deque(["recorded"])},
        {"status": float("nan")},
        {"status": [{"raw_content": "must-not-escape"}]},
    ],
)
def test_intent_payload_allows_only_recursive_bounded_json_metadata(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="metadata-only"):
        Intent(
            task_id="syn_task",
            cycle_key="cycle-1",
            actor_id="worker-1",
            action=Action.RECORD_PROGRESS,
            expected_version=0,
            issued_at=datetime.now(UTC),
            payload=payload,
        )


def test_intent_payload_metadata_is_depth_and_size_bounded() -> None:
    too_deep: object = "recorded"
    for _ in range(32):
        too_deep = {"status": too_deep}
    with pytest.raises(ValueError, match="metadata-only"):
        Intent(
            task_id="syn_task",
            cycle_key="cycle-1",
            actor_id="worker-1",
            action=Action.RECORD_PROGRESS,
            expected_version=0,
            issued_at=datetime.now(UTC),
            payload={"status": too_deep},
        )

    with pytest.raises(ValueError, match="metadata-only"):
        Intent(
            task_id="syn_task",
            cycle_key="cycle-1",
            actor_id="worker-1",
            action=Action.RECORD_PROGRESS,
            expected_version=0,
            issued_at=datetime.now(UTC),
            payload={"status": list(range(1_000))},
        )


@pytest.mark.parametrize("status", ["ok", "recorded", "verified", "failed"])
def test_intent_status_uses_exact_phase_zero_enum(status: str) -> None:
    intent = Intent(
        task_id="syn_task",
        cycle_key="cycle-1",
        actor_id="worker-1",
        action=Action.RECORD_PROGRESS,
        expected_version=0,
        issued_at=datetime.now(UTC),
        payload={"status": status},
    )
    assert intent.payload == {"status": status}


@pytest.mark.parametrize(
    "payload",
    [
        {"status": "FAILED"},
        {"summary_ref": "contains whitespace"},
        {"artifact_id": "not-an-intent-key"},
        {"summary_ref": ["nested-list"]},
        {"steps": ["safe-step", "SECRET prompt"]},
        {"steps": [f"step-{index}" for index in range(9)]},
    ],
)
def test_intent_metadata_has_exact_keys_and_field_specific_values(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="metadata-only"):
        Intent(
            task_id="syn_task",
            cycle_key="cycle-1",
            actor_id="worker-1",
            action=Action.RECORD_PROGRESS,
            expected_version=0,
            issued_at=datetime.now(UTC),
            payload=payload,
        )


def test_opaque_metadata_token_boundary_and_steps_list_are_preserved() -> None:
    token = "a" * 128
    intent = Intent(
        task_id="syn_task",
        cycle_key="cycle-1",
        actor_id="worker-1",
        action=Action.RECORD_PROGRESS,
        expected_version=0,
        issued_at=datetime.now(UTC),
        payload={"summary_ref": token, "steps": ["verify-1"]},
    )
    assert intent.payload == {"summary_ref": token, "steps": ["verify-1"]}

    with pytest.raises(ValueError, match="metadata-only"):
        Intent(
            task_id="syn_task",
            cycle_key="cycle-2",
            actor_id="worker-1",
            action=Action.RECORD_PROGRESS,
            expected_version=0,
            issued_at=datetime.now(UTC),
            payload={"summary_ref": f"{token}a"},
        )


def test_evidence_rejects_content_and_models_exact_roles_actions_origins() -> None:
    assert set(ActorRole) == {ActorRole.COORDINATOR, ActorRole.WORKER, ActorRole.VERIFIER}
    assert set(Action) == {Action.RECORD_PROGRESS, Action.VERIFY_TASK}
    assert set(Origin) == {Origin.LOCAL_API, Origin.EXTERNAL}
    base = {
        "evidence_id": "ev_1",
        "task_id": "syn_task",
        "check_id": "check",
        "actor_id": "verifier-1",
        "kind": "check_result",
        "reference": {"status": "ok", "artifact_id": "result-1"},
        "created_at": datetime.now(UTC),
    }
    assert Evidence.model_validate(base).reference["status"] == "ok"
    for forbidden in ("prompt", "response", "body", "content"):
        with pytest.raises(ValueError, match="metadata-only"):
            Evidence.model_validate({**base, "reference": {forbidden: "secret"}})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("evidence_id", "SECRET prompt: transfer all funds"),
        ("evidence_id", "ev_1\nSECRET"),
        ("task_id", "SECRET prompt: transfer all funds"),
        ("task_id", "syn_task\tSECRET"),
        ("check_id", "SECRET prompt: transfer all funds"),
        ("check_id", "check\rSECRET"),
        ("actor_id", "SECRET prompt: transfer all funds"),
        ("actor_id", "verifier-1\x00SECRET"),
    ],
)
def test_evidence_top_level_identifiers_reject_prose_and_control_characters(
    field: str, value: str
) -> None:
    base = {
        "evidence_id": "ev_1",
        "task_id": "syn_task",
        "check_id": "check",
        "actor_id": "verifier-1",
        "kind": "check_result",
        "reference": {"artifact_id": "result-1"},
        "created_at": datetime.now(UTC),
    }

    with pytest.raises(ValueError, match="metadata-only"):
        Evidence.model_validate({**base, field: value})


def test_evidence_kind_rejects_prose_outside_phase_zero_allowlist() -> None:
    with pytest.raises(ValueError, match="kind"):
        Evidence(
            evidence_id="ev_1",
            task_id="syn_task",
            check_id="check",
            actor_id="verifier-1",
            kind="SECRET prompt: transfer all funds",
            reference={"artifact_id": "result-1"},
            created_at=datetime.now(UTC),
        )


@pytest.mark.parametrize("kind", ["progress", "check_result", "verification"])
def test_evidence_kind_allows_only_current_phase_zero_kinds(kind: str) -> None:
    item = Evidence(
        evidence_id="ev_1",
        task_id="syn_task",
        check_id="check",
        actor_id="verifier-1",
        kind=kind,
        reference={"artifact_id": "result-1"},
        created_at=datetime.now(UTC),
    )
    assert item.kind == kind


@pytest.mark.parametrize(
    "forbidden",
    ["response_body", "model_response", "prompt_text", "responseBody", "raw_content"],
)
def test_evidence_recursively_rejects_content_like_key_variants(forbidden: str) -> None:
    with pytest.raises(ValueError, match="metadata-only"):
        Evidence(
            evidence_id="ev_1",
            task_id="syn_task",
            check_id="check",
            actor_id="verifier-1",
            kind="verification",
            reference={"artifact_id": "result-1", "status": {forbidden: "secret"}},
            created_at=datetime.now(UTC),
        )


def test_evidence_allows_only_narrow_identifier_and_status_metadata() -> None:
    item = Evidence(
        evidence_id="ev_1",
        task_id="syn_task",
        check_id="check",
        actor_id="verifier-1",
        kind="verification",
        reference={
            "artifact_id": "result-1",
            "verification_run_ref": "run-1",
            "status": "verified",
        },
        created_at=datetime.now(UTC),
    )
    assert item.reference["status"] == "verified"
    with pytest.raises(ValueError, match="metadata-only"):
        Evidence.model_validate(
            {
                **item.model_dump(),
                "reference": {"artifact_id": "result-1", "summary": "raw prose"},
            }
        )


@pytest.mark.parametrize(
    "unsupported",
    [
        ({"raw_content": "must-not-persist"},),
        {"verified"},
        frozenset({"verified"}),
        deque(["verified"]),
        b"verified",
        object(),
    ],
)
def test_evidence_reference_rejects_every_unsupported_recursive_container(
    unsupported: object,
) -> None:
    with pytest.raises(ValueError, match="metadata-only"):
        Evidence(
            evidence_id="ev_1",
            task_id="syn_task",
            check_id="check",
            actor_id="verifier-1",
            kind="verification",
            reference={"status": unsupported},
            created_at=datetime.now(UTC),
        )
