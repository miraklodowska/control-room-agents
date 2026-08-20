from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from control_room.domain import Action, ActorRole, Evidence, Intent, Origin, Task, TaskState
from control_room.policy import PolicyDenied
from control_room.state import (
    ActionFailed,
    BudgetExhausted,
    CycleOutcome,
    EvidenceConflict,
    IdempotencyConflict,
    MemoryBacking,
    MemoryStateStore,
    VersionConflict,
)

NOW = datetime(2026, 8, 17, 10, 0, tzinfo=UTC)


def task(task_id: str = "syn_task") -> Task:
    return Task.new(task_id, "demo", ("check",), Origin.LOCAL_API)


def intent(*, cycle_key: str = "cycle-1", version: int = 0, summary: str = "ok") -> Intent:
    return Intent(
        task_id="syn_task",
        cycle_key=cycle_key,
        actor_id="worker-1",
        action=Action.RECORD_PROGRESS,
        expected_version=version,
        issued_at=NOW,
        payload={"summary_ref": summary},
    )


def evidence(evidence_id: str = "ev_1") -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        task_id="syn_task",
        check_id="check",
        actor_id="verifier-1",
        kind="check_result",
        reference={"artifact_id": "result-1"},
        created_at=NOW,
    )


def verifier_intent(*, version: int = 1) -> Intent:
    return Intent(
        task_id="syn_task",
        cycle_key="verify-1",
        actor_id="verifier-1",
        action=Action.VERIFY_TASK,
        expected_version=version,
        issued_at=NOW,
        payload={"verification_run_id": "run-1"},
    )


def register_cycle_actors(store: MemoryStateStore) -> None:
    store.register_actor("worker-1", ActorRole.WORKER, expected_version=0)
    store.register_actor("verifier-1", ActorRole.VERIFIER, expected_version=1)


def registered_store() -> MemoryStateStore:
    store = MemoryStateStore()
    register_cycle_actors(store)
    return store


def test_append_only_evidence() -> None:
    store = MemoryStateStore()
    store.create_task(task())
    updated = store.append_evidence(evidence(), expected_version=0)
    assert updated.version == 1
    assert store.list_evidence("syn_task") == (evidence(),)
    with pytest.raises(EvidenceConflict):
        store.append_evidence(evidence(), expected_version=1)
    assert store.list_evidence("syn_task") == (evidence(),)


def test_optimistic_version_check_has_no_mutation() -> None:
    store = MemoryStateStore()
    store.create_task(task())
    with pytest.raises(VersionConflict):
        store.append_evidence(evidence(), expected_version=9)
    assert store.get_task("syn_task") == task()
    assert store.list_evidence("syn_task") == ()


def test_duplicate_cycle_key_is_idempotent() -> None:
    store = registered_store()
    store.create_task(task())
    calls = 0

    def execute(_admitted: Task) -> CycleOutcome:
        nonlocal calls
        calls += 1
        return CycleOutcome(target_state=TaskState.RUNNING)

    first = store.apply_cycle(intent(), execute)
    second = store.apply_cycle(intent(), execute)
    assert second == first
    assert calls == 1
    assert store.get_task("syn_task").tool_actions == 1


def test_conflicting_reuse_fails_without_mutation() -> None:
    store = registered_store()
    store.create_task(task())
    store.apply_cycle(intent(), lambda _: CycleOutcome(target_state=TaskState.RUNNING))
    before = store.get_task("syn_task")
    with pytest.raises(IdempotencyConflict):
        store.apply_cycle(
            intent(version=before.version, summary="different"),
            lambda _: CycleOutcome(target_state=TaskState.RUNNING),
        )
    assert store.get_task("syn_task") == before


def test_canonical_operation_never_replaces_canonical_intent() -> None:
    store = registered_store()
    store.create_task(task())
    operation = '{"kind":"normal","task_id":"syn_task","cycle_key":"cycle-1"}'
    first = store.apply_cycle(
        intent(),
        lambda _: CycleOutcome(target_state=TaskState.RUNNING),
        canonical_operation=operation,
    )
    changed = intent(version=first.task.version, summary="different")

    with pytest.raises(IdempotencyConflict, match="different intent"):
        store.apply_cycle(
            changed,
            lambda _: pytest.fail("conflicting intent must not execute"),
            canonical_operation=operation,
        )

    assert store.get_task("syn_task") == first.task


def test_shared_backing_and_fresh_store_isolation() -> None:
    backing = MemoryBacking()
    first = MemoryStateStore(backing)
    second = MemoryStateStore(backing)
    isolated = MemoryStateStore()
    first.create_task(task())
    first.set_baton("syn_task", {"next_step": "step-1"}, expected_version=0)
    first.register_actor("worker-1", "worker", expected_version=0)
    assert second.get_task("syn_task").version == 1
    assert second.get_baton("syn_task") == {"next_step": "step-1"}
    assert second.get_actor_role("worker-1") == "worker"
    with pytest.raises(KeyError):
        isolated.get_task("syn_task")


def test_task_status_is_one_defensive_snapshot() -> None:
    store = MemoryStateStore()
    store.create_task(task())
    store.set_baton("syn_task", {"steps": ["step-1"]}, expected_version=0)

    status_task, status_baton = store.get_task_status("syn_task")

    assert status_task.version == 1
    assert status_baton == {"steps": ["step-1"]}
    assert status_baton is not None
    status_baton["steps"][0] = "tampered"
    assert store.get_status("syn_task") == (
        status_task,
        {"steps": ["step-1"]},
    )


def test_admitted_action_increments_tool_count_once() -> None:
    store = registered_store()
    store.create_task(task())
    result = store.apply_cycle(
        intent(), lambda admitted: CycleOutcome(target_state=admitted.state)
    )
    assert result.task.tool_actions == 1
    assert result.task.version == 1


def test_failed_action_increments_failure_count_once() -> None:
    store = registered_store()
    store.create_task(task())
    result = store.apply_cycle(
        intent(), lambda _: CycleOutcome.failed("typed_failure")
    )
    assert result.task.failed_actions == 1
    assert result.failure_code == "typed_failure"
    assert store.apply_cycle(intent(), lambda _: pytest.fail("must replay")) == result


def test_raised_action_failure_increments_failure_count_once() -> None:
    store = registered_store()
    store.create_task(task())

    def fail(_admitted: Task) -> CycleOutcome:
        raise ActionFailed("raised_failure")

    result = store.apply_cycle(intent(), fail)
    assert result.task.failed_actions == 1
    assert result.failure_code == "raised_failure"


def test_malformed_raised_action_failure_is_atomic_sanitized_and_idempotent() -> None:
    store = registered_store()
    original = task()
    store.create_task(original)
    calls = 0

    def fail(_admitted: Task) -> CycleOutcome:
        nonlocal calls
        calls += 1
        raise ActionFailed({"raw_content": "must-not-persist"})

    try:
        result = store.apply_cycle(intent(), fail)
    except ValidationError as exc:
        assert store.get_task("syn_task") == original
        assert store.get_cycle_result("syn_task", "cycle-1") is None
        assert calls == 1
        pytest.fail(f"Pydantic ValidationError escaped guarded apply_cycle: {exc}")

    assert result.failure_code == "invalid_outcome"
    assert result.task.state is TaskState.NEEDS_OPERATOR
    assert result.task.tool_actions == 1
    assert result.task.failed_actions == 1
    assert result.evidence == ()
    assert store.list_evidence("syn_task") == ()
    assert store.get_baton("syn_task") is None
    assert "raw_content" not in result.model_dump_json()
    assert store.apply_cycle(intent(), fail) == result
    assert calls == 1


def test_first_failure_atomically_moves_task_to_needs_operator() -> None:
    store = registered_store()
    store.create_task(task())
    result = store.apply_cycle(intent(), lambda _: CycleOutcome.failed("failed"))
    assert result.task.state is TaskState.NEEDS_OPERATOR
    assert result.task.tool_actions == 1
    assert result.task.failed_actions == 1


def test_exhausted_budget_denies_before_execution() -> None:
    store = MemoryStateStore()
    exhausted = task().model_copy(update={"tool_actions": 3})
    store.create_task(exhausted)
    executed = False

    def execute(_admitted: Task) -> CycleOutcome:
        nonlocal executed
        executed = True
        return CycleOutcome(target_state=TaskState.RUNNING)

    with pytest.raises(BudgetExhausted):
        store.apply_cycle(intent(), execute)
    assert executed is False
    assert store.get_task("syn_task") == exhausted


def test_worker_cycle_cannot_authorize_pass_even_with_verifier_evidence() -> None:
    store = MemoryStateStore()
    store.create_task(task())
    register_cycle_actors(store)
    result = store.apply_cycle(
        intent(),
        lambda _: CycleOutcome(target_state=TaskState.PASSED, evidence=(evidence(),)),
    )
    assert result.failure_code == "pass_requires_verifier"
    assert result.task.state is TaskState.NEEDS_OPERATOR
    assert store.list_evidence("syn_task") == ()


def test_pass_requires_typed_verification_evidence_from_registered_verifier() -> None:
    store = MemoryStateStore()
    store.create_task(task())
    register_cycle_actors(store)
    running = store.apply_cycle(
        intent(), lambda _: CycleOutcome(target_state=TaskState.RUNNING)
    ).task

    denied = store.apply_cycle(
        verifier_intent(version=running.version),
        lambda _: CycleOutcome(target_state=TaskState.PASSED, evidence=(evidence(),)),
    )
    assert denied.failure_code == "missing_evidence"
    assert denied.task.state is TaskState.NEEDS_OPERATOR

    store = MemoryStateStore()
    store.create_task(task())
    register_cycle_actors(store)
    running = store.apply_cycle(
        intent(), lambda _: CycleOutcome(target_state=TaskState.RUNNING)
    ).task
    verified = evidence().model_copy(
        update={
            "kind": "verification",
            "reference": {"artifact_id": "result-1", "status": "verified"},
        }
    )
    result = store.apply_cycle(
        verifier_intent(version=running.version),
        lambda _: CycleOutcome(target_state=TaskState.PASSED, evidence=(verified,)),
    )
    assert result.task.state is TaskState.PASSED
    assert result.task.worker_actor_id == "worker-1"


@pytest.mark.parametrize(
    "reference",
    [
        {"artifact_id": "result-1"},
        {"artifact_id": "result-1", "status": "failed"},
        {"artifact_id": "result-1", "status": "ok"},
        {"artifact_id": "result-1", "status": True},
        {"artifact_id": "result-1", "status": {"code": "verified"}},
    ],
)
def test_pass_rejects_verification_without_exact_verified_status(
    reference: dict[str, object],
) -> None:
    store = registered_store()
    store.create_task(task())
    running = store.apply_cycle(
        intent(), lambda _: CycleOutcome(target_state=TaskState.RUNNING)
    ).task
    unverified = evidence().model_copy(
        update={"kind": "verification", "reference": reference}
    )

    result = store.apply_cycle(
        verifier_intent(version=running.version),
        lambda _: CycleOutcome(target_state=TaskState.PASSED, evidence=(unverified,)),
    )

    assert result.failure_code == "invalid_outcome"
    assert result.task.state is TaskState.NEEDS_OPERATOR
    assert result.task.tool_actions == 2
    assert result.task.failed_actions == 1
    assert store.list_evidence("syn_task") == ()


@pytest.mark.parametrize("terminal", [TaskState.PASSED, TaskState.FAILED])
def test_failure_outcome_cannot_mutate_terminal_task(terminal: TaskState) -> None:
    store = MemoryStateStore()
    terminal_task = task().model_copy(update={"state": terminal})
    store.create_task(terminal_task)
    with pytest.raises(ValueError, match="terminal"):
        store.apply_cycle(intent(), lambda _: CycleOutcome.failed("late_failure"))
    assert store.get_task("syn_task") == terminal_task
    assert store.list_evidence("syn_task") == ()
    assert store.get_cycle_result("syn_task", "cycle-1") is None


@pytest.mark.parametrize("terminal", [TaskState.PASSED, TaskState.FAILED])
@pytest.mark.parametrize("operation", ["append_evidence", "set_baton"])
def test_direct_task_mutations_reject_terminal_without_any_mutation(
    terminal: TaskState, operation: str
) -> None:
    store = MemoryStateStore()
    terminal_task = task().model_copy(update={"state": terminal, "version": 7})
    store.create_task(terminal_task)

    with pytest.raises(ValueError, match="terminal"):
        if operation == "append_evidence":
            store.append_evidence(evidence(), expected_version=7)
        else:
            store.set_baton("syn_task", {"next_step": "forbidden"}, expected_version=7)

    assert store.get_task("syn_task") == terminal_task
    assert store.list_evidence("syn_task") == ()
    assert store.get_baton("syn_task") is None


def test_failed_target_without_failure_code_is_counted_and_replayed_as_invalid() -> None:
    store = registered_store()
    store.create_task(task())

    result = store.apply_cycle(
        intent(), lambda _: CycleOutcome(target_state=TaskState.FAILED)
    )

    assert result.failure_code == "invalid_outcome"
    assert result.task.state is TaskState.NEEDS_OPERATOR
    assert result.task.tool_actions == 1
    assert result.task.failed_actions == 1
    assert store.get_task("syn_task") == result.task
    assert store.apply_cycle(intent(), lambda _: pytest.fail("must replay")) == result


def test_any_execution_exception_is_one_atomic_idempotent_failed_action() -> None:
    store = registered_store()
    store.create_task(task())
    calls = 0

    def explode(_admitted: Task) -> CycleOutcome:
        nonlocal calls
        calls += 1
        raise RuntimeError("untyped worker failure")

    result = store.apply_cycle(intent(), explode)
    assert result.failure_code == "action_exception"
    assert result.task.state is TaskState.NEEDS_OPERATOR
    assert result.task.tool_actions == 1
    assert result.task.failed_actions == 1
    assert store.list_evidence("syn_task") == ()
    assert store.get_baton("syn_task") is None
    assert store.apply_cycle(intent(), explode) == result
    assert calls == 1


def test_constructed_outcome_with_non_string_failure_code_is_atomic_and_idempotent() -> None:
    store = registered_store()
    store.create_task(task())
    calls = 0

    def malformed(_admitted: Task) -> CycleOutcome:
        nonlocal calls
        calls += 1
        return CycleOutcome.model_construct(
            target_state=TaskState.NEEDS_OPERATOR,
            evidence=(),
            baton=None,
            failure_code={"raw_content": "unvalidated"},
        )

    result = store.apply_cycle(intent(), malformed)

    assert result.failure_code == "invalid_outcome"
    assert result.task.state is TaskState.NEEDS_OPERATOR
    assert result.task.tool_actions == 1
    assert result.task.failed_actions == 1
    assert store.list_evidence("syn_task") == ()
    assert store.get_baton("syn_task") is None
    assert store.apply_cycle(intent(), malformed) == result
    assert calls == 1


def test_post_execution_evidence_validation_failure_is_atomic_and_idempotent() -> None:
    store = registered_store()
    store.create_task(task())
    wrong_task_evidence = evidence().model_copy(update={"task_id": "syn_other"})
    result = store.apply_cycle(
        intent(),
        lambda _: CycleOutcome(
            target_state=TaskState.RUNNING,
            evidence=(wrong_task_evidence,),
            baton={"next_step": "must-not-persist"},
        ),
    )
    assert result.failure_code == "invalid_outcome"
    assert result.task.state is TaskState.NEEDS_OPERATOR
    assert result.task.tool_actions == 1
    assert result.task.failed_actions == 1
    assert result.evidence == ()
    assert store.list_evidence("syn_task") == ()
    assert store.get_baton("syn_task") is None
    assert store.apply_cycle(intent(), lambda _: pytest.fail("must replay")) == result


def test_unknown_actor_and_verifier_cannot_execute_worker_action() -> None:
    store = MemoryStateStore()
    store.create_task(task())
    register_cycle_actors(store)
    executed = False

    def execute(_admitted: Task) -> CycleOutcome:
        nonlocal executed
        executed = True
        return CycleOutcome(target_state=TaskState.RUNNING)

    for actor_id in ("unknown-1", "verifier-1"):
        unauthorized = intent().model_copy(update={"actor_id": actor_id})
        with pytest.raises(PolicyDenied):
            store.apply_cycle(unauthorized, execute)
        assert store.get_task("syn_task") == task()
    assert executed is False


def test_verifier_action_does_not_overwrite_worker_actor_id() -> None:
    store = MemoryStateStore()
    store.create_task(task())
    register_cycle_actors(store)
    running = store.apply_cycle(
        intent(), lambda _: CycleOutcome(target_state=TaskState.RUNNING)
    ).task
    result = store.apply_cycle(
        verifier_intent(version=running.version),
        lambda _: CycleOutcome(target_state=TaskState.RUNNING),
    )
    assert result.task.worker_actor_id == "worker-1"


def test_callers_cannot_mutate_stored_nested_evidence_reference() -> None:
    store = MemoryStateStore()
    store.create_task(task())
    item = evidence().model_copy(
        update={"reference": {"artifact_id": "result-1", "status": "verified"}}
    )
    store.append_evidence(item, expected_version=0)
    item.reference["status"] = "failed"
    first_read = store.list_evidence("syn_task")
    assert first_read[0].reference["status"] == "verified"
    first_read[0].reference["status"] = "failed"
    assert store.list_evidence("syn_task")[0].reference["status"] == "verified"


def test_store_revalidates_copied_evidence_that_bypassed_model_validation() -> None:
    store = registered_store()
    store.create_task(task())
    unsafe = evidence().model_copy(update={"reference": {"prompt_text": "secret"}})
    with pytest.raises(ValueError, match="metadata-only"):
        store.append_evidence(unsafe, expected_version=0)
    assert store.get_task("syn_task") == task()
    assert store.list_evidence("syn_task") == ()


@pytest.mark.parametrize("construction", ["copy", "construct"])
def test_append_evidence_revalidates_invalid_top_level_fields_before_mutation(
    construction: str,
) -> None:
    store = MemoryStateStore()
    original = task()
    store.create_task(original)
    secret = "SECRET prompt: transfer all funds"
    unsafe = (
        evidence().model_copy(update={"kind": secret})
        if construction == "copy"
        else Evidence.model_construct(
            evidence_id="ev_unsafe",
            task_id="syn_task",
            check_id="check\nSECRET",
            actor_id="verifier-1",
            kind="check_result",
            reference={"artifact_id": "result-1"},
            created_at=NOW,
        )
    )

    with pytest.raises(ValueError):
        store.append_evidence(unsafe, expected_version=0)

    assert store.get_task("syn_task") == original
    assert store.list_evidence("syn_task") == ()
    assert secret not in repr(store._backing)
    assert "check\nSECRET" not in repr(store._backing)


@pytest.mark.parametrize("construction", ["copy", "construct"])
def test_apply_cycle_revalidates_invalid_top_level_evidence_atomically(
    construction: str,
) -> None:
    store = registered_store()
    original = task()
    store.create_task(original)
    secret = "SECRET prompt: transfer all funds"
    unsafe = (
        evidence().model_copy(update={"kind": secret})
        if construction == "copy"
        else Evidence.model_construct(
            evidence_id="ev_unsafe",
            task_id="syn_task",
            check_id="check",
            actor_id="verifier-1\x00SECRET",
            kind="check_result",
            reference={"artifact_id": "result-1"},
            created_at=NOW,
        )
    )

    result = store.apply_cycle(
        intent(),
        lambda _: CycleOutcome(
            target_state=TaskState.RUNNING,
            evidence=(unsafe,),
            baton={"next_step": "must-not-persist"},
        ),
    )

    assert result.failure_code == "invalid_outcome"
    assert result.task.state is TaskState.NEEDS_OPERATOR
    assert result.task.version == original.version + 1
    assert result.task.tool_actions == original.tool_actions + 1
    assert result.task.failed_actions == original.failed_actions + 1
    assert result.evidence == ()
    assert store.list_evidence("syn_task") == ()
    assert store.get_baton("syn_task") is None
    assert secret not in repr(store._backing)
    assert "verifier-1\\x00SECRET" not in repr(store._backing)


@pytest.mark.parametrize(
    "field",
    ["status", "summary_ref", "artifact_id", "failure_code"],
)
def test_sensitive_prose_under_allowed_evidence_scalar_key_rejects_before_mutation(
    field: str,
) -> None:
    store = MemoryStateStore()
    original = task()
    store.create_task(original)
    secret = "SECRET prompt: transfer all funds"
    unsafe = evidence().model_construct(
        evidence_id="ev_secret",
        task_id="syn_task",
        check_id="check",
        actor_id="verifier-1",
        kind="check_result",
        reference={field: secret},
        created_at=NOW,
    )

    with pytest.raises(ValueError, match="metadata-only"):
        store.append_evidence(unsafe, expected_version=0)

    assert store.get_task("syn_task") == original
    assert store.list_evidence("syn_task") == ()
    assert secret not in repr(store._backing)


def test_sensitive_prose_in_direct_baton_rejects_before_mutation() -> None:
    store = MemoryStateStore()
    original = task()
    store.create_task(original)
    secret = "SECRET prompt: transfer all funds"

    with pytest.raises(ValueError, match="metadata-only"):
        store.set_baton("syn_task", {"next_step": secret}, expected_version=0)

    assert store.get_task("syn_task") == original
    assert store.get_baton("syn_task") is None
    assert secret not in repr(store._backing)


def test_sensitive_prose_in_outcome_baton_fails_closed_without_persistence() -> None:
    store = registered_store()
    store.create_task(task())
    secret = "SECRET prompt: transfer all funds"

    result = store.apply_cycle(
        intent(),
        lambda _: CycleOutcome.model_construct(
            target_state=TaskState.RUNNING,
            baton={"next_step": secret},
        ),
    )

    assert result.failure_code == "invalid_outcome"
    assert result.task.state is TaskState.NEEDS_OPERATOR
    assert store.get_baton("syn_task") is None
    assert secret not in repr(store._backing)


def test_tuple_wrapped_raw_content_never_serializes_or_persists_as_evidence() -> None:
    store = registered_store()
    store.create_task(task())
    unsafe = evidence().model_copy(
        update={"reference": {"status": ({"raw_content": "secret"},)}}
    )

    with pytest.raises(ValueError, match="metadata-only"):
        store.append_evidence(unsafe, expected_version=0)
    assert store.get_task("syn_task") == task()
    assert store.list_evidence("syn_task") == ()

    result = store.apply_cycle(
        intent(),
        lambda _: CycleOutcome(target_state=TaskState.RUNNING, evidence=(unsafe,)),
    )
    assert result.failure_code == "invalid_outcome"
    assert result.evidence == ()
    assert store.list_evidence("syn_task") == ()
    assert "raw_content" not in result.model_dump_json()

    result = store.apply_cycle(
        intent(),
        lambda _: CycleOutcome(target_state=TaskState.RUNNING, evidence=(unsafe,)),
    )
    assert result.failure_code == "invalid_outcome"
    assert result.task.tool_actions == 1
    assert result.task.failed_actions == 1
    assert store.list_evidence("syn_task") == ()


def test_baton_write_requires_matching_expected_version_without_mutation() -> None:
    store = MemoryStateStore()
    store.create_task(task())
    with pytest.raises(VersionConflict):
        store.set_baton("syn_task", {"next_step": "wrong"}, expected_version=9)
    assert store.get_task("syn_task") == task()
    assert store.get_baton("syn_task") is None
    updated = store.set_baton("syn_task", {"next_step": "step-1"}, expected_version=0)
    assert updated.version == 1
    assert store.get_baton("syn_task") == {"next_step": "step-1"}


def test_registry_write_requires_matching_expected_version_without_mutation() -> None:
    store = MemoryStateStore()
    assert store.get_registry_version() == 0
    with pytest.raises(VersionConflict):
        store.register_actor("worker-1", ActorRole.WORKER, expected_version=1)
    assert store.list_actors() == {}
    assert store.get_registry_version() == 0
    assert store.register_actor(
        "worker-1", ActorRole.WORKER, expected_version=0
    ) == 1
    assert store.get_registry_version() == 1
