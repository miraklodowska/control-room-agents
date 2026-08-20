from datetime import UTC, datetime, timedelta

import pytest

from control_room.domain import Action, ActorRole, Evidence, Intent, Origin, Task, TaskState
from control_room.policy import PolicyDenied, authorize_intent, require_pass_evidence

NOW = datetime(2026, 8, 17, 10, 0, tzinfo=UTC)


def task(**updates: object) -> Task:
    base = Task.new("syn_task", "demo", ("quality", "safety"), Origin.LOCAL_API)
    return base.model_copy(update=updates)


def raw_intent(**updates: object) -> dict[str, object]:
    value: dict[str, object] = {
        "task_id": "syn_task",
        "cycle_key": "cycle-1",
        "actor_id": "worker-1",
        "action": Action.RECORD_PROGRESS,
        "expected_version": 0,
        "issued_at": NOW,
        "payload": {"summary_ref": "one-bounded-action"},
    }
    value.update(updates)
    return value


def deny(raw: dict[str, object], current: Task, role: ActorRole) -> PolicyDenied:
    with pytest.raises(PolicyDenied) as caught:
        authorize_intent(raw, current, role, now=NOW)
    assert caught.value.target_state is TaskState.NEEDS_OPERATOR
    return caught.value


def test_valid_typed_intent_is_authorized() -> None:
    parsed = authorize_intent(raw_intent(), task(), ActorRole.WORKER, now=NOW)
    assert isinstance(parsed, Intent)


def test_unknown_action_and_malformed_output_are_denied() -> None:
    assert deny(raw_intent(action="delete_task"), task(), ActorRole.WORKER).code == "malformed"
    assert deny({"not": "an intent"}, task(), ActorRole.WORKER).code == "malformed"


def test_stale_intent_is_denied() -> None:
    stale = NOW - timedelta(seconds=301)
    assert deny(raw_intent(issued_at=stale), task(), ActorRole.WORKER).code == "stale_intent"


def test_intent_exactly_300_seconds_old_is_stale() -> None:
    boundary = NOW - timedelta(seconds=300)
    assert (
        deny(raw_intent(issued_at=boundary), task(), ActorRole.WORKER).code
        == "stale_intent"
    )


def test_actor_action_mismatch_is_denied() -> None:
    assert (
        deny(raw_intent(action=Action.VERIFY_TASK), task(), ActorRole.WORKER).code
        == "actor_action_mismatch"
    )
    assert (
        deny(raw_intent(), task(), ActorRole.VERIFIER).code == "actor_action_mismatch"
    )


def test_stale_version_is_denied() -> None:
    assert deny(raw_intent(expected_version=1), task(), ActorRole.WORKER).code == "stale_version"


def test_excess_steps_are_denied() -> None:
    raw = raw_intent(payload={"steps": ["one", "two"]})
    assert deny(raw, task(), ActorRole.WORKER).code == "excess_steps"


def test_caller_and_proposed_cycle_keys_must_match() -> None:
    with pytest.raises(PolicyDenied) as caught:
        authorize_intent(
            raw_intent(cycle_key="coordinator-key"),
            task(),
            ActorRole.WORKER,
            now=NOW,
            cycle_key="caller-key",
        )
    assert caught.value.code == "cycle_key_mismatch"
    assert caught.value.target_state is TaskState.NEEDS_OPERATOR


@pytest.mark.parametrize(
    ("mutation", "failure_code"),
    [
        ({"prompt_text": "raw model prompt"}, "malformed"),
        ({"raw_content": "raw model response"}, "malformed"),
        ({"steps": ["one", "two"]}, "excess_steps"),
    ],
)
def test_existing_intent_payload_mutation_cannot_bypass_authorization(
    mutation: dict[str, object], failure_code: str
) -> None:
    parsed = Intent.model_validate(raw_intent(payload={"summary_ref": "safe"}))
    parsed.payload.update(mutation)

    with pytest.raises(PolicyDenied) as caught:
        authorize_intent(parsed, task(), ActorRole.WORKER, now=NOW)

    assert caught.value.code == failure_code


def test_budget_exhaustion_halts() -> None:
    assert (
        deny(raw_intent(), task(tool_actions=3), ActorRole.WORKER).code
        == "budget_exhausted"
    )
    assert (
        deny(raw_intent(), task(failed_actions=1), ActorRole.WORKER).code
        == "budget_exhausted"
    )


def check_evidence(check_id: str, actor_id: str = "verifier-1") -> Evidence:
    return Evidence(
        evidence_id=f"ev_{check_id}",
        task_id="syn_task",
        check_id=check_id,
        actor_id=actor_id,
        kind="verification",
        reference={"status": "verified", "artifact_id": f"artifact-{check_id}"},
        created_at=NOW,
    )


def test_worker_cannot_self_verify() -> None:
    running = task(state=TaskState.RUNNING, worker_actor_id="worker-1")
    evidence = (check_evidence("quality", "worker-1"), check_evidence("safety", "worker-1"))
    with pytest.raises(PolicyDenied, match="distinct"):
        require_pass_evidence(running, evidence, verifier_id="worker-1")


def test_missing_evidence_forbids_pass() -> None:
    running = task(state=TaskState.RUNNING, worker_actor_id="worker-1")
    with pytest.raises(PolicyDenied, match="missing evidence"):
        require_pass_evidence(running, (check_evidence("quality"),), verifier_id="verifier-1")


def test_all_required_checks_and_distinct_verifier_allow_pass() -> None:
    running = task(state=TaskState.RUNNING, worker_actor_id="worker-1")
    evidence = (check_evidence("quality"), check_evidence("safety"))
    require_pass_evidence(running, evidence, verifier_id="verifier-1")
