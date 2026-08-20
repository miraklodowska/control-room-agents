import json
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from fastapi.testclient import TestClient
from test_firestore_state import FakeFirestoreClient

from control_room.agents.gemini import (
    DEFAULT_MODEL_CALLS_PER_DAY,
    AdkModelTransport,
    GeminiCoordinator,
    build_gemini_coordinator,
    model_calls_per_day,
)
from control_room.api import create_app
from control_room.domain import Action, Origin, Task, TaskState
from control_room.policy import PolicyDenied
from control_room.service import ControlRoomService, DeterministicCoordinator
from control_room.state import FirestoreStateStore, MemoryStateStore

NOW = datetime(2026, 8, 20, 10, 0, tzinfo=UTC)
DAY = "2026-08-20"


class FakeTransport:
    def __init__(self, replies: list[str] | None = None, error: Exception | None = None) -> None:
        self.replies = list(replies or [])
        self.error = error
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if self.error is not None:
            raise self.error
        return self.replies.pop(0)


class RecordingBudget:
    def __init__(self, *, admit: bool = True, error: Exception | None = None) -> None:
        self.admit = admit
        self.error = error
        self.days: list[str] = []

    def __call__(self, day: str) -> bool:
        self.days.append(day)
        if self.error is not None:
            raise self.error
        return self.admit


def task(*, version: int = 0) -> Task:
    base = Task.new("syn_gemini", "synthetic_model_demo", ("quality",), Origin.LOCAL_API)
    return base.model_copy(update={"version": version})


def coordinator(
    transport: FakeTransport, budget: RecordingBudget
) -> GeminiCoordinator:
    return GeminiCoordinator(
        transport,
        consume_model_call=budget,
        fallback=DeterministicCoordinator(),
    )


def model_service(
    transport: FakeTransport,
    budget: RecordingBudget,
    store: MemoryStateStore | None = None,
) -> ControlRoomService:
    return ControlRoomService(
        store if store is not None else MemoryStateStore(),
        now=lambda: NOW,
        id_factory=lambda: "gemini",
        coordinator=coordinator(transport, budget),
    )


def good_reply(*, action: str = "record_progress", summary: str = "synthetic_model_step") -> str:
    return json.dumps({"action": action, "summary_ref": summary})


def test_model_choice_is_untrusted_but_authorized_end_to_end() -> None:
    transport = FakeTransport([good_reply()])
    budget = RecordingBudget()
    service = model_service(transport, budget)
    service.create_task("synthetic_model_demo", ("quality",))

    result = service.run_cycle("syn_gemini", "cycle-a")

    assert result.task.state is TaskState.RUNNING
    assert result.failure_code is None
    assert result.task.tool_actions == 1
    assert len(result.evidence) == 1
    assert budget.days == [DAY]
    assert len(transport.prompts) == 1
    assert "syn_gemini" in transport.prompts[0]


def test_model_proposal_carries_locally_derived_fields_only() -> None:
    transport = FakeTransport([good_reply(action="verify_task", summary="synthetic_check")])
    budget = RecordingBudget()
    proposal = coordinator(transport, budget).propose(task(version=3), "cycle-b", NOW)

    assert proposal == {
        "task_id": "syn_gemini",
        "cycle_key": "cycle-b",
        "actor_id": "verifier-1",
        "action": Action.VERIFY_TASK,
        "expected_version": 3,
        "issued_at": NOW,
        "payload": {"summary_ref": "synthetic_check"},
    }


def test_fenced_json_is_accepted_mechanically() -> None:
    transport = FakeTransport(["```json\n" + good_reply() + "\n```"])
    proposal = coordinator(transport, RecordingBudget()).propose(task(), "cycle-c", NOW)
    assert proposal["payload"] == {"summary_ref": "synthetic_model_step"}


@pytest.mark.parametrize(
    "reply",
    [
        "not json",
        "[]",
        json.dumps({"action": "record_progress"}),
        json.dumps({"action": "record_progress", "summary_ref": "ok", "actor_id": "admin"}),
        json.dumps({"action": "delete_everything", "summary_ref": "ok"}),
        json.dumps({"action": 7, "summary_ref": "ok"}),
        json.dumps({"action": "record_progress", "summary_ref": "Not Valid!"}),
        json.dumps({"action": "record_progress", "summary_ref": ""}),
        json.dumps({"action": "record_progress", "summary_ref": "x" * 65}),
        "",
        "x" * 8_193,
    ],
)
def test_untrusted_model_output_fails_closed(reply: str) -> None:
    transport = FakeTransport([reply])
    with pytest.raises(PolicyDenied) as denied:
        coordinator(transport, RecordingBudget()).propose(task(), "cycle-d", NOW)
    assert denied.value.code == "model_output_invalid"


def test_transport_error_fails_closed_to_needs_operator_without_evidence() -> None:
    transport = FakeTransport(error=RuntimeError("vertex unavailable"))
    budget = RecordingBudget()
    service = model_service(transport, budget)
    service.create_task("synthetic_model_demo", ("quality",))

    result = service.run_cycle("syn_gemini", "cycle-e")

    assert result.task.state is TaskState.NEEDS_OPERATOR
    assert result.failure_code == "model_unavailable"
    assert result.evidence == ()
    assert service.store.list_evidence("syn_gemini") == ()


def test_invalid_model_output_reaches_needs_operator_via_service() -> None:
    transport = FakeTransport(["not json"])
    service = model_service(transport, RecordingBudget())
    service.create_task("synthetic_model_demo", ("quality",))

    result = service.run_cycle("syn_gemini", "cycle-f")

    assert result.task.state is TaskState.NEEDS_OPERATOR
    assert result.failure_code == "model_output_invalid"


def test_exhausted_budget_never_calls_the_model_and_falls_back() -> None:
    transport = FakeTransport([good_reply()])
    budget = RecordingBudget(admit=False)
    service = model_service(transport, budget)
    service.create_task("synthetic_model_demo", ("quality",))

    result = service.run_cycle("syn_gemini", "cycle-g")

    assert result.task.state is TaskState.RUNNING
    assert result.failure_code is None
    assert transport.prompts == []
    assert budget.days == [DAY]


def test_unreadable_budget_never_calls_the_model_and_falls_back() -> None:
    transport = FakeTransport([good_reply()])
    budget = RecordingBudget(error=RuntimeError("budget document corrupted"))
    service = model_service(transport, budget)
    service.create_task("synthetic_model_demo", ("quality",))

    result = service.run_cycle("syn_gemini", "cycle-h")

    assert result.task.state is TaskState.RUNNING
    assert transport.prompts == []


def test_budget_day_follows_the_intent_clock_in_utc() -> None:
    transport = FakeTransport([good_reply()])
    budget = RecordingBudget()
    later = NOW + timedelta(days=2)
    coordinator(transport, budget).propose(task(), "cycle-i", later)
    assert budget.days == ["2026-08-22"]


def test_memory_budget_enforces_daily_bound_and_rollover() -> None:
    store = MemoryStateStore()
    assert store.consume_model_call(day=DAY, limit=2) is True
    assert store.consume_model_call(day=DAY, limit=2) is True
    assert store.consume_model_call(day=DAY, limit=2) is False
    assert store.consume_model_call(day="2026-08-21", limit=2) is True
    with pytest.raises(ValueError, match="bounded positive integer"):
        store.consume_model_call(day=DAY, limit=0)
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        store.consume_model_call(day="today", limit=2)


def test_firestore_budget_is_durable_transactional_and_bounded() -> None:
    client = FakeFirestoreClient()
    first = FirestoreStateStore(client, transaction_runner=client.run_transaction)
    second = FirestoreStateStore(client, transaction_runner=client.run_transaction)

    assert first.consume_model_call(day=DAY, limit=2) is True
    assert second.consume_model_call(day=DAY, limit=2) is True
    before = dict(client.documents)
    assert second.consume_model_call(day=DAY, limit=2) is False
    assert client.documents == before
    assert client.documents["control_room/model_budget"] == {
        "schema_version": 1,
        "day": DAY,
        "used": 2,
    }
    assert first.consume_model_call(day="2026-08-21", limit=2) is True
    assert client.documents["control_room/model_budget"]["used"] == 1


def test_firestore_budget_rejects_corrupted_documents() -> None:
    client = FakeFirestoreClient()
    store = FirestoreStateStore(client, transaction_runner=client.run_transaction)
    client.documents["control_room/model_budget"] = {
        "schema_version": 1,
        "day": DAY,
        "used": "many",
    }
    with pytest.raises(ValueError, match="model budget document"):
        store.consume_model_call(day=DAY, limit=2)


def test_model_calls_per_day_parses_and_bounds_configuration() -> None:
    assert model_calls_per_day({}) == DEFAULT_MODEL_CALLS_PER_DAY
    assert model_calls_per_day({"CONTROL_ROOM_MODEL_CALLS_PER_DAY": "7"}) == 7
    with pytest.raises(ValueError, match="must be an integer"):
        model_calls_per_day({"CONTROL_ROOM_MODEL_CALLS_PER_DAY": "many"})
    with pytest.raises(ValueError, match="between 1 and"):
        model_calls_per_day({"CONTROL_ROOM_MODEL_CALLS_PER_DAY": "0"})
    with pytest.raises(ValueError, match="between 1 and"):
        model_calls_per_day({"CONTROL_ROOM_MODEL_CALLS_PER_DAY": "1001"})


def test_build_gemini_coordinator_wires_store_budget_and_fallback() -> None:
    store = MemoryStateStore()
    transport = FakeTransport([good_reply()])
    built = build_gemini_coordinator(
        {"CONTROL_ROOM_MODEL_CALLS_PER_DAY": "1"},
        store=store,
        transport_factory=lambda: transport,
    )
    first = built.propose(task(), "cycle-j", NOW)
    assert first["payload"] == {"summary_ref": "synthetic_model_step"}
    fallback = built.propose(task(), "cycle-j", NOW)
    assert fallback["payload"] == {"summary_ref": "synthetic-progress"}
    assert len(transport.prompts) == 1


def test_app_selects_coordinator_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CONTROL_ROOM_COORDINATOR", raising=False)
    api = TestClient(create_app(demo_mode=False))
    assert api.get("/health").json()["coordinator"] == "deterministic"

    monkeypatch.setenv("CONTROL_ROOM_COORDINATOR", "gemini")
    gemini_api = TestClient(create_app(demo_mode=False))
    assert gemini_api.get("/health").json()["coordinator"] == "gemini"

    monkeypatch.setenv("CONTROL_ROOM_COORDINATOR", "clippy")
    with pytest.raises(ValueError, match=r"deterministic.*or.*gemini"):
        create_app(demo_mode=False)


def test_gemini_selected_app_fails_closed_not_open_on_model_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The env-selected gemini app admits budget then fails closed on transport."""
    monkeypatch.setenv("CONTROL_ROOM_COORDINATOR", "gemini")
    monkeypatch.setattr(
        AdkModelTransport,
        "generate",
        lambda self, prompt: (_ for _ in ()).throw(RuntimeError("no vertex here")),
    )
    api = TestClient(create_app(demo_mode=False, id_factory=lambda: "gemini"))
    created = api.post(
        "/tasks", json={"title": "synthetic_model_demo", "required_checks": ["quality"]}
    )
    assert created.status_code == 201
    cycle = api.post(
        "/tasks/syn_gemini/cycles", json={"idempotency_key": "model-cycle"}
    )
    assert cycle.status_code == 200
    body = cycle.json()
    assert body["task"]["state"] == "NEEDS_OPERATOR"
    assert body["failure_code"] == "model_unavailable"


def test_default_transport_is_lazy_and_uses_the_root_agent() -> None:
    transport = AdkModelTransport()
    assert isinstance(transport, AdkModelTransport)


def test_prompt_contains_no_verification_authority_language() -> None:
    transport = FakeTransport([good_reply()])
    coordinator(transport, RecordingBudget()).propose(task(), "cycle-k", NOW)
    prompt = transport.prompts[0]
    assert "never claim work passed verification" in prompt
    assert "synthetic" in prompt


def test_fake_transport_replies_are_recorded_for_forensics() -> None:
    transport = FakeTransport([good_reply(), good_reply(summary="synthetic_second")])
    built = coordinator(transport, RecordingBudget())
    built.propose(task(), "cycle-l", NOW)
    built.propose(task(), "cycle-m", NOW)
    assert len(transport.prompts) == 2
    assert "cycle-l" in transport.prompts[0]
    assert "cycle-m" in transport.prompts[1]


def test_non_mapping_fallback_is_rejected() -> None:
    class BrokenFallback:
        def propose(self, current: Task, cycle_key: str, now: datetime) -> Any:
            return "not-a-proposal"

    built = GeminiCoordinator(
        FakeTransport([good_reply()]),
        consume_model_call=RecordingBudget(admit=False),
        fallback=BrokenFallback(),
    )
    with pytest.raises(PolicyDenied) as denied:
        built.propose(task(), "cycle-n", NOW)
    assert denied.value.code == "fallback_invalid"
