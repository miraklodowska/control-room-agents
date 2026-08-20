import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from threading import Barrier, Event, Lock, RLock, get_ident
from typing import Any

import pytest
from fastapi.testclient import TestClient

from control_room.api import create_app
from control_room.domain import Action, ActionResult, Intent, Origin, Task, TaskState
from control_room.policy import PolicyDenied
from control_room.service import ControlRoomService
from control_room.state import MemoryBacking, MemoryStateStore

NOW = datetime(2026, 8, 17, 10, 0, tzinfo=UTC)


def client(
    *,
    backing: MemoryBacking | None = None,
    demo_mode: bool = False,
    coordinator: object | None = None,
) -> tuple[TestClient, MemoryBacking]:
    shared = backing if backing is not None else MemoryBacking()
    app = create_app(
        backing=shared,
        demo_mode=demo_mode,
        now=lambda: NOW,
        id_factory=lambda: "fixed123",
        coordinator=coordinator,
    )
    return TestClient(app), shared


def create_task(api: TestClient) -> dict[str, object]:
    response = api.post(
        "/tasks", json={"title": "Synthetic handoff", "required_checks": ["quality"]}
    )
    assert response.status_code == 201
    return response.json()


def test_health_endpoint_and_create_task_enforce_synthetic_provenance() -> None:
    api, _ = client()
    dashboard = api.get("/")
    assert dashboard.status_code == 200
    assert "Control Room" in dashboard.text
    assert api.get("/health").json() == {
        "status": "ok",
        "demo_mode": False,
        "state_backend": "memory",
        "coordinator": "deterministic",
    }
    created = create_task(api)
    assert created["task_id"] == "syn_fixed123"
    assert created["origin"] == "LOCAL_API"
    assert created["state"] == "QUEUED"
    spoof = api.post(
        "/tasks",
        json={
            "title": "spoof",
            "required_checks": ["quality"],
            "origin": "EXTERNAL",
        },
    )
    assert spoof.status_code == 422


def test_create_task_request_accepts_exact_bounds() -> None:
    api, _ = client()
    response = api.post(
        "/tasks",
        json={
            "title": "t" * 64,
            "required_checks": ["c" * 64, *[f"check-{index}" for index in range(7)]],
        },
    )

    assert response.status_code == 201


@pytest.mark.parametrize(
    "payload",
    [
        {"title": "t" * 65, "required_checks": ["quality"]},
        {
            "title": "bounded",
            "required_checks": [f"check-{index}" for index in range(9)],
        },
        {"title": "bounded", "required_checks": ["c" * 65]},
    ],
)
def test_create_task_request_rejects_excess_bounds_without_echoing_input(
    payload: dict[str, object],
) -> None:
    api, _ = client()
    response = api.post("/tasks", json=payload)

    assert response.status_code == 422
    assert "input" not in response.json()["detail"][0]
    for value in payload.values():
        if isinstance(value, str) and len(value) > 64:
            assert value not in response.text
        if isinstance(value, list):
            for item in value:
                if len(value) > 8 or len(item) > 64:
                    assert item not in response.text


def test_home_exposes_minimal_operator_dashboard_contract() -> None:
    api, _ = client()
    response = api.get("/")
    assert response.status_code == 200
    html = response.text
    for element_id in (
        'id="task-id"',
        'id="load-task"',
        'id="task-state"',
        'id="evidence"',
        'id="tool-budget"',
        'id="failure-budget"',
        'id="breaker-status"',
    ):
        assert element_id in html
    assert "fetch(`/tasks/${taskId}`)" in html
    assert "fetch(`/tasks/${taskId}/evidence`)" in html
    assert "tool_actions" in html
    assert "failed_actions" in html


@pytest.mark.parametrize("demo_mode", [False, True])
def test_dashboard_reports_breaker_availability_only_from_health(
    demo_mode: bool,
) -> None:
    api, _ = client(demo_mode=demo_mode)
    html = api.get("/").text

    assert "Breaker availability" in html
    assert 'fetch("/health")' in html
    assert 'health.demo_mode ? "DEMO ENABLED" : "DEMO DISABLED"' in html
    assert html.count('text("breaker-status"') == 1
    assert 'task.state === "NEEDS_OPERATOR"' not in html
    assert '"TRIPPED"' not in html
    assert '"CLEAR"' not in html


def test_execute_one_cycle_status_evidence_idempotency_and_baton_persist() -> None:
    api, _ = client()
    created = create_task(api)
    task_id = str(created["task_id"])
    first = api.post(f"/tasks/{task_id}/cycles", json={"idempotency_key": "cycle-1"})
    assert first.status_code == 200
    assert first.json()["task"]["state"] == "RUNNING"
    assert first.json()["task"]["tool_actions"] == 1
    replay = api.post(f"/tasks/{task_id}/cycles", json={"idempotency_key": "cycle-1"})
    assert replay.status_code == 200
    assert replay.json() == first.json()
    status = api.get(f"/tasks/{task_id}").json()
    assert status["task"]["tool_actions"] == 1
    cycle_key = sha256(b"cycle-1").hexdigest()
    assert status["baton"] == {"last_cycle_key": cycle_key, "next_step": "verify"}
    evidence = api.get(f"/tasks/{task_id}/evidence").json()
    assert len(evidence) == 1
    assert evidence[0]["kind"] == "progress"
    assert evidence[0]["reference"] == {"cycle_key": cycle_key, "status": "recorded"}


def test_caller_idempotency_key_is_opaque_everywhere_and_replays() -> None:
    raw_key = "prompt=approve transfer; secret=correct-horse-battery-staple"
    other_raw_key = "prompt=approve transfer; secret=different"
    expected = sha256(raw_key.encode()).hexdigest()
    other_expected = sha256(other_raw_key.encode()).hexdigest()
    api, backing = client()
    task_id = str(create_task(api)["task_id"])

    first = api.post(f"/tasks/{task_id}/cycles", json={"idempotency_key": raw_key})
    replay = api.post(f"/tasks/{task_id}/cycles", json={"idempotency_key": raw_key})
    second = api.post(
        f"/tasks/{task_id}/cycles", json={"idempotency_key": other_raw_key}
    )

    assert first.status_code == replay.status_code == second.status_code == 200
    assert replay.json() == first.json()
    assert expected != other_expected
    serialized_outputs = json.dumps(
        {
            "first": first.json(),
            "replay": replay.json(),
            "second": second.json(),
            "status": api.get(f"/tasks/{task_id}").json(),
            "evidence": api.get(f"/tasks/{task_id}/evidence").json(),
            "telemetry": api.app.state.service.telemetry.events,
        },
        default=str,
        sort_keys=True,
    )
    internal_records = repr(
        (backing.tasks, backing.evidence, backing.cycles, backing.batons)
    )
    for secret in (raw_key, other_raw_key):
        assert secret not in serialized_outputs
        assert secret not in internal_records
    for digest in (expected, other_expected):
        assert digest in serialized_outputs
        assert digest in internal_records


@pytest.mark.parametrize("raw_key", ["", "   ", "x" * 257])
def test_caller_idempotency_key_must_be_nonblank_and_bounded(raw_key: str) -> None:
    api, _ = client()
    task_id = str(create_task(api)["task_id"])
    response = api.post(
        f"/tasks/{task_id}/cycles", json={"idempotency_key": raw_key}
    )
    assert response.status_code == 422
    if raw_key:
        assert raw_key not in response.text


def test_caller_idempotency_key_accepts_exactly_256_characters() -> None:
    raw_key = "k" * 256
    api, _ = client()
    task_id = str(create_task(api)["task_id"])
    response = api.post(
        f"/tasks/{task_id}/cycles", json={"idempotency_key": raw_key}
    )
    assert response.status_code == 200
    assert raw_key not in response.text


def test_task_status_registry_and_shared_backing_persist_across_app_instances() -> None:
    first, backing = client()
    created = create_task(first)
    task_id = str(created["task_id"])
    first.post(f"/tasks/{task_id}/cycles", json={"idempotency_key": "cycle-1"})
    second, _ = client(backing=backing)
    assert second.get(f"/tasks/{task_id}").json()["task"]["state"] == "RUNNING"
    registry = second.get("/registry").json()
    assert registry == {
        "coordinator-1": "coordinator",
        "verifier-1": "verifier",
        "worker-1": "worker",
    }
    assert second.get(f"/tasks/{task_id}").json()["baton"]["next_step"] == "verify"


def test_task_status_endpoint_cannot_mix_task_and_baton_snapshots() -> None:
    api, backing = client()
    task_id = str(create_task(api)["task_id"])
    store = MemoryStateStore(backing)
    first_release = Event()
    writer_done = Event()

    class HandoffRLock:
        def __init__(self) -> None:
            self._lock = RLock()
            self._depths: dict[int, int] = {}
            self._gate_used = False
            self._metadata_lock = Lock()

        def __enter__(self) -> "HandoffRLock":
            self._lock.acquire()
            identity = get_ident()
            with self._metadata_lock:
                self._depths[identity] = self._depths.get(identity, 0) + 1
            return self

        def __exit__(
            self,
            _exc_type: object,
            _exc_value: object,
            _traceback: object,
        ) -> None:
            identity = get_ident()
            with self._metadata_lock:
                depth = self._depths[identity] - 1
                if depth:
                    self._depths[identity] = depth
                else:
                    del self._depths[identity]
                handoff = not depth and not self._gate_used
                if handoff:
                    self._gate_used = True
            self._lock.release()
            if handoff:
                first_release.set()
                writer_done.wait()

    backing.lock = HandoffRLock()  # type: ignore[assignment]

    def write_post_snapshot_state() -> None:
        first_release.wait()
        store.set_baton(task_id, {"next_step": "post"}, expected_version=0)
        writer_done.set()

    with ThreadPoolExecutor(max_workers=2) as executor:
        writer = executor.submit(write_post_snapshot_state)
        response = executor.submit(api.get, f"/tasks/{task_id}").result()
        writer.result()

    assert response.status_code == 200
    assert response.json()["task"]["version"] == 0
    assert response.json()["baton"] is None
    assert store.get_status(task_id) == (
        store.get_task(task_id),
        {"next_step": "post"},
    )


def test_concurrent_same_cycle_across_shared_service_instances_executes_once() -> None:
    backing = MemoryBacking()
    lookup_barrier = Barrier(2)
    start_barrier = Barrier(3)

    class CoordinatedLookupStore(MemoryStateStore):
        def get_cycle_result(
            self,
            task_id: str,
            cycle_key: str,
            *,
            canonical_operation: str | None = None,
        ) -> ActionResult | None:
            result = super().get_cycle_result(
                task_id, cycle_key, canonical_operation=canonical_operation
            )
            lookup_barrier.wait()
            return result

    class CountingCoordinator:
        def __init__(self) -> None:
            self.calls = 0
            self._lock = Lock()

        def propose(self, task: Task, cycle_key: str, now: datetime) -> dict[str, Any]:
            with self._lock:
                self.calls += 1
            return {
                "task_id": task.task_id,
                "cycle_key": cycle_key,
                "actor_id": "worker-1",
                "action": Action.RECORD_PROGRESS,
                "expected_version": task.version,
                "issued_at": now,
                "payload": {"summary_ref": "synthetic-progress"},
            }

    coordinator = CountingCoordinator()
    services = tuple(
        ControlRoomService(
            CoordinatedLookupStore(backing),
            now=lambda: NOW,
            id_factory=lambda: "unused",
            coordinator=coordinator,
        )
        for _ in range(2)
    )
    task_id = services[0].create_task("Concurrent", ("quality",)).task_id

    def run(service: ControlRoomService) -> object:
        start_barrier.wait()
        return service.run_cycle(task_id, "same-caller-key")

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(run, service) for service in services]
        start_barrier.wait()
        results = tuple(future.result() for future in futures)

    assert results[0] == results[1]
    assert coordinator.calls == 1
    assert MemoryStateStore(backing).get_task(task_id).tool_actions == 1
    assert len(MemoryStateStore(backing).list_evidence(task_id)) == 1
    assert sum(len(service.telemetry.events) for service in services) == 1


def test_concurrent_same_cycle_across_shared_app_instances_replays_identically() -> None:
    backing = MemoryBacking()
    start_barrier = Barrier(3)

    class CountingCoordinator:
        def __init__(self) -> None:
            self.calls = 0
            self._lock = Lock()

        def propose(self, task: Task, cycle_key: str, now: datetime) -> dict[str, Any]:
            with self._lock:
                self.calls += 1
            return {
                "task_id": task.task_id,
                "cycle_key": cycle_key,
                "actor_id": "worker-1",
                "action": Action.RECORD_PROGRESS,
                "expected_version": task.version,
                "issued_at": now,
                "payload": {"summary_ref": "synthetic-progress"},
            }

    coordinator = CountingCoordinator()
    first, _ = client(backing=backing, coordinator=coordinator)
    second, _ = client(backing=backing, coordinator=coordinator)
    task_id = str(create_task(first)["task_id"])

    def request(api: TestClient) -> object:
        start_barrier.wait()
        return api.post(
            f"/tasks/{task_id}/cycles", json={"idempotency_key": "same-app-key"}
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(request, api) for api in (first, second)]
        start_barrier.wait()
        responses = tuple(future.result() for future in futures)

    assert all(response.status_code == 200 for response in responses)
    assert responses[0].json() == responses[1].json()
    assert coordinator.calls == 1
    assert MemoryStateStore(backing).get_task(task_id).tool_actions == 1
    assert len(MemoryStateStore(backing).list_evidence(task_id)) == 1


def test_concurrent_breaker_across_shared_services_executes_and_emits_once() -> None:
    backing = MemoryBacking()
    lookup_barrier = Barrier(2)
    start_barrier = Barrier(3)

    class CoordinatedLookupStore(MemoryStateStore):
        def get_cycle_result(
            self,
            task_id: str,
            cycle_key: str,
            *,
            canonical_operation: str | None = None,
        ) -> ActionResult | None:
            result = super().get_cycle_result(
                task_id, cycle_key, canonical_operation=canonical_operation
            )
            lookup_barrier.wait()
            return result

    services = tuple(
        ControlRoomService(
            CoordinatedLookupStore(backing),
            now=lambda: NOW,
            id_factory=lambda: "unused",
        )
        for _ in range(2)
    )
    task_id = services[0].create_task("Concurrent breaker", ("quality",)).task_id

    def trip(service: ControlRoomService) -> ActionResult:
        start_barrier.wait()
        return service.trip_demo_breaker(task_id, "same-breaker-key")

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(trip, service) for service in services]
        start_barrier.wait()
        results = tuple(future.result() for future in futures)

    assert results[0] == results[1]
    assert results[0].task.tool_actions == 1
    assert results[0].task.failed_actions == 1
    assert sum(len(service.telemetry.events) for service in services) == 1
    assert MemoryStateStore(backing).get_task(task_id) == results[0].task


def test_concurrent_service_initialization_is_atomic_and_idempotent() -> None:
    backing = MemoryBacking()
    version_barrier = Barrier(2)
    start_barrier = Barrier(3)

    class CoordinatedRegistryStore(MemoryStateStore):
        def __init__(self, backing: MemoryBacking) -> None:
            super().__init__(backing)
            self._first_version_read = True

        def get_registry_version(self) -> int:
            version = super().get_registry_version()
            if self._first_version_read:
                self._first_version_read = False
                version_barrier.wait()
            return version

    def construct() -> ControlRoomService:
        start_barrier.wait()
        return ControlRoomService(
            CoordinatedRegistryStore(backing),
            now=lambda: NOW,
            id_factory=lambda: "unused",
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(construct) for _ in range(2)]
        start_barrier.wait()
        services = tuple(future.result() for future in futures)

    assert len(services) == 2
    store = MemoryStateStore(backing)
    assert store.list_actors() == {
        "coordinator-1": "coordinator",
        "verifier-1": "verifier",
        "worker-1": "worker",
    }
    assert store.get_registry_version() == 3


def test_breaker_route_is_absent_by_default() -> None:
    api, _ = client(demo_mode=False)
    created = create_task(api)
    response = api.post(
        f"/demo/tasks/{created['task_id']}/breaker",
        json={"idempotency_key": "break-1"},
    )
    assert response.status_code == 404


@pytest.mark.parametrize(
    ("demo_mode", "route"),
    [
        (False, "/tasks/{task_id}/cycles"),
        (True, "/demo/tasks/{task_id}/breaker"),
    ],
)
def test_policy_denied_from_injected_store_maps_to_conflict(
    demo_mode: bool, route: str
) -> None:
    class PolicyDenyingStore(MemoryStateStore):
        def run_cycle_transaction(
            self, *_args: object, **_kwargs: object
        ) -> ActionResult:
            raise PolicyDenied("injected_denial", "injected policy denial")

    app = create_app(
        store=PolicyDenyingStore(),
        demo_mode=demo_mode,
        now=lambda: NOW,
        id_factory=lambda: "fixed123",
    )
    api = TestClient(app, raise_server_exceptions=False)
    task_id = str(create_task(api)["task_id"])

    response = api.post(
        route.format(task_id=task_id),
        json={"idempotency_key": "cycle-1"},
    )

    assert response.status_code == 409
    assert response.json() == {"detail": "injected policy denial"}


def test_breaker_rejections_make_no_mutation() -> None:
    api, backing = client(demo_mode=True)
    store = MemoryStateStore(backing)
    non_synthetic = Task.new("plain_task", "plain", ("quality",), Origin.LOCAL_API)
    spoofed = Task.new("syn_spoof", "spoof", ("quality",), Origin.EXTERNAL)
    store.create_task(non_synthetic)
    store.create_task(spoofed)

    unknown = api.post(
        "/demo/tasks/syn_unknown/breaker", json={"idempotency_key": "break-1"}
    )
    assert unknown.status_code == 404
    for stored in (non_synthetic, spoofed):
        before = store.get_task(stored.task_id)
        response = api.post(
            f"/demo/tasks/{stored.task_id}/breaker",
            json={"idempotency_key": "break-1"},
        )
        assert response.status_code == 403
        assert store.get_task(stored.task_id) == before
        assert store.list_evidence(stored.task_id) == ()
        assert store.get_baton(stored.task_id) is None


def test_breaker_accepts_only_stored_local_synthetic_task_and_fails_atomically() -> None:
    api, _ = client(demo_mode=True)
    created = create_task(api)
    task_id = str(created["task_id"])
    response = api.post(
        f"/demo/tasks/{task_id}/breaker", json={"idempotency_key": "break-1"}
    )
    assert response.status_code == 200
    result = response.json()
    assert result["failure_code"] == "demo_missing_evidence"
    assert result["task"]["state"] == "NEEDS_OPERATOR"
    assert result["task"]["tool_actions"] == 1
    assert result["task"]["failed_actions"] == 1
    replay = api.post(
        f"/demo/tasks/{task_id}/breaker", json={"idempotency_key": "break-1"}
    )
    assert replay.status_code == 200
    assert replay.json() == result


def test_breaker_rejects_terminal_task_without_mutation() -> None:
    api, backing = client(demo_mode=True)
    store = MemoryStateStore(backing)
    terminal = Task.new("syn_terminal", "done", ("quality",), Origin.LOCAL_API).model_copy(
        update={"state": TaskState.FAILED}
    )
    store.create_task(terminal)
    response = api.post(
        "/demo/tasks/syn_terminal/breaker", json={"idempotency_key": "break-late"}
    )
    assert response.status_code == 409
    assert store.get_task("syn_terminal") == terminal
    assert store.list_evidence("syn_terminal") == ()
    assert store.get_baton("syn_terminal") is None
    assert store.get_cycle_result("syn_terminal", "break-late") is None


def test_normal_cycle_rejects_terminal_task_without_mutation() -> None:
    api, backing = client()
    store = MemoryStateStore(backing)
    terminal = Task.new("syn_terminal", "done", ("quality",), Origin.LOCAL_API).model_copy(
        update={"state": TaskState.FAILED}
    )
    store.create_task(terminal)
    response = api.post(
        "/tasks/syn_terminal/cycles", json={"idempotency_key": "cycle-late"}
    )
    assert response.status_code == 409
    assert store.get_task("syn_terminal") == terminal
    assert store.list_evidence("syn_terminal") == ()
    assert store.get_cycle_result("syn_terminal", "cycle-late") is None


class StaticCoordinator:
    def __init__(self, proposal: dict[str, Any]) -> None:
        self.proposal = proposal

    def propose(self, task: Task, cycle_key: str, now: datetime) -> dict[str, Any]:
        return dict(self.proposal)


class CountingCoordinator:
    def __init__(self, proposal: object) -> None:
        self.proposal = proposal
        self.calls = 0

    def propose(self, task: Task, cycle_key: str, now: datetime) -> object:
        self.calls += 1
        return self.proposal


def denied_proposal(**updates: object) -> dict[str, object]:
    proposal: dict[str, object] = {
        "task_id": "syn_fixed123",
        "cycle_key": sha256(b"denied-1").hexdigest(),
        "actor_id": "worker-1",
        "action": Action.RECORD_PROGRESS,
        "expected_version": 0,
        "issued_at": NOW,
        "payload": {"summary_ref": "progress-1"},
    }
    proposal.update(updates)
    return proposal


@pytest.mark.parametrize(
    "unsafe_payload",
    [
        {"summary_ref": object()},
        {"status": (object(),)},
        {"status": ({"raw_content": "must-not-escape"},)},
        {"status": frozenset({"recorded"})},
    ],
)
def test_non_json_or_content_like_coordinator_payload_fails_before_admission(
    unsafe_payload: dict[str, object],
) -> None:
    cycle_key = sha256(b"unsafe-payload").hexdigest()
    constructed = Intent.model_construct(
        task_id="syn_fixed123",
        cycle_key=cycle_key,
        actor_id="worker-1",
        action=Action.RECORD_PROGRESS,
        expected_version=0,
        issued_at=NOW,
        payload=unsafe_payload,
    )
    coordinator = CountingCoordinator(constructed)
    api, backing = client(coordinator=coordinator)
    task_id = str(create_task(api)["task_id"])

    first = api.post(
        f"/tasks/{task_id}/cycles", json={"idempotency_key": "unsafe-payload"}
    )
    replay = api.post(
        f"/tasks/{task_id}/cycles", json={"idempotency_key": "unsafe-payload"}
    )

    assert first.status_code == replay.status_code == 200
    assert replay.json() == first.json()
    assert first.json()["failure_code"] == "malformed"
    assert first.json()["task"]["state"] == "NEEDS_OPERATOR"
    assert first.json()["task"]["tool_actions"] == 0
    assert first.json()["task"]["failed_actions"] == 0
    assert coordinator.calls == 1
    assert len(backing.cycles) == 1
    assert "raw_content" not in first.text


def test_sensitive_status_from_untrusted_coordinator_never_reaches_canonical_persistence() -> None:
    secret = "SECRET prompt: transfer all funds"
    proposal = denied_proposal(
        cycle_key=sha256(b"secret-status").hexdigest(), payload={"status": secret}
    )
    api, backing = client(coordinator=StaticCoordinator(proposal))
    task_id = str(create_task(api)["task_id"])

    response = api.post(
        f"/tasks/{task_id}/cycles", json={"idempotency_key": "secret-status"}
    )

    assert response.status_code == 200
    assert response.json()["failure_code"] == "malformed"
    assert response.json()["task"]["state"] == "NEEDS_OPERATOR"
    assert response.json()["task"]["tool_actions"] == 0
    assert secret not in response.text
    record = next(iter(backing.cycles.values()))
    assert record.canonical_intent is None
    assert secret not in repr(backing)


@pytest.mark.parametrize(
    ("proposal", "failure_code"),
    [
        ({"not": "an intent"}, "malformed"),
        (denied_proposal(task_id="syn_other"), "task_mismatch"),
        (denied_proposal(action=Action.VERIFY_TASK), "actor_action_mismatch"),
        (denied_proposal(actor_id="verifier-1"), "actor_action_mismatch"),
        (denied_proposal(actor_id="unknown-1"), "unknown_actor"),
        (denied_proposal(issued_at=NOW - timedelta(seconds=300)), "stale_intent"),
        (denied_proposal(expected_version=1), "stale_version"),
        (denied_proposal(payload={"steps": ["one", "two"]}), "excess_steps"),
    ],
)
def test_malformed_and_policy_denied_cycles_persist_fail_closed(
    proposal: dict[str, Any], failure_code: str
) -> None:
    api, backing = client(coordinator=StaticCoordinator(proposal))
    created = create_task(api)
    task_id = str(created["task_id"])
    response = api.post(f"/tasks/{task_id}/cycles", json={"idempotency_key": "denied-1"})
    assert response.status_code == 200
    result = response.json()
    assert result["failure_code"] == failure_code
    assert result["task"]["state"] == "NEEDS_OPERATOR"
    assert result["task"]["tool_actions"] == 0
    assert result["task"]["failed_actions"] == 0
    assert MemoryStateStore(backing).get_task(task_id).state is TaskState.NEEDS_OPERATOR
    replay = api.post(f"/tasks/{task_id}/cycles", json={"idempotency_key": "denied-1"})
    assert replay.status_code == 200
    assert replay.json() == result


def test_cycle_key_mismatch_fails_before_admission_and_replays_exactly() -> None:
    proposal = denied_proposal(cycle_key="coordinator-key")
    api, backing = client(coordinator=StaticCoordinator(proposal))
    task_id = str(create_task(api)["task_id"])

    first = api.post(
        f"/tasks/{task_id}/cycles", json={"idempotency_key": "caller-key"}
    )

    assert first.status_code == 200
    result = first.json()
    assert result["failure_code"] == "cycle_key_mismatch"
    assert result["task"]["state"] == "NEEDS_OPERATOR"
    assert result["task"]["tool_actions"] == 0
    assert result["task"]["failed_actions"] == 0
    assert MemoryStateStore(backing).get_task(task_id).model_dump(mode="json") == result["task"]
    replay = api.post(
        f"/tasks/{task_id}/cycles", json={"idempotency_key": "caller-key"}
    )
    assert replay.status_code == 200
    assert replay.json() == result


class RaisingCoordinator:
    def propose(self, task: Task, cycle_key: str, now: datetime) -> dict[str, Any]:
        raise RuntimeError("coordinator failed")


class NonMappingCoordinator:
    def propose(self, task: Task, cycle_key: str, now: datetime) -> Any:
        return ["not", "an", "intent"]


def test_non_mapping_model_output_is_persisted_as_malformed() -> None:
    api, backing = client(coordinator=NonMappingCoordinator())
    task_id = str(create_task(api)["task_id"])
    response = api.post(f"/tasks/{task_id}/cycles", json={"idempotency_key": "shape-1"})
    assert response.status_code == 200
    assert response.json()["failure_code"] == "malformed"
    assert MemoryStateStore(backing).get_task(task_id).state is TaskState.NEEDS_OPERATOR


def test_coordinator_exception_persists_idempotent_fail_closed_result() -> None:
    api, backing = client(coordinator=RaisingCoordinator())
    task_id = str(create_task(api)["task_id"])
    response = api.post(f"/tasks/{task_id}/cycles", json={"idempotency_key": "crash-1"})
    assert response.status_code == 200
    result = response.json()
    assert result["failure_code"] == "coordinator_exception"
    assert result["task"]["state"] == "NEEDS_OPERATOR"
    assert result["task"]["tool_actions"] == 0
    assert result["task"]["failed_actions"] == 0
    assert MemoryStateStore(backing).get_task(task_id).state is TaskState.NEEDS_OPERATOR
    replay = api.post(f"/tasks/{task_id}/cycles", json={"idempotency_key": "crash-1"})
    assert replay.status_code == 200
    assert replay.json() == result


def test_budget_policy_denial_persists_fail_closed_without_admission() -> None:
    backing = MemoryBacking()
    store = MemoryStateStore(backing)
    exhausted = Task.new(
        "syn_budget", "budget", ("quality",), Origin.LOCAL_API
    ).model_copy(update={"state": TaskState.RUNNING, "tool_actions": 3})
    store.create_task(exhausted)
    proposal = denied_proposal(
        task_id="syn_budget", cycle_key=sha256(b"budget-1").hexdigest()
    )
    api, _ = client(backing=backing, coordinator=StaticCoordinator(proposal))
    response = api.post(
        "/tasks/syn_budget/cycles", json={"idempotency_key": "budget-1"}
    )
    assert response.status_code == 200
    result = response.json()
    assert result["failure_code"] == "budget_exhausted"
    assert result["task"]["state"] == "NEEDS_OPERATOR"
    assert result["task"]["tool_actions"] == 3
    assert result["task"]["failed_actions"] == 0
    replay = api.post(
        "/tasks/syn_budget/cycles", json={"idempotency_key": "budget-1"}
    )
    assert replay.status_code == 200
    assert replay.json() == result


@pytest.mark.parametrize("first_operation", ["normal", "breaker"])
def test_same_key_for_normal_and_breaker_operations_conflicts(
    first_operation: str,
) -> None:
    api, backing = client(demo_mode=True)
    task_id = str(create_task(api)["task_id"])
    paths = {
        "normal": f"/tasks/{task_id}/cycles",
        "breaker": f"/demo/tasks/{task_id}/breaker",
    }
    first = api.post(paths[first_operation], json={"idempotency_key": "shared-key"})
    assert first.status_code == 200
    before = MemoryStateStore(backing).get_task(task_id)
    other = "breaker" if first_operation == "normal" else "normal"
    conflict = api.post(paths[other], json={"idempotency_key": "shared-key"})
    assert conflict.status_code == 409
    assert "different operation" in conflict.json()["detail"]
    assert MemoryStateStore(backing).get_task(task_id) == before
    replay = api.post(paths[first_operation], json={"idempotency_key": "shared-key"})
    assert replay.status_code == 200
    assert replay.json() == first.json()
