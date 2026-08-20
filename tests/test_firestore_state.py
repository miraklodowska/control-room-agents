from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from importlib.metadata import version
from threading import Barrier, RLock
from typing import Any, TypeVar

import pytest
from fastapi.testclient import TestClient
from google.api_core.exceptions import Aborted

import control_room.state as state_module
from control_room.api import create_app
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
from control_room.service import ControlRoomService, opaque_cycle_key
from control_room.state import (
    CycleOutcome,
    EvidenceConflict,
    FirestoreStateStore,
    IdempotencyConflict,
    MemoryStateStore,
    VersionConflict,
    build_state_store,
)
from control_room.telemetry import TelemetrySink

NOW = datetime(2026, 8, 18, 10, 0, tzinfo=UTC)
T = TypeVar("T")


class FakeSnapshot:
    def __init__(self, value: dict[str, Any] | None) -> None:
        self.exists = value is not None
        self._value = deepcopy(value)

    def to_dict(self) -> dict[str, Any] | None:
        return deepcopy(self._value)


class FakeDocument:
    def __init__(self, client: FakeFirestoreClient, path: str) -> None:
        self._client = client
        self.path = path

    def get(self, *, transaction: FakeTransaction | None = None) -> FakeSnapshot:
        source = transaction.staged if transaction is not None else self._client.documents
        return FakeSnapshot(source.get(self.path))


class FakeCollection:
    def __init__(self, client: FakeFirestoreClient, name: str) -> None:
        self._client = client
        self._name = name

    def document(self, name: str) -> FakeDocument:
        return FakeDocument(self._client, f"{self._name}/{name}")


class FakeTransaction:
    def __init__(self, client: FakeFirestoreClient) -> None:
        self._client = client
        self.staged = deepcopy(client.documents)

    def set(self, document: FakeDocument, value: dict[str, Any]) -> None:
        self.staged[document.path] = deepcopy(value)

    def delete(self, document: FakeDocument) -> None:
        self.staged.pop(document.path, None)


class FakeFirestoreClient:
    def __init__(self) -> None:
        self.documents: dict[str, dict[str, Any]] = {}
        self._lock = RLock()

    def collection(self, name: str) -> FakeCollection:
        return FakeCollection(self, name)

    def run_transaction(self, callback: Callable[[FakeTransaction], T]) -> T:
        with self._lock:
            transaction = FakeTransaction(self)
            result = callback(transaction)
            self.documents = transaction.staged
            return result


class SharedBackend:
    def __init__(self) -> None:
        self.documents: dict[str, dict[str, Any]] = {}
        self.lock = RLock()


class DistinctFakeClient(FakeFirestoreClient):
    """A distinct client object sharing only the simulated Firestore backend."""

    def __init__(self, backend: SharedBackend) -> None:
        self._backend = backend
        self.documents = backend.documents
        self._lock = backend.lock

    def run_transaction(self, callback: Callable[[FakeTransaction], T]) -> T:
        with self._backend.lock:
            self.documents = self._backend.documents
            transaction = FakeTransaction(self)
            result = callback(transaction)
            self._backend.documents.clear()
            self._backend.documents.update(transaction.staged)
            self.documents = self._backend.documents
            return result


class AbortThenReplayRunner:
    """Replay one transaction callback after an aborted first commit."""

    def __init__(self, client: DistinctFakeClient) -> None:
        self.client = client
        self.callback_calls = 0

    def __call__(self, callback: Callable[[FakeTransaction], T]) -> T:
        with self.client._backend.lock:
            first = FakeTransaction(self.client)
            self.callback_calls += 1
            callback(first)
            replay = FakeTransaction(self.client)
            self.callback_calls += 1
            result = callback(replay)
            self.client._backend.documents.clear()
            self.client._backend.documents.update(replay.staged)
            return result


def make_store(client: FakeFirestoreClient) -> FirestoreStateStore:
    return FirestoreStateStore(client, transaction_runner=client.run_transaction)


def task() -> Task:
    return Task.new("syn_firestore", "synthetic_durable_demo", ("check",), Origin.LOCAL_API)


def evidence() -> Evidence:
    return Evidence(
        evidence_id="ev_firestore_1",
        task_id="syn_firestore",
        check_id="check",
        actor_id="worker-1",
        kind="progress",
        reference={"artifact_id": "result-1"},
        created_at=NOW,
    )


def intent(*, version: int = 0, summary: str = "ok") -> Intent:
    return Intent(
        task_id="syn_firestore",
        cycle_key="cycle-1",
        actor_id="worker-1",
        action=Action.RECORD_PROGRESS,
        expected_version=version,
        issued_at=NOW,
        payload={"summary_ref": summary},
    )


def test_firestore_persists_tasks_across_store_instances() -> None:
    client = FakeFirestoreClient()
    first = make_store(client)
    second = make_store(client)

    first.create_task(task())

    assert second.get_task("syn_firestore") == task()
    assert second.get_task_status("syn_firestore") == (task(), None)
    assert client.documents["control_room/counters"] == {
        "schema_version": 1,
        "created_tasks": 1,
    }


def test_firestore_global_task_cap_is_durable_across_store_instances() -> None:
    client = FakeFirestoreClient()
    client.documents["control_room/counters"] = {
        "schema_version": 1,
        "created_tasks": state_module.MAX_FIRESTORE_TASKS - 1,
    }
    first = make_store(client)
    second = make_store(client)

    first.create_task(task())

    assert client.documents["control_room/counters"]["created_tasks"] == (
        state_module.MAX_FIRESTORE_TASKS
    )
    rejected = Task.new(
        "syn_firestore_over_cap",
        "synthetic_over_cap",
        ("check",),
        Origin.LOCAL_API,
    )
    before = deepcopy(client.documents)
    with pytest.raises(ValueError, match="global task count bound"):
        second.create_task(rejected)

    assert client.documents == before
    assert "control_room/task__syn_firestore_over_cap" not in client.documents


def test_firestore_global_task_cap_maps_to_http_conflict_without_partial_write() -> None:
    client = FakeFirestoreClient()
    client.documents["control_room/counters"] = {
        "schema_version": 1,
        "created_tasks": state_module.MAX_FIRESTORE_TASKS,
    }
    store = make_store(client)
    app = create_app(
        store=store,
        demo_mode=False,
        now=lambda: NOW,
        id_factory=lambda: "over_cap",
    )
    api = TestClient(app)
    before = deepcopy(client.documents)

    response = api.post(
        "/tasks",
        json={"title": "synthetic_over_cap", "required_checks": ["check"]},
    )

    assert response.status_code == 409
    assert response.json() == {"detail": "global task count bound exceeded"}
    assert client.documents == before
    assert "control_room/task__syn_over_cap" not in client.documents


def test_firestore_optimistic_version_conflict_has_no_mutation() -> None:
    client = FakeFirestoreClient()
    store = make_store(client)
    store.create_task(task())

    with pytest.raises(VersionConflict):
        store.append_evidence(evidence(), expected_version=9)

    assert make_store(client).get_task("syn_firestore") == task()
    assert make_store(client).list_evidence("syn_firestore") == ()


def test_firestore_evidence_is_append_only_and_durable() -> None:
    client = FakeFirestoreClient()
    first = make_store(client)
    first.create_task(task())
    updated = first.append_evidence(evidence(), expected_version=0)

    second = make_store(client)
    assert updated.version == 1
    assert second.list_evidence("syn_firestore") == (evidence(),)
    with pytest.raises(EvidenceConflict):
        second.append_evidence(evidence(), expected_version=1)
    assert first.list_evidence("syn_firestore") == (evidence(),)


def test_firestore_cycle_idempotency_is_durable_and_conflicts_fail_closed() -> None:
    client = FakeFirestoreClient()
    first = make_store(client)
    first.create_task(task())
    first.register_actor("worker-1", ActorRole.WORKER, expected_version=0)
    calls = 0

    def execute(_admitted: Task) -> CycleOutcome:
        nonlocal calls
        calls += 1
        return CycleOutcome(target_state=TaskState.RUNNING, evidence=(evidence(),))

    result = first.apply_cycle(intent(), execute)
    second = make_store(client)
    assert second.apply_cycle(intent(), execute) == result
    assert calls == 1

    with pytest.raises(IdempotencyConflict):
        second.apply_cycle(
            intent(version=result.task.version, summary="changed"),
            execute,
        )

    assert first.get_task("syn_firestore") == result.task
    assert first.list_evidence("syn_firestore") == (evidence(),)
    assert calls == 1


def test_store_selection_defaults_to_memory() -> None:
    assert isinstance(build_state_store({}), MemoryStateStore)


@pytest.mark.parametrize("value", ["", "MEMORY", "redis", "fire-store"])
def test_store_selection_rejects_invalid_explicit_configuration(value: str) -> None:
    with pytest.raises(ValueError, match="CONTROL_ROOM_STATE_STORE"):
        build_state_store({"CONTROL_ROOM_STATE_STORE": value})


def test_firestore_selection_requires_an_explicit_project() -> None:
    with pytest.raises(ValueError, match="GOOGLE_CLOUD_PROJECT"):
        build_state_store({"CONTROL_ROOM_STATE_STORE": "firestore"})


def test_firestore_selection_injects_project_and_database() -> None:
    client = FakeFirestoreClient()
    calls: list[tuple[str, str]] = []

    def factory(*, project: str, database: str) -> FakeFirestoreClient:
        calls.append((project, database))
        return client

    store = build_state_store(
        {
            "CONTROL_ROOM_STATE_STORE": "firestore",
            "GOOGLE_CLOUD_PROJECT": "local-test-project",
            "CONTROL_ROOM_FIRESTORE_DATABASE": "control-room-test",
        },
        firestore_client_factory=factory,
        transaction_runner=client.run_transaction,
    )

    assert isinstance(store, FirestoreStateStore)
    assert calls == [("local-test-project", "control-room-test")]


def test_application_uses_environment_driven_store_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import control_room.api as api_module

    selected = MemoryStateStore()
    observed: list[dict[str, str]] = []

    def select(environ: dict[str, str]) -> MemoryStateStore:
        observed.append(dict(environ))
        return selected

    monkeypatch.setenv("CONTROL_ROOM_STATE_STORE", "memory")
    monkeypatch.setattr(api_module, "build_state_store", select)

    application = api_module.create_app()

    assert application.state.service.store is selected
    assert observed[0]["CONTROL_ROOM_STATE_STORE"] == "memory"


def test_firestore_aborted_claim_replay_does_not_repeat_effects() -> None:
    backend = SharedBackend()
    client = DistinctFakeClient(backend)
    runner = AbortThenReplayRunner(client)
    telemetry = TelemetrySink()

    class CountingCoordinator:
        calls = 0

        def propose(self, current: Task, cycle_key: str, now: datetime) -> dict[str, Any]:
            self.calls += 1
            return {
                "task_id": current.task_id,
                "cycle_key": cycle_key,
                "actor_id": "worker-1",
                "action": Action.RECORD_PROGRESS,
                "expected_version": current.version,
                "issued_at": now,
                "payload": {"summary_ref": "synthetic-progress"},
            }

    coordinator = CountingCoordinator()
    store = FirestoreStateStore(client, transaction_runner=runner)
    service = ControlRoomService(
        store,
        now=lambda: NOW,
        id_factory=lambda: "firestore",
        coordinator=coordinator,
        telemetry=telemetry,
    )
    service.create_task("synthetic_replay", ("quality",))

    result = service.run_cycle("syn_firestore", "same-key")

    assert runner.callback_calls >= 2
    assert coordinator.calls == 1
    assert len(telemetry.events) == 1
    assert result.task.tool_actions == 1
    assert len(store.list_evidence("syn_firestore")) == 1


def test_distinct_clients_same_key_execute_once_but_different_tasks_do_not_contend() -> None:
    backend = SharedBackend()
    clients = (DistinctFakeClient(backend), DistinctFakeClient(backend))
    coordinators: list[int] = []

    class CountingCoordinator:
        def propose(self, current: Task, cycle_key: str, now: datetime) -> dict[str, Any]:
            coordinators.append(1)
            return {
                "task_id": current.task_id,
                "cycle_key": cycle_key,
                "actor_id": "worker-1",
                "action": Action.RECORD_PROGRESS,
                "expected_version": current.version,
                "issued_at": now,
                "payload": {"summary_ref": "synthetic-progress"},
            }

    services = tuple(
        ControlRoomService(
            FirestoreStateStore(client, transaction_runner=client.run_transaction),
            now=lambda: NOW,
            id_factory=lambda: "unused",
            coordinator=CountingCoordinator(),
        )
        for client in clients
    )
    services[0].store.create_task(
        Task.new("syn_a", "synthetic_a", ("quality",), Origin.LOCAL_API)
    )
    services[0].store.create_task(
        Task.new("syn_b", "synthetic_b", ("quality",), Origin.LOCAL_API)
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        same = tuple(
            executor.map(
                lambda service: service.run_cycle("syn_a", "shared"), services
            )
        )
    assert same[0] == same[1]
    assert len(coordinators) == 1
    assert sum(len(service.telemetry.events) for service in services) == 1

    with ThreadPoolExecutor(max_workers=2) as executor:
        different = tuple(
            executor.map(
                lambda pair: pair[0].run_cycle(pair[1], "different"),
                zip(services, ("syn_a", "syn_b"), strict=True),
            )
        )
    assert {result.task.task_id for result in different} == {"syn_a", "syn_b"}
    assert len(coordinators) == 3


def test_distinct_clients_different_task_keys_reach_effects_concurrently() -> None:
    backend = SharedBackend()
    effect_barrier = Barrier(2)

    class BarrierCoordinator:
        def propose(self, current: Task, cycle_key: str, now: datetime) -> dict[str, Any]:
            effect_barrier.wait(timeout=1)
            return {
                "task_id": current.task_id,
                "cycle_key": cycle_key,
                "actor_id": "worker-1",
                "action": Action.RECORD_PROGRESS,
                "expected_version": current.version,
                "issued_at": now,
                "payload": {"summary_ref": "synthetic-progress"},
            }

    clients = (DistinctFakeClient(backend), DistinctFakeClient(backend))
    services = tuple(
        ControlRoomService(
            FirestoreStateStore(client, transaction_runner=client.run_transaction),
            now=lambda: NOW,
            id_factory=lambda: "unused",
            coordinator=BarrierCoordinator(),
        )
        for client in clients
    )
    for task_id in ("syn_parallel_a", "syn_parallel_b"):
        services[0].store.create_task(
            Task.new(task_id, f"synthetic_{task_id}", ("quality",), Origin.LOCAL_API)
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(
            executor.map(
                lambda pair: pair[0].run_cycle(pair[1], f"key-{pair[1]}"),
                zip(services, ("syn_parallel_a", "syn_parallel_b"), strict=True),
            )
        )

    assert all(result.task.state is TaskState.RUNNING for result in results)


def test_firestore_uses_bounded_per_task_and_registry_documents() -> None:
    client = FakeFirestoreClient()
    store = make_store(client)
    store.initialize_actors({"worker-1": ActorRole.WORKER})
    for index in range(40):
        store.create_task(
            Task.new(
                f"syn_{index}",
                f"synthetic_task_{index}",
                ("quality",),
                Origin.LOCAL_API,
            )
        )

    assert "control_room/state_v1" not in client.documents
    task_documents = {
        path: value for path, value in client.documents.items() if "/task__" in path
    }
    assert len(task_documents) == 40
    assert max(len(repr(value).encode()) for value in task_documents.values()) < 64_000
    assert any(path.endswith("/registry") for path in client.documents)


@pytest.mark.parametrize(
    "title,checks",
    [
        ("password=correct-horse", ("quality",)),
        ("Contact alice@example.com", ("quality",)),
        ("ordinary user prose", ("quality",)),
        ("synthetic_ok", ("send customer records",)),
        ("synthetic_ok", ("api_key_sk-live-secret",)),
    ],
)
def test_firestore_rejects_non_synthetic_public_fields_before_persistence(
    title: str, checks: tuple[str, ...]
) -> None:
    client = FakeFirestoreClient()
    store = make_store(client)
    unsafe = Task.new("syn_private", title, checks, Origin.LOCAL_API)

    with pytest.raises(ValueError, match="synthetic-only"):
        store.create_task(unsafe)

    assert not any("syn_private" in repr(value) for value in client.documents.values())


def test_cloud_run_requires_explicit_firestore_configuration() -> None:
    with pytest.raises(ValueError, match="Cloud Run requires Firestore"):
        build_state_store({"K_SERVICE": "control-room"})
    with pytest.raises(ValueError, match="Cloud Run requires Firestore"):
        build_state_store(
            {"K_SERVICE": "control-room", "CONTROL_ROOM_STATE_STORE": "memory"}
        )


def test_google_cloud_firestore_2281_transaction_wrapper_replays_only_callback() -> None:
    assert version("google-cloud-firestore") == "2.28.1"

    class WrapperTransaction:
        _read_only = False
        _max_attempts = 2

        def __init__(self) -> None:
            self._id: bytes | None = None
            self.commits = 0
            self.rollbacks = 0

        def _clean_up(self) -> None:
            self._id = None

        def _begin(self, retry_id: bytes | None = None) -> None:
            self._id = retry_id or b"transaction-1"

        def _commit(self) -> None:
            self.commits += 1
            if self.commits == 1:
                raise Aborted("synthetic aborted commit")
            self._clean_up()

        def _rollback(self) -> None:
            self.rollbacks += 1
            self._clean_up()

    transaction = WrapperTransaction()

    class WrapperClient:
        def __init__(self) -> None:
            self.kwargs: list[dict[str, Any]] = []

        def collection(self, _name: str) -> FakeCollection:
            return FakeCollection(FakeFirestoreClient(), "unused")

        def transaction(self, **kwargs: Any) -> WrapperTransaction:
            self.kwargs.append(kwargs)
            return transaction

    client = WrapperClient()
    store = FirestoreStateStore(client)
    calls = 0

    def pure_callback(_transaction: object) -> str:
        nonlocal calls
        calls += 1
        return "committed"

    assert store._run_google_transaction(pure_callback) == "committed"
    assert calls == 2
    assert transaction.commits == 2
    assert client.kwargs == [{}]


def test_firestore_worker_callback_runs_once_when_finalize_transaction_replays() -> None:
    backend = SharedBackend()
    client = DistinctFakeClient(backend)
    runner = AbortThenReplayRunner(client)
    store = FirestoreStateStore(client, transaction_runner=runner)
    store.create_task(task())
    store.initialize_actors({"worker-1": ActorRole.WORKER})
    worker_calls = 0

    def execute(_admitted: Task) -> CycleOutcome:
        nonlocal worker_calls
        worker_calls += 1
        return CycleOutcome(target_state=TaskState.RUNNING, evidence=(evidence(),))

    result = store.apply_cycle(intent(), execute)

    assert result.task.tool_actions == 1
    assert worker_calls == 1
    assert store.apply_cycle(intent(), execute) == result
    assert worker_calls == 1


def test_firestore_evidence_storage_is_hard_bounded() -> None:
    client = FakeFirestoreClient()
    store = make_store(client)
    store.create_task(task())
    current = task()
    for index in range(24):
        item = evidence().model_copy(update={"evidence_id": f"ev_{index}"})
        current = store.append_evidence(item, expected_version=current.version)

    overflow = evidence().model_copy(update={"evidence_id": "ev_overflow"})
    with pytest.raises(ValueError, match="evidence bound"):
        store.append_evidence(overflow, expected_version=current.version)
    assert len(store.list_evidence(task().task_id)) == 24


def test_firestore_rejects_over_bound_corrupted_evidence_before_item_decode() -> None:
    client = FakeFirestoreClient()
    store = make_store(client)
    store.create_task(task())
    task_ref = store._task_ref(task().task_id)
    corrupted = deepcopy(client.documents[task_ref.path])
    corrupted["evidence"] = [
        evidence().model_copy(update={"evidence_id": f"ev_{index}"}).model_dump(mode="json")
        for index in range(state_module.MAX_FIRESTORE_EVIDENCE)
    ] + [{"corrupted": "must not be decoded"}]
    client.documents[task_ref.path] = corrupted

    with pytest.raises(ValueError, match="evidence bound"):
        store.list_evidence(task().task_id)


def test_failed_external_cycle_releases_claim_without_mutating_task() -> None:
    client = FakeFirestoreClient()
    store = make_store(client)
    store.create_task(task())

    def explode(_task: Task) -> ActionResult:
        raise RuntimeError("synthetic failure")

    with pytest.raises(RuntimeError, match="synthetic failure"):
        store.run_cycle_transaction(
            task().task_id,
            "failed-claim",
            canonical_operation="failed-operation",
            transact=explode,
        )

    assert store.get_cycle_result(task().task_id, "failed-claim") is None
    assert store.get_task(task().task_id) == task()
    assert not any(
        value.get("active_cycle") is not None
        for path, value in client.documents.items()
        if "/task__" in path
    )


def test_firestore_registry_storage_is_hard_bounded() -> None:
    client = FakeFirestoreClient()
    store = make_store(client)
    version = 0
    for index in range(64):
        version = store.register_actor(
            f"worker-{index}", ActorRole.WORKER, expected_version=version
        )

    with pytest.raises(ValueError, match="registry actor bound"):
        store.register_actor("worker-overflow", ActorRole.WORKER, expected_version=version)
    assert len(store.list_actors()) == 64


def test_firestore_rejects_corrupted_registry_with_65_actors_on_decode() -> None:
    client = FakeFirestoreClient()
    client.documents["control_room/registry"] = {
        "schema_version": 1,
        "actors": {f"worker-{index}": "worker" for index in range(65)},
        "registry_version": 65,
    }
    store = make_store(client)

    with pytest.raises(ValueError, match="registry actor bound"):
        store.list_actors()


def test_firestore_health_discloses_backend_and_public_api_rejects_private_fields() -> None:
    client = FakeFirestoreClient()
    store = make_store(client)
    api = TestClient(
        create_app(
            store=store,
            now=lambda: NOW,
            id_factory=lambda: "private",
        )
    )

    assert api.get("/health").json()["state_backend"] == "firestore"
    response = api.post(
        "/tasks",
        json={
            "title": "alice@example.com secret customer request",
            "required_checks": ["quality"],
        },
    )
    assert response.status_code == 409
    assert "alice@example.com" not in repr(client.documents)


def test_claim_owner_is_unique_across_retries_and_claim_timestamps_are_utc() -> None:
    client = FakeFirestoreClient()
    observed: list[tuple[str, datetime, datetime]] = []
    store = FirestoreStateStore(
        client,
        transaction_runner=client.run_transaction,
        now=lambda: NOW,
    )
    store.create_task(task())

    def observe_then_fail(_task: Task) -> ActionResult:
        cycle = client.documents[store._cycle_ref("syn_firestore", "retry").path]
        observed.append((cycle["owner"], cycle["created_at"], cycle["expires_at"]))
        raise RuntimeError("retry me")

    for _ in range(2):
        with pytest.raises(RuntimeError, match="retry me"):
            store.run_cycle_transaction(
                "syn_firestore",
                "retry",
                canonical_operation="same-operation",
                transact=observe_then_fail,
            )

    assert observed[0][0] != observed[1][0]
    assert all(created.tzinfo is UTC and expires.tzinfo is UTC for _, created, expires in observed)
    assert all(expires > created for _, created, expires in observed)


def test_multiprocess_style_distinct_client_recovers_expired_claim_atomically() -> None:
    backend = SharedBackend()
    first_client = DistinctFakeClient(backend)
    second_client = DistinctFakeClient(backend)
    first = FirestoreStateStore(
        first_client,
        transaction_runner=first_client.run_transaction,
        now=lambda: NOW,
    )
    first.create_task(task())
    task_ref = first._task_ref("syn_firestore")
    stale_ref = first._cycle_ref("syn_firestore", "abandoned")
    task_value = deepcopy(backend.documents[task_ref.path])
    task_value["active_cycle"] = "abandoned"
    backend.documents[task_ref.path] = task_value
    backend.documents[stale_ref.path] = {
        "status": "claimed",
        "owner": "stale-owner",
        "canonical_operation": "old-operation",
        "canonical_intent": None,
        "created_at": NOW - timedelta(minutes=2),
        "expires_at": NOW - timedelta(minutes=1),
    }
    second = FirestoreStateStore(
        second_client,
        transaction_runner=second_client.run_transaction,
        now=lambda: NOW,
    )

    result = second.run_cycle_transaction(
        "syn_firestore",
        "replacement",
        canonical_operation="new-operation",
        transact=lambda current: second.fail_cycle(
            current.task_id,
            "replacement",
            canonical_operation="new-operation",
            expected_version=current.version,
            failure_code="synthetic_failure",
        ),
    )

    assert result.failure_code == "synthetic_failure"
    assert backend.documents[task_ref.path]["active_cycle"] is None
    assert stale_ref.path not in backend.documents


def test_firestore_lease_blocks_takeover_until_120_seconds_then_allows_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeFirestoreClient()
    clock = [NOW]

    class RetainFailedClaimStore(FirestoreStateStore):
        def _abort_claim(self, task_id: str, cycle_key: str, owner: str) -> None:
            return None

    store = RetainFailedClaimStore(
        client,
        transaction_runner=client.run_transaction,
        now=lambda: clock[0],
    )
    store.create_task(task())
    monkeypatch.setattr(state_module, "sleep", lambda _seconds: None)

    with pytest.raises(RuntimeError, match="retain claim"):
        store.run_cycle_transaction(
            "syn_firestore",
            "leased-cycle",
            canonical_operation="leased-operation",
            transact=lambda _task: (_ for _ in ()).throw(RuntimeError("retain claim")),
        )

    cycle_ref = store._cycle_ref("syn_firestore", "leased-cycle")
    original_claim = deepcopy(client.documents[cycle_ref.path])
    assert original_claim["expires_at"] - original_claim["created_at"] == timedelta(
        seconds=120
    )

    clock[0] = NOW + timedelta(seconds=119)
    with pytest.raises(VersionConflict, match="already in progress"):
        store.run_cycle_transaction(
            "syn_firestore",
            "leased-cycle",
            canonical_operation="leased-operation",
            transact=lambda _task: (_ for _ in ()).throw(
                AssertionError("must not take over before lease expiry")
            ),
        )
    assert client.documents[cycle_ref.path] == original_claim

    clock[0] = NOW + timedelta(seconds=120)
    result = store.run_cycle_transaction(
        "syn_firestore",
        "leased-cycle",
        canonical_operation="leased-operation",
        transact=lambda current: store.fail_cycle(
            current.task_id,
            "leased-cycle",
            canonical_operation="leased-operation",
            expected_version=current.version,
            failure_code="takeover_after_expiry",
        ),
    )

    assert result.failure_code == "takeover_after_expiry"
    assert client.documents[cycle_ref.path]["status"] == "finalized"


def test_service_takeover_replays_bound_intent_after_crash_with_new_timestamp() -> None:
    backend = SharedBackend()
    clock = [NOW]

    class CrashAfterBindingStore(FirestoreStateStore):
        crash_after_binding = True

        def _bind_claim_intent(
            self,
            task_id: str,
            cycle_key: str,
            canonical_operation: str,
            canonical_intent: str,
        ) -> None:
            super()._bind_claim_intent(
                task_id,
                cycle_key,
                canonical_operation,
                canonical_intent,
            )
            if self.crash_after_binding:
                self.crash_after_binding = False
                raise RuntimeError("service crashed after intent binding")

        def _abort_claim(self, task_id: str, cycle_key: str, owner: str) -> None:
            return None

    first_client = DistinctFakeClient(backend)
    first_store = CrashAfterBindingStore(
        first_client,
        transaction_runner=first_client.run_transaction,
        now=lambda: clock[0],
    )
    first_service = ControlRoomService(
        first_store,
        now=lambda: clock[0],
        id_factory=lambda: "firestore",
    )
    first_service.create_task("synthetic_takeover", ("quality",))

    with pytest.raises(RuntimeError, match="crashed after intent binding"):
        first_service.run_cycle("syn_firestore", "durable-key")

    cycle_key = opaque_cycle_key("durable-key")
    cycle_ref = first_store._cycle_ref("syn_firestore", cycle_key)
    bound_claim = deepcopy(backend.documents[cycle_ref.path])
    assert bound_claim["status"] == "claimed"
    assert Intent.model_validate_json(bound_claim["canonical_intent"]).issued_at == NOW

    clock[0] = NOW + timedelta(seconds=120)
    second_client = DistinctFakeClient(backend)
    second_store = FirestoreStateStore(
        second_client,
        transaction_runner=second_client.run_transaction,
        now=lambda: clock[0],
    )
    second_service = ControlRoomService(
        second_store,
        now=lambda: clock[0],
        id_factory=lambda: "unused",
    )

    result = second_service.run_cycle("syn_firestore", "durable-key")
    replay = second_service.run_cycle("syn_firestore", "durable-key")

    assert result == replay
    assert result.task.tool_actions == 1
    assert len(second_store.list_evidence("syn_firestore")) == 1
    finalized = backend.documents[cycle_ref.path]
    assert finalized["status"] == "finalized"
    assert finalized["canonical_intent"] == bound_claim["canonical_intent"]
    assert Intent.model_validate_json(finalized["canonical_intent"]).issued_at != clock[0]


def test_direct_apply_takeover_keeps_bound_intent_over_incoming_intent() -> None:
    client = FakeFirestoreClient()
    clock = [NOW]

    class RetainBoundClaimStore(FirestoreStateStore):
        def _bind_claim_intent(
            self,
            task_id: str,
            cycle_key: str,
            canonical_operation: str,
            canonical_intent: str,
        ) -> None:
            super()._bind_claim_intent(
                task_id,
                cycle_key,
                canonical_operation,
                canonical_intent,
            )
            raise RuntimeError("crash after binding")

        def _abort_claim(self, task_id: str, cycle_key: str, owner: str) -> None:
            return None

    first = RetainBoundClaimStore(
        client,
        transaction_runner=client.run_transaction,
        now=lambda: clock[0],
    )
    first.create_task(task())
    first.initialize_actors({"worker-1": ActorRole.WORKER})
    original = intent()

    with pytest.raises(RuntimeError, match="crash after binding"):
        first.apply_cycle(original, lambda _task: CycleOutcome.failed("must_not_run"))

    clock[0] = NOW + timedelta(seconds=120)
    takeover = FirestoreStateStore(
        client,
        transaction_runner=client.run_transaction,
        now=lambda: clock[0],
    )
    regenerated = original.model_copy(update={"issued_at": clock[0]})
    result = takeover.apply_cycle(
        regenerated,
        lambda _task: CycleOutcome.failed("taken_over"),
    )

    assert result.failure_code == "taken_over"
    cycle = client.documents[takeover._cycle_ref(original.task_id, original.cycle_key).path]
    assert Intent.model_validate_json(cycle["canonical_intent"]) == original


def test_firestore_expired_claim_owner_cannot_finalize() -> None:
    client = FakeFirestoreClient()
    clock = [NOW]
    store = FirestoreStateStore(
        client,
        transaction_runner=client.run_transaction,
        now=lambda: clock[0],
    )
    store.create_task(task())

    def finalize_at_expiry(current: Task) -> ActionResult:
        clock[0] = NOW + timedelta(seconds=120)
        return store.fail_cycle(
            current.task_id,
            "expires-before-finalize",
            canonical_operation="expiring-operation",
            expected_version=current.version,
            failure_code="must_not_persist",
        )

    with pytest.raises(VersionConflict, match="claim expired"):
        store.run_cycle_transaction(
            "syn_firestore",
            "expires-before-finalize",
            canonical_operation="expiring-operation",
            transact=finalize_at_expiry,
        )

    assert store.get_task("syn_firestore") == task()
    assert store.get_cycle_result("syn_firestore", "expires-before-finalize") is None


def test_competing_direct_apply_intents_conflict_while_claim_is_active() -> None:
    backend = SharedBackend()
    clients = (DistinctFakeClient(backend), DistinctFakeClient(backend))
    stores = tuple(
        FirestoreStateStore(
            client,
            transaction_runner=client.run_transaction,
            now=lambda: NOW,
        )
        for client in clients
    )
    stores[0].create_task(task())
    stores[0].initialize_actors({"worker-1": ActorRole.WORKER})
    entered = Barrier(2)

    def hold_claim(_task: Task) -> CycleOutcome:
        entered.wait(timeout=1)
        entered.wait(timeout=1)
        return CycleOutcome.failed("held_failure")

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_result = executor.submit(stores[0].apply_cycle, intent(summary="first"), hold_claim)
        entered.wait(timeout=1)
        with pytest.raises(IdempotencyConflict, match="different intent"):
            stores[1].apply_cycle(
                intent(summary="second"),
                lambda _task: CycleOutcome.failed("must_not_run"),
            )
        entered.wait(timeout=1)
        assert first_result.result(timeout=1).failure_code == "held_failure"


def test_abort_cleanup_failure_never_masks_original_transaction_exception() -> None:
    client = FakeFirestoreClient()

    class CleanupFailureStore(FirestoreStateStore):
        def _abort_claim(self, task_id: str, cycle_key: str, owner: str) -> None:
            raise RuntimeError("cleanup failure")

    store = CleanupFailureStore(client, transaction_runner=client.run_transaction, now=lambda: NOW)
    store.create_task(task())

    with pytest.raises(LookupError, match="original failure"):
        store.run_cycle_transaction(
            "syn_firestore",
            "cleanup",
            canonical_operation="cleanup-operation",
            transact=lambda _task: (_ for _ in ()).throw(LookupError("original failure")),
        )


def test_finalize_failure_is_not_masked_by_abort_cleanup_failure() -> None:
    client = FakeFirestoreClient()

    class FinalizeAndCleanupFailureStore(FirestoreStateStore):
        def _finalize_claim(
            self,
            task_id: str,
            cycle_key: str,
            canonical_operation: str,
            result: ActionResult,
            backing: state_module.MemoryBacking,
        ) -> ActionResult:
            raise LookupError("finalize failure")

        def _abort_claim(self, task_id: str, cycle_key: str, owner: str) -> None:
            raise RuntimeError("cleanup failure")

    store = FinalizeAndCleanupFailureStore(
        client,
        transaction_runner=client.run_transaction,
        now=lambda: NOW,
    )
    store.create_task(task())
    store.initialize_actors({"worker-1": ActorRole.WORKER})

    with pytest.raises(LookupError, match="finalize failure"):
        store.apply_cycle(intent(), lambda _task: CycleOutcome.failed("synthetic_failure"))


def test_firestore_cycle_count_is_bounded_for_fail_cycle() -> None:
    client = FakeFirestoreClient()
    store = FirestoreStateStore(client, transaction_runner=client.run_transaction, now=lambda: NOW)
    store.create_task(task())
    current = task()
    for index in range(state_module.MAX_FIRESTORE_CYCLES_PER_TASK):
        result = store.fail_cycle(
            current.task_id,
            f"failure-{index}",
            canonical_operation=f"failure-operation-{index}",
            expected_version=current.version,
            failure_code="synthetic_failure",
        )
        current = result.task

    with pytest.raises(ValueError, match="cycle count bound"):
        store.fail_cycle(
            current.task_id,
            "failure-overflow",
            canonical_operation="failure-operation-overflow",
            expected_version=current.version,
            failure_code="synthetic_failure",
        )
    assert store.get_task(current.task_id) == current


def test_firestore_rejects_long_canonical_cycle_fields_before_effects() -> None:
    client = FakeFirestoreClient()
    store = FirestoreStateStore(client, transaction_runner=client.run_transaction, now=lambda: NOW)
    store.create_task(task())
    effects = 0

    def effect(_task: Task) -> ActionResult:
        nonlocal effects
        effects += 1
        raise AssertionError("must not execute")

    operation = "x" * (state_module.MAX_FIRESTORE_CANONICAL_FIELD_LENGTH + 1)
    with pytest.raises(ValueError, match="canonical_operation"):
        store.run_cycle_transaction(
            "syn_firestore",
            "oversized-operation",
            canonical_operation=operation,
            transact=effect,
        )
    assert effects == 0
    assert store.get_task("syn_firestore") == task()


def test_firestore_cycle_document_bytes_are_bounded_atomically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeFirestoreClient()
    store = FirestoreStateStore(client, transaction_runner=client.run_transaction, now=lambda: NOW)
    store.create_task(task())
    monkeypatch.setattr(state_module, "MAX_FIRESTORE_CYCLE_BYTES", 64)

    with pytest.raises(ValueError, match="cycle document size bound"):
        store.fail_cycle(
            "syn_firestore",
            "oversized-document",
            canonical_operation="operation",
            expected_version=0,
            failure_code="synthetic_failure",
        )
    assert store.get_task("syn_firestore") == task()


def test_firestore_registry_bytes_and_actor_ids_are_bounded_atomically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeFirestoreClient()
    store = make_store(client)

    with pytest.raises(ValueError, match="actor ID"):
        store.register_actor("a" * 129, ActorRole.WORKER, expected_version=0)
    assert store.list_actors() == {}

    monkeypatch.setattr(state_module, "MAX_FIRESTORE_REGISTRY_BYTES", 64)
    with pytest.raises(ValueError, match="registry document size bound"):
        store.initialize_actors({"worker-1": ActorRole.WORKER})
    assert store.list_actors() == {}
