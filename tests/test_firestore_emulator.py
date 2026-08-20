"""Opt-in integration suite against the real Firestore emulator.

The in-memory fake in test_firestore_state.py simulates serializable transactions
but cannot reproduce two real-semantics hazards:

* reads inside a transaction observe a snapshot, never your own buffered writes
  (the google client raises ReadAfterWriteError on any read after a write), and
* commits are optimistic — contended callbacks are aborted and replayed.

Running the concurrency-critical paths (claim/lease/takeover/fencing/idempotency/
global task cap) against ``gcloud emulators firestore`` therefore enforces the
ALL-reads-before-ALL-writes invariant for every transaction callback exercised.

Opt in with ``CONTROL_ROOM_EMULATOR_TESTS=1``. The suite skips cleanly when the
emulator cannot be provisioned (missing gcloud SDK, missing Java runtime), or
reuses an externally managed emulator via ``FIRESTORE_EMULATOR_HOST``.
"""

from __future__ import annotations

import os
import shutil
import signal
import socket
import subprocess
import urllib.request
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from threading import Barrier
from time import monotonic, sleep
from typing import Any
from uuid import uuid4

import pytest

import control_room.state as state_module
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
    VersionConflict,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("CONTROL_ROOM_EMULATOR_TESTS") != "1",
    reason="emulator integration suite is opt-in: set CONTROL_ROOM_EMULATOR_TESTS=1",
)

NOW = datetime(2026, 8, 18, 10, 0, tzinfo=UTC)
EMULATOR_BOOT_TIMEOUT_SECONDS = 120.0
BARRIER_TIMEOUT_SECONDS = 30.0


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _emulator_ready(host: str) -> bool:
    try:
        with urllib.request.urlopen(f"http://{host}/", timeout=2) as response:
            return bool(response.status == 200)
    except OSError:
        return False


@pytest.fixture(scope="session")
def emulator_host(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    external = os.environ.get("FIRESTORE_EMULATOR_HOST", "").strip()
    if external:
        if not _emulator_ready(external):
            pytest.skip(f"FIRESTORE_EMULATOR_HOST={external} is not answering")
        yield external
        return

    gcloud = os.environ.get("CONTROL_ROOM_GCLOUD", "").strip() or shutil.which("gcloud")
    if not gcloud:
        pytest.skip("gcloud SDK is unavailable; cannot start the Firestore emulator")

    host = f"127.0.0.1:{_free_port()}"
    log_path = tmp_path_factory.mktemp("emulator") / "firestore-emulator.log"
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            [gcloud, "emulators", "firestore", "start", f"--host-port={host}"],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    try:
        deadline = monotonic() + EMULATOR_BOOT_TIMEOUT_SECONDS
        while monotonic() < deadline:
            if _emulator_ready(host):
                break
            if process.poll() is not None:
                pytest.skip(
                    "Firestore emulator exited during startup "
                    f"(is a Java runtime installed?): {log_path.read_text()[-500:]}"
                )
            sleep(0.25)
        else:
            pytest.skip("Firestore emulator did not become ready in time")
        yield host
    finally:
        if process.poll() is None:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)


@pytest.fixture
def make_client(
    emulator_host: str, monkeypatch: pytest.MonkeyPatch
) -> Callable[[], Any]:
    """Build real Firestore clients against an isolated per-test project."""
    monkeypatch.setenv("FIRESTORE_EMULATOR_HOST", emulator_host)
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    project = f"emulator-{uuid4().hex[:12]}"

    def factory() -> Any:
        from google.auth.credentials import AnonymousCredentials
        from google.cloud import firestore

        return firestore.Client(project=project, credentials=AnonymousCredentials())

    return factory


def make_store(client: Any, *, now: Callable[[], datetime] | None = None) -> FirestoreStateStore:
    return FirestoreStateStore(client, now=now)


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


def counting_coordinator(calls: list[int]) -> Any:
    class CountingCoordinator:
        def propose(self, current: Task, cycle_key: str, now: datetime) -> dict[str, Any]:
            calls.append(1)
            return {
                "task_id": current.task_id,
                "cycle_key": cycle_key,
                "actor_id": "worker-1",
                "action": Action.RECORD_PROGRESS,
                "expected_version": current.version,
                "issued_at": now,
                "payload": {"summary_ref": "synthetic-progress"},
            }

    return CountingCoordinator()


def test_real_client_rejects_reads_after_writes_in_transactions(
    make_client: Callable[[], Any],
) -> None:
    """Prove the harness enforces the invariant the in-memory fake cannot see."""
    from google.cloud import firestore
    from google.cloud.firestore_v1._helpers import ReadAfterWriteError

    client = make_client()
    document = client.collection("control_room").document("meta__read_after_write")
    document.set({"value": 1})
    observed: dict[str, Any] = {}

    @firestore.transactional
    def violate(transaction: Any) -> None:
        observed["before"] = document.get(transaction=transaction).to_dict()
        transaction.set(document, {"value": 2})
        document.get(transaction=transaction)

    with pytest.raises(ReadAfterWriteError):
        violate(client.transaction())
    assert observed["before"] == {"value": 1}


def test_persists_tasks_and_enforces_global_cap_durably(
    make_client: Callable[[], Any],
) -> None:
    client = make_client()
    client.collection("control_room").document("counters").set(
        {"schema_version": 1, "created_tasks": state_module.MAX_FIRESTORE_TASKS - 1}
    )
    first = make_store(client)
    second = make_store(make_client())

    first.create_task(task())

    assert second.get_task("syn_firestore") == task()
    assert second.get_task_status("syn_firestore") == (task(), None)
    counters = client.collection("control_room").document("counters").get().to_dict()
    assert counters == {
        "schema_version": 1,
        "created_tasks": state_module.MAX_FIRESTORE_TASKS,
    }

    rejected = Task.new(
        "syn_firestore_over_cap", "synthetic_over_cap", ("check",), Origin.LOCAL_API
    )
    with pytest.raises(ValueError, match="global task count bound"):
        second.create_task(rejected)
    over_cap_ref = client.collection("control_room").document(
        "task__syn_firestore_over_cap"
    )
    assert not over_cap_ref.get().exists
    counters = client.collection("control_room").document("counters").get().to_dict()
    assert counters["created_tasks"] == state_module.MAX_FIRESTORE_TASKS


def test_evidence_is_append_only_and_version_conflicts_do_not_mutate(
    make_client: Callable[[], Any],
) -> None:
    first = make_store(make_client())
    first.create_task(task())

    with pytest.raises(VersionConflict):
        first.append_evidence(evidence(), expected_version=9)
    assert first.get_task("syn_firestore") == task()
    assert first.list_evidence("syn_firestore") == ()

    updated = first.append_evidence(evidence(), expected_version=0)
    second = make_store(make_client())
    assert updated.version == 1
    assert second.list_evidence("syn_firestore") == (evidence(),)
    with pytest.raises(EvidenceConflict):
        second.append_evidence(evidence(), expected_version=1)
    assert first.list_evidence("syn_firestore") == (evidence(),)


def test_cycle_idempotency_is_durable_and_conflicts_fail_closed(
    make_client: Callable[[], Any],
) -> None:
    first = make_store(make_client())
    first.create_task(task())
    first.register_actor("worker-1", ActorRole.WORKER, expected_version=0)
    calls = 0

    def execute(_admitted: Task) -> CycleOutcome:
        nonlocal calls
        calls += 1
        return CycleOutcome(target_state=TaskState.RUNNING, evidence=(evidence(),))

    result = first.apply_cycle(intent(), execute)
    second = make_store(make_client())
    assert second.apply_cycle(intent(), execute) == result
    assert calls == 1

    with pytest.raises(IdempotencyConflict):
        second.apply_cycle(intent(version=result.task.version, summary="changed"), execute)

    assert first.get_task("syn_firestore") == result.task
    assert first.list_evidence("syn_firestore") == (evidence(),)
    assert calls == 1


def test_claim_owner_is_unique_across_retries_and_timestamps_are_utc(
    make_client: Callable[[], Any],
) -> None:
    client = make_client()
    store = make_store(client, now=lambda: NOW)
    store.create_task(task())
    observed: list[tuple[str, datetime, datetime]] = []

    def observe_then_fail(_task: Task) -> ActionResult:
        raw = store._cycle_ref("syn_firestore", "retry").get().to_dict()
        observed.append((raw["owner"], raw["created_at"], raw["expires_at"]))
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
    assert all(
        created.tzinfo is not None and expires.tzinfo is not None
        for _, created, expires in observed
    )
    assert all(
        expires - created == timedelta(seconds=state_module.FIRESTORE_CLAIM_LEASE_SECONDS)
        for _, created, expires in observed
    )


def test_lease_blocks_takeover_until_expiry_then_allows_it(
    make_client: Callable[[], Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    client = make_client()
    clock = [NOW]

    class RetainFailedClaimStore(FirestoreStateStore):
        def _abort_claim(self, task_id: str, cycle_key: str, owner: str) -> None:
            return None

    store = RetainFailedClaimStore(client, now=lambda: clock[0])
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
    original_claim = cycle_ref.get().to_dict()
    assert original_claim["status"] == "claimed"

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
    assert cycle_ref.get().to_dict()["owner"] == original_claim["owner"]

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
    assert cycle_ref.get().to_dict()["status"] == "finalized"


def test_expired_claim_owner_cannot_finalize(make_client: Callable[[], Any]) -> None:
    client = make_client()
    clock = [NOW]
    store = make_store(client, now=lambda: clock[0])
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


def test_multiprocess_style_distinct_client_recovers_expired_claim_atomically(
    make_client: Callable[[], Any],
) -> None:
    client = make_client()
    first = make_store(client, now=lambda: NOW)
    first.create_task(task())
    task_ref = first._task_ref("syn_firestore")
    stale_ref = first._cycle_ref("syn_firestore", "abandoned")
    task_value = task_ref.get().to_dict()
    task_value["active_cycle"] = "abandoned"
    task_ref.set(task_value)
    stale_ref.set(
        {
            "schema_version": 1,
            "status": "claimed",
            "owner": "stale-owner",
            "canonical_operation": "old-operation",
            "canonical_intent": None,
            "created_at": NOW - timedelta(minutes=2),
            "expires_at": NOW - timedelta(minutes=1),
        }
    )
    second = make_store(make_client(), now=lambda: NOW)

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
    assert task_ref.get().to_dict()["active_cycle"] is None
    assert not stale_ref.get().exists


def test_service_takeover_replays_bound_intent_after_crash(
    make_client: Callable[[], Any],
) -> None:
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
                task_id, cycle_key, canonical_operation, canonical_intent
            )
            if self.crash_after_binding:
                self.crash_after_binding = False
                raise RuntimeError("service crashed after intent binding")

        def _abort_claim(self, task_id: str, cycle_key: str, owner: str) -> None:
            return None

    first_store = CrashAfterBindingStore(make_client(), now=lambda: clock[0])
    first_service = ControlRoomService(
        first_store, now=lambda: clock[0], id_factory=lambda: "firestore"
    )
    first_service.create_task("synthetic_takeover", ("quality",))

    with pytest.raises(RuntimeError, match="crashed after intent binding"):
        first_service.run_cycle("syn_firestore", "durable-key")

    cycle_key = opaque_cycle_key("durable-key")
    cycle_ref = first_store._cycle_ref("syn_firestore", cycle_key)
    bound_claim = cycle_ref.get().to_dict()
    assert bound_claim["status"] == "claimed"
    assert Intent.model_validate_json(bound_claim["canonical_intent"]).issued_at == NOW

    clock[0] = NOW + timedelta(seconds=120)
    second_store = make_store(make_client(), now=lambda: clock[0])
    second_service = ControlRoomService(
        second_store, now=lambda: clock[0], id_factory=lambda: "unused"
    )

    result = second_service.run_cycle("syn_firestore", "durable-key")
    replay = second_service.run_cycle("syn_firestore", "durable-key")

    assert result == replay
    assert result.task.tool_actions == 1
    assert len(second_store.list_evidence("syn_firestore")) == 1
    finalized = cycle_ref.get().to_dict()
    assert finalized["status"] == "finalized"
    assert finalized["canonical_intent"] == bound_claim["canonical_intent"]


def test_distinct_clients_same_key_execute_once_and_tasks_do_not_contend(
    make_client: Callable[[], Any],
) -> None:
    calls: list[int] = []
    services = tuple(
        ControlRoomService(
            make_store(make_client(), now=lambda: NOW),
            now=lambda: NOW,
            id_factory=lambda: "unused",
            coordinator=counting_coordinator(calls),
        )
        for _ in range(2)
    )
    services[0].store.create_task(
        Task.new("syn_a", "synthetic_a", ("quality",), Origin.LOCAL_API)
    )
    services[0].store.create_task(
        Task.new("syn_b", "synthetic_b", ("quality",), Origin.LOCAL_API)
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        same = tuple(
            executor.map(lambda service: service.run_cycle("syn_a", "shared"), services)
        )
    assert same[0] == same[1]
    assert len(calls) == 1
    assert sum(len(service.telemetry.events) for service in services) == 1

    with ThreadPoolExecutor(max_workers=2) as executor:
        different = tuple(
            executor.map(
                lambda pair: pair[0].run_cycle(pair[1], "different"),
                zip(services, ("syn_a", "syn_b"), strict=True),
            )
        )
    assert {result.task.task_id for result in different} == {"syn_a", "syn_b"}
    assert len(calls) == 3


def test_model_call_budget_is_transactional_at_the_bound(
    make_client: Callable[[], Any],
) -> None:
    from google.api_core.exceptions import Aborted

    limit = 5
    day = "2026-08-20"
    stores = tuple(make_store(make_client()) for _ in range(2))

    def consume(index: int) -> bool:
        try:
            return stores[index % 2].consume_model_call(day=day, limit=limit)
        except (Aborted, ValueError) as exc:
            # Emulator lock contention (Aborted, or the client's "Failed to
            # commit transaction in 5 attempts" ValueError): nothing committed,
            # so the call was safely NOT admitted. The bound must stay exact.
            if isinstance(exc, ValueError) and "attempts" not in str(exc):
                raise
            return False

    with ThreadPoolExecutor(max_workers=4) as executor:
        admitted = sum(executor.map(consume, range(12)))

    # Drain sequentially: every remaining slot under the bound is still grantable,
    # and the bound is never exceeded in total.
    while stores[0].consume_model_call(day=day, limit=limit):
        admitted += 1
    assert admitted == limit
    assert stores[1].consume_model_call(day=day, limit=limit) is False
    assert stores[1].consume_model_call(day="2026-08-21", limit=limit) is True


def test_competing_direct_apply_intents_conflict_while_claim_is_active(
    make_client: Callable[[], Any],
) -> None:
    stores = tuple(make_store(make_client(), now=lambda: NOW) for _ in range(2))
    stores[0].create_task(task())
    stores[0].initialize_actors({"worker-1": ActorRole.WORKER})
    entered = Barrier(2)

    def hold_claim(_task: Task) -> CycleOutcome:
        entered.wait(timeout=BARRIER_TIMEOUT_SECONDS)
        entered.wait(timeout=BARRIER_TIMEOUT_SECONDS)
        return CycleOutcome.failed("held_failure")

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_result = executor.submit(
            stores[0].apply_cycle, intent(summary="first"), hold_claim
        )
        entered.wait(timeout=BARRIER_TIMEOUT_SECONDS)
        with pytest.raises(IdempotencyConflict, match="different intent"):
            stores[1].apply_cycle(
                intent(summary="second"),
                lambda _task: CycleOutcome.failed("must_not_run"),
            )
        entered.wait(timeout=BARRIER_TIMEOUT_SECONDS)
        assert (
            first_result.result(timeout=BARRIER_TIMEOUT_SECONDS).failure_code
            == "held_failure"
        )
