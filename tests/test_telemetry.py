import pytest

from control_room.telemetry import MAX_TELEMETRY_EVENTS, TelemetrySink


def test_telemetry_contains_only_allowlisted_metadata_fields() -> None:
    sink = TelemetrySink()
    sink.emit(
        "cycle.completed",
        task_id="syn_task",
        actor_id="worker-1",
        action="record_progress",
        outcome="RUNNING",
        version=1,
    )
    assert sink.events == (
        {
            "event": "cycle.completed",
            "task_id": "syn_task",
            "actor_id": "worker-1",
            "action": "record_progress",
            "outcome": "RUNNING",
            "version": 1,
        },
    )
    flattened = repr(sink.events).lower()
    for forbidden in ("prompt", "response", "body", "content", "summary"):
        assert forbidden not in flattened


def test_telemetry_rejects_non_allowlisted_or_content_fields() -> None:
    sink = TelemetrySink()
    for field in ("prompt", "response", "body", "content", "summary", "unknown"):
        try:
            sink.emit("unsafe", **{field: "secret"})
        except ValueError as exc:
            assert "not allowed" in str(exc)
        else:
            raise AssertionError(f"telemetry accepted forbidden field: {field}")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("task_id", {"nested": "syn_task"}),
        ("actor_id", ["worker-1"]),
        ("action", ("record_progress",)),
        ("outcome", b"RUNNING"),
        ("version", True),
        ("version", 1.0),
        ("version", "1"),
        ("failure_code", {"raw_content": "secret"}),
    ],
)
def test_telemetry_rejects_container_content_and_wrong_scalar_types(
    field: str, value: object
) -> None:
    sink = TelemetrySink()
    with pytest.raises(ValueError, match="invalid telemetry value"):
        sink.emit("cycle.completed", **{field: value})
    assert sink.events == ()


def test_telemetry_rejects_non_string_event_name() -> None:
    sink = TelemetrySink()
    with pytest.raises(ValueError, match="invalid telemetry event"):
        sink.emit({"raw_content": "secret"})  # type: ignore[arg-type]
    assert sink.events == ()


def test_telemetry_reads_cannot_mutate_stored_events() -> None:
    sink = TelemetrySink()
    sink.emit("cycle.completed", task_id="syn_task", version=1)
    first = sink.events
    first[0]["task_id"] = "tampered"
    first[0]["version"] = 999
    assert sink.events == (
        {"event": "cycle.completed", "task_id": "syn_task", "version": 1},
    )


def test_telemetry_keeps_only_the_most_recent_bounded_events() -> None:
    sink = TelemetrySink()

    for version in range(MAX_TELEMETRY_EVENTS + 5):
        sink.emit("cycle.completed", task_id="syn_task", version=version)

    assert len(sink.events) == MAX_TELEMETRY_EVENTS
    assert sink.events[0]["version"] == 5
    assert sink.events[-1]["version"] == MAX_TELEMETRY_EVENTS + 4
