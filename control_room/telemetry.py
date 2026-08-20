from __future__ import annotations

from copy import deepcopy
from threading import RLock
from typing import Any

ALLOWED_ATTRIBUTES = frozenset(
    {"task_id", "actor_id", "action", "outcome", "version", "failure_code"}
)
STRING_ATTRIBUTES = ALLOWED_ATTRIBUTES - {"version"}
MAX_TELEMETRY_EVENTS = 1_000


class TelemetrySink:
    """Metadata-only in-memory events bounded to the most recent 1,000 entries."""

    def __init__(self) -> None:
        self._events: list[dict[str, Any]] = []
        self._lock = RLock()

    @property
    def events(self) -> tuple[dict[str, Any], ...]:
        with self._lock:
            return deepcopy(tuple(self._events))

    def emit(self, event: str, **attributes: Any) -> None:
        if type(event) is not str or not event.strip():
            raise ValueError("invalid telemetry event")
        unexpected = set(attributes) - ALLOWED_ATTRIBUTES
        if unexpected:
            names = ", ".join(sorted(unexpected))
            raise ValueError(f"telemetry fields not allowed: {names}")
        for field, value in attributes.items():
            if field in STRING_ATTRIBUTES:
                valid = type(value) is str and bool(value.strip())
            else:
                valid = type(value) is int and value >= 0
            if not valid:
                raise ValueError(f"invalid telemetry value for field: {field}")
        with self._lock:
            self._events.append(deepcopy({"event": event, **attributes}))
            if len(self._events) > MAX_TELEMETRY_EVENTS:
                del self._events[: len(self._events) - MAX_TELEMETRY_EVENTS]
