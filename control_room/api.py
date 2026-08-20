from __future__ import annotations

import os
import uuid
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from control_room.agents.agent import Coordinator
from control_room.agents.gemini import build_gemini_coordinator
from control_room.policy import PolicyDenied
from control_room.service import (
    MAX_CALLER_IDEMPOTENCY_KEY_LENGTH,
    ControlRoomService,
)
from control_room.state import (
    BudgetExhausted,
    IdempotencyConflict,
    MemoryBacking,
    MemoryStateStore,
    StateStore,
    VersionConflict,
    build_state_store,
)


def _select_coordinator(
    environ: Mapping[str, str], store: StateStore
) -> tuple[Coordinator | None, str]:
    configured = environ.get("CONTROL_ROOM_COORDINATOR", "").strip() or "deterministic"
    if configured == "deterministic":
        return None, "deterministic"
    if configured == "gemini":
        return build_gemini_coordinator(environ, store=store), "gemini"
    raise ValueError("CONTROL_ROOM_COORDINATOR must be 'deterministic' or 'gemini'")


class CreateTaskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=64)
    required_checks: tuple[str, ...] = Field(min_length=1, max_length=8)

    @field_validator("required_checks")
    @classmethod
    def required_checks_are_bounded(
        cls, value: tuple[str, ...]
    ) -> tuple[str, ...]:
        if any(len(check) > 64 for check in value):
            raise ValueError("required checks must contain at most 64 characters")
        return value


class CycleRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    idempotency_key: str = Field(
        min_length=1, max_length=MAX_CALLER_IDEMPOTENCY_KEY_LENGTH
    )

    @field_validator("idempotency_key")
    @classmethod
    def idempotency_key_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("idempotency key must not be blank")
        return value


def create_app(
    *,
    backing: MemoryBacking | None = None,
    store: StateStore | None = None,
    demo_mode: bool | None = None,
    now: Callable[[], datetime] | None = None,
    id_factory: Callable[[], str] | None = None,
    coordinator: Coordinator | None = None,
) -> FastAPI:
    if backing is not None and store is not None:
        raise ValueError("backing and store are mutually exclusive")
    enabled = (
        os.getenv("CONTROL_ROOM_DEMO_MODE", "").lower() == "true"
        if demo_mode is None
        else demo_mode
    )
    selected_store = (
        store
        if store is not None
        else MemoryStateStore(backing)
        if backing is not None
        else build_state_store(os.environ)
    )
    if coordinator is not None:
        coordinator_name = "injected"
    else:
        coordinator, coordinator_name = _select_coordinator(os.environ, selected_store)
    service = ControlRoomService(
        selected_store,
        now=now or (lambda: datetime.now(UTC)),
        id_factory=id_factory or (lambda: uuid.uuid4().hex),
        coordinator=coordinator,
    )
    application = FastAPI(title="Control Room", version="0.1.0")
    application.state.service = service

    @application.exception_handler(RequestValidationError)
    async def sanitized_validation_error(
        _request: object, exc: RequestValidationError
    ) -> JSONResponse:
        errors = [
            {key: value for key, value in error.items() if key != "input"}
            for error in exc.errors()
        ]
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={"detail": jsonable_encoder(errors)},
        )

    @application.get("/", response_class=HTMLResponse)
    def dashboard() -> str:
        return (Path(__file__).parent.parent / "web" / "index.html").read_text()

    @application.get("/health")
    def health() -> dict[str, object]:
        return {
            "status": "ok",
            "demo_mode": enabled,
            "state_backend": selected_store.backend_name,
            "coordinator": coordinator_name,
        }

    @application.post("/tasks", status_code=status.HTTP_201_CREATED)
    def create_task(request: CreateTaskRequest) -> object:
        try:
            return service.create_task(request.title, request.required_checks)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @application.get("/tasks/{task_id}")
    def task_status(task_id: str) -> dict[str, object]:
        try:
            task, baton = service.store.get_task_status(task_id)
            return {
                "task": task,
                "baton": baton,
            }
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @application.get("/tasks/{task_id}/evidence")
    def task_evidence(task_id: str) -> object:
        try:
            return service.store.list_evidence(task_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @application.get("/registry")
    def registry() -> object:
        return service.store.list_actors()

    @application.post("/tasks/{task_id}/cycles")
    def execute_cycle(task_id: str, request: CycleRequest) -> object:
        try:
            return service.run_cycle(task_id, request.idempotency_key)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (
            BudgetExhausted,
            IdempotencyConflict,
            PolicyDenied,
            ValueError,
            VersionConflict,
        ) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    if enabled:

        @application.post("/demo/tasks/{task_id}/breaker")
        def trip_breaker(task_id: str, request: CycleRequest) -> object:
            try:
                return service.trip_demo_breaker(task_id, request.idempotency_key)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except PermissionError as exc:
                raise HTTPException(status_code=403, detail=str(exc)) from exc
            except (
                BudgetExhausted,
                IdempotencyConflict,
                PolicyDenied,
                ValueError,
                VersionConflict,
            ) as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc

    return application


app = create_app()
