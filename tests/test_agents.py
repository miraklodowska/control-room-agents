from datetime import UTC, datetime
from typing import Any

from google.adk.agents.llm_agent import Agent

from control_room.agents import root_agent
from control_room.agents.agent import propose_authorized_intent
from control_room.domain import Action, ActorRole, Origin, Task

NOW = datetime(2026, 8, 17, 10, 0, tzinfo=UTC)


class FakeCoordinator:
    def __init__(self) -> None:
        self.calls = 0

    def propose(self, task: Task, cycle_key: str, now: datetime) -> dict[str, Any]:
        self.calls += 1
        return {
            "task_id": task.task_id,
            "cycle_key": cycle_key,
            "actor_id": "worker-1",
            "action": Action.RECORD_PROGRESS,
            "expected_version": task.version,
            "issued_at": now,
            "payload": {"summary_ref": "deterministic-fake-progress"},
        }


def test_root_agent_uses_documented_adk_agent_and_model() -> None:
    assert isinstance(root_agent, Agent)
    assert root_agent.name == "control_room_coordinator"
    assert root_agent.model == "gemini-3.5-flash"


def test_root_agent_generation_is_bounded_and_deterministic() -> None:
    config = root_agent.generate_content_config
    assert config is not None
    assert config.temperature == 0.0
    assert config.max_output_tokens == 512
    assert config.thinking_config is not None
    assert config.thinking_config.thinking_budget == 0


def test_injected_fake_agent_produces_one_deterministic_authorized_cycle() -> None:
    fake = FakeCoordinator()
    task = Task.new("syn_task", "demo", ("check",), Origin.LOCAL_API)
    first = propose_authorized_intent(
        fake, task, "cycle-1", ActorRole.WORKER, now=NOW
    )
    second = propose_authorized_intent(
        fake, task, "cycle-1", ActorRole.WORKER, now=NOW
    )
    assert first == second
    assert fake.calls == 2
    assert first.action is Action.RECORD_PROGRESS
    assert first.payload == {"summary_ref": "deterministic-fake-progress"}
