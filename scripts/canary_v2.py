"""Canary v2: exactly one bounded real Gemini request through the coordinator seam.

Implements the minimum safe canary design recorded in the Phase 1 contract:
one request, retries=0, temperature 0, thinking disabled AND
max_output_tokens >= 256, assert model_version + finish_reason STOP + a
deterministic content check, routed through GeminiCoordinator ->
authorize_intent (integration proof of the production seam).

Prints a JSON metadata report to stdout (never response bodies; a SHA-256 of
the raw response is recorded instead). Exits non-zero if any assertion fails.
The google-genai SDK performs no automatic retries for generate_content by
default; this script makes exactly one API call per invocation.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from hashlib import sha256
from typing import Any

from control_room.agents.agent import (
    GEMINI_MAX_OUTPUT_TOKENS,
    GEMINI_MODEL,
    GEMINI_TEMPERATURE,
    propose_authorized_intent,
)
from control_room.agents.gemini import GeminiCoordinator
from control_room.domain import Action, ActorRole, Origin, Task
from control_room.service import DeterministicCoordinator
from control_room.state import MemoryStateStore

PROJECT = "gen-lang-client-0960551791"
# gemini-3.5-flash is not served from europe-west1; Vertex's recommended
# "global" endpoint hosts it (verified via control-plane models.get).
LOCATION = "global"


class RecordingVertexTransport:
    """One bounded Vertex request; records metadata, never persists bodies."""

    def __init__(self) -> None:
        self.metadata: dict[str, Any] = {}
        self.calls = 0

    def generate(self, prompt: str) -> str:
        from google import genai
        from google.genai import types

        self.calls += 1
        if self.calls > 1:
            raise RuntimeError("canary transport permits exactly one request")
        client = genai.Client(vertexai=True, project=PROJECT, location=LOCATION)
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=GEMINI_TEMPERATURE,
                max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,
                candidate_count=1,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        candidate = (response.candidates or [None])[0]
        usage = response.usage_metadata
        text = response.text or ""
        self.metadata = {
            "model_requested": GEMINI_MODEL,
            "model_version": response.model_version,
            "response_id": response.response_id,
            "finish_reason": (
                candidate.finish_reason.name
                if candidate is not None and candidate.finish_reason is not None
                else None
            ),
            "prompt_tokens": getattr(usage, "prompt_token_count", None),
            "thoughts_tokens": getattr(usage, "thoughts_token_count", None),
            "candidate_tokens": getattr(usage, "candidates_token_count", None),
            "total_tokens": getattr(usage, "total_token_count", None),
            "response_sha256": sha256(text.encode()).hexdigest(),
            "response_chars": len(text),
        }
        return text


def main() -> int:
    now = datetime.now(UTC)
    store = MemoryStateStore()
    transport = RecordingVertexTransport()
    coordinator = GeminiCoordinator(
        transport,
        consume_model_call=lambda day: store.consume_model_call(day=day, limit=1),
        fallback=DeterministicCoordinator(),
    )
    task = Task.new(
        "syn_canary_v2", "synthetic_canary_v2", ("quality",), Origin.LOCAL_API
    )
    roles = {"worker-1": ActorRole.WORKER, "verifier-1": ActorRole.VERIFIER}

    failures: list[str] = []
    intent = None
    try:
        intent = propose_authorized_intent(
            coordinator,
            task,
            "canary-v2-cycle",
            lambda actor_id: roles[actor_id],
            now=now,
        )
    except Exception as exc:  # record honestly; retries are forbidden
        chain: list[str] = []
        cause: BaseException | None = exc
        while cause is not None:
            chain.append(f"{type(cause).__name__}: {str(cause)[:200]}")
            cause = cause.__cause__
        failures.append("seam_rejected: " + " <- ".join(chain))

    metadata = dict(transport.metadata)
    metadata["event"] = "bounded_real_gemini_canary_v2"
    metadata["timestamp"] = now.isoformat()
    metadata["request_count"] = transport.calls
    metadata["retries"] = 0
    metadata["temperature"] = GEMINI_TEMPERATURE
    metadata["max_output_tokens"] = GEMINI_MAX_OUTPUT_TOKENS
    metadata["thinking_budget"] = 0
    metadata["response_body_persisted"] = False
    metadata["routed_through"] = "GeminiCoordinator -> authorize_intent"

    if transport.calls != 1:
        failures.append(f"expected exactly one request, made {transport.calls}")
    version = metadata.get("model_version") or ""
    if GEMINI_MODEL not in version:
        failures.append(f"model_version {version!r} does not match {GEMINI_MODEL}")
    if metadata.get("finish_reason") != "STOP":
        failures.append(f"finish_reason {metadata.get('finish_reason')!r} != STOP")
    if intent is not None:
        # Deterministic content check: the untrusted output strict-parsed into a
        # bounded choice and survived full intent authorization.
        if intent.action not in {Action.RECORD_PROGRESS, Action.VERIFY_TASK}:
            failures.append(f"unexpected authorized action {intent.action}")
        metadata["authorized_action"] = intent.action.value
        metadata["authorized_actor"] = intent.actor_id
        metadata["deterministic_content_check"] = "strict_parse_and_authorize_passed"
    metadata["assertions_passed"] = not failures
    if failures:
        metadata["failures"] = failures

    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
