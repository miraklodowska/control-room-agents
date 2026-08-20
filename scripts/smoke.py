#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def request_json(
    base_url: str, method: str, path: str, payload: dict[str, Any] | None = None
) -> tuple[int, Any]:
    body = None if payload is None else json.dumps(payload).encode()
    request = Request(
        f"{base_url.rstrip('/')}{path}",
        data=body,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read())
    except HTTPError as exc:
        return exc.code, json.loads(exc.read())


def run(base_url: str, mode: str) -> None:
    status, health = request_json(base_url, "GET", "/health")
    assert status == 200 and health["status"] == "ok", health
    assert health["demo_mode"] is (mode == "breaker"), health
    assert health["state_backend"] in {"memory", "firestore"}, health

    status, task = request_json(
        base_url,
        "POST",
        "/tasks",
        {"title": f"synthetic_{mode}_smoke", "required_checks": ["quality"]},
    )
    assert status == 201 and task["task_id"].startswith("syn_"), task
    assert task["origin"] == "LOCAL_API", task

    task_id = task["task_id"]
    if mode == "normal":
        path = f"/tasks/{task_id}/cycles"
        expected_state = "RUNNING"
        expected_failure = None
    else:
        path = f"/demo/tasks/{task_id}/breaker"
        expected_state = "NEEDS_OPERATOR"
        expected_failure = "demo_missing_evidence"

    status, result = request_json(
        base_url, "POST", path, {"idempotency_key": f"smoke-{mode}"}
    )
    assert status == 200, result
    assert result["task"]["state"] == expected_state, result
    assert result["failure_code"] == expected_failure, result
    assert result["task"]["tool_actions"] == 1, result

    status, persisted = request_json(base_url, "GET", f"/tasks/{task_id}")
    assert status == 200 and persisted["task"]["state"] == expected_state, persisted
    print(f"{mode} smoke passed: {task_id} -> {expected_state}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Control Room loopback smoke")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--mode", choices=("normal", "breaker"), default="normal")
    arguments = parser.parse_args()
    run(arguments.base_url, arguments.mode)


if __name__ == "__main__":
    main()
