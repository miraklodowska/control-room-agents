from pathlib import Path

ROOT = Path(__file__).parent.parent


def test_dockerfile_is_bounded_nonroot_cloud_run_runtime() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text()

    assert (
        "FROM python:3.12-slim@sha256:"
        "876416ecde9aca2bcc90e1fb0c7a9500bbf749f5788b70f82d4c5a5c2357f8b4"
        in dockerfile
    )
    assert "uv sync --frozen --no-dev" in dockerfile
    assert "USER control-room" in dockerfile
    assert "0.0.0.0" in dockerfile
    assert "${PORT:-8080}" in dockerfile
    assert "--timeout-keep-alive 5" in dockerfile
    assert "CONTROL_ROOM_STATE_STORE" not in dockerfile
    assert "GOOGLE_APPLICATION_CREDENTIALS" not in dockerfile


def test_docker_context_excludes_local_and_sensitive_artifacts() -> None:
    ignored = set((ROOT / ".dockerignore").read_text().splitlines())

    assert {
        ".git",
        ".venv",
        ".env",
        ".env.*",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "tests",
    } <= ignored


def test_runtime_smoke_has_bounded_timeout_and_synthetic_only_title() -> None:
    smoke = (ROOT / "scripts" / "smoke.py").read_text()

    assert "urlopen(request, timeout=5)" in smoke
    assert 'f"synthetic_{mode}_smoke"' in smoke
