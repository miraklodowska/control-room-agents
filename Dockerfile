FROM python:3.12-slim@sha256:876416ecde9aca2bcc90e1fb0c7a9500bbf749f5788b70f82d4c5a5c2357f8b4

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_NO_PROGRESS=1 \
    PATH=/app/.venv/bin:$PATH \
    HOME=/tmp

RUN pip install --no-cache-dir uv==0.11.2 \
    && useradd --create-home --uid 10001 --shell /usr/sbin/nologin control-room

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY control_room ./control_room
COPY web ./web
RUN uv sync --frozen --no-dev

USER control-room
EXPOSE 8080
CMD ["sh", "-c", "exec uvicorn control_room.api:app --host 0.0.0.0 --port ${PORT:-8080} --timeout-keep-alive 5"]
