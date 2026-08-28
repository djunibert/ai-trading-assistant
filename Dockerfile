FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
COPY scripts/prepare_requirements.py ./scripts/prepare_requirements.py

RUN python -m pip install --upgrade pip \
    && python scripts/prepare_requirements.py \
        --output requirements-linux.txt \
        --exclude-package pywin32 \
    && sed -i \
        -e '/^pywin32/d' \
        -e '/^pywinpty/d' \
        -e '/^windows-/d' \
        requirements-linux.txt \
    && python -m pip install -r requirements-linux.txt

COPY backend ./backend
COPY src ./src

RUN mkdir -p \
    /app/models \
    /app/data \
    /app/reports \
    /app/logs

EXPOSE 8000

HEALTHCHECK \
    --interval=30s \
    --timeout=5s \
    --start-period=30s \
    --retries=3 \
    CMD curl --fail http://127.0.0.1:${PORT}/health || exit 1

CMD ["sh", "-c", "uvicorn backend.app:app --host 0.0.0.0 --port ${PORT}"]