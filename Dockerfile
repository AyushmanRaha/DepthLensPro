# syntax=docker/dockerfile:1

FROM python:3.10-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        build-essential \
        gcc \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /build
COPY backend/requirements.txt ./requirements.txt
RUN pip install --upgrade pip wheel setuptools \
    && pip wheel --wheel-dir /tmp/wheels -r requirements.txt \
    && pip install --no-index --find-links=/tmp/wheels -r requirements.txt

FROM python:3.10-slim AS runner

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    HOST=0.0.0.0 \
    PORT=8000 \
    LOG_LEVEL=info \
    DEBUG=false

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system depthlens \
    && useradd --system --gid depthlens --home-dir /app --shell /usr/sbin/nologin depthlens

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY backend ./backend

RUN chown -R depthlens:depthlens /app
USER depthlens

EXPOSE 8000

CMD ["sh", "-c", "exec uvicorn backend.app:app --host ${HOST} --port ${PORT} --log-level $(printf '%s' "$LOG_LEVEL" | tr '[:upper:]' '[:lower:]')"]
