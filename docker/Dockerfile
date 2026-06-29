# Dockerfile for the FIFA Fantasy WC 2026 prediction system.
#
# Two roles:
#   1. Interactive: run any module on demand
#        docker run --rm -v $PWD/data:/app/data fifa-fantasy \
#            python -m fifa_fantasy.optimizer --stage R32
#   2. Scheduled: run the daily snapshot loop until the WC ends
#        docker compose --profile snapshot up -d
#
# The image is built on a slim Python base, installs the project
# editable, and lets you mount data/ and results/ as volumes so anything
# the running container writes survives a clean teardown.

FROM python:3.12-slim AS base

# CBC solver is bundled with the PuLP wheel on Linux x86_64. If running
# on ARM, install coinor-cbc separately.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        tzdata \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (better build caching).
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Project source (editable install so volume-mounted code changes apply).
COPY pyproject.toml /app/pyproject.toml
COPY src/ /app/src/
COPY scripts/ /app/scripts/
COPY tests/ /app/tests/
RUN pip install --no-cache-dir -e .

# Default writable volume targets. Mount from host to persist.
VOLUME ["/app/data", "/app/results"]

ENV PYTHONUNBUFFERED=1 \
    TZ=UTC

# Default command: print help. Override at runtime for actual work.
CMD ["python", "-m", "fifa_fantasy.collector", "--help"]
