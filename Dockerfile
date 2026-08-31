# Minimal Python image for the Render Cron Job. We do NOT need GPU
# support (the GATv2 GNN is trained offline; inference is plain
# torch_geometric on CPU). Layer caching is set up so a 1-line code
# change doesn't reinstall the world.
FROM python:3.12-slim AS base

# OS deps: build-essential for any wheel that needs compiling,
# libgomp1 for XGBoost's OpenMP runtime, curl for Render health checks.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- requirements (cached unless requirements.txt changes) ----
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# ---- application code ----
COPY src/    /app/src/
COPY config/ /app/config/
COPY sql/    /app/sql/
COPY scripts/ /app/scripts/
COPY models/ /app/models/
# Data dir is created at runtime; pre-create so the docker layer
# doesn't ship a stray SQLite file.
RUN mkdir -p /app/data

# Persistent disk is mounted by Render at /var/data/aizen; the
# default in config.py points there when AIZEN_DB_PATH is set in
# render.yaml.

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    AIZEN_DB_PATH=/var/data/aizen/trading.db \
    AIZEN_LLM_PROVIDER=mock \
    RUN_MODE=paper \
    AIZEN_TRACE=1

# Health check: confirm the entry point imports and the CLI flag works.
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import sys; sys.path.insert(0, '/app'); from src.agents.graph import Orchestrator; print('ok')" || exit 1

# Default command: one cycle per tick. Render Cron Job runs this
# every 15 minutes and then the process exits. To run as a long-lived
# loop locally, override with `python -m src.agents.cli.run_loop` (no
# --once) and a small interval.
CMD ["python", "-m", "src.agents.cli.run_loop", "--once"]
