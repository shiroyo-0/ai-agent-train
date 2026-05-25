FROM python:3.12-slim AS base

WORKDIR /app
ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl build-essential && \
    rm -rf /var/lib/apt/lists/* && \
    pip install uv

# Dependencies layer (cached)
COPY pyproject.toml .
RUN uv pip install --system -e .

# Application layer
COPY src/ src/
COPY configs/ configs/

# --- CLI target ---
FROM base AS cli
ENTRYPOINT ["ai"]

# --- API server target ---
FROM base AS api
EXPOSE 8080
CMD ["python", "-m", "uvicorn", "ai_agent.api:app", "--host", "0.0.0.0", "--port", "8080"]

# --- Training target (with GPU deps) ---
FROM base AS training
RUN uv pip install --system -e ".[training]"
ENTRYPOINT ["ai", "train"]
