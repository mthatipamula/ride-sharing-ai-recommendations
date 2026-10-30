# Recommendation service image (CPU PyTorch). Models and feature data are mounted, not baked in, so a new model
# version is a new artifact directory, not a new image.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.4 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app

# dependencies first: this layer is reused until uv.lock changes
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-editable

ENV PATH=/app/.venv/bin:$PATH \
    ARTIFACTS_DIR=/app/artifacts \
    DATA_DIR=/app/data/riders-5000-seed-7
RUN useradd --uid 10001 --no-create-home app
USER 10001
EXPOSE 8080
# one worker per container: scale with replicas, and Prometheus metrics stay per-process correct
CMD ["uvicorn", "rideshare_recs.serving.app:app", "--host", "0.0.0.0", "--port", "8080", "--no-access-log"]
