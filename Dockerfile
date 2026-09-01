FROM python:3.11-slim AS build
WORKDIR /build
RUN python -m pip install --no-cache-dir build
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN python -m build --wheel

FROM python:3.11-slim
RUN useradd --create-home --uid 10001 appuser
WORKDIR /app
COPY --from=build /build/dist/*.whl /tmp/
RUN python -m pip install --no-cache-dir /tmp/*.whl && rm -f /tmp/*.whl
USER appuser
ENV MODELFORGE_ENGINE=fixture \
    MODELFORGE_LOG_INPUTS=false \
    MODELFORGE_MAX_CONCURRENCY=2 \
    MODELFORGE_QUEUE_TIMEOUT_SECONDS=5 \
    MODELFORGE_SHUTDOWN_GRACE_SECONDS=35 \
    MODELFORGE_MAX_REQUEST_BODY_BYTES=32768
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3)"]
ENTRYPOINT ["modelforge", "serve", "--host", "0.0.0.0", "--port", "8000"]
