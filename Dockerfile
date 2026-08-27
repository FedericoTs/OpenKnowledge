# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# pdfplumber needs libgomp for its image backend; everything else in the
# parsing stack is pure Python.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first so application edits do not invalidate the layer.
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir ".[anthropic]"

COPY web/ ./web/
COPY documents/ ./documents/

# Run unprivileged. /app/data is the only path that needs to be writable.
RUN useradd --create-home --uid 10001 openknowledge \
    && mkdir -p /app/data \
    && chown -R openknowledge:openknowledge /app/data
USER openknowledge

ENV OK_DATA_DIR=/app/data \
    OK_DOCUMENTS_DIR=/app/documents

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=4).status==200 else 1)"

CMD ["openknowledge", "serve", "--host", "0.0.0.0", "--port", "8080"]
