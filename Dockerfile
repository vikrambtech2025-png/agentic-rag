FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libmupdf-dev && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 appuser

COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv && uv sync --frozen --no-dev

COPY app/ ./app/
COPY evals/ ./evals/
COPY tests/ ./tests/

RUN chown -R appuser:appuser /app
USER appuser

ENV DATA_DIR=/data
VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health').read()" || exit 1

CMD ["uv", "run", "chainlit", "run", "app/ui/app.py", "-w", "--host", "0.0.0.0", "--port", "8000"]
