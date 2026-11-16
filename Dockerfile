FROM python:3.12-slim

WORKDIR /app

# system deps for pymupdf
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmupdf-dev && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv && uv sync --frozen --no-dev

COPY app/ ./app/
COPY evals/ ./evals/
COPY tests/ ./tests/

ENV DATA_DIR=/data
VOLUME ["/data"]
EXPOSE 8000

CMD ["uv", "run", "chainlit", "run", "app/ui/app.py", "-w", "--host", "0.0.0.0", "--port", "8000"]
