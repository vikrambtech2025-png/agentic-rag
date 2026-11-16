# Agentic RAG Buddy

[![CI](https://github.com/vikrambtech2025-png/agentic-rag-buddy/actions/workflows/ci.yml/badge.svg)](https://github.com/vikrambtech2025-png/agentic-rag-buddy/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A production-minded, mixed-corpus **agentic RAG** assistant. Ingest PDFs, Markdown, text, URLs, YouTube videos, and entire codebases, then ask questions and get **grounded, cited answers** with a visible reasoning trace.

Built with **LangGraph**, **LanceDB**, **FastAPI/Chainlit**, and **OpenAI-compatible LLMs**.

---

## ✨ Features

- **Mixed corpus** — docs + web + code in one knowledge base
- **Agentic loop** — route → retrieve → grade → rewrite → generate → verify, with hop budget
- **Hybrid retrieval** — LanceDB vectors + SQLite FTS5, fused with RRF
- **Citations** — inline `[n]` references with source + page, verified against passages
- **Multi-hop** — rewrites and re-retrieves when context is insufficient
- **Web search** — Tavily with caching
- **Code indexer** — tree-sitter symbol-aware chunking with `file:line` citations
- **Local embeddings** — fastembed (offline, no API cost); OpenAI embeddings optional
- **UI** — Chainlit chat with streaming tokens, visible Steps, side-panel citation previews
- **CLI** — `python -m app.cli "question" --trace`
- **Memory** — `remember:` facts persisted across sessions
- **Eval harness** — recall@k, citation precision, refusal accuracy
- **Production hardening** — retries, validation, SSRF protection, rate limiting, structured logging, Docker

## 🏗️ Architecture

```
┌─────────────┐     ┌──────────┐     ┌──────────┐
│  Chainlit   │────▶│ LangGraph│────▶│  LanceDB │
│     UI      │     │  agent   │     │  + FTS5  │
└─────────────┘     └──────────┘     └──────────┘
                          │
                          ├── Tavily (web)
                          ├── fastembed (embeddings)
                          └── OpenAI-compatible LLM
```

Graph: `route → retrieve → grade ─┬─ sufficient → generate → verify → END`
`                                    └─ gaps & hops<MAX → rewrite ↺ retrieve`

## 🚀 Quick start

```powershell
git clone https://github.com/vikrambtech2025-png/agentic-rag-buddy.git
cd agentic-rag-buddy
cp .env.example .env   # fill OPENROUTER_API_KEY or GEMINI_API_KEY, TAVILY_API_KEY
uv sync
uv run chainlit run app/ui/app.py -w
```
Open http://localhost:8000

## 🧪 Tests

```powershell
uv run python -m pytest -q          # 8 unit tests
uv run python -m app.smoke          # health check
uv run python evals/run.py          # eval metrics
```

## 📦 Ingest a codebase

```powershell
uv run python -m app.ingest.code C:\path\to\repo
```

## 🐳 Docker

```powershell
docker compose up --build
```

## 🔧 Configuration

Key env vars (see `.env.example`):

| Var | Purpose |
|---|---|
| `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL` | Generic LLM override (highest priority) |
| `OPENROUTER_API_KEY` | OpenRouter (default provider) |
| `OPENAI_MODEL` | Model name (e.g. `gpt-5` → `openai/gpt-5`) |
| `EMBED_MODEL` / `EMBED_DIM` | Embeddings (default: local fastembed) |
| `TAVILY_API_KEY` | Web search |
| `DATA_DIR` | Data directory (default `./data`) |

## 📊 Eval targets

| Metric | Target |
|---|---|
| recall@k | ≥ 0.85 |
| citation precision | ≥ 0.90 |
| refusal accuracy | 1.00 |

## 🛡️ Production notes

- SQLite + LanceDB are single-node; move to Postgres + pgvector for scale
- Add real auth (currently open access) and per-user isolation
- Set `LOG_FORMAT=json` for structured logs
- `/health` endpoint available for orchestrators

## 📄 License

MIT — see [LICENSE](LICENSE).
