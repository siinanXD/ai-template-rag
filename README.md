# ai-template-rag

Specialized RAG template derived from frozen [`ai-starter`](https://github.com/siinanXD/ai-starter) `7f91e3f394536164ffefcf320b4356ad24092702`.

One page ingests a short text document and asks a question. FastAPI chunks and embeds the text, PostgreSQL + pgvector stores the corpus, `ai-core` produces a grounded structured answer with citations, and the UI shows the result. Query text is hashed, not stored.

## 1. Clone

```bash
git clone https://github.com/siinanXD/ai-template-rag.git
cd ai-template-rag
```

## 2. Env setup

```bash
cp .env.example .env
```

Set `OPENAI_API_KEY` for live ingest and query. Leave Langfuse unset unless you want traces. Leave `RERANK_ENABLED=false` unless you have eval evidence to turn it on.

## 3. Start PostgreSQL + pgvector

```bash
docker compose up -d
```

The Compose image is `pgvector/pgvector:pg16`. Alembic creates the `vector` extension.

## 4. Migrations

```bash
cd apps/api
python -m venv .venv
.venv/Scripts/activate   # Windows
# source .venv/bin/activate  # macOS/Linux
pip install -e ".[dev]"
alembic upgrade head
```

## 5. Start FastAPI

From `apps/api`:

```bash
uvicorn app.main:app --reload --port 8000
```

`GET /health` is liveness. `GET /ready` checks the database.

## 6. Start Next.js

```bash
cd apps/web
npm install
npm run dev
```

Open http://localhost:3000. The page posts to `NEXT_PUBLIC_API_URL` (default `http://localhost:8000`).

## 7. Tests

From `apps/api`. OpenAI is mocked. Embeddings use a deterministic hash embedder.

```bash
ruff check .
ruff format --check .
pytest
```

pgvector integration is opt-in:

```bash
RUN_PGVECTOR_IT=1 pytest tests/test_integration_pgvector.py
```

## 8. Evals

Default CI evals are deterministic and do not call OpenAI.

From the repo root, with `apps/api` installed and `PYTHONPATH=apps/api`:

```bash
pip install "agent-eval-harness @ git+https://github.com/siinanXD/agent-eval-harness.git@4b2cb9b7839da8970bdbf271769cde41d7258b60"
agent-eval-harness evals/suites/rag.json \
  --target evals.target:build_target \
  --baseline rag-baseline \
  --root .evals
```

Live OpenAI evals are opt-in only:

```bash
RUN_OPENAI_EVAL=1 OPENAI_API_KEY=... agent-eval-harness evals/suites/rag.json \
  --target evals.live_target:build_target \
  --providers openai \
  --no-gate
```

## 9. Architecture

See `docs/ARCHITECTURE.md`.

- `ai-core` is imported for OpenAI completions, structured output, retry, wrapping, redaction, cost, and optional Langfuse.
- Embeddings use `build_openai_client` from `ai-core`. They are not a second completion wrapper.
- `agent-eval-harness` is the only eval runner.
- Compose runs PostgreSQL + pgvector only.
- Reranking stays off by default; the golden set cannot yet measure whether overlap rerank helps.

## 10. Railway

See `docs/RAILWAY.md`. Dockerfiles exist for `web` and `api`. Both listen on Railway's `$PORT` (local fallbacks 8000 / 3000). This repository does not deploy.

Pinned runtime libraries (both public):

- `ai-core` @ `9fb7f568640346d7ba31eeb6e4d366f6a0e022f1`
- `agent-eval-harness` @ `4b2cb9b7839da8970bdbf271769cde41d7258b60`

`pip install -e ".[dev]"` from `apps/api` is enough. The API image installs `git` so the same pin works in Docker/Railway.
