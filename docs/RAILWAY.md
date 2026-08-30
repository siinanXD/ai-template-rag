# Railway preparation

Do not deploy from this repository automatically.

`POST /api/v1/documents` and `POST /api/v1/query` are unauthenticated and spend the OpenAI key. Do not expose the API on a public URL without a gateway or other access control.

Create three Railway services from the same GitHub repo:

| Service | Source | Root Directory | Notes |
| --- | --- | --- | --- |
| postgres | Railway PostgreSQL plugin | — | Enable the `vector` extension before or during Alembic (`CREATE EXTENSION IF NOT EXISTS vector`). Railway's `DATABASE_URL` (`postgresql://…`) is rewritten to `postgresql+psycopg://` at startup |
| api | `Dockerfile` | `apps/api` | Installs `git` so `ai-core` can be fetched. Listens on `$PORT` (fallback 8000). Runs Alembic then uvicorn. Two replicas can race on the Alembic lock. |
| web | `Dockerfile` | `apps/web` | Listens on `$PORT` (fallback 3000) at `0.0.0.0`. Set `NEXT_PUBLIC_API_URL` to the public API URL **at build time** |

API environment:

- `DATABASE_URL` (Railway Postgres plugin value is fine)
- `OPENAI_API_KEY`
- `OPENAI_MODEL` (optional, default `gpt-4o-mini`)
- `OPENAI_EMBEDDING_MODEL` (optional, default `text-embedding-3-small`)
- `CORS_ORIGINS` (the web origin)
- `RERANK_ENABLED` (optional, default `false`)
- optional Langfuse keys
- optional `OPENAI_INPUT_USD_PER_MTOK` / `OPENAI_OUTPUT_USD_PER_MTOK`

Web environment:

- `NEXT_PUBLIC_API_URL`

Required extension: `vector`.

Migration step: `alembic upgrade head` (the API image runs this on boot).

No Redis, workers, or extra datastores. `ai-core` is public.
