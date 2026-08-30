# Architecture

Derived from frozen `ai-starter` at `7f91e3f394536164ffefcf320b4356ad24092702`.

One vertical slice:

```
Next.js form
  → FastAPI POST /api/v1/documents
  → normalize → chunk → embed (OpenAI client from ai-core)
  → store chunks + embeddings in PostgreSQL + pgvector

Next.js question
  → FastAPI POST /api/v1/query
  → embed question
  → vector retrieval
  → optional overlap rerank (off by default)
  → ai-core structured OpenAI call
  → validate citations against retrieved chunks
  → persist query metadata only
  → show grounded answer + citations
```

`ai-core` owns provider, retry, structured output, redaction, and optional Langfuse generation traces. This repo does not reimplement those.

Embeddings are not in `ai-core`. The template calls `embeddings.create` on the client from `build_openai_client`.

PostgreSQL stores:

- document metadata and chunk text (the RAG corpus)
- embeddings
- query run metadata (hash, model, latency, tokens, cost — never raw query text)

`agent-eval-harness` owns scoring and the regression gate. The deterministic target uses a hash embedder and a fake provider. Stored run files under `.evals/` contain the full query response — do not commit live runs of real customer questions.

Reranking is implemented as a cheap token-overlap reorder of vector hits. It did not beat the vector baseline on the golden set, so `RERANK_ENABLED` defaults to false.
