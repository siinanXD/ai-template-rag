"use client";

import { FormEvent, useState } from "react";

import { addDocument, queryRag, type DocumentResult, type QueryResult } from "@/lib/api";

export function RagPage() {
  const [sourceName, setSourceName] = useState("");
  const [text, setText] = useState("");
  const [question, setQuestion] = useState("");
  const [document, setDocument] = useState<DocumentResult | null>(null);
  const [result, setResult] = useState<QueryResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [adding, setAdding] = useState(false);
  const [asking, setAsking] = useState(false);

  async function onAdd(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setAdding(true);
    setError(null);
    try {
      setDocument(await addDocument(sourceName, text));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Document ingest failed");
    } finally {
      setAdding(false);
    }
  }

  async function onAsk(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setAsking(true);
    setError(null);
    setResult(null);
    try {
      setResult(await queryRag(question));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Query failed");
    } finally {
      setAsking(false);
    }
  }

  return (
    <main className="page">
      <h1>RAG</h1>
      <p className="lede">
        Add a short text document, then ask a question. The API retrieves chunks
        from PostgreSQL + pgvector and answers only from that context.
      </p>

      <section>
        <h2>Document</h2>
        <form onSubmit={onAdd}>
          <label htmlFor="source_name">Source name</label>
          <input
            id="source_name"
            name="source_name"
            value={sourceName}
            onChange={(event) => setSourceName(event.target.value)}
            maxLength={256}
            required
          />
          <label htmlFor="text">Document text</label>
          <textarea
            id="text"
            name="text"
            rows={6}
            value={text}
            onChange={(event) => setText(event.target.value)}
            maxLength={20000}
            required
          />
          <button type="submit" disabled={adding || text.trim().length === 0}>
            {adding ? "Adding…" : "Add document"}
          </button>
        </form>
        {document ? (
          <p className="note">
            Stored {document.source_name} as {document.chunk_count} chunk
            {document.chunk_count === 1 ? "" : "s"}.
          </p>
        ) : null}
      </section>

      <section>
        <h2>Query</h2>
        <form onSubmit={onAsk}>
          <label htmlFor="question">Question</label>
          <textarea
            id="question"
            name="question"
            rows={3}
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            maxLength={2000}
            required
          />
          <button type="submit" disabled={asking || question.trim().length === 0}>
            {asking ? "Asking…" : "Ask"}
          </button>
        </form>
      </section>

      {error ? <p className="error" role="alert">{error}</p> : null}

      {result ? (
        <article className="card">
          {result.insufficient_context ? (
            <p className="warn">Insufficient context. The model refused to invent an answer.</p>
          ) : null}
          <p>
            <strong>Answer</strong>
            {result.answer}
          </p>
          <p>
            <strong>Confidence</strong>
            {result.confidence}
          </p>
          <p>
            <strong>Citations</strong>
            {result.citations.length === 0
              ? "none"
              : result.citations
                  .map((citation) => `${citation.source_name}#${citation.chunk_index}`)
                  .join(", ")}
          </p>
          <dl className="meta">
            <div>
              <dt>Model</dt>
              <dd>{result.model}</dd>
            </div>
            <div>
              <dt>Latency</dt>
              <dd>{result.latency_ms} ms</dd>
            </div>
            <div>
              <dt>Retrieval</dt>
              <dd>{result.retrieval_latency_ms} ms</dd>
            </div>
            <div>
              <dt>Rerank</dt>
              <dd>
                {result.rerank_used
                  ? `${result.rerank_latency_ms ?? 0} ms`
                  : "off"}
              </dd>
            </div>
            <div>
              <dt>Tokens</dt>
              <dd>
                {result.input_tokens ?? "—"} in / {result.output_tokens ?? "—"} out
              </dd>
            </div>
            <div>
              <dt>Estimated cost</dt>
              <dd>
                {result.estimated_cost_usd == null
                  ? "unknown"
                  : `$${result.estimated_cost_usd.toFixed(6)}`}
              </dd>
            </div>
          </dl>
          {result.retrieved_chunks.length ? (
            <p className="note">
              Retrieved {result.retrieved_count}:{" "}
              {result.retrieved_chunks
                .map((chunk) => `${chunk.source_name}#${chunk.chunk_index}`)
                .join(", ")}
            </p>
          ) : null}
        </article>
      ) : null}
    </main>
  );
}
