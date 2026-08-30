export type DocumentResult = {
  id: string;
  source_name: string;
  content_hash: string;
  chunk_count: number;
  created_at: string;
};

export type Citation = {
  document_id: string;
  source_name: string;
  chunk_index: number;
};

export type RetrievedChunk = {
  document_id: string;
  source_name: string;
  chunk_index: number;
  score: number;
};

export type QueryResult = {
  id: string;
  answer: string;
  confidence: number;
  insufficient_context: boolean;
  citations: Citation[];
  retrieved_chunks: RetrievedChunk[];
  model: string;
  latency_ms: number;
  retrieval_latency_ms: number;
  rerank_latency_ms: number | null;
  generation_latency_ms: number;
  input_tokens: number | null;
  output_tokens: number | null;
  estimated_cost_usd: number | null;
  retrieved_count: number;
  rerank_used: boolean;
  created_at: string;
};

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

async function readError(response: Response): Promise<string> {
  let detail = `Request failed (${response.status})`;
  try {
    const payload = (await response.json()) as { detail?: unknown };
    if (typeof payload.detail === "string") {
      detail = payload.detail;
    }
  } catch {
    // Keep the status message when the body is not JSON.
  }
  return detail;
}

export async function addDocument(sourceName: string, text: string): Promise<DocumentResult> {
  const response = await fetch(`${API_URL}/api/v1/documents`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ source_name: sourceName, text }),
  });
  if (!response.ok) {
    throw new Error(await readError(response));
  }
  return (await response.json()) as DocumentResult;
}

export async function queryRag(question: string): Promise<QueryResult> {
  const response = await fetch(`${API_URL}/api/v1/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  if (!response.ok) {
    throw new Error(await readError(response));
  }
  return (await response.json()) as QueryResult;
}
