"""Optional lexical overlap rerank of already-retrieved vector hits.

No extra model. Default is off until evals show a clear gain.
"""

import re

from app.core.observe import trace_stage
from app.services.retrieve import Retrieved


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def overlap_score(question: str, content: str) -> float:
    query_tokens = _tokens(question)
    if not query_tokens:
        return 0.0
    return len(query_tokens & _tokens(content)) / len(query_tokens)


def rerank_overlap(question: str, hits: list[Retrieved]) -> list[Retrieved]:
    with trace_stage("reranking", {"candidate_count": len(hits), "method": "token_overlap"}):
        scored = [
            Retrieved(
                chunk_id=hit.chunk_id,
                document_id=hit.document_id,
                source_name=hit.source_name,
                chunk_index=hit.chunk_index,
                content=hit.content,
                score=overlap_score(question, hit.content),
            )
            for hit in hits
        ]
        return sorted(scored, key=lambda item: item.score, reverse=True)
