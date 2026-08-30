import math
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from app.core.observe import trace_stage
from app.models.document import Chunk


@dataclass(frozen=True)
class Retrieved:
    chunk_id: UUID
    document_id: UUID
    source_name: str
    chunk_index: int
    content: str
    score: float


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def retrieve_vector(db: Session, embedding: list[float], top_k: int) -> list[Retrieved]:
    with trace_stage("retrieval", {"top_k": top_k, "mode": "vector"}):
        dialect = db.bind.dialect.name if db.bind is not None else "sqlite"
        if dialect == "postgresql":
            from sqlalchemy import text

            rows = db.execute(
                text(
                    """
                    SELECT c.id, c.document_id, d.source_name, c.chunk_index, c.content,
                           1 - (c.embedding <=> CAST(:embedding AS vector)) AS score
                    FROM chunks c
                    JOIN documents d ON d.id = c.document_id
                    ORDER BY c.embedding <=> CAST(:embedding AS vector)
                    LIMIT :top_k
                    """
                ),
                {"embedding": str(embedding), "top_k": top_k},
            ).mappings()
            return [
                Retrieved(
                    chunk_id=row["id"],
                    document_id=row["document_id"],
                    source_name=row["source_name"],
                    chunk_index=row["chunk_index"],
                    content=row["content"],
                    score=float(row["score"]),
                )
                for row in rows
            ]
        chunks = db.execute(select(Chunk).options(joinedload(Chunk.document))).scalars().all()
        ranked = sorted(
            (
                Retrieved(
                    chunk_id=chunk.id,
                    document_id=chunk.document_id,
                    source_name=chunk.document.source_name,
                    chunk_index=chunk.chunk_index,
                    content=chunk.content,
                    score=cosine_similarity(embedding, chunk.embedding),
                )
                for chunk in chunks
            ),
            key=lambda item: item.score,
            reverse=True,
        )
        return ranked[:top_k]
