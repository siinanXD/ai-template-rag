import hashlib
from uuid import uuid4

from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

from app.core.observe import trace_stage
from app.models.document import Chunk, Document
from app.schemas.document import DocumentResponse
from app.services.chunk import chunk_text
from app.services.embed import Embedder
from app.services.normalize import normalize_text


def hash_content(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _persist(db: Session, document: Document, chunks: list[Chunk]) -> None:
    db.add(document)
    db.add_all(chunks)
    db.commit()
    db.refresh(document)


async def ingest_document(
    db: Session, embedder: Embedder, source_name: str, text: str
) -> DocumentResponse:
    with trace_stage("ingest", {"source_chars": len(text)}):
        normalized = normalize_text(text)
        pieces = chunk_text(normalized)
        embeddings = await embedder.embed_texts(pieces)
        document = Document(
            id=uuid4(),
            source_name=source_name,
            content_hash=hash_content(normalized),
        )
        chunks = [
            Chunk(
                id=uuid4(),
                document_id=document.id,
                chunk_index=index,
                content=piece,
                embedding=vector,
            )
            for index, (piece, vector) in enumerate(zip(pieces, embeddings, strict=True))
        ]
        await run_in_threadpool(_persist, db, document, chunks)
        return DocumentResponse(
            id=document.id,
            source_name=document.source_name,
            content_hash=document.content_hash,
            chunk_count=len(chunks),
            created_at=document.created_at,
        )
