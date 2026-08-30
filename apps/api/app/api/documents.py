from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.core.dependencies import get_embedder
from app.schemas.document import DocumentCreate, DocumentResponse
from app.services.embed import Embedder
from app.services.ingest import ingest_document

router = APIRouter()


@router.post("/api/v1/documents", response_model=DocumentResponse)
async def create_document(
    payload: DocumentCreate,
    db: Session = Depends(get_db),
    embedder: Embedder = Depends(get_embedder),
) -> DocumentResponse:
    return await ingest_document(db, embedder, payload.source_name, payload.text)
