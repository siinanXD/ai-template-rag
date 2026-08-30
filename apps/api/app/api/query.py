from ai_core import LLMProvider
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.core.dependencies import get_embedder, get_provider
from app.core.settings import Settings, get_settings
from app.schemas.query import QueryRequest, QueryResponse
from app.services.embed import Embedder
from app.services.query import answer_question

router = APIRouter()


@router.post("/api/v1/query", response_model=QueryResponse)
async def query(
    payload: QueryRequest,
    db: Session = Depends(get_db),
    provider: LLMProvider = Depends(get_provider),
    embedder: Embedder = Depends(get_embedder),
    settings: Settings = Depends(get_settings),
) -> QueryResponse:
    return await answer_question(db, provider, embedder, settings, payload.question)
