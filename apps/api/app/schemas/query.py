from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)

    @field_validator("question")
    @classmethod
    def not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be blank")
        return stripped


class Citation(BaseModel):
    document_id: UUID
    source_name: str
    chunk_index: int


class RetrievedChunk(BaseModel):
    document_id: UUID
    source_name: str
    chunk_index: int
    score: float


class GroundedAnswer(BaseModel):
    """Structured model output. Used by ai-core complete_structured."""

    answer: str = Field(min_length=1, max_length=4000)
    confidence: float = Field(ge=0, le=1)
    insufficient_context: bool
    citations: list[Citation] = Field(default_factory=list)


class QueryResponse(BaseModel):
    id: UUID
    answer: str
    confidence: float
    insufficient_context: bool
    citations: list[Citation]
    retrieved_chunks: list[RetrievedChunk]
    model: str
    latency_ms: int
    retrieval_latency_ms: int
    rerank_latency_ms: int | None
    generation_latency_ms: int
    input_tokens: int | None
    output_tokens: int | None
    estimated_cost_usd: float | None
    retrieved_count: int
    rerank_used: bool
    created_at: datetime
