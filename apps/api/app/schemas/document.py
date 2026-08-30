from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class DocumentCreate(BaseModel):
    source_name: str = Field(min_length=1, max_length=256)
    text: str = Field(min_length=1, max_length=20000)

    @field_validator("source_name", "text")
    @classmethod
    def not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be blank")
        return stripped


class DocumentResponse(BaseModel):
    id: UUID
    source_name: str
    content_hash: str
    chunk_count: int
    created_at: datetime
