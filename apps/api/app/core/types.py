import json
from typing import Any

from sqlalchemy.engine import Dialect
from sqlalchemy.types import TEXT, TypeDecorator

from app.core.settings import EMBEDDING_DIMENSIONS


class EmbeddingColumn(TypeDecorator):
    """pgvector on PostgreSQL, JSON text on SQLite for unit tests."""

    impl = TEXT
    cache_ok = True

    def load_dialect_impl(self, dialect: Dialect) -> Any:
        if dialect.name == "postgresql":
            from pgvector.sqlalchemy import Vector

            return dialect.type_descriptor(Vector(EMBEDDING_DIMENSIONS))
        return dialect.type_descriptor(TEXT())

    def process_bind_param(self, value: list[float] | None, dialect: Dialect) -> Any:
        if value is None:
            return None
        if dialect.name == "postgresql":
            return value
        return json.dumps(value)

    def process_result_value(self, value: Any, dialect: Dialect) -> list[float] | None:
        if value is None:
            return None
        if dialect.name == "postgresql":
            return list(value)
        if isinstance(value, str):
            parsed = json.loads(value)
            return [float(item) for item in parsed]
        return list(value)
