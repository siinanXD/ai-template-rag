"""Embedding helpers.

OpenAI embeddings use ai-core's client builder. This is not a second completion wrapper.
"""

from __future__ import annotations

import math
import re
import zlib
from typing import Protocol

from app.core.settings import EMBEDDING_DIMENSIONS


class Embedder(Protocol):
    model: str

    async def embed_texts(self, texts: list[str]) -> list[list[float]]: ...


class OpenAIEmbedder:
    def __init__(self, client: object, model: str) -> None:
        self._client = client
        self.model = model

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = await self._client.embeddings.create(model=self.model, input=texts)
        return [list(item.embedding) for item in response.data]


class HashEmbedder:
    """Deterministic bag-of-words vectors for tests and CI evals. No paid API."""

    model = "hash-embedder"

    def __init__(self, dimensions: int = EMBEDDING_DIMENSIONS) -> None:
        self.dimensions = dimensions

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_one(text) for text in texts]

    def embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in _tokens(text):
            vector[zlib.crc32(token.encode("utf-8")) % self.dimensions] += 1.0
        return _normalize(vector)


_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "be",
        "can",
        "do",
        "does",
        "for",
        "get",
        "have",
        "how",
        "i",
        "in",
        "is",
        "like",
        "my",
        "of",
        "on",
        "or",
        "the",
        "to",
        "we",
        "what",
        "who",
    }
)


def _tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        if token in _STOPWORDS or len(token) < 2:
            continue
        tokens.append(token)
        if token.endswith("s") and len(token) > 3:
            tokens.append(token[:-1])
    return tokens


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]
