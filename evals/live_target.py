"""Optional live OpenAI target. Refuses to run unless explicitly enabled."""

from __future__ import annotations

import asyncio
import os

from app.core.dependencies import get_embedder, get_provider
from app.core.settings import get_settings
from app.services.ingest import ingest_document
from app.services.query import answer_question
from evals.corpus import CORPUS
from evals.db import memory_session


def build_target():
    if os.getenv("RUN_OPENAI_EVAL") != "1":
        raise RuntimeError("refusing live OpenAI evals without RUN_OPENAI_EVAL=1")
    get_settings.cache_clear()
    get_provider.cache_clear()
    get_embedder.cache_clear()
    provider = get_provider()
    embedder = get_embedder()
    settings = get_settings()

    def target(case, _provider):
        session = memory_session()
        try:
            for source_name, text in CORPUS:
                asyncio.run(ingest_document(session, embedder, source_name, text))
            result = asyncio.run(answer_question(session, provider, embedder, settings, case.input))
            return result.model_dump_json()
        finally:
            session.close()

    return target
