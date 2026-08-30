import asyncio
import os

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from app.core.settings import EMBEDDING_DIMENSIONS
from app.services.embed import HashEmbedder
from app.services.ingest import ingest_document
from app.services.retrieve import retrieve_vector
from tests.conftest import BILLING_DOC

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_PGVECTOR_IT") != "1",
    reason="pgvector integration is opt-in via RUN_PGVECTOR_IT=1",
)


def test_pgvector_stores_and_retrieves() -> None:
    url = os.getenv("DATABASE_URL", "postgresql+psycopg://starter:starter@localhost:5432/starter")
    engine = create_engine(url, pool_pre_ping=True)
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.execute(text("DROP TABLE IF EXISTS chunks"))
        conn.execute(text("DROP TABLE IF EXISTS documents"))
        conn.execute(text("DROP TABLE IF EXISTS query_runs"))
        conn.commit()
    from app.core.db import Base
    from app.models import Chunk, Document, QueryRun  # noqa: F401

    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine, autoflush=False, autocommit=False)()
    try:
        embedder = HashEmbedder(dimensions=EMBEDDING_DIMENSIONS)
        created = asyncio.run(ingest_document(session, embedder, "billing-policy", BILLING_DOC))
        assert created.chunk_count >= 1
        embedding = asyncio.run(embedder.embed_texts(["invoice INV-88421 overdue"]))[0]
        hits = retrieve_vector(session, embedding, 3)
        assert hits
        assert hits[0].source_name == "billing-policy"
    finally:
        session.close()
        engine.dispose()
