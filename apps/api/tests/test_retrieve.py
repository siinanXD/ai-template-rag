import asyncio

from sqlalchemy.orm import Session

from app.services.embed import HashEmbedder
from app.services.retrieve import retrieve_vector
from tests.conftest import seed_corpus


def test_vector_retrieval_ranks_billing_for_invoice_query(
    db_session: Session,
    hash_embedder: HashEmbedder,
) -> None:
    asyncio.run(seed_corpus(db_session, hash_embedder))
    embedding = asyncio.run(hash_embedder.embed_texts(["Is invoice INV-88421 overdue?"]))[0]
    hits = retrieve_vector(db_session, embedding, 3)
    assert hits
    assert any("INV-88421" in hit.content or hit.source_name == "billing-policy" for hit in hits)


def test_vector_retrieval_ranks_legal_for_nda_query(
    db_session: Session,
    hash_embedder: HashEmbedder,
) -> None:
    asyncio.run(seed_corpus(db_session, hash_embedder))
    embedding = asyncio.run(hash_embedder.embed_texts(["Who reviews the NDA?"]))[0]
    hits = retrieve_vector(db_session, embedding, 3)
    assert hits
    assert any(hit.source_name == "legal-nda" for hit in hits)
