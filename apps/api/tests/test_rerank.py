import asyncio

from sqlalchemy.orm import Session

from app.core.settings import Settings
from app.services.embed import HashEmbedder
from app.services.rerank import rerank_overlap
from app.services.retrieve import retrieve_vector
from tests.conftest import seed_corpus


def test_rerank_is_optional_and_does_not_clearly_beat_vector(
    db_session: Session,
    hash_embedder: HashEmbedder,
) -> None:
    asyncio.run(seed_corpus(db_session, hash_embedder))
    cases = [
        ("Is invoice INV-88421 overdue?", "billing-policy"),
        ("How do I reset my password?", "support-notes"),
        ("What is the Pro plan price?", "sales-plans"),
        ("Who reviews the NDA?", "legal-nda"),
    ]
    vector_top1 = 0
    rerank_top1 = 0
    for question, source in cases:
        embedding = asyncio.run(hash_embedder.embed_texts([question]))[0]
        vector_hits = retrieve_vector(db_session, embedding, 4)
        reranked = rerank_overlap(question, vector_hits)
        if vector_hits and vector_hits[0].source_name == source:
            vector_top1 += 1
        if reranked and reranked[0].source_name == source:
            rerank_top1 += 1
    assert Settings().rerank_enabled is False
    assert vector_top1 >= 3
    assert rerank_top1 <= vector_top1
