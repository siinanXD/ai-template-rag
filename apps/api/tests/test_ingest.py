from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.models.document import Chunk, Document
from tests.conftest import BILLING_DOC


def test_document_insert_creates_chunks_and_embeddings(
    client: TestClient, db_session: Session
) -> None:
    response = client.post(
        "/api/v1/documents",
        json={"source_name": "billing-policy", "text": BILLING_DOC},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["source_name"] == "billing-policy"
    assert body["chunk_count"] >= 1
    assert len(body["content_hash"]) == 64
    documents = db_session.query(Document).all()
    chunks = db_session.query(Chunk).all()
    assert len(documents) == 1
    assert len(chunks) == body["chunk_count"]
    assert all(chunk.embedding for chunk in chunks)
    assert all(chunk.content for chunk in chunks)
    assert BILLING_DOC.split()[0] in " ".join(chunk.content for chunk in chunks)


def test_duplicate_content_is_idempotent(client: TestClient, db_session: Session) -> None:
    payload = {"source_name": "billing-policy", "text": BILLING_DOC}
    first = client.post("/api/v1/documents", json=payload)
    second = client.post(
        "/api/v1/documents",
        json={"source_name": "billing-policy-retry", "text": BILLING_DOC},
    )
    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["id"] == second.json()["id"]
    assert first.json()["content_hash"] == second.json()["content_hash"]
    assert first.json()["source_name"] == second.json()["source_name"] == "billing-policy"
    assert db_session.query(Document).count() == 1
    assert db_session.query(Chunk).count() == first.json()["chunk_count"]


def test_document_invalid_empty_text(client: TestClient) -> None:
    response = client.post("/api/v1/documents", json={"source_name": "x", "text": "   "})
    assert response.status_code == 422
    assert response.json() == {"detail": "invalid_request"}
