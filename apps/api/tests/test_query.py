import asyncio
import hashlib

from ai_core import ProviderError, RetryExhaustedError, StructuredOutputError
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.core.dependencies import get_embedder, get_provider
from app.core.settings import Settings
from app.main import app
from app.models.query import QueryRun
from app.schemas.query import CitationRef, GroundedAnswer
from tests.conftest import (
    BILLING_DOC,
    ContextAwareProvider,
    FailingProvider,
    FakeProvider,
    seed_corpus,
)

UNIQUE = "UNIQUE-CUSTOMER-QUERY-INV-88421-overdue"


def test_query_with_relevant_context(
    client: TestClient,
    db_session: Session,
    hash_embedder,
    generation,
) -> None:
    asyncio.run(seed_corpus(db_session, hash_embedder))
    app.dependency_overrides[get_provider] = lambda: ContextAwareProvider(generation)
    response = client.post("/api/v1/query", json={"question": "Is invoice INV-88421 overdue?"})
    assert response.status_code == 200
    body = response.json()
    assert body["insufficient_context"] is False
    assert "INV-88421" in body["answer"]
    assert body["retrieved_count"] >= 1
    assert any(item["source_name"] == "billing-policy" for item in body["retrieved_chunks"])


def test_query_with_insufficient_context(
    client: TestClient,
    db_session: Session,
    hash_embedder,
    generation,
) -> None:
    asyncio.run(seed_corpus(db_session, hash_embedder))
    app.dependency_overrides[get_provider] = lambda: ContextAwareProvider(generation)
    response = client.post(
        "/api/v1/query", json={"question": "What is the weather in Berlin tomorrow?"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["insufficient_context"] is True
    assert body["citations"] == []


def test_citations_are_validated_against_retrieved_chunks(
    client: TestClient,
    db_session: Session,
    hash_embedder,
    generation,
) -> None:
    asyncio.run(seed_corpus(db_session, hash_embedder))
    invented = GroundedAnswer(
        answer="Invoice INV-88421 is overdue and the CEO is Jane Doe.",
        confidence=0.95,
        insufficient_context=False,
        citations=[CitationRef(ref=99)],
    )
    app.dependency_overrides[get_provider] = lambda: FakeProvider(
        generation.__class__(
            text=invented.model_dump_json(),
            provider=generation.provider,
            model=generation.model,
            latency_ms=generation.latency_ms,
            usage=generation.usage,
            cost=generation.cost,
            parsed=invented,
        )
    )
    response = client.post("/api/v1/query", json={"question": "Is invoice INV-88421 overdue?"})
    assert response.status_code == 200
    body = response.json()
    assert body["citations"] == []
    assert body["insufficient_context"] is True
    assert "Jane Doe" not in body["answer"]
    assert "insufficient" in body["answer"].lower()


def test_valid_citation_ref_is_mapped_to_retrieved_chunk(
    client: TestClient,
    db_session: Session,
    hash_embedder,
    generation,
) -> None:
    asyncio.run(seed_corpus(db_session, hash_embedder))

    class CitingProvider(FakeProvider):
        async def complete_structured(self, system: str, user: str, schema: type):
            parsed = GroundedAnswer(
                answer="Invoice INV-88421 is overdue.",
                confidence=0.9,
                insufficient_context=False,
                citations=[CitationRef(ref=1)],
            )
            return generation.__class__(
                text=parsed.model_dump_json(),
                provider=generation.provider,
                model=generation.model,
                latency_ms=generation.latency_ms,
                usage=generation.usage,
                cost=generation.cost,
                parsed=parsed,
            )

    app.dependency_overrides[get_provider] = lambda: CitingProvider(generation)
    response = client.post("/api/v1/query", json={"question": "Is invoice INV-88421 overdue?"})
    assert response.status_code == 200
    body = response.json()
    assert body["insufficient_context"] is False
    assert len(body["citations"]) == 1
    assert body["citations"][0]["source_name"] == body["retrieved_chunks"][0]["source_name"]
    assert body["citations"][0]["document_id"] == body["retrieved_chunks"][0]["document_id"]
    assert body["citations"][0]["chunk_index"] == body["retrieved_chunks"][0]["chunk_index"]


def test_raw_query_is_not_persisted(
    client: TestClient,
    db_session: Session,
    hash_embedder,
) -> None:
    asyncio.run(seed_corpus(db_session, hash_embedder))
    response = client.post("/api/v1/query", json={"question": UNIQUE})
    assert response.status_code == 200
    rows = db_session.query(QueryRun).all()
    assert len(rows) == 1
    dumped = " ".join(
        str(value)
        for value in (
            rows[0].id,
            rows[0].query_hash,
            rows[0].model,
            rows[0].latency_ms,
            rows[0].retrieval_latency_ms,
            rows[0].rerank_latency_ms,
            rows[0].input_tokens,
            rows[0].output_tokens,
            rows[0].estimated_cost_usd,
            rows[0].created_at,
        )
    )
    assert UNIQUE not in dumped
    assert rows[0].query_hash == hashlib.sha256(UNIQUE.encode("utf-8")).hexdigest()
    assert not hasattr(rows[0], "question")
    assert not hasattr(rows[0], "query")


def test_query_invalid_empty(client: TestClient) -> None:
    response = client.post("/api/v1/query", json={"question": "   "})
    assert response.status_code == 422
    assert response.json() == {"detail": "invalid_request"}
    assert UNIQUE not in response.text


def test_provider_failure(
    client: TestClient,
    db_session: Session,
    hash_embedder,
    generation,
) -> None:
    asyncio.run(seed_corpus(db_session, hash_embedder))
    app.dependency_overrides[get_provider] = lambda: FailingProvider(generation)
    response = client.post("/api/v1/query", json={"question": UNIQUE})
    assert response.status_code == 502
    assert response.json() == {"detail": "provider_failed"}
    assert UNIQUE not in response.text


def test_retry_exhausted_is_provider_failure(
    client: TestClient,
    db_session: Session,
    hash_embedder,
    generation,
) -> None:
    asyncio.run(seed_corpus(db_session, hash_embedder))

    class ExhaustingProvider(FakeProvider):
        async def complete_structured(self, system: str, user: str, schema: type):
            raise RetryExhaustedError(
                "gave up after 3 attempts: APITimeoutError",
                attempts=3,
                last_error=TimeoutError("APITimeoutError"),
            )

    app.dependency_overrides[get_provider] = lambda: ExhaustingProvider(generation)
    response = client.post("/api/v1/query", json={"question": UNIQUE})
    assert response.status_code == 502
    assert response.json() == {"detail": "provider_failed"}


def test_structured_output_error_is_provider_failure(
    client: TestClient,
    db_session: Session,
    hash_embedder,
    generation,
) -> None:
    asyncio.run(seed_corpus(db_session, hash_embedder))

    class BadStructuredProvider(FakeProvider):
        async def complete_structured(self, system: str, user: str, schema: type):
            raise StructuredOutputError("could not parse")

    app.dependency_overrides[get_provider] = lambda: BadStructuredProvider(generation)
    response = client.post("/api/v1/query", json={"question": UNIQUE})
    assert response.status_code == 502
    assert response.json() == {"detail": "provider_failed"}


def test_missing_openai_key_returns_503(db_session: Session, monkeypatch) -> None:
    monkeypatch.setattr(
        "app.core.dependencies.get_settings",
        lambda: Settings(openai_api_key="", database_url="sqlite://"),
    )
    get_provider.cache_clear()
    get_embedder.cache_clear()

    def _db():
        yield db_session

    app.dependency_overrides.clear()
    app.dependency_overrides[get_db] = _db
    with TestClient(app) as test_client:
        response = test_client.post("/api/v1/query", json={"question": UNIQUE})
    app.dependency_overrides.clear()
    get_provider.cache_clear()
    get_embedder.cache_clear()
    assert response.status_code == 503
    assert response.json() == {"detail": "openai_not_configured"}


def test_database_error_returns_503(client: TestClient) -> None:
    from sqlalchemy.exc import OperationalError

    class BoomSession:
        def add(self, *_args, **_kwargs):
            raise OperationalError("INSERT", {}, Exception("down"))

        def execute(self, *_args, **_kwargs):
            raise OperationalError("SELECT", {}, Exception("down"))

        def close(self) -> None:
            return None

    def _db():
        yield BoomSession()

    app.dependency_overrides[get_db] = _db
    response = client.post("/api/v1/documents", json={"source_name": "x", "text": BILLING_DOC})
    assert response.status_code == 503
    assert response.json() == {"detail": "database_unavailable"}


def test_provider_error_type_is_ai_core() -> None:
    assert issubclass(ProviderError, Exception)
