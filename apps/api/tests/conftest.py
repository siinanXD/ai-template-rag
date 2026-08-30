from collections.abc import Generator

import pytest
from ai_core import CostEstimate, Generation, Usage
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.db import Base, get_db
from app.core.dependencies import get_embedder, get_provider
from app.core.settings import Settings
from app.main import app
from app.models import Chunk, Document, QueryRun  # noqa: F401
from app.schemas.query import GroundedAnswer
from app.services.embed import HashEmbedder

BILLING_DOC = (
    "Harborline billing policy. Invoice INV-88421 is overdue. "
    "Refunds are allowed within 14 days of payment. "
    "Contact billing@harborline.example for invoice status."
)
SUPPORT_DOC = (
    "Harborline support notes. Password resets happen in the dashboard. "
    "MFA is enabled by default after the April update. "
    "Login errors after an update should open a support ticket."
)
SALES_DOC = (
    "Harborline plans. The Pro plan is 49 USD per month. "
    "The Team plan is 199 USD per month and includes a demo. "
    "Request a demo through sales@harborline.example."
)
LEGAL_DOC = (
    "Harborline legal process. NDA review is handled by legal@harborline.example. "
    "Standard review takes 10 business days. Do not sign before legal review."
)


@pytest.fixture
def grounded_answer() -> GroundedAnswer:
    return GroundedAnswer(
        answer="Invoice INV-88421 is overdue.",
        confidence=0.91,
        insufficient_context=False,
        citations=[],
    )


@pytest.fixture
def generation(grounded_answer: GroundedAnswer) -> Generation:
    return Generation(
        text=grounded_answer.model_dump_json(),
        provider="openai",
        model="gpt-4o-mini",
        latency_ms=18,
        usage=Usage(input_tokens=11, output_tokens=22, total_tokens=33),
        cost=CostEstimate("gpt-4o-mini", 11, 22, 0.0002, "known"),
        parsed=grounded_answer,
    )


@pytest.fixture
def db_session() -> Generator[Session, None, None]:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    session = factory()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


class FakeProvider:
    provider = "openai"
    model = "gpt-4o-mini"

    def __init__(self, generation: Generation) -> None:
        self._generation = generation
        self.calls: list[tuple[str, str, type]] = []

    async def complete(self, system: str, user: str) -> Generation:
        raise AssertionError("complete should not be used")

    async def complete_structured(self, system: str, user: str, schema: type) -> Generation:
        self.calls.append((system, user, schema))
        return self._generation


class FailingProvider(FakeProvider):
    async def complete_structured(self, system: str, user: str, schema: type) -> Generation:
        from ai_core import ProviderError

        raise ProviderError("upstream failed")


class ContextAwareProvider(FakeProvider):
    async def complete_structured(self, system: str, user: str, schema: type) -> Generation:
        self.calls.append((system, user, schema))
        question = user.split("Question:", 1)[-1].split("Context:", 1)[0]
        context = user.split("Context:", 1)[-1] if "Context:" in user else ""
        insufficient = "weather" in question.lower() or "INV-88421" not in context
        if insufficient:
            parsed = GroundedAnswer(
                answer="The retrieved context is insufficient to answer this question.",
                confidence=0.2,
                insufficient_context=True,
                citations=[],
            )
        else:
            parsed = GroundedAnswer(
                answer="Invoice INV-88421 is overdue.",
                confidence=0.9,
                insufficient_context=False,
                citations=[],
            )
        return Generation(
            text=parsed.model_dump_json(),
            provider=self.provider,
            model=self.model,
            latency_ms=12,
            usage=Usage(input_tokens=9, output_tokens=8, total_tokens=17),
            cost=CostEstimate(self.model, 9, 8, None, "unknown"),
            parsed=parsed,
        )


@pytest.fixture
def fake_provider(generation: Generation) -> FakeProvider:
    return FakeProvider(generation)


@pytest.fixture
def hash_embedder() -> HashEmbedder:
    return HashEmbedder()


@pytest.fixture
def settings() -> Settings:
    return Settings(openai_api_key="test-key", rerank_enabled=False, retrieval_top_k=5)


@pytest.fixture
def client(
    db_session: Session,
    fake_provider: FakeProvider,
    hash_embedder: HashEmbedder,
) -> Generator[TestClient, None, None]:
    def _db() -> Generator[Session, None, None]:
        yield db_session

    app.dependency_overrides[get_db] = _db
    app.dependency_overrides[get_provider] = lambda: fake_provider
    app.dependency_overrides[get_embedder] = lambda: hash_embedder
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


async def seed_corpus(db_session: Session, hash_embedder: HashEmbedder) -> None:
    from app.services.ingest import ingest_document

    await ingest_document(db_session, hash_embedder, "billing-policy", BILLING_DOC)
    await ingest_document(db_session, hash_embedder, "support-notes", SUPPORT_DOC)
    await ingest_document(db_session, hash_embedder, "sales-plans", SALES_DOC)
    await ingest_document(db_session, hash_embedder, "legal-nda", LEGAL_DOC)
