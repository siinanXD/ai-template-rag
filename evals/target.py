"""Deterministic harness target. Calls ingest + answer_question. No paid API calls."""

from __future__ import annotations

import asyncio
import re

from ai_core import CostEstimate, Generation, Usage

from app.core.settings import Settings
from app.schemas.query import CitationRef, GroundedAnswer
from app.services.embed import HashEmbedder
from app.services.ingest import ingest_document
from app.services.query import answer_question
from evals.corpus import CORPUS
from evals.db import memory_session

_REF = re.compile(r"\[(\d+)\] source_name=(\S+)")


def _citations_for(context: str, *source_names: str) -> list[CitationRef]:
    wanted = set(source_names)
    return [
        CitationRef(ref=int(match.group(1)))
        for match in _REF.finditer(context)
        if match.group(2) in wanted
    ]


def classify(question: str, context: str) -> GroundedAnswer:
    lowered_q = question.lower()
    lowered_c = context.lower()
    if "invented citation" in lowered_q:
        return GroundedAnswer(
            answer="Invoice INV-88421 is overdue and the CEO is Jane Doe.",
            confidence=0.95,
            insufficient_context=False,
            citations=[CitationRef(ref=99)],
        )
    if "weather" in lowered_q or "ceo" in lowered_q or "berlin" in lowered_q:
        return GroundedAnswer(
            answer="The retrieved context is insufficient to answer this question.",
            confidence=0.2,
            insufficient_context=True,
            citations=[],
        )
    if "inv-88421" in lowered_c and (
        "overdue" in lowered_q or "invoice" in lowered_q or "inv-88421" in lowered_q
    ):
        return GroundedAnswer(
            answer="Invoice INV-88421 is overdue.",
            confidence=0.9,
            insufficient_context=False,
            citations=_citations_for(context, "billing-policy"),
        )
    if "refund" in lowered_q and "14 days" in lowered_c:
        return GroundedAnswer(
            answer="Refunds are allowed within 14 days of payment.",
            confidence=0.88,
            insufficient_context=False,
            citations=_citations_for(context, "billing-policy"),
        )
    if "password" in lowered_q and "dashboard" in lowered_c:
        return GroundedAnswer(
            answer="Password resets happen in the dashboard.",
            confidence=0.86,
            insufficient_context=False,
            citations=_citations_for(context, "support-notes"),
        )
    if "mfa" in lowered_q and "default" in lowered_c:
        return GroundedAnswer(
            answer="MFA is enabled by default after the April update.",
            confidence=0.87,
            insufficient_context=False,
            citations=_citations_for(context, "support-notes"),
        )
    if "pro plan" in lowered_q and "49" in lowered_c:
        return GroundedAnswer(
            answer="The Pro plan is 49 USD per month.",
            confidence=0.9,
            insufficient_context=False,
            citations=_citations_for(context, "sales-plans"),
        )
    if ("team plan" in lowered_q or "demo" in lowered_q) and "199" in lowered_c:
        return GroundedAnswer(
            answer="The Team plan is 199 USD per month and includes a demo.",
            confidence=0.88,
            insufficient_context=False,
            citations=_citations_for(context, "sales-plans"),
        )
    if "business days" in lowered_q and "10 business days" in lowered_c:
        return GroundedAnswer(
            answer="Standard NDA review takes 10 business days.",
            confidence=0.86,
            insufficient_context=False,
            citations=_citations_for(context, "legal-nda"),
        )
    if "nda" in lowered_q and "legal@harborline.example" in lowered_c:
        return GroundedAnswer(
            answer="NDA review is handled by legal@harborline.example.",
            confidence=0.9,
            insufficient_context=False,
            citations=_citations_for(context, "legal-nda"),
        )
    return GroundedAnswer(
        answer="The retrieved context is insufficient to answer this question.",
        confidence=0.35,
        insufficient_context=True,
        citations=[],
    )


class _DeterministicProvider:
    provider = "openai"
    model = "eval-fake"

    async def complete(self, system: str, user: str) -> Generation:
        raise AssertionError("complete should not be used")

    async def complete_structured(self, system: str, user: str, schema: type) -> Generation:
        question = user.split("Question:", 1)[-1].split("Context:", 1)[0]
        context = user.split("Context:", 1)[-1] if "Context:" in user else ""
        parsed = classify(question, context)
        return Generation(
            text=parsed.model_dump_json(),
            provider=self.provider,
            model=self.model,
            latency_ms=1,
            usage=Usage(input_tokens=1, output_tokens=1, total_tokens=2),
            cost=CostEstimate(self.model, 1, 1, None, "unknown"),
            parsed=parsed,
        )


async def _seed(session) -> None:
    embedder = HashEmbedder()
    for source_name, text in CORPUS:
        await ingest_document(session, embedder, source_name, text)


def _payload(result) -> dict:
    return {
        "output": result.model_dump_json(),
        "retrieved": [chunk.source_name for chunk in result.retrieved_chunks],
    }


def build_target():
    provider = _DeterministicProvider()
    embedder = HashEmbedder()
    settings = Settings(openai_api_key="eval", rerank_enabled=False, retrieval_top_k=2)

    def target(case, _provider):
        session = memory_session()
        try:
            asyncio.run(_seed(session))
            result = asyncio.run(answer_question(session, provider, embedder, settings, case.input))
            return _payload(result)
        finally:
            session.close()

    return target
