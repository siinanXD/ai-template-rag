from functools import lru_cache

from ai_core import LLMProvider, ModelPricing, OpenAIProvider, build_openai_client

from app.core.settings import get_settings
from app.services.embed import Embedder, HashEmbedder, OpenAIEmbedder


class OpenAINotConfiguredError(Exception):
    """OPENAI_API_KEY is missing. The app can still serve health checks."""


@lru_cache
def get_provider() -> LLMProvider:
    settings = get_settings()
    if not settings.openai_api_key:
        raise OpenAINotConfiguredError
    pricing = None
    if (
        settings.openai_input_usd_per_mtok is not None
        and settings.openai_output_usd_per_mtok is not None
    ):
        pricing = ModelPricing(
            model=settings.openai_model,
            input_usd_per_mtok=settings.openai_input_usd_per_mtok,
            output_usd_per_mtok=settings.openai_output_usd_per_mtok,
        )
    client = build_openai_client(
        settings.openai_api_key,
        timeout_seconds=settings.openai_timeout_seconds,
    )
    return OpenAIProvider(client, settings.openai_model, pricing=pricing)


@lru_cache
def get_embedder() -> Embedder:
    settings = get_settings()
    if not settings.openai_api_key:
        raise OpenAINotConfiguredError
    client = build_openai_client(
        settings.openai_api_key,
        timeout_seconds=settings.openai_timeout_seconds,
    )
    return OpenAIEmbedder(client, settings.openai_embedding_model)


def hash_embedder() -> HashEmbedder:
    return HashEmbedder()
