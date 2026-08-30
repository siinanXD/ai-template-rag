from ai_core import ProviderError, RetryExhaustedError, StructuredOutputError
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError

from app.api.documents import router as documents_router
from app.api.health import router as health_router
from app.api.query import router as query_router
from app.core.dependencies import OpenAINotConfiguredError
from app.core.settings import get_settings

app = FastAPI(title="ai-template-rag", version="0.1.0")
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
app.include_router(health_router)
app.include_router(documents_router)
app.include_router(query_router)


async def _provider_failed(_request: Request, _exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=502, content={"detail": "provider_failed"})


app.add_exception_handler(ProviderError, _provider_failed)
app.add_exception_handler(RetryExhaustedError, _provider_failed)
app.add_exception_handler(StructuredOutputError, _provider_failed)


@app.exception_handler(OpenAINotConfiguredError)
async def openai_not_configured(_request: Request, _exc: OpenAINotConfiguredError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": "openai_not_configured"})


@app.exception_handler(SQLAlchemyError)
async def database_unavailable(_request: Request, _exc: SQLAlchemyError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": "database_unavailable"})


@app.exception_handler(RequestValidationError)
async def invalid_request(_request: Request, _exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": "invalid_request"})
