"""Fail-open RAG stage traces through ai-core's Langfuse client.

Never send raw query or document text. Metadata only.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from ai_core import get_langfuse, redact


@contextmanager
def trace_stage(name: str, metadata: dict[str, Any] | None = None) -> Iterator[None]:
    client = None
    span = None
    started = time.monotonic()
    safe = redact(metadata or {})
    try:
        client = get_langfuse()
        if client is not None:
            start = getattr(client, "start_as_current_observation", None)
            if start is not None:
                span = start(name=name, as_type="span", metadata=safe)
            else:
                start_span = getattr(client, "start_span", None)
                if start_span is not None:
                    span = start_span(name=name, metadata=safe)
    except Exception:
        span = None
    try:
        yield
    except Exception as exc:
        _finish(span, safe, started, error=type(exc).__name__)
        raise
    else:
        _finish(span, safe, started, error=None)


def _finish(span: Any, metadata: dict[str, Any], started: float, error: str | None) -> None:
    latency_ms = int((time.monotonic() - started) * 1000)
    update = {**metadata, "latency_ms": latency_ms}
    if error:
        update["error"] = error
    if span is None:
        return
    try:
        updater = getattr(span, "update", None)
        if updater is not None:
            updater(metadata=redact(update))
        ender = getattr(span, "end", None)
        if ender is not None:
            ender()
    except Exception:
        return
