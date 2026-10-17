"""
tracing.py — OpenTelemetry spans for each step of answering a question.

A question's trace shows where its time went: the follow-up rewrite, the
planner, the calculation or the retrieval (semantic and keyword search, the
cross-encoder, the LLM reranker), then the answer and its check. Uploads and
fetches by ticker are traced too: indexing, figure extraction, the XBRL check.

Off unless configured, and then the spans cost next to nothing:
  OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:4318   send traces over OTLP/HTTP to a
                                                   collector you run (compose has a
                                                   Jaeger under the "tracing" profile)
  TRACE_CONSOLE=1                                  print each span instead
OTEL_SERVICE_NAME names the service (default "financial-rag"); the exporter
reads the other standard OTEL_EXPORTER_OTLP_* settings itself.

Spans carry timings, routes, metric keys, counts and file names. Never the
question, the answer, or who asked: traces may be kept and read by operators,
and a user's questions are theirs.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from functools import wraps

from opentelemetry import trace

tracer = trace.get_tracer("financial_rag")

_lock = threading.Lock()
_configured = False
_enabled = False


def setup(service: str = "financial-rag") -> bool:
    """Install an exporter if the environment asks for one; True if tracing is on.
    Safe to call more than once: only the first call configures anything."""
    global _configured, _enabled
    with _lock:
        if _configured:
            return _enabled
        _configured = True
        endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
        console = os.getenv("TRACE_CONSOLE") == "1"
        if not endpoint and not console:
            return False

        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor,
            ConsoleSpanExporter,
            SimpleSpanProcessor,
        )

        name = os.getenv("OTEL_SERVICE_NAME", "").strip() or service
        provider = TracerProvider(resource=Resource.create({"service.name": name}))
        if endpoint:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

            # Reads OTEL_EXPORTER_OTLP_ENDPOINT and appends /v1/traces
            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        if console:
            provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
        trace.set_tracer_provider(provider)
        _enabled = True
        return True


def collect():
    """
    Keep finished spans in memory too, and return the exporter holding them, so a
    benchmark can report where the time went (eval_answers.py does). Works with or
    without a collector configured.
    """
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    setup()
    provider = trace.get_tracer_provider()
    if not isinstance(provider, TracerProvider):
        provider = TracerProvider()
        trace.set_tracer_provider(provider)
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return exporter


def step_timings(spans) -> dict[str, tuple[int, float]]:
    """Per span name: how many there were and their median duration in seconds."""
    durations: dict[str, list[float]] = {}
    for span in spans:
        durations.setdefault(span.name, []).append((span.end_time - span.start_time) / 1e9)
    return {name: (len(values), sorted(values)[len(values) // 2])
            for name, values in durations.items()}


def traced(name: str) -> Callable:
    """Run the decorated function inside a span of this name."""
    def decorate(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args, **kwargs):
            with tracer.start_as_current_span(name):
                return fn(*args, **kwargs)
        return wrapper
    return decorate


def note(**attributes) -> None:
    """Set attributes on the current span (a no-op when tracing is off).
    None values are skipped; lists are stored as lists of strings."""
    span = trace.get_current_span()
    if not span.is_recording():
        return
    for key, value in attributes.items():
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            value = [str(v) for v in value]
        span.set_attribute(key, value)
