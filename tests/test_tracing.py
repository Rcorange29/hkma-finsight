"""Tests for tracing (tracing.py): which steps get spans, and what they may say.

Spans are captured in memory; no collector is needed.
"""

from decimal import Decimal

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from financial_rag import financials, ingest, rag, router, tracing
from financial_rag.financials import LINE_ITEMS, Fact
from financial_rag.ingest import PUBLIC_OWNER, Filing
from financial_rag.router import QuestionPlan

_EXPORTER = InMemorySpanExporter()


@pytest.fixture(scope="module", autouse=True)
def provider():
    """One SDK provider for the module: OpenTelemetry allows setting it only once."""
    if not isinstance(trace.get_tracer_provider(), TracerProvider):
        sdk = TracerProvider()
        sdk.add_span_processor(SimpleSpanProcessor(_EXPORTER))
        trace.set_tracer_provider(sdk)


@pytest.fixture
def spans():
    _EXPORTER.clear()
    return lambda: {s.name: s for s in _EXPORTER.get_finished_spans()}


QUESTION = "What was Microsoft's secret-sounding operating margin in fiscal 2024?"


@pytest.fixture
def msft(monkeypatch):
    facts = [Fact(PUBLIC_OWNER, "msft.htm", metric, 2024, Decimal(v), "USD",
                  LINE_ITEMS[metric].name, 55, "row")
             for metric, v in (("revenue", 245122), ("operating_income", 109433))]
    financials.replace_facts(PUBLIC_OWNER, "msft.htm", facts)
    monkeypatch.setattr(ingest, "filings", lambda owners: [
        Filing("msft.htm", PUBLIC_OWNER, 10, "Microsoft Corporation")])


def _planner(route="calculation", metrics=("operating_margin",)):
    plan = QuestionPlan(route=route, metrics=list(metrics), fiscal_years=[2024],
                        year_over_year=False, companies=["Microsoft"], qualifier=None)
    return lambda question: plan


def test_a_calculation_is_traced_step_by_step(msft, spans):
    router.answer(QUESTION, "alice", planner=_planner())
    got = spans()
    assert {"answer", "plan", "calculate"} <= set(got)
    assert got["plan"].parent.span_id == got["answer"].context.span_id
    assert got["answer"].attributes["route"] == "calculation"
    assert got["plan"].attributes["metrics"] == ("operating_margin",)
    assert got["plan"].attributes["companies"] == 1
    assert got["calculate"].attributes["results"] == 1


def test_a_follow_up_traces_its_rewrite(msft, spans):
    router.answer("And in 2024?", "alice", planner=_planner(),
                  history=[("user", "Revenue in 2023?"), ("assistant", "…")],
                  rewriter=lambda q, h: QUESTION)
    assert spans()["answer"].attributes["follow_up"] is True
    assert "rewrite_follow_up" in spans()


def test_a_text_answer_records_its_route(msft, spans, monkeypatch):
    monkeypatch.setattr(rag, "query", lambda question, owner, filing=None, filings=None:
                        rag.RAGResponse(answer="from text", sources=[]))
    router.answer(QUESTION, "alice", planner=_planner(route="text", metrics=()))
    assert spans()["answer"].attributes["route"] == "text"


def test_spans_never_carry_the_question_or_the_answer(msft, spans):
    result = router.answer(QUESTION, "alice", planner=_planner())
    for span in spans().values():
        for value in span.attributes.values():
            text = " ".join(value) if isinstance(value, tuple) else str(value)
            assert "secret-sounding" not in text
            assert "alice" not in text
            assert result.answer[:40] not in text


def test_note_skips_none_and_stores_lists_as_strings(spans):
    with tracing.tracer.start_as_current_span("unit"):
        tracing.note(years=[2024, 2023], missing=None, count=3)
    assert dict(spans()["unit"].attributes) == {"years": ("2024", "2023"), "count": 3}


def test_setup_is_off_without_a_collector(monkeypatch):
    monkeypatch.setattr(tracing, "_configured", False)
    monkeypatch.setattr(tracing, "_enabled", False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("TRACE_CONSOLE", raising=False)
    assert tracing.setup() is False
    assert tracing.setup() is False     # configured once only
