"""Tests for question routing (router.py) and the analysis parts of the API.

The planner (a model call) is replaced by fixed plans, and text retrieval by a
stub, so these run offline. Figures are stored in the per-test database.
"""

from decimal import Decimal

import pytest
from fastapi.testclient import TestClient

import api
import financials
import rag
import router
from financials import LINE_ITEMS, Fact
from ingest import PUBLIC_OWNER
from router import QuestionPlan

M = Decimal(1_000_000)


def _store(owner, source, figures):
    """figures: {(metric, year): millions}"""
    facts = []
    for (metric, year), millions in figures.items():
        item = LINE_ITEMS[metric]
        facts.append(Fact(owner, source, metric, year, Decimal(millions) * M, item.unit,
                          item.name, 66, f"{item.name}   {millions:,}"))
    financials.replace_facts(owner, source, facts)


MSFT = {("revenue", 2024): 245122, ("revenue", 2023): 211915,
        ("operating_income", 2024): 109433, ("operating_income", 2023): 88523}


@pytest.fixture
def text_calls(monkeypatch):
    calls = []

    def fake_query(question, owner):
        calls.append((question, owner))
        return rag.RAGResponse(answer="from text", sources=[
            {"source_file": "msft.pdf", "page": 12, "excerpt": "prose"}])

    monkeypatch.setattr(rag, "query", fake_query)
    return calls


def _plan(route="calculation", metrics=("operating_margin",), years=(), yoy=False, source=None):
    plan = QuestionPlan(route=route, metrics=list(metrics), fiscal_years=list(years),
                        year_over_year=yoy, source_file=source)
    return lambda question, documents: plan


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def test_calculation_is_answered_deterministically(text_calls):
    _store(PUBLIC_OWNER, "msft.pdf", MSFT)
    result = router.answer("What was the operating margin?", "alice", planner=_plan())
    assert result.route == "calculation"
    assert "**FY2024: 44.6%**" in result.answer
    assert "Operating income FY2024: $109,433 million (msft.pdf, p.66)" in result.answer
    assert {s["excerpt"] for s in result.sources} == {
        "Operating income   109,433", "Total revenue   245,122"}
    assert text_calls == []


def test_text_questions_go_to_retrieval(text_calls):
    _store(PUBLIC_OWNER, "msft.pdf", MSFT)
    result = router.answer("Why did revenue grow?", "alice", planner=_plan(route="text", metrics=()))
    assert (result.route, result.answer) == ("text", "from text")
    assert text_calls == [("Why did revenue grow?", "alice")]


def test_no_figures_skips_the_planner(text_calls):
    def boom(question, documents):
        raise AssertionError("planner called")

    assert router.answer("Operating margin?", "alice", planner=boom).route == "text"


def test_planner_failure_falls_back_to_text(text_calls):
    _store(PUBLIC_OWNER, "msft.pdf", MSFT)

    def broken(question, documents):
        raise RuntimeError("model unavailable")

    assert router.answer("Operating margin?", "alice", planner=broken).route == "text"


def test_uncomputable_calculation_explains_then_answers_from_text(text_calls):
    _store(PUBLIC_OWNER, "msft.pdf", MSFT)
    result = router.answer("Current ratio?", "alice", planner=_plan(metrics=["current_ratio"]))
    assert result.route == "text"
    assert result.answer.startswith("I couldn't calculate this from the structured figures:")
    assert "Current ratio FY2024: needs Total current assets" in result.answer
    assert result.answer.endswith("From the filing text instead:\n\nfrom text")


def test_planner_sees_only_the_callers_documents(text_calls):
    _store(PUBLIC_OWNER, "msft.pdf", MSFT)
    _store("alice", "acme.pdf", {("revenue", 2024): 10})
    _store("bob", "secret.pdf", {("revenue", 2024): 99})
    seen = {}

    def spy(question, documents):
        seen.update(documents)
        return _plan(route="text", metrics=())(question, documents)

    router.answer("q", "alice", planner=spy)
    assert set(seen) == {"msft.pdf", "acme.pdf"}
    assert "(sample filing)" in seen["msft.pdf"]
    assert "fiscal years 2023-2024" in seen["msft.pdf"]


def test_other_users_figures_are_never_used(text_calls):
    _store("bob", "secret.pdf", {("revenue", 2024): 99})
    result = router.answer("Revenue?", "alice", planner=_plan(metrics=["revenue"]))
    assert result.route == "text"   # alice has no figures at all
    assert "secret.pdf" not in result.answer


def test_source_file_restricts_the_calculation(text_calls):
    _store(PUBLIC_OWNER, "msft.pdf", MSFT)
    _store("alice", "acme.pdf", {("revenue", 2024): 10})
    result = router.answer("Acme revenue?", "alice",
                           planner=_plan(metrics=["revenue"], source="acme.pdf"))
    assert [s.source_file for s in result.series] == ["acme.pdf"]
    # An unknown file name from the planner is ignored rather than trusted
    result = router.answer("Revenue?", "alice",
                           planner=_plan(metrics=["revenue"], source="nope.pdf"))
    assert sorted(s.source_file for s in result.series) == ["acme.pdf", "msft.pdf"]


def test_own_file_shadows_a_sample_of_the_same_name(text_calls):
    _store(PUBLIC_OWNER, "msft.pdf", MSFT)
    _store("alice", "msft.pdf", {("revenue", 2024): 1})
    facts = router.visible_facts("alice")
    assert {(f.owner, f.value) for f in facts} == {("alice", Decimal(1) * M)}


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    return TestClient(api.app)


def test_query_returns_structured_calculations(client, text_calls, monkeypatch):
    _store(PUBLIC_OWNER, "msft.pdf", MSFT)
    original = router.answer
    monkeypatch.setattr(router, "answer",
                        lambda q, owner: original(q, owner, planner=_plan(yoy=True)))
    body = client.post("/query", json={"question": "Operating margin?"},
                       headers={"X-User-Id": "alice"}).json()
    assert body["route"] == "calculation"
    fy24 = next(c for c in body["calculations"] if c["fiscal_year"] == 2024)
    assert fy24["display"] == "44.6%"
    assert Decimal(fy24["value"]) == Decimal(109433) / Decimal(245122)
    assert fy24["formula"] == "Operating income ÷ Total revenue"
    [op, rev] = fy24["inputs"]
    assert (op["metric"], op["value"], op["page"], op["shared"]) == (
        "operating_income", "109433000000", 66, True)
    assert rev["line"] == "Total revenue   245,122"


def test_financials_lists_only_visible_figures(client):
    _store(PUBLIC_OWNER, "msft.pdf", {("revenue", 2024): 245122})
    _store("alice", "acme.pdf", {("revenue", 2024): 10})
    _store("bob", "secret.pdf", {("revenue", 2024): 99})
    body = client.get("/financials", headers={"X-User-Id": "alice"}).json()
    assert sorted((f["source_file"], f["shared"]) for f in body["figures"]) == [
        ("acme.pdf", False), ("msft.pdf", True)]
    one = client.get("/financials", params={"source_file": "msft.pdf"},
                     headers={"X-User-Id": "alice"}).json()["figures"]
    assert [(f["display"], f["fiscal_year"]) for f in one] == [("$245,122 million", 2024)]
