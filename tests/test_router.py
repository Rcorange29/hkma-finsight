"""Tests for question routing (router.py) and the analysis parts of the API.

The planner (a model call) is replaced by fixed plans, and text retrieval by a
stub, so these run offline. Figures are stored in the per-test database.
"""

from decimal import Decimal

import pytest
from fastapi.testclient import TestClient

import api
import financials
import ingest
import rag
import router
from financials import LINE_ITEMS, Fact
from ingest import PUBLIC_OWNER, Filing
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


COMPANIES = {"msft.pdf": "Microsoft Corporation", "acme.pdf": "Acme Inc.",
             "secret.pdf": "Secret Corp"}


@pytest.fixture(autouse=True)
def text_only(monkeypatch):
    """
    Stand-in for the vector store's filing list: every filing with stored figures,
    plus any (owner, name) appended to the returned list, for filings whose
    statements yielded no figures.
    """
    extra: list[tuple[str, str]] = []

    def fake_filings(owners):
        keys = {(f.owner, f.source_file) for f in financials.load_facts(owners)}
        keys |= {k for k in extra if k[0] in owners}
        return [Filing(name, owner, 10, COMPANIES.get(name)) for owner, name in sorted(keys)]

    monkeypatch.setattr(ingest, "filings", fake_filings)
    return extra


@pytest.fixture
def text_calls(monkeypatch):
    calls = []

    def fake_query(question, owner, filing=None):
        calls.append((question, owner, filing))
        return rag.RAGResponse(answer="from text", sources=[
            {"source_file": "msft.pdf", "page": 12, "excerpt": "prose"}])

    monkeypatch.setattr(rag, "query", fake_query)
    return calls


def _plan(route="calculation", metrics=("operating_margin",), years=(), yoy=False, company=None):
    plan = QuestionPlan(route=route, metrics=list(metrics), fiscal_years=list(years),
                        year_over_year=yoy, company=company)
    return lambda question: plan


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def test_calculation_is_answered_deterministically(text_calls):
    _store(PUBLIC_OWNER, "msft.pdf", MSFT)
    result = router.answer("What was the operating margin?", "alice", planner=_plan())
    assert result.route == "calculation"
    assert "**FY2024: 44.6%**" in result.answer
    assert "Operating income FY2024: $109,433 million (msft.pdf, p.67)" in result.answer
    assert {s["excerpt"] for s in result.sources} == {
        "Operating income   109,433", "Total revenue   245,122"}
    assert text_calls == []


def test_text_questions_go_to_retrieval(text_calls):
    _store(PUBLIC_OWNER, "msft.pdf", MSFT)
    result = router.answer("Why did revenue grow?", "alice", planner=_plan(route="text", metrics=()))
    assert (result.route, result.answer, result.filing) == ("text", "from text", None)
    assert text_calls == [("Why did revenue grow?", "alice", None)]


def test_no_figures_skips_the_planner(text_calls):
    def boom(question):
        raise AssertionError("planner called")

    assert router.answer("Operating margin?", "alice", planner=boom).route == "text"


def test_planner_failure_falls_back_to_text(text_calls):
    _store(PUBLIC_OWNER, "msft.pdf", MSFT)

    def broken(question):
        raise RuntimeError("model unavailable")

    assert router.answer("Operating margin?", "alice", planner=broken).route == "text"


def test_uncomputable_calculation_explains_then_answers_from_text(text_calls):
    _store(PUBLIC_OWNER, "msft.pdf", MSFT)
    result = router.answer("Current ratio?", "alice", planner=_plan(metrics=["current_ratio"]))
    assert result.route == "text"
    assert result.answer.startswith("I couldn't calculate this from the structured figures:")
    assert "Current ratio FY2024: needs Total current assets" in result.answer
    assert result.answer.endswith("From the filing text instead:\n\nfrom text")


def test_only_the_callers_filings_and_the_samples_can_be_picked(text_calls):
    _store(PUBLIC_OWNER, "msft.pdf", MSFT)
    _store("alice", "acme.pdf", {("revenue", 2024): 10})
    _store("bob", "secret.pdf", {("revenue", 2024): 99})
    assert set(router.visible_filings("alice")) == {"msft.pdf", "acme.pdf"}
    # Naming another user's company narrows nothing: their filing isn't a candidate
    result = router.answer("Secret Corp's risks?", "alice",
                           planner=_plan(route="text", metrics=(), company="Secret Corp"))
    assert (result.filing, text_calls[-1][2]) == (None, None)


def test_other_users_figures_are_never_used(text_calls):
    _store("bob", "secret.pdf", {("revenue", 2024): 99})
    result = router.answer("Revenue?", "alice", planner=_plan(metrics=["revenue"]))
    assert result.route == "text"   # alice has no figures at all
    assert "secret.pdf" not in result.answer


def test_source_file_restricts_the_calculation(text_calls):
    _store(PUBLIC_OWNER, "msft.pdf", MSFT)
    _store("alice", "acme.pdf", {("revenue", 2024): 10})
    result = router.answer("Acme revenue?", "alice",
                           planner=_plan(metrics=["revenue"], company="Acme"))
    assert [s.source_file for s in result.series] == ["acme.pdf"]
    # A company that matches no filing narrows nothing
    result = router.answer("Revenue?", "alice",
                           planner=_plan(metrics=["revenue"], company="Apple"))
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
    monkeypatch.setattr(router, "answer", lambda q, owner, source_file=None: original(
        q, owner, source_file, planner=_plan(yoy=True)))
    body = client.post("/query", json={"question": "Operating margin?"},
                       headers={"X-User-Id": "alice"}).json()
    assert body["route"] == "calculation"
    fy24 = next(c for c in body["calculations"] if c["fiscal_year"] == 2024)
    assert fy24["display"] == "44.6%"
    assert Decimal(fy24["value"]) == Decimal(109433) / Decimal(245122)
    assert fy24["formula"] == "Operating income ÷ Total revenue"
    [op, rev] = fy24["inputs"]
    assert (op["metric"], op["value"], op["page"], op["shared"]) == (
        "operating_income", "109433000000", 67, True)
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


# ---------------------------------------------------------------------------
# Which filing a question is about
# ---------------------------------------------------------------------------

def test_a_text_question_naming_a_company_searches_its_filing_only(text_calls):
    _store(PUBLIC_OWNER, "msft.pdf", MSFT)
    _store("alice", "acme.pdf", {("revenue", 2024): 10})
    result = router.answer("What risks does Acme describe?", "alice",
                           planner=_plan(route="text", metrics=(), company="Acme"))
    assert text_calls == [("What risks does Acme describe?", "alice", ("alice", "acme.pdf"))]
    assert result.filing == "acme.pdf"


def test_the_picked_filing_overrides_the_planner(text_calls):
    _store(PUBLIC_OWNER, "msft.pdf", MSFT)
    _store("alice", "acme.pdf", {("revenue", 2024): 10})
    result = router.answer("Revenue?", "alice", source_file="msft.pdf",
                           planner=_plan(metrics=["revenue"], company="Acme"))
    assert [s.source_file for s in result.series] == ["msft.pdf"]
    assert result.filing == "msft.pdf"
    router.answer("Why?", "alice", source_file="msft.pdf",
                  planner=_plan(route="text", metrics=(), company="Acme"))
    assert text_calls[-1][2] == (PUBLIC_OWNER, "msft.pdf")


@pytest.mark.parametrize("name", ["nope.pdf", "secret.pdf"])
def test_picking_an_unknown_or_someone_elses_filing_is_refused(text_calls, name):
    _store(PUBLIC_OWNER, "msft.pdf", MSFT)
    _store("bob", "secret.pdf", {("revenue", 2024): 99})
    with pytest.raises(router.UnknownFiling):
        router.answer("Revenue?", "alice", source_file=name, planner=_plan())
    assert text_calls == []


def test_a_picked_name_resolves_to_the_users_own_copy(text_calls):
    _store(PUBLIC_OWNER, "msft.pdf", MSFT)
    _store("alice", "msft.pdf", {("revenue", 2024): 1})
    router.answer("Why?", "alice", source_file="msft.pdf",
                  planner=_plan(route="text", metrics=()))
    assert text_calls[-1][2] == ("alice", "msft.pdf")


def test_filings_without_figures_still_get_narrowed(text_calls, text_only):
    text_only.extend([(PUBLIC_OWNER, "msft.pdf"), ("alice", "acme.pdf")])
    result = router.answer("Acme's risks?", "alice",
                           planner=_plan(route="text", metrics=(), company="Acme"))
    assert result.filing == "acme.pdf"
    assert text_calls[-1][2] == ("alice", "acme.pdf")


def test_a_question_that_names_no_filing_searches_them_all(text_calls):
    _store(PUBLIC_OWNER, "msft.pdf", MSFT)
    _store("alice", "acme.pdf", {("revenue", 2024): 10})
    result = router.answer("Revenue?", "alice", planner=_plan(metrics=["revenue"]))
    assert sorted(s.source_file for s in result.series) == ["acme.pdf", "msft.pdf"]
    assert result.filing is None


@pytest.fixture
def offline_planner(monkeypatch):
    """The API calls router.answer with the real planner; give it a fixed text plan."""
    original = router.answer
    monkeypatch.setattr(router, "answer", lambda q, owner, source_file=None: original(
        q, owner, source_file, planner=_plan(route="text", metrics=())))


def test_query_api_passes_the_picked_filing_and_reports_it(client, text_calls, offline_planner):
    _store(PUBLIC_OWNER, "msft.pdf", MSFT)
    body = client.post("/query", json={"question": "Why?", "source_file": "msft.pdf"},
                       headers={"X-User-Id": "alice"})
    assert body.status_code == 200, body.text
    assert body.json()["filing"] == "msft.pdf"
    assert text_calls[-1][2] == (PUBLIC_OWNER, "msft.pdf")


def test_query_api_refuses_a_filing_the_caller_cant_see(client, text_calls, offline_planner):
    _store("bob", "secret.pdf", {("revenue", 2024): 99})
    response = client.post("/query", json={"question": "Revenue?", "source_file": "secret.pdf"},
                           headers={"X-User-Id": "alice"})
    assert response.status_code == 404
    assert text_calls == []


@pytest.mark.parametrize(
    ("company", "expected"),
    [
        ("Microsoft", "microsoft-sec-10k.pdf"),
        ("Microsoft's", "microsoft-sec-10k.pdf"),
        ("MICROSOFT CORP", "microsoft-sec-10k.pdf"),
        ("Acme Holdings", "acme.pdf"),          # matched on the company name
        ("acme", "acme.pdf"),
        ("Apple", None),                        # not listed
        ("Holdings Inc", None),                 # only generic words
        (None, None),
        ("", None),
    ],
)
def test_match_filing(company, expected):
    filings = {
        "microsoft-sec-10k.pdf": Filing("microsoft-sec-10k.pdf", "public", 1, "Microsoft Corporation"),
        "acme.pdf": Filing("acme.pdf", "alice", 1, "Acme Holdings, Inc."),
        "q3-report.pdf": Filing("q3-report.pdf", "alice", 1, None),
    }
    assert router.match_filing(company, filings) == expected


def test_a_name_matching_several_filings_narrows_nothing():
    filings = {
        "msft-2023.pdf": Filing("msft-2023.pdf", "alice", 1, "Microsoft Corporation"),
        "msft-2024.pdf": Filing("msft-2024.pdf", "alice", 1, "Microsoft Corporation"),
    }
    assert router.match_filing("Microsoft", filings) is None
