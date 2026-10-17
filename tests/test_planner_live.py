"""Routing accuracy of the real planner model. Opt-in: it calls OpenAI.

    RUN_LIVE_TESTS=1 pytest tests/test_planner_live.py

The planner is the one model-dependent step between a question and the
deterministic engine, so this pins its behaviour on representative questions.
Re-run it after changing PLANNER_PROMPT or config.CHAT_MODEL.
"""

import os

import pytest

from financial_rag import router
from financial_rag.ingest import Filing

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_LIVE_TESTS") != "1", reason="calls OpenAI; set RUN_LIVE_TESTS=1"
)


# question -> (route, metrics, fiscal_years, year_over_year); None = don't care
CASES = {
    "What is the current ratio?": ("calculation", {"current_ratio"}, [], None),
    "What's Microsoft's return on equity?": ("calculation", {"return_on_equity"}, [], None),
    "What was total revenue?": ("calculation", {"revenue"}, [], False),
    "What was net income in fiscal 2023?": ("calculation", {"net_income"}, [2023], False),
    "Operating margin in 2024 vs the prior year": ("calculation", {"operating_margin"}, None, True),
    "What was the operating margin in fiscal 2024, and how did it change from the prior year?":
        ("calculation", {"operating_margin"}, None, True),
    "How much did revenue grow from 2022 to 2024?": ("calculation", {"revenue"}, [2022, 2023, 2024], None),
    "What was free cash flow in 2024?": ("calculation", {"free_cash_flow"}, [2024], False),
    "Gross and net margin for fiscal 2023": ("calculation", {"gross_margin", "net_margin"}, [2023], False),
    "How much does Microsoft spend on R&D relative to revenue?": ("calculation", {"rd_intensity"}, [], None),
    "What was the effective tax rate?": ("calculation", {"effective_tax_rate"}, [], None),
    "Did diluted EPS increase last year?": ("calculation", {"eps_diluted"}, None, True),
    "What share of net income was paid out as dividends?": ("calculation", {"payout_ratio"}, [], None),
    "How leveraged is the balance sheet (long-term debt to equity)?":
        ("calculation", {"debt_to_equity"}, [], None),
    "Why did operating margin improve?": ("text", None, None, None),
    # Planned either way: Azure is a qualifier no filing part matches, so code sends it to text
    "How fast did Azure grow?": (None, None, None, None),
    "What are the main risk factors related to AI?": ("text", None, None, None),
    "Who is the CEO?": ("text", None, None, None),
    "What does the company say about its revenue recognition policy?": ("text", None, None, None),
    "How many employees does Microsoft have?": ("text", None, None, None),
    # Planned as diluted_shares 8 times in 8 before the headcount rule
    "How many full-time equivalent employees did Apple have at the end of fiscal 2025?":
        ("text", None, None, None),
    "How many diluted shares did Apple have in fiscal 2025?":
        ("calculation", {"diluted_shares"}, [2025], None),
    # Planned as share_repurchases (cash spent) 2 times in 8 before its example
    "How much was left in the share repurchase program at June 30, 2024?":
        ("text", None, None, None),
    "How much did Microsoft spend repurchasing shares in fiscal 2024?":
        ("calculation", {"share_repurchases"}, [2024], None),
}


@pytest.mark.parametrize("question", list(CASES))
def test_planner_routes_question(question):
    route, metrics, years, yoy = CASES[question]
    plan = router.plan_question(question)
    if route is not None:
        assert plan.route == route
    if route == "calculation":
        assert plan.qualifier is None    # a qualifier would send it to text
    if metrics is not None:
        assert set(plan.metrics) == metrics
    if years is not None:
        assert sorted(plan.fiscal_years) == years
    if yoy is not None:
        assert plan.year_over_year == yoy


# question -> (route, filing it should narrow to), with these two filings visible
WHICH_FILING = {
    "What risks does Acme describe?": ("text", "acme-10k.pdf"),
    "What was Microsoft's operating margin in 2024?": ("calculation", "msft.pdf"),
    "How did Acme Holdings' revenue change from 2023?": ("calculation", "acme-10k.pdf"),
    "Who is Microsoft's auditor?": ("text", "msft.pdf"),
    "How fast did Azure grow?": (None, "msft.pdf"),            # a product names its company
    "What was net income in fiscal 2024?": ("calculation", None),
    "What are the main risk factors?": ("text", None),
    "What was Apple's revenue?": (None, None),                 # not listed: answered as unavailable
    # Other companies mentioned aren't the company asked about
    "How much goodwill was recorded for the Activision Blizzard acquisition?": ("text", None),
    "How does Microsoft compete with Google?": ("text", "msft.pdf"),
    "Who are Acme's largest customers?": ("text", "acme-10k.pdf"),
}
FILINGS = {
    "msft.pdf": Filing("msft.pdf", "public", 1, "Microsoft Corporation"),
    "acme-10k.pdf": Filing("acme-10k.pdf", "alice", 1, "Acme Holdings, Inc."),
}


@pytest.mark.parametrize("question", list(WHICH_FILING))
def test_planner_picks_the_filing_a_question_names(question):
    route, filing = WHICH_FILING[question]
    plan = router.plan_question(question)
    if route is not None:
        assert plan.route == route
    named = [name for company in plan.companies
             for name in router.filings_named(company, FILINGS)]
    assert named == ([filing] if filing else []), plan


# question -> (route, companies named); a comparison names each company it compares
COMPARED = {
    "Compare Apple's and Microsoft's operating margins": ("calculation", {"apple", "microsoft"}),
    "Which had more revenue in 2024, Microsoft or Acme?": ("calculation", {"microsoft", "acme"}),
    "Apple vs Microsoft: net income": ("calculation", {"apple", "microsoft"}),
    "How do Apple's and Microsoft's AI risks differ?": ("text", {"apple", "microsoft"}),
    "Who audits Apple and who audits Microsoft?": ("text", {"apple", "microsoft"}),
    # A competitor mentioned is not a company compared
    "How does Microsoft compete with Google?": ("text", {"microsoft"}),
}


@pytest.mark.parametrize("question", list(COMPARED))
def test_planner_names_every_company_compared(question):
    route, companies = COMPARED[question]
    plan = router.plan_question(question)
    assert plan.route == route, plan
    assert {c.lower().split()[0] for c in plan.companies} == companies, plan


# question -> whether the question qualifies a listed metric (segment, variant, ...)
QUALIFIED = {
    "What was Intelligent Cloud revenue in fiscal 2024?": True,
    "What was LinkedIn revenue?": True,
    "What were dividends declared in fiscal 2024?": True,
    "How much revenue came from the United States in fiscal 2024?": True,
    "What was adjusted operating income in fiscal 2023?": True,
    "What was revenue in the fourth quarter of fiscal 2024?": True,
    "What was the operating margin in fiscal 2024?": False,
    "What was Microsoft's net income?": False,          # a company isn't a qualifier
    "What was free cash flow in 2024?": False,
    "What were dividends paid in fiscal 2024?": False,
    "How fast did Azure grow?": True,
    "What was the Intelligent Cloud operating margin?": True,
}


# Segment questions are planned as calculations, with the part as the qualifier
SEGMENT_PLANS = {
    "What was Intelligent Cloud revenue in fiscal 2024?": ({"revenue"}, "intelligent cloud"),
    "What was the Intelligent Cloud operating margin?": ({"operating_margin"}, "intelligent cloud"),
    "How much revenue came from the United States in fiscal 2024?": ({"revenue"}, "united states"),
    "How did Gaming revenue change from 2023 to 2024?": ({"revenue"}, "gaming"),
}


@pytest.mark.parametrize("question", list(SEGMENT_PLANS))
def test_planner_plans_segment_questions_as_calculations(question):
    metrics, part = SEGMENT_PLANS[question]
    plan = router.plan_question(question)
    assert plan.route == "calculation", plan
    assert set(plan.metrics) == metrics, plan
    assert part in (plan.qualifier or "").lower(), plan


@pytest.mark.parametrize("question", list(QUALIFIED))
def test_planner_flags_qualified_metrics(question):
    plan = router.plan_question(question)
    assert (plan.qualifier is not None) == QUALIFIED[question], plan
