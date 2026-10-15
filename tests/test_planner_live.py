"""Routing accuracy of the real planner model. Opt-in: it calls OpenAI.

    RUN_LIVE_TESTS=1 pytest tests/test_planner_live.py

The planner is the one model-dependent step between a question and the
deterministic engine, so this pins its behaviour on representative questions.
Re-run it after changing PLANNER_PROMPT or PLANNER_MODEL.
"""

import os

import pytest

import router

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_LIVE_TESTS") != "1", reason="calls OpenAI; set RUN_LIVE_TESTS=1"
)

DOCUMENTS = {"msft.pdf": "- msft.pdf (sample filing): fiscal years 2022-2024"}

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
    "How fast did Azure grow?": ("text", None, None, None),
    "What are the main risk factors related to AI?": ("text", None, None, None),
    "Who is the CEO?": ("text", None, None, None),
    "What does the company say about its revenue recognition policy?": ("text", None, None, None),
    "How many employees does Microsoft have?": ("text", None, None, None),
}


@pytest.mark.parametrize("question", list(CASES))
def test_planner_routes_question(question):
    route, metrics, years, yoy = CASES[question]
    plan = router.plan_question(question, DOCUMENTS)
    assert plan.route == route
    if metrics is not None:
        assert set(plan.metrics) == metrics
    if years is not None:
        assert sorted(plan.fiscal_years) == years
    if yoy is not None:
        assert plan.year_over_year == yoy
