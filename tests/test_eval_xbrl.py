"""Tests for generating benchmark questions from XBRL data (eval_xbrl.py)."""

from decimal import Decimal

from financial_rag import eval_xbrl, xbrl

ACCN = "0000320193-25-000079"
OLD = "0000320193-24-000123"
FY25 = {"start": "2024-09-29", "end": "2025-09-27", "form": "10-K", "accn": ACCN}
END25 = {"end": "2025-09-27", "form": "10-K", "accn": ACCN}


def _tag(unit, *rows):
    return {"units": {unit: [dict(period, val=val) for period, val in rows]}}


FACTS = {"facts": {"us-gaap": {
    "RevenueFromContractWithCustomerExcludingAssessedTax": _tag(
        "USD", (FY25, 416161000000),
        # A later filing restated it; the tested filing's own value wins
        (dict(FY25, accn="0000320193-26-000001"), 416000000000)),
    "OperatingIncomeLoss": _tag("USD", (FY25, 133050000000)),
    "NetIncomeLoss": _tag("USD", (FY25, 112010000000)),
    "EarningsPerShareDiluted": _tag("USD/shares", (FY25, 7.46)),
    "NetCashProvidedByUsedInOperatingActivities": _tag("USD", (FY25, 111482000000)),
    "PaymentsToAcquirePropertyPlantAndEquipment": _tag("USD", (FY25, 12715000000)),
    "AssetsCurrent": _tag("USD", (END25, 147957000000)),
    "LiabilitiesCurrent": _tag("USD", (END25, 165631000000)),
    # No total assets for 2025: that question is skipped, never guessed
    "Assets": _tag("USD", (dict(END25, end="2024-09-28", accn=OLD), 364980000000)),
}}}


def test_cases_come_from_the_filings_own_xbrl_values():
    cases = {c.metric: c for c in eval_xbrl.make_cases("AAPL", "Apple Inc.", 2025, FACTS, ACCN)}
    assert set(cases) == {"revenue", "operating_income", "net_income", "eps_diluted",
                          "operating_cash_flow", "operating_margin", "net_margin",
                          "free_cash_flow", "current_ratio"}
    assert cases["revenue"].question == "What was Apple Inc.'s total revenue in fiscal 2025?"
    assert cases["revenue"].expected == "$416,161 million"
    assert cases["eps_diluted"].expected == "$7.46 per share"
    assert cases["operating_margin"].expected == "32.0%"
    assert cases["net_margin"].expected == "26.9%"
    assert cases["free_cash_flow"].expected == "$98,767 million"
    assert cases["current_ratio"].expected == "0.89x"


def test_company_names_read_naturally():
    assert eval_xbrl.company_name("MICROSOFT CORP") == "Microsoft Corp"
    assert eval_xbrl.company_name("Apple Inc.") == "Apple Inc."


def test_annual_values_can_be_limited_to_one_filing():
    assert xbrl.annual_values(FACTS, "revenue", 2025) == {
        Decimal(416161000000), Decimal(416000000000)}
    assert xbrl.annual_values(FACTS, "revenue", 2025, ACCN) == {Decimal(416161000000)}


def test_a_value_without_the_filings_own_is_used_only_if_unambiguous():
    assert eval_xbrl.xbrl_value(FACTS, "total_assets", 2024, ACCN) == Decimal(364980000000)
    assert eval_xbrl.xbrl_value(FACTS, "total_assets", 2025, ACCN) is None


def test_scoring_needs_the_figure_the_route_and_a_citation(monkeypatch):
    from financial_rag import router

    case = eval_xbrl.Case("AAPL", "revenue", 2025, "Revenue?", "$416,161 million")
    monkeypatch.setattr(router, "answer", lambda q, owner: router.Answer(
        "**FY2025: $416,161 million** (aapl-10k-2025-09-27.htm, p.31)", [], "calculation"))
    row = eval_xbrl.run(case)
    assert (row["correct"], row["route_ok"], row["cited"]) == (True, True, True)
    monkeypatch.setattr(router, "answer", lambda q, owner: router.Answer(
        "Revenue was $416.2 billion.", [], "text"))
    row = eval_xbrl.run(case)
    assert (row["correct"], row["route_ok"], row["cited"]) == (False, False, False)
