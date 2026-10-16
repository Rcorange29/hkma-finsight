"""
xbrl.py — Check extracted figures against the company's own XBRL data.

Every 10-K tags its financial statements in XBRL, and the SEC publishes the tagged
values per company (EDGAR's companyfacts API). Those are the company's official
numbers, so a figure read from the filing's text can be checked against them:

  match     XBRL reports the same value for that line item and fiscal year
  mismatch  XBRL reports a different value: the reading is wrong, so the figure is
            dropped rather than stored
  absent    XBRL has no value to compare with (a tag this map doesn't know, or a
            filing too new or too old): the figure stands on the page check alone

Each line item maps to the us-gaap tags companies use for it; a figure matches if
any of them reports its value. A fiscal year is the year of the period's end
date, which is how the statements' own column headers read. Annual flows
(income, cash) must cover about a year; balances are as of the year end.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from datetime import date
from decimal import Decimal

from financial_rag import edgar
from financial_rag.financials import LINE_ITEMS, Fact

# Line item -> us-gaap tags that report it, most common first
TAGS: dict[str, tuple[str, ...]] = {
    "revenue": ("Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                "SalesRevenueNet", "RevenueFromContractWithCustomerIncludingAssessedTax"),
    "cost_of_revenue": ("CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold"),
    "gross_profit": ("GrossProfit",),
    "research_and_development": ("ResearchAndDevelopmentExpense",
                                 "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost"),
    "operating_income": ("OperatingIncomeLoss",),
    "income_before_taxes": (
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
    ),
    "income_tax_expense": ("IncomeTaxExpenseBenefit",),
    "net_income": ("NetIncomeLoss",),
    "eps_basic": ("EarningsPerShareBasic",),
    "eps_diluted": ("EarningsPerShareDiluted",),
    "diluted_shares": ("WeightedAverageNumberOfDilutedSharesOutstanding",),
    "cash_and_equivalents": ("CashAndCashEquivalentsAtCarryingValue",),
    "total_current_assets": ("AssetsCurrent",),
    "total_assets": ("Assets",),
    "total_current_liabilities": ("LiabilitiesCurrent",),
    "total_liabilities": ("Liabilities",),
    "long_term_debt": ("LongTermDebtNoncurrent", "LongTermDebtAndCapitalLeaseObligations"),
    "total_equity": ("StockholdersEquity",
                     "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"),
    "operating_cash_flow": ("NetCashProvidedByUsedInOperatingActivities",
                            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"),
    "capital_expenditures": ("PaymentsToAcquirePropertyPlantAndEquipment",
                             "PaymentsToAcquireProductiveAssets"),
    "dividends_paid": ("PaymentsOfDividends", "PaymentsOfDividendsCommonStock"),
    "share_repurchases": ("PaymentsForRepurchaseOfCommonStock",),
}

_UNITS = {"USD": "USD", "USD/share": "USD/shares", "shares": "shares"}
_ANNUAL_DAYS = range(350, 380)      # 52/53-week years included
MATCH, MISMATCH, ABSENT = "match", "mismatch", "absent"


def annual_values(facts_json: dict, metric: str, fiscal_year: int) -> set[Decimal]:
    """Every value 10-K XBRL reports for a line item's fiscal year (restatements too)."""
    item = LINE_ITEMS[metric]
    gaap = facts_json.get("facts", {}).get("us-gaap", {})
    flow = item.statement in ("income", "cash")
    values = set()
    for tag in TAGS.get(metric, ()):
        for row in gaap.get(tag, {}).get("units", {}).get(_UNITS[item.unit], []):
            if not str(row.get("form", "")).startswith("10-K"):
                continue
            end = date.fromisoformat(row["end"])
            if end.year != fiscal_year:
                continue
            if flow:
                if "start" not in row:
                    continue
                if (end - date.fromisoformat(row["start"])).days not in _ANNUAL_DAYS:
                    continue    # a quarter or a stub period
            elif "start" in row:
                continue
            values.add(Decimal(str(row["val"])))
    return values


def check(fact: Fact, facts_json: dict) -> str:
    """match, mismatch or absent: how XBRL compares with one extracted figure."""
    if fact.segment or fact.metric not in TAGS:
        return ABSENT       # breakdowns are dimensional XBRL, not in companyfacts
    reported = annual_values(facts_json, fact.metric, fact.fiscal_year)
    if not reported:
        return ABSENT
    magnitude = LINE_ITEMS[fact.metric].magnitude
    ours = abs(fact.value) if magnitude else fact.value
    return MATCH if any((abs(v) if magnitude else v) == ours for v in reported) else MISMATCH


@dataclass(frozen=True)
class Report:
    facts: list[Fact]           # what to store: matches marked, mismatches removed
    counts: Counter
    mismatches: list[tuple[Fact, set[Decimal]]]
    cik: int | None = None
    skipped: str = ""           # why no check ran, if it didn't


def crosscheck(facts: list[Fact], company: str | None, cik: int | None = None) -> Report:
    """Mark figures XBRL confirms, and drop any it contradicts.

    Never raises for EDGAR trouble: without a contact address, a known company or
    a network, the figures are returned unchanged with the reason in `skipped`.
    """
    try:
        cik = cik or edgar.cik_for_name(company)
        if cik is None:
            return Report(facts, Counter(), [], skipped=f"no single SEC company named {company!r}")
        facts_json = edgar.companyfacts(cik)
    except edgar.EdgarUnavailable as exc:
        return Report(facts, Counter(), [], skipped=str(exc))

    kept: list[Fact] = []
    counts: Counter = Counter()
    mismatches = []
    for fact in facts:
        result = check(fact, facts_json)
        counts[result] += 1
        if result == MISMATCH:
            mismatches.append((fact, annual_values(facts_json, fact.metric, fact.fiscal_year)))
            continue
        kept.append(replace(fact, xbrl=result) if result == MATCH else fact)
    return Report(kept, counts, mismatches, cik)


def summary(report: Report, source_file: str) -> str:
    if report.skipped:
        return f"XBRL check skipped for {source_file}: {report.skipped}"
    c = report.counts
    line = (f"XBRL check for {source_file} (CIK {report.cik}): {c[MATCH]} match, "
            f"{c[MISMATCH]} contradicted and dropped, {c[ABSENT]} not in XBRL")
    for fact, reported in report.mismatches:
        shown = ", ".join(str(v) for v in sorted(reported))
        line += f"\n  dropped {fact.metric} FY{fact.fiscal_year}: read {fact.value}, XBRL {shown}"
    return line
