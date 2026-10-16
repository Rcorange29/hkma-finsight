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
any of them reports its value. Annual flows (income, cash) must cover about a
year; balances are as of the year end.

A fiscal year is named as the company names it, which is how its income
statement's column headers read. Usually that's the year the period ends in, but
not always: Target's year to January 31, 2026 is its "2025", and Johnson &
Johnson's year to January 1, 2023 was its "2022". Each 10-K's XBRL says which
fiscal year it reports (its `fy`), so a period end date takes the name of the
year whose 10-K it closes (fiscal_year_labels).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
from datetime import date
from decimal import Decimal

from financial_rag import edgar
from financial_rag.financials import LINE_ITEMS, Fact, page_number, row_printing

# Line item -> us-gaap tags that report it, most common first
TAGS: dict[str, tuple[str, ...]] = {
    "revenue": ("Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                "SalesRevenueNet", "RevenueFromContractWithCustomerIncludingAssessedTax"),
    "cost_of_revenue": ("CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold"),
    "gross_profit": ("GrossProfit",),
    "net_interest_income": ("InterestIncomeExpenseNet",),
    "noninterest_income": ("NoninterestIncome",),
    "provision_for_credit_losses": ("ProvisionForLoanLeaseAndOtherLosses",
                                    "ProvisionForLoanAndLeaseLosses",
                                    "ProvisionForLoanLossesExpensed"),
    "noninterest_expense": ("NoninterestExpense",),
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


def fiscal_year_labels(facts_json: dict) -> dict[date, int]:
    """Each fiscal year end a 10-K reports, mapped to the year the company names it.

    A 10-K's latest period end is the year end it reports, and its `fy` is that
    year's name.
    """
    latest: dict[str | None, tuple[date, int]] = {}
    for tag in facts_json.get("facts", {}).get("us-gaap", {}).values():
        for rows in tag.get("units", {}).values():
            for row in rows:
                if not str(row.get("form", "")).startswith("10-K") or row.get("fp") != "FY":
                    continue
                if not isinstance(row.get("fy"), int):
                    continue
                end = date.fromisoformat(row["end"])
                accession = row.get("accn")
                if accession not in latest or end > latest[accession][0]:
                    latest[accession] = (end, row["fy"])
    return dict(latest.values())


def year_name(end: date, labels: dict[date, int]) -> int:
    """The fiscal year a period ending on `end` belongs to, as the company names it.

    A year end no 10-K reports as its own (one only shown as a comparative) is
    named the way the nearest reported year end is.
    """
    if end in labels:
        return labels[end]
    if not labels:
        return end.year
    nearest = min(labels, key=lambda known: abs((known - end).days))
    return end.year + labels[nearest] - nearest.year


def annual_values(facts_json: dict, metric: str, fiscal_year: int,
                  accession: str | None = None,
                  labels: dict[date, int] | None = None) -> set[Decimal]:
    """Every value 10-K XBRL reports for a line item's fiscal year (restatements too),
    or only those one filing reported, given its accession number.

    `labels` is fiscal_year_labels(facts_json), for callers asking many times.
    """
    item = LINE_ITEMS[metric]
    gaap = facts_json.get("facts", {}).get("us-gaap", {})
    if labels is None:
        labels = fiscal_year_labels(facts_json)
    flow = item.statement in ("income", "cash")
    values = set()
    for tag in TAGS.get(metric, ()):
        for row in gaap.get(tag, {}).get("units", {}).get(_UNITS[item.unit], []):
            if not str(row.get("form", "")).startswith("10-K"):
                continue
            if accession is not None and row.get("accn") != accession:
                continue
            end = date.fromisoformat(row["end"])
            if year_name(end, labels) != fiscal_year:
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


def check(fact: Fact, facts_json: dict, labels: dict[date, int] | None = None) -> str:
    """match, mismatch or absent: how XBRL compares with one extracted figure."""
    if fact.segment or fact.metric not in TAGS:
        return ABSENT       # breakdowns are dimensional XBRL, not in companyfacts
    reported = annual_values(facts_json, fact.metric, fact.fiscal_year, labels=labels)
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
    reread: list[Fact] = field(default_factory=list)   # one figure per line item re-read


def crosscheck(facts: list[Fact], company: str | None, cik: int | None = None,
               pages: dict[int, str] | None = None) -> Report:
    """Mark figures XBRL confirms, and drop any it contradicts.

    Given the statement pages, a contradicted line item is re-read from the one
    row there that prints XBRL's values, if there is one (financials.row_printing);
    that row's figures replace all the line item's others.

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

    labels = fiscal_year_labels(facts_json)
    kept: list[Fact] = []
    counts: Counter = Counter()
    mismatches = []
    for fact in facts:
        result = check(fact, facts_json, labels)
        counts[result] += 1
        if result == MISMATCH:
            mismatches.append((fact, annual_values(facts_json, fact.metric, fact.fiscal_year,
                                                   labels=labels)))
            continue
        kept.append(replace(fact, xbrl=result) if result == MATCH else fact)

    reread: list[Fact] = []
    years = {fact.fiscal_year for fact in facts}
    for metric in dict.fromkeys(fact.metric for fact, _ in mismatches) if pages else ():
        expected = {year: annual_values(facts_json, metric, year, labels=labels)
                    for year in years}
        first = next(fact for fact, _ in mismatches if fact.metric == metric)
        rows = row_printing(pages, metric, expected, first.owner, first.source_file)
        if rows:
            kept = [f for f in kept if f.metric != metric or f.segment] + rows
            reread.append(rows[0])
    return Report(kept, counts, mismatches, cik, reread=reread)


def summary(report: Report, source_file: str) -> str:
    if report.skipped:
        return f"XBRL check skipped for {source_file}: {report.skipped}"
    c = report.counts
    line = (f"XBRL check for {source_file} (CIK {report.cik}): {c[MATCH]} match, "
            f"{c[MISMATCH]} contradicted and dropped, {c[ABSENT]} not in XBRL")
    for fact, reported in report.mismatches:
        shown = ", ".join(str(v) for v in sorted(reported))
        line += f"\n  dropped {fact.metric} FY{fact.fiscal_year}: read {fact.value}, XBRL {shown}"
    for fact in report.reread:
        line += (f"\n  re-read {fact.metric} from {fact.label!r} (p.{page_number(fact.page)}),"
                 " the row printing XBRL's values")
    return line
