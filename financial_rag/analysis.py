"""
analysis.py — Deterministic financial calculations over extracted figures.

Margins, ratios, growth rates and year-over-year comparisons are computed here
in Python with Decimal arithmetic, never by a language model. Every result
keeps the figures it was computed from, and every figure keeps the file, page
and row it was read from (see financials.py), so answers cite each input.

Nothing is estimated: if an input is missing for a year, the result says which
one instead of producing a number.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from itertools import pairwise

from financial_rag.financials import (
    BREAKDOWNS,
    ITEM_NAMES,
    LINE_ITEMS,
    Fact,
    page_number,
    same_words,
)

# ---------------------------------------------------------------------------
# Metric catalog
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Formula:
    inputs: tuple[str, ...]                 # line item keys, in argument order
    compute: Callable[..., Decimal]
    text: str


@dataclass(frozen=True)
class Metric:
    key: str
    name: str
    unit: str           # USD | USD/share | shares | percent | multiple
    description: str
    formulas: tuple[Formula, ...] = ()      # empty: a line item, read directly
    # For breakdowns: segment | geography | product. Computed once per part,
    # with every input read for that same part.
    dimension: str | None = None

    @property
    def is_line_item(self) -> bool:
        return not self.formulas


def _ratio(key, name, numerator, denominator, unit, description):
    num, den = LINE_ITEMS[numerator].name, LINE_ITEMS[denominator].name
    return Metric(key, name, unit, description,
                  (Formula((numerator, denominator), lambda a, b: a / b, f"{num} ÷ {den}"),))


_DERIVED = [
    Metric("gross_margin", "Gross margin", "percent", "Gross profit as a share of revenue.", (
        Formula(("gross_profit", "revenue"), lambda g, r: g / r, "Gross profit ÷ Total revenue"),
        Formula(("revenue", "cost_of_revenue"), lambda r, c: (r - c) / r,
                "(Total revenue − Cost of revenue) ÷ Total revenue"),
    )),
    _ratio("operating_margin", "Operating margin", "operating_income", "revenue", "percent",
           "Operating income as a share of revenue."),
    _ratio("net_margin", "Net profit margin", "net_income", "revenue", "percent",
           "Net income as a share of revenue."),
    _ratio("rd_intensity", "R&D as a share of revenue", "research_and_development", "revenue",
           "percent", "Research and development expense as a share of revenue."),
    _ratio("effective_tax_rate", "Effective tax rate", "income_tax_expense",
           "income_before_taxes", "percent", "Income tax provision as a share of pre-tax income."),
    Metric("free_cash_flow", "Free cash flow", "USD",
           "Operating cash flow minus capital expenditures.", (
               Formula(("operating_cash_flow", "capital_expenditures"), lambda o, c: o - c,
                       "Net cash from operating activities − Capital expenditures"),
           )),
    Metric("fcf_margin", "Free cash flow margin", "percent",
           "Free cash flow as a share of revenue.", (
               Formula(("operating_cash_flow", "capital_expenditures", "revenue"),
                       lambda o, c, r: (o - c) / r,
                       "(Net cash from operating activities − Capital expenditures) ÷ Total revenue"),
           )),
    _ratio("current_ratio", "Current ratio", "total_current_assets",
           "total_current_liabilities", "multiple", "Current assets per dollar of current liabilities."),
    _ratio("debt_to_equity", "Long-term debt to equity", "long_term_debt", "total_equity",
           "multiple", "Long-term debt (non-current) relative to stockholders' equity."),
    _ratio("liabilities_to_equity", "Total liabilities to equity", "total_liabilities",
           "total_equity", "multiple", "Total liabilities relative to stockholders' equity."),
    _ratio("return_on_equity", "Return on equity", "net_income", "total_equity", "percent",
           "Net income relative to year-end stockholders' equity."),
    _ratio("return_on_assets", "Return on assets", "net_income", "total_assets", "percent",
           "Net income relative to year-end total assets."),
    _ratio("payout_ratio", "Dividend payout ratio", "dividends_paid", "net_income", "percent",
           "Cash dividends paid as a share of net income."),
]

_BREAKDOWN_METRICS = [
    *(Metric(b.key, b.name, "USD",
             f"{ITEM_NAMES[b.total]} for one {b.dimension}, from the filing's breakdown.",
             dimension=b.dimension)
      for b in BREAKDOWNS.values()),
    Metric("segment_operating_margin", "Segment operating margin", "percent",
           "A segment's operating income as a share of its revenue.", (
               Formula(("segment_operating_income", "segment_revenue"), lambda o, r: o / r,
                       "Segment operating income ÷ Segment revenue"),
           ), dimension="segment"),
]

METRICS: dict[str, Metric] = {
    **{k: Metric(k, item.name, item.unit, item.hint) for k, item in LINE_ITEMS.items()},
    **{m.key: m for m in _DERIVED},
    **{m.key: m for m in _BREAKDOWN_METRICS},
}

# The breakdown metric answering "<company-wide metric> for <some part>"
BREAKDOWN_OF: dict[str, list[str]] = {
    "revenue": ["segment_revenue", "geographic_revenue", "product_revenue"],
    "operating_income": ["segment_operating_income"],
    "operating_margin": ["segment_operating_margin"],
}


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------


@dataclass
class Result:
    metric: Metric
    source_file: str
    fiscal_year: int
    value: Decimal | None
    formula: str | None = None              # None for line items
    inputs: list[Fact] = field(default_factory=list)
    problem: str | None = None              # why value is None


@dataclass
class Change:
    from_year: int
    to_year: int
    absolute: Decimal
    relative: Decimal | None                # None for ratios, or a non-positive base


@dataclass
class Series:
    metric: Metric
    source_file: str
    results: list[Result]
    changes: list[Change]
    cagr: tuple[int, int, Decimal] | None = None     # (from year, to year, rate)
    part: str = ""                                   # for breakdowns: the segment etc.


FactIndex = dict[tuple[str, str, int], Fact]


def index_facts(facts: list[Fact]) -> dict[str, FactIndex]:
    """Group facts by source file, keyed by (metric, segment, fiscal_year)."""
    by_file: dict[str, FactIndex] = {}
    for fact in facts:
        key = (fact.metric, fact.segment, fact.fiscal_year)
        by_file.setdefault(fact.source_file, {})[key] = fact
    return by_file


def parts_of(metric: Metric, index: FactIndex) -> list[str]:
    """The parts (segments, areas, product lines) a breakdown metric has figures for."""
    if metric.dimension is None:
        return []
    key = metric.formulas[0].inputs[0] if metric.formulas else metric.key
    return sorted({part for k, part, _ in index if k == key})


def compute(
    metric: Metric, index: FactIndex, source_file: str, year: int, part: str = ""
) -> Result:
    """One metric for one fiscal year of one filing (and one part, for breakdowns)."""
    part = part if metric.dimension else ""
    of_part = f" for {part}" if part else ""
    if metric.is_line_item:
        fact = index.get((metric.key, part, year))
        if fact is None:
            return Result(metric, source_file, year, None,
                          problem=f"{metric.name}{of_part} for FY{year} is not in the "
                                  "extracted figures.")
        return Result(metric, source_file, year, fact.value, inputs=[fact])

    # Formulas are tried in order; the first whose inputs all exist is used.
    for formula in metric.formulas:
        facts = [index.get((key, part, year)) for key in formula.inputs]
        if any(f is None for f in facts):
            continue
        try:
            value = formula.compute(*(f.value for f in facts))
        except (ZeroDivisionError, ArithmeticError):
            zero = next((f for f in facts if f.value == 0), None)
            what = f"{ITEM_NAMES[zero.metric]} is zero" if zero else "division by zero"
            return Result(metric, source_file, year, None, formula.text, facts,
                          problem=f"not meaningful: {what}.")
        return Result(metric, source_file, year, value, formula.text, facts)

    primary = metric.formulas[0]
    missing = [ITEM_NAMES[k] + of_part for k in primary.inputs if (k, part, year) not in index]
    return Result(metric, source_file, year, None, primary.text,
                  problem=f"needs {', '.join(missing)} for FY{year}, which "
                          f"{'is' if len(missing) == 1 else 'are'} not in the extracted figures.")


def _change(metric: Metric, older: Result, newer: Result) -> Change:
    absolute = newer.value - older.value
    relative = None
    if metric.unit in ("USD", "USD/share", "shares") and older.value > 0:
        relative = absolute / older.value
    return Change(older.fiscal_year, newer.fiscal_year, absolute, relative)


def analyze(
    facts: list[Fact],
    metric_keys: list[str],
    fiscal_years: list[int],
    year_over_year: bool,
    parts: list[str] | None = None,
) -> list[Series]:
    """
    Compute each metric for each filing that has figures.

    Breakdown metrics get one series per part: the `parts` asked for that the
    filing has, or every part it has when none are asked for.

    Years: the requested ones, or else the latest year the metric can be
    computed for. With year_over_year and a single year, the year before is
    added so the change can be shown; several requested years are already
    compared with each other. Changes are reported between consecutive years
    shown, plus a compound annual growth rate when the span is over a year.
    """
    series: list[Series] = []
    for source_file, index in index_facts(facts).items():
        available = sorted({year for _, _, year in index}, reverse=True)
        for key in metric_keys:
            metric = METRICS[key]
            if metric.dimension is None:
                targets = [""]
            else:
                # Parts asked for are shown even when a figure is missing, with the
                # reason; a filing with no such breakdown at all is skipped.
                present = parts_of(metric, index)
                targets = (list(parts) if present else []) if parts else present
            for part in targets:
                series.extend(_series(metric, index, source_file, available, part,
                                      fiscal_years, year_over_year))
    return series


def _series(
    metric: Metric, index: FactIndex, source_file: str, available: list[int], part: str,
    fiscal_years: list[int], year_over_year: bool,
) -> list[Series]:
    if fiscal_years:
        years = set(fiscal_years)
    else:
        latest = next(
            (y for y in available
             if compute(metric, index, source_file, y, part).value is not None),
            available[0] if available else None,
        )
        years = {latest} if latest is not None else set()
    if year_over_year and len(years) == 1:
        years |= {min(years) - 1}
    # A year that can't be computed stays in, with the reason, so a missing
    # comparison is explained rather than silently absent.
    results = [compute(metric, index, source_file, y, part) for y in sorted(years)]
    valued = [r for r in results if r.value is not None]
    changes = [_change(metric, a, b) for a, b in pairwise(valued)]
    cagr = None
    if (
        len(valued) >= 2
        and metric.unit in ("USD", "USD/share", "shares")
        and valued[-1].fiscal_year - valued[0].fiscal_year > 1
        and valued[0].value > 0
        and valued[-1].value > 0
    ):
        first, last = valued[0], valued[-1]
        span = last.fiscal_year - first.fiscal_year
        rate = (last.value / first.value) ** (Decimal(1) / span) - 1
        cagr = (first.fiscal_year, last.fiscal_year, rate)
    return [Series(metric, source_file, results, changes, cagr, part)]


# ---------------------------------------------------------------------------
# Presentation
# ---------------------------------------------------------------------------


def _round(value: Decimal, places: int) -> Decimal:
    return value.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)


def _signed(text: str, negative: bool) -> str:
    return f"-{text}" if negative else text


def format_value(value: Decimal, unit: str) -> str:
    if unit == "percent":
        return f"{_round(value * 100, 1)}%"
    if unit == "multiple":
        return f"{_round(value, 2)}x"
    if unit == "USD/share":
        return _signed(f"${_round(abs(value), 2):,} per share", value < 0)
    if unit == "shares":
        return f"{_round(value / 1_000_000, 1).normalize():,f} million shares"
    # USD: filings report in millions, so show millions unless the amount is tiny
    millions = abs(value) / 1_000_000
    if millions < 1:
        return _signed(f"${_round(abs(value), 0):,}", value < 0)
    places = 0 if millions == millions.to_integral_value() else 1
    return _signed(f"${_round(millions, places):,} million", value < 0)


def format_change(metric: Metric, change: Change) -> str:
    sign = "+" if change.absolute >= 0 else "-"
    if metric.unit == "percent":
        return f"{sign}{_round(abs(change.absolute) * 100, 1)} percentage points"
    if metric.unit == "multiple":
        return f"{sign}{_round(abs(change.absolute), 2)}x"
    text = f"{sign}{format_value(abs(change.absolute), metric.unit)}"
    if change.relative is not None:
        text += f" ({sign}{_round(abs(change.relative) * 100, 1)}%)"
    return text


def cite(fact: Fact) -> str:
    return f"({fact.source_file}, p.{page_number(fact.page)})"


def _describe_input(fact: Fact) -> str:
    name = ITEM_NAMES[fact.metric]
    printed = "" if same_words(fact.label, name) else f" (“{fact.label}”)"
    return f"{name}{printed} FY{fact.fiscal_year}: {format_value(fact.value, fact.unit)} {cite(fact)}"


FOOTER = (
    "_Calculated in Python from figures read from the filing's financial statements. "
    "Each input was matched to the cited row and fiscal-year column; nothing is estimated._"
)


def render(series: list[Series]) -> tuple[str, list[dict]]:
    """Markdown answer plus one source citation per input row used."""
    blocks: list[str] = []
    sources: dict[tuple[str, int, str], dict] = {}
    for s in series:
        title = f"{s.metric.name}: {s.part}" if s.part else s.metric.name
        lines = [f"**{title}** · {s.source_file}"]
        for r in s.results:
            if r.value is None:
                lines.append(f"- FY{r.fiscal_year}: not calculated: {r.problem}")
                continue
            shown = format_value(r.value, s.metric.unit)
            if s.metric.is_line_item:
                fact = r.inputs[0]
                printed = (
                    "" if same_words(fact.label, s.metric.name) or fact.label == s.part
                    else f" (“{fact.label}”)"
                )
                lines.append(f"- **FY{r.fiscal_year}: {shown}**{printed} {cite(fact)}")
            else:
                lines.append(f"- **FY{r.fiscal_year}: {shown}** = {r.formula}, from:")
                lines += [f"    - {_describe_input(f)}" for f in r.inputs]
            for f in r.inputs:
                sources[(f.source_file, f.page, f.line)] = {
                    "source_file": f.source_file, "page": page_number(f.page),
                    "excerpt": f.line,
                }
        for c in s.changes:
            lines.append(f"- Change FY{c.from_year} → FY{c.to_year}: {format_change(s.metric, c)}")
        if s.cagr is not None:
            first, last, rate = s.cagr
            lines.append(
                f"- Compound annual growth FY{first} → FY{last}: {_round(rate * 100, 1)}% a year"
            )
        blocks.append("\n".join(lines))
    blocks.append(FOOTER)
    return "\n\n".join(blocks), list(sources.values())
