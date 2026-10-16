"""Tests for the deterministic calculation engine (analysis.py).

Figures are Microsoft's FY2022-FY2024 reported amounts (in millions), so the
expected results can be checked by hand against the 10-K.
"""

from decimal import Decimal

import pytest

from analysis import METRICS, analyze, compute, format_value, index_facts, render
from financials import LINE_ITEMS, Fact

M = Decimal(1_000_000)

# metric: {fiscal year: value in millions}, page
MSFT = {
    "revenue": ({2024: 245122, 2023: 211915, 2022: 198270}, 66),
    "cost_of_revenue": ({2024: 74114, 2023: 65863, 2022: 62650}, 66),
    "gross_profit": ({2024: 171008, 2023: 146052, 2022: 135620}, 66),
    "operating_income": ({2024: 109433, 2023: 88523, 2022: 83383}, 66),
    "net_income": ({2024: 88136, 2023: 72361, 2022: 72738}, 66),
    "total_current_assets": ({2024: 159734, 2023: 184257}, 68),
    "total_current_liabilities": ({2024: 125286, 2023: 104149}, 68),
    "total_equity": ({2024: 268477, 2023: 206223}, 68),
    "operating_cash_flow": ({2024: 118548, 2023: 87582, 2022: 89035}, 70),
    "capital_expenditures": ({2024: 44477, 2023: 28107, 2022: 23886}, 70),
}


def _facts(source="msft.pdf", owner="public", only=None, overrides=None):
    facts = []
    for metric, (values, page) in MSFT.items():
        if only and metric not in only:
            continue
        for year, millions in values.items():
            if overrides and (metric, year) in overrides:
                millions = overrides[(metric, year)]
                if millions is None:
                    continue
            item = LINE_ITEMS[metric]
            facts.append(Fact(owner, source, metric, year, Decimal(millions) * M, item.unit,
                              item.name, page, f"{item.name}   {millions:,}"))
    return facts


def _one(metric, year, **kw):
    return compute(METRICS[metric], index_facts(_facts(**kw))["msft.pdf"], "msft.pdf", year)


# ---------------------------------------------------------------------------
# Single results
# ---------------------------------------------------------------------------

def test_line_item_is_read_directly_with_its_source():
    r = _one("revenue", 2024)
    assert r.value == Decimal("245122") * M
    assert r.formula is None
    assert [(f.metric, f.page) for f in r.inputs] == [("revenue", 66)]


def test_ratio_is_exact_decimal_arithmetic():
    r = _one("operating_margin", 2024)
    assert r.value == Decimal(109433) / Decimal(245122)
    assert r.formula == "Operating income ÷ Total revenue"
    assert [f.metric for f in r.inputs] == ["operating_income", "revenue"]


def test_gross_margin_falls_back_to_revenue_minus_cost():
    r = _one("gross_margin", 2024, overrides={("gross_profit", 2024): None})
    assert r.value == (Decimal(245122) - Decimal(74114)) / Decimal(245122)
    assert r.formula == "(Total revenue − Cost of revenue) ÷ Total revenue"


def test_free_cash_flow_subtracts_capex():
    assert _one("free_cash_flow", 2024).value == (Decimal(118548) - Decimal(44477)) * M


def test_missing_input_is_reported_not_estimated():
    r = _one("current_ratio", 2022)
    assert r.value is None
    assert r.inputs == []
    assert "Total current assets, Total current liabilities for FY2022" in r.problem


def test_zero_denominator_is_not_meaningful():
    r = _one("operating_margin", 2024, overrides={("revenue", 2024): 0})
    assert r.value is None
    assert r.problem == "not meaningful: Total revenue is zero."


# ---------------------------------------------------------------------------
# Series: years, changes, growth
# ---------------------------------------------------------------------------

def test_defaults_to_the_latest_year_the_metric_can_be_computed_for():
    [s] = analyze(_facts(), ["current_ratio"], [], False)
    assert [r.fiscal_year for r in s.results] == [2024]


def test_year_over_year_adds_the_prior_year_and_the_change():
    [s] = analyze(_facts(), ["revenue"], [], True)
    assert [r.fiscal_year for r in s.results] == [2023, 2024]
    [change] = s.changes
    assert change.absolute == (Decimal(245122) - Decimal(211915)) * M
    assert change.relative == (Decimal(245122) - Decimal(211915)) / Decimal(211915)


def test_several_requested_years_are_compared_without_adding_another():
    # A planner may list 2023 and also ask for year-over-year; that must not pull in 2022
    [s] = analyze(_facts(), ["operating_margin"], [2024, 2023], True)
    assert [r.fiscal_year for r in s.results] == [2023, 2024]
    [s] = analyze(_facts(), ["revenue"], [2022, 2023, 2024], True)
    assert [r.fiscal_year for r in s.results] == [2022, 2023, 2024]


def test_missing_comparison_year_is_explained_not_dropped():
    # Balance sheets cover two years: FY2023's comparison year (2022) isn't there
    [s] = analyze(_facts(), ["current_ratio"], [2023], True)
    assert [(r.fiscal_year, r.value is None) for r in s.results] == [(2022, True), (2023, False)]
    assert "not in the extracted figures" in s.results[0].problem
    assert s.changes == []


def test_explicitly_requested_missing_year_is_reported():
    [s] = analyze(_facts(), ["revenue"], [2021, 2024], False)
    assert [(r.fiscal_year, r.value is None) for r in s.results] == [(2021, True), (2024, False)]


def test_compound_growth_over_a_multi_year_span():
    [s] = analyze(_facts(), ["revenue"], [2022, 2023, 2024], False)
    first, last, rate = s.cagr
    assert (first, last) == (2022, 2024)
    assert abs(rate - ((Decimal(245122) / Decimal(198270)).sqrt() - 1)) < Decimal("1e-20")
    assert len(s.changes) == 2


def test_ratio_changes_have_no_percentage_growth():
    [s] = analyze(_facts(), ["operating_margin"], [], True)
    assert s.changes[0].relative is None
    assert s.cagr is None


def test_each_filing_gets_its_own_series():
    facts = _facts("msft.pdf") + _facts("other.pdf", owner="alice", only={"revenue"})
    series = analyze(facts, ["revenue"], [2024], False)
    assert sorted(s.source_file for s in series) == ["msft.pdf", "other.pdf"]


# ---------------------------------------------------------------------------
# Presentation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("value", "unit", "text"),
    [
        (Decimal(245122) * M, "USD", "$245,122 million"),
        (Decimal(-1646) * M, "USD", "-$1,646 million"),
        (Decimal(123456) * 1000, "USD", "$123.5 million"),
        (Decimal(250000), "USD", "$250,000"),
        (Decimal("11.80"), "USD/share", "$11.80 per share"),
        (Decimal(7469) * M, "shares", "7,469 million shares"),
        (Decimal(109433) / Decimal(245122), "percent", "44.6%"),
        (Decimal(159734) / Decimal(125286), "multiple", "1.27x"),
    ],
)
def test_format_value(value, unit, text):
    assert format_value(value, unit) == text


def test_render_cites_every_input_number():
    series = analyze(_facts(), ["operating_margin"], [], True)
    text, sources = render(series)
    assert "**FY2024: 44.6%** = Operating income ÷ Total revenue" in text
    assert "Operating income FY2024: $109,433 million (msft.pdf, p.66)" in text
    assert "Total revenue FY2024: $245,122 million (msft.pdf, p.66)" in text
    assert "Total revenue FY2023: $211,915 million (msft.pdf, p.66)" in text
    # 44.644% - 41.773% = 2.871 points: computed from exact values, not the
    # rounded ones shown (44.6 - 41.8 would give 2.8)
    assert "Change FY2023 → FY2024: +2.9 percentage points" in text
    # One source per input row, each with the row as printed
    assert {s["excerpt"] for s in sources} == {
        "Operating income   109,433", "Total revenue   245,122",
        "Operating income   88,523", "Total revenue   211,915",
    }


def test_render_line_item_growth():
    text, _ = render(analyze(_facts(), ["revenue"], [2022, 2023, 2024], False))
    assert "**FY2024: $245,122 million** (msft.pdf, p.66)" in text
    assert "Change FY2023 → FY2024: +$33,207 million (+15.7%)" in text
    assert "Compound annual growth FY2022 → FY2024: 11.2% a year" in text


def test_render_explains_what_is_missing():
    text, _ = render(analyze(_facts(), ["current_ratio"], [2022], False))
    assert "FY2022: not calculated: needs Total current assets, Total current liabilities" in text


def test_render_shows_the_printed_label_when_it_differs():
    facts = [f if f.metric != "capital_expenditures" else
             Fact(**{**f.__dict__, "label": "Additions to property and equipment"})
             for f in _facts()]
    text, _ = render(analyze(facts, ["free_cash_flow"], [2024], False))
    assert ("Capital expenditures (“Additions to property and equipment”) FY2024: "
            "$44,477 million (msft.pdf, p.70)") in text
