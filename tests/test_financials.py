"""Tests for extracting structured figures from statement text (financials.py).

The page text below reproduces rows from Microsoft's FY2024 10-K as PyPDF
extracts them: figures right-aligned in year columns, negatives printed as
"(1,646 )", and labels that contain numbers or wrap onto a second line.
No API key needed: the model's row mapping is replaced by a fake.
"""

from decimal import Decimal

import pytest

import financials
from financials import (
    Fact,
    RowMapping,
    extract_facts,
    header_years,
    page_scale,
    parse_number,
    split_row,
    statement_pages,
    verify_row,
)

INCOME = """PART II
Item 8
ITEM 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA
INCOME STATEMENTS
(In millions, except per share amounts)
Year Ended June 30,  2024   2023   2022
Revenue:
Product  $ 64,773   $ 64,699   $ 72,732
Service and other   180,349    147,216    125,538
Total revenue   245,122    211,915    198,270
Total cost of revenue   74,114    65,863    62,650
Gross margin   171,008    146,052    135,620
Operating income   109,433    88,523    83,383
Other income (expense), net   (1,646 )   788    333
Net income  $ 88,136   $ 72,361   $ 72,738
Earnings per share:
Basic  $ 11.86   $ 9.72   $ 9.70
Diluted  $ 11.80   $ 9.68   $ 9.65
Weighted average shares outstanding:
Basic   7,431    7,446    7,496
Diluted   7,469    7,472    7,540
"""

BALANCE = """PART II
Item 8
BALANCE SHEETS
(In millions)
June 30,  2024   2023
Cash and cash equivalents  $ 18,315   $ 34,704
Accounts receivable, net of allowance for doubtful accounts of $830 and $650   56,924    48,688
Total current assets   159,734    184,257
Total assets  $ 512,163   $ 411,976
Total current liabilities   125,286    104,149
Total liabilities   243,686    205,753
Total stockholders� equity   268,477    206,223
"""

CASH = """PART II
Item 8
CASH FLOWS STATEMENTS
(In millions)
Year Ended June 30,  2024   2023   2022
Net cash from operations   118,548    87,582    89,035
Common stock cash dividends paid   (21,771 )   (19,800 )   (18,135 )
Other, net   (1,309 )   (1,006 )   (863 )
Additions to property and equipment   (44,477 )   (28,107 )   (23,886 )
Acquisition of companies, net of cash acquired, and purchases of
intangible and other assets
(69,132 )   (1,670 )   (22,038 )
Other, net   (1,298 )   (3,116 )   (2,825 )
Net cash used in investing   (96,970 )   (22,680 )   (30,311 )
"""

PAGES = {66: INCOME, 68: BALANCE, 70: CASH}


# ---------------------------------------------------------------------------
# Reading numbers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("245,122", Decimal("245122")),
        ("$ 11.80", Decimal("11.80")),
        ("$650", Decimal("650")),
        ("(1,646)", Decimal("-1646")),
        ("-", Decimal("0")),
        ("—", Decimal("0")),
        ("0", Decimal("0")),
    ],
)
def test_parse_number_reads_printed_figures(token, expected):
    assert parse_number(token) == expected


@pytest.mark.parametrize("token", ["16%", "net", "1,23", "(a)", "2,4,6", ""])
def test_parse_number_rejects_non_figures(token):
    assert parse_number(token) is None


def test_split_row_reads_trailing_columns_and_parenthesised_negatives():
    label, values = split_row("Other income (expense), net   (1,646 )   788    333  ")
    assert label == "Other income (expense), net"
    assert values == [Decimal(-1646), Decimal(788), Decimal(333)]


def test_split_row_ignores_dollar_signs():
    assert split_row("Net income  $ 88,136   $ 72,361   $ 72,738") == (
        "Net income", [Decimal(88136), Decimal(72361), Decimal(72738)]
    )


def test_header_years_finds_the_nearest_header_above():
    lines = INCOME.splitlines()
    row = next(i for i, ln in enumerate(lines) if ln.startswith("Total revenue"))
    assert header_years(lines, row) == [2024, 2023, 2022]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("(In millions, except per share amounts)", Decimal(1_000_000)),
        ("(in thousands)", Decimal(1_000)),
        ("(Dollars in billions)", Decimal(1_000_000_000)),
        ("Total revenue 245,122", None),
    ],
)
def test_page_scale(text, expected):
    assert page_scale(text) == expected


# ---------------------------------------------------------------------------
# Verifying a mapping against the page
# ---------------------------------------------------------------------------

def test_verify_row_confirms_the_column_and_reads_every_year():
    row = verify_row(INCOME, "Total revenue", "245,122", 2024)
    assert row.years == [2024, 2023, 2022]
    assert row.values == [Decimal(245122), Decimal(211915), Decimal(198270)]
    assert row.label == "Total revenue"
    assert row.excerpt == "Total revenue   245,122    211,915    198,270"


def test_verify_row_accepts_a_sample_from_any_year_column():
    assert verify_row(INCOME, "Total revenue", "198,270", 2022).values[0] == Decimal(245122)


def test_verify_row_rejects_a_value_in_the_wrong_year_column():
    # 245,122 is FY2024's revenue; claiming it for FY2023 must fail
    assert verify_row(INCOME, "Total revenue", "245,122", 2023) is None


def test_verify_row_rejects_a_year_the_statement_does_not_cover():
    assert verify_row(BALANCE, "Total assets", "512,163", 2022) is None


def test_verify_row_rejects_a_label_that_does_not_match_the_row():
    assert verify_row(INCOME, "Operating income", "245,122", 2024) is None


def test_verify_row_rejects_a_value_not_on_the_page():
    assert verify_row(INCOME, "Total revenue", "245,123", 2024) is None


def test_verify_row_ignores_numbers_inside_the_label():
    row = verify_row(BALANCE, "Accounts receivable, net", "56,924", 2024)
    assert row.values == [Decimal(56924), Decimal(48688)]


def test_verify_row_matches_wrapped_labels():
    row = verify_row(
        CASH, "Acquisition of companies, net of cash acquired, and purchases of "
              "intangible and other assets", "(69,132 )", 2024,
    )
    assert row.values == [Decimal(-69132), Decimal(-1670), Decimal(-22038)]
    assert row.excerpt.startswith("Acquisition of companies")
    assert row.excerpt.endswith("(22,038 )")


def test_verify_row_tolerates_misencoded_characters_in_the_label():
    row = verify_row(BALANCE, "Total stockholders' equity", "268,477", 2024)
    assert row.values == [Decimal(268477), Decimal(206223)]


def test_verify_row_rejects_ambiguous_rows():
    # Two rows are labelled "Other, net", but their FY2024 values differ, so the
    # sample value picks out exactly one of them...
    assert verify_row(CASH, "Other, net", "(1,309 )", 2024).values[1] == Decimal(-1006)
    # ...but if two equally good rows carry the same value and differ in other
    # years, picking one would be a guess.
    page = CASH + "Other, net   (1,309 )   (5,555 )   (6,666 )\n"
    assert verify_row(page, "Other, net", "(1,309 )", 2024) is None


# ---------------------------------------------------------------------------
# Choosing statement pages
# ---------------------------------------------------------------------------

def test_statement_pages_picks_the_three_statements():
    toc = "INDEX TO FINANCIAL STATEMENTS\nIncome Statements 66\nBalance Sheets 68\n"
    note = "\n".join(["NOTE 12 - INCOME TAXES"] + ["text"] * 10 + ["see balance sheets"]
                     + [f"Item {i}  1,{i:03d}  2,{i:03d}" for i in range(20)])
    comprehensive = INCOME.replace("INCOME STATEMENTS", "COMPREHENSIVE INCOME STATEMENTS")
    pages = {3: toc, 66: INCOME, 67: comprehensive, 68: BALANCE, 70: CASH, 98: note}
    assert list(statement_pages(pages)) == [66, 68, 70]


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def _mapper(*mappings):
    return lambda pages: [RowMapping(**m) for m in mappings]


def test_extract_facts_scales_and_normalises_signs():
    facts = extract_facts(PAGES, "msft.pdf", "public", mapper=_mapper(
        dict(metric="revenue", page=66, label="Total revenue", value="245,122", fiscal_year=2024),
        dict(metric="eps_diluted", page=66, label="Diluted", value="$ 11.80", fiscal_year=2024),
        dict(metric="diluted_shares", page=66, label="Diluted", value="7,469", fiscal_year=2024),
        dict(metric="capital_expenditures", page=70, label="Additions to property and equipment",
             value="(44,477 )", fiscal_year=2024),
        dict(metric="total_equity", page=68, label="Total stockholders' equity",
             value="268,477", fiscal_year=2024),
    ))
    got = {(f.metric, f.fiscal_year): f for f in facts}
    assert got[("revenue", 2022)].value == Decimal("198270000000")
    assert got[("revenue", 2024)].page == 66
    assert got[("revenue", 2024)].line == "Total revenue   245,122    211,915    198,270"
    # Per-share figures are not scaled by "(In millions, except per share amounts)"
    assert got[("eps_diluted", 2023)].value == Decimal("9.68")
    assert got[("eps_diluted", 2023)].unit == "USD/share"
    # Share counts are: they're in millions
    assert got[("diluted_shares", 2024)].value == Decimal("7469000000")
    # Outflows printed in parentheses are stored as positive amounts
    assert got[("capital_expenditures", 2024)].value == Decimal("44477000000")
    assert len(facts) == 3 + 3 + 3 + 3 + 2


def test_extract_facts_drops_mappings_that_fail_verification():
    facts = extract_facts(PAGES, "msft.pdf", "public", mapper=_mapper(
        # wrong year for the value
        dict(metric="revenue", page=66, label="Total revenue", value="245,122", fiscal_year=2023),
        # a number the model made up
        dict(metric="net_income", page=66, label="Net income", value="90,000", fiscal_year=2024),
        # a page that isn't a statement page
        dict(metric="total_assets", page=12, label="Total assets", value="512,163",
             fiscal_year=2024),
        # this one is right
        dict(metric="operating_income", page=66, label="Operating income", value="109,433",
             fiscal_year=2024),
    ))
    assert {f.metric for f in facts} == {"operating_income"}


def test_extract_facts_drops_a_metric_mapped_to_conflicting_rows():
    facts = extract_facts(PAGES, "msft.pdf", "public", mapper=_mapper(
        dict(metric="net_income", page=66, label="Net income", value="88,136", fiscal_year=2024),
        dict(metric="net_income", page=66, label="Operating income", value="109,433",
             fiscal_year=2024),
        dict(metric="revenue", page=66, label="Total revenue", value="245,122", fiscal_year=2024),
    ))
    assert {f.metric for f in facts} == {"revenue"}


def test_extract_facts_requires_a_stated_scale_for_amounts():
    unscaled = {66: INCOME.replace("(In millions, except per share amounts)", "")}
    facts = extract_facts(unscaled, "x.pdf", "public", mapper=_mapper(
        dict(metric="revenue", page=66, label="Total revenue", value="245,122", fiscal_year=2024),
        dict(metric="eps_basic", page=66, label="Basic", value="$ 11.86", fiscal_year=2024),
    ))
    # Without "(In millions ...)" revenue's scale is unknown; per-share needs none
    assert {f.metric for f in facts} == {"eps_basic"}


def test_extract_facts_skips_the_model_when_no_statement_pages_exist():
    def boom(pages):
        raise AssertionError("model called")

    assert extract_facts({1: "Risk factors\nprose only"}, "x.pdf", "public", mapper=boom) == []


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

def _fact(owner="alice", source="a.pdf", metric="revenue", year=2024, value="245122000000"):
    return Fact(owner, source, metric, year, Decimal(value), "USD", "Total revenue", 66,
                "Total revenue   245,122")


def test_store_round_trips_exact_decimals():
    financials.replace_facts("alice", "a.pdf", [_fact(), _fact(metric="eps_diluted", value="11.80")])
    loaded = {f.metric: f for f in financials.load_facts(["alice"])}
    assert loaded["eps_diluted"].value == Decimal("11.80")
    assert loaded["revenue"] == _fact()


def test_store_replaces_per_owner_and_file():
    financials.replace_facts("alice", "a.pdf", [_fact(), _fact(year=2023)])
    financials.replace_facts("bob", "a.pdf", [_fact(owner="bob")])
    financials.replace_facts("alice", "a.pdf", [_fact(year=2022)])
    assert [f.fiscal_year for f in financials.load_facts(["alice"])] == [2022]
    assert len(financials.load_facts(["bob"])) == 1
    assert financials.count_facts("alice") == {"a.pdf": 1}
    assert financials.load_facts(["carol"]) == []


def test_extract_and_store_clears_old_figures_even_if_extraction_fails(monkeypatch):
    financials.replace_facts("alice", "a.pdf", [_fact()])

    def fail(*args, **kwargs):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(financials, "extract_facts", fail)
    with pytest.raises(RuntimeError):
        financials.extract_and_store(PAGES, "a.pdf", "alice")
    assert financials.load_facts(["alice"]) == []
