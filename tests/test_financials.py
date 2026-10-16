"""Tests for extracting structured figures from statement text (financials.py).

The page text below reproduces rows from Microsoft's FY2024 10-K as PyPDF
extracts them: figures right-aligned in year columns, negatives printed as
"(1,646 )", and labels that contain numbers or wrap onto a second line.
No API key needed: the model's row mapping is replaced by a fake.
"""

from decimal import Decimal

import pytest

from financial_rag import financials
from financial_rag.financials import (
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
        # Target: the scale word alone in the note
        ("(millions, except per share data) 2025 2024 2023", Decimal(1_000_000)),
        ("(millions) 2025 2024 2023", Decimal(1_000_000)),
        ("Total revenue 245,122", None),
        ("Revenue grew as (millions of) new customers joined", None),
    ],
)
def test_page_scale(text, expected):
    assert page_scale(text) == expected


# Apple's statements, as pypdf extracts them: each column's date on two lines
APPLE_INCOME = """Apple Inc.
CONSOLIDATED STATEMENTS OF OPERATIONS
(In millions, except number of shares, which are reflected in thousands, and per-share amounts)
Years ended
September 27,
2025
September 28,
2024
September 30,
2023
Net sales:
   Products $ 307,003 $ 294,866 $ 298,085
   Services  109,158  96,169  85,200
Total net sales  416,161  391,035  383,285
Net income $ 112,010 $ 93,736 $ 96,995
Shares used in computing earnings per share:
Diluted  15,004,697  15,408,095  15,812,547
"""


def test_header_years_reads_one_year_per_line():
    lines = APPLE_INCOME.splitlines()
    for label in ("   Products", "Total net sales", "Diluted"):
        row = next(i for i, ln in enumerate(lines) if ln.startswith(label))
        assert header_years(lines, row) == [2025, 2024, 2023]


def test_header_years_ignores_a_lone_year():
    lines = ["Fiscal", "2024", "Revenue 1,000 900"]
    assert header_years(lines, 2) is None


def test_verify_row_with_stacked_year_header():
    row = verify_row(APPLE_INCOME, "Total net sales", "391,035", 2024)
    assert row is not None
    assert row.years == [2025, 2024, 2023]
    assert row.values == [Decimal(416161), Decimal(391035), Decimal(383285)]


@pytest.mark.parametrize(
    ("note", "dollars", "shares"),
    [
        # Apple: shares have their own scale
        ("(In millions, except number of shares, which are reflected in thousands, "
         "and per-share amounts)", Decimal(1_000_000), Decimal(1_000)),
        # Microsoft: shares follow the dollars
        ("(In millions, except per share amounts)", Decimal(1_000_000), Decimal(1_000_000)),
        ("(In millions, Except Per Share Amounts)", Decimal(1_000_000), Decimal(1_000_000)),
        # Shares excepted from the scale: reported as is
        ("(In thousands, except share and per share data)", Decimal(1_000), Decimal(1)),
        ("(in thousands, except shares)", Decimal(1_000), Decimal(1)),
        ("(In millions, except per share amounts; shares in thousands)",
         Decimal(1_000_000), Decimal(1_000)),
        ("(millions, except per share data)", Decimal(1_000_000), Decimal(1_000_000)),
    ],
)
def test_page_scale_for_share_counts(note, dollars, shares):
    assert page_scale(note) == dollars
    assert page_scale(note, "shares") == shares
    assert page_scale(note, "USD/share") == dollars    # the caller ignores scale for EPS


def test_apple_share_counts_are_in_thousands():
    def mapper(pages):
        return [RowMapping(metric="diluted_shares", page=0, label="Diluted",
                           value="15,004,697", fiscal_year=2025)]

    facts = extract_facts({0: APPLE_INCOME}, "apple.pdf", "public", mapper=mapper)
    by_year = {f.fiscal_year: f.value for f in facts}
    assert by_year[2025] == Decimal(15_004_697_000)


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


def test_a_row_naming_a_statement_is_not_its_heading():
    # Travelers: the comprehensive income statement sits between the income
    # statement and the balance sheet, and one of its first rows names the
    # income statement. Taken for a heading, it made a tighter run of pages
    # than the real income statement and replaced it.
    comprehensive = "\n".join([
        "THE TRAVELERS COMPANIES, INC. AND SUBSIDIARIES",
        "CONSOLIDATED STATEMENT OF COMPREHENSIVE INCOME",
        "(in millions)",
        "For the year ended December 31, 2025 2024 2023",
        "Net income $ 6,288 $ 4,999 $ 2,991",
        "Other comprehensive income (loss):",
        "Having no credit losses recognized in the consolidated statement of income"
        " 2,744 ( 644 ) 2,249",
    ] + [f"Item {i}  1,{i:03d}  2,{i:03d}" for i in range(12)])
    income = INCOME.replace("INCOME STATEMENTS", "CONSOLIDATED STATEMENT OF INCOME")
    pages = {118: income, 119: comprehensive, 120: BALANCE, 122: CASH}
    assert list(statement_pages(pages)) == [118, 120, 122]


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


def test_every_line_item_says_how_its_row_is_printed():
    assert set(financials.PRINTED_AS) == set(financials.LINE_ITEMS)


@pytest.mark.parametrize(
    ("metric", "label"),
    [
        # Rows XBRL confirmed, in each company's wording
        ("revenue", "Sales to customers"),                      # Johnson & Johnson
        ("revenue", "Total net revenue"),                       # JPMorgan
        ("cost_of_revenue", "Cost of products sold"),
        ("gross_profit", "Gross margin"),                       # Apple, Microsoft
        ("operating_income", "Income from operations"),         # Alphabet
        ("income_before_taxes", "Earnings before provision for taxes on income"),
        ("income_tax_expense", "Income taxes"),                 # Coca-Cola
        ("net_income", "Net Income Attributable to Shareowners of The Coca-Cola Company"),
        ("eps_diluted", "Diluted"),
        ("diluted_shares", "Weighted-average diluted shares"),
        ("total_equity", "Total shareholders' investment"),     # Target
        ("long_term_debt", "Term debt"),                        # Apple
        ("capital_expenditures", "Purchases related to property and equipment and "
                                 "intangible assets"),          # NVIDIA
        ("dividends_paid", "Cash distributions to common stockholders"),   # Realty Income
        ("share_repurchases", "Purchases of stock for treasury"),          # Coca-Cola
        ("share_repurchases", "Common stock repurchased"),                 # Microsoft
    ],
)
def test_printed_labels_that_fit_the_line_item(metric, label):
    assert financials.PRINTED_AS[metric].search(label)


@pytest.mark.parametrize(
    ("metric", "label"),
    [
        # Substitutions the model made for line items these statements lack
        ("cost_of_revenue", "Total noninterest expense"),            # JPMorgan
        ("cost_of_revenue", "Total claims and expenses"),            # Travelers
        ("operating_income", "Income before income tax expense"),    # JPMorgan
        ("operating_income", "Earnings before provision for taxes on income"),   # J&J
        ("gross_profit", "Income before income taxes"),              # Travelers
        ("research_and_development", "Other operating charges"),     # Coca-Cola
        ("research_and_development", "Amortization of deferred acquisition costs"),
        ("total_current_assets", "Total assets"),                    # banks, insurers, REITs
        ("total_current_liabilities", "Total liabilities"),
        ("total_liabilities", "Total Liabilities and Equity"),       # Coca-Cola
        ("total_equity", "Total liabilities and shareholders' equity"),
        ("capital_expenditures", "Purchases of investments: Fixed maturities"),  # Travelers
        ("capital_expenditures", "Purchases"),                       # JPMorgan
    ],
)
def test_printed_labels_that_do_not(metric, label):
    assert not financials.PRINTED_AS[metric].search(label)


def test_extract_facts_rejects_a_row_printed_as_another_line_item():
    # Asked for total current assets, the model points at "Total assets"
    facts = extract_facts(PAGES, "msft.pdf", "public", mapper=_mapper(
        dict(metric="total_current_assets", page=68, label="Total assets", value="512,163",
             fiscal_year=2024),
        dict(metric="total_assets", page=68, label="Total assets", value="512,163",
             fiscal_year=2024),
    ))
    assert {f.metric for f in facts} == {"total_assets"}


def test_extract_facts_drops_a_metric_mapped_to_conflicting_rows():
    # Both rows are printed "Diluted": earnings per share, and the share count
    facts = extract_facts(PAGES, "msft.pdf", "public", mapper=_mapper(
        dict(metric="eps_diluted", page=66, label="Diluted", value="$ 11.80", fiscal_year=2024),
        dict(metric="eps_diluted", page=66, label="Diluted", value="7,469", fiscal_year=2024),
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


# Target's statements, as html_filing reads them: the income statement names the
# year to January 31, 2026 "2025"; the balance sheet is dated
TARGET_INCOME = """Consolidated Statements of Operations
(millions, except per share data) 2025 2024 2023
Net sales $ 104,780 $ 106,566 $ 107,412
Cost of sales 75,511 76,502 77,828
Operating income 5,117 5,566 5,707
Net earnings $ 3,705 $ 4,091 $ 4,138
"""
TARGET_BALANCE = """Consolidated Statements of Financial Position
(millions, except footnotes) January 31, 2026 February 1, 2025
Cash and cash equivalents $ 5,488 $ 4,762
Inventory 12,304 12,740
Total current assets 20,005 19,454
Total assets $ 59,490 $ 57,769
Total current liabilities 21,230 20,799
Total shareholders' investment 16,165 14,666
"""
TARGET_CASH = """Consolidated Statements of Cash Flows
(millions) 2025 2024 2023
Net earnings $ 3,705 $ 4,091 $ 4,138
Depreciation and amortization 3,134 2,981 2,801
Cash provided by operating activities 6,562 7,367 8,621
Expenditures for property and equipment ( 3,727 ) ( 2,891 ) ( 4,806 )
Dividends paid ( 2,053 ) ( 2,046 ) ( 2,011 )
Cash used in investing activities ( 3,649 ) ( 2,860 ) ( 4,760 )
"""


def test_balance_sheet_years_follow_the_income_statements_names():
    pages = {45: TARGET_INCOME, 47: TARGET_BALANCE, 48: TARGET_CASH}
    facts = extract_facts(pages, "tgt.htm", "public", mapper=_mapper(
        dict(metric="revenue", page=45, label="Net sales", value="104,780", fiscal_year=2025),
        dict(metric="total_assets", page=47, label="Total assets", value="59,490",
             fiscal_year=2026),
        dict(metric="operating_cash_flow", page=48, label="Cash provided by operating activities",
             value="6,562", fiscal_year=2025),
    ))
    got = {(f.metric, f.fiscal_year): f.value for f in facts}
    assert got[("revenue", 2025)] == Decimal("104780000000")
    # The balance at January 31, 2026 closes the year Target calls 2025
    assert got[("total_assets", 2025)] == Decimal("59490000000")
    assert got[("total_assets", 2024)] == Decimal("57769000000")
    assert ("total_assets", 2026) not in got


def test_balance_sheet_years_stay_when_the_statements_agree():
    facts = extract_facts(PAGES, "msft.pdf", "public", mapper=_mapper(
        dict(metric="revenue", page=66, label="Total revenue", value="245,122", fiscal_year=2024),
        dict(metric="total_assets", page=68, label="Total assets", value="512,163",
             fiscal_year=2024),
    ))
    assert {(f.metric, f.fiscal_year) for f in facts} == {
        ("revenue", 2024), ("revenue", 2023), ("revenue", 2022),
        ("total_assets", 2024), ("total_assets", 2023)}


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



# ---------------------------------------------------------------------------
# Breakdowns: segments, geography, product lines
# ---------------------------------------------------------------------------

SEGMENTS = """PART II
Item 8
Segment revenue and operating income were as follows during the periods presented:
(In millions)
Year Ended June 30,   2024    2023    2022
Revenue
Productivity and Business Processes  $ 77,728   $ 69,274   $ 63,364
Intelligent Cloud   105,362    87,907    74,965
More Personal Computing   62,032    54,734    59,941
Total  $ 245,122   $ 211,915   $ 198,270
Operating Income
Productivity and Business Processes  $ 40,540   $ 34,189   $ 29,690
Intelligent Cloud   49,584    37,884    33,203
More Personal Computing   19,309    16,450    20,490
Total  $ 109,433   $ 88,523   $ 83,383
Long-lived assets, classified by the location of the controlling statutory company:
(In millions)
June 30,  2024   2023   2022
United States  $ 186,106   $ 114,380   $ 106,430
Other countries   115,263    72,859    59,938
Total  $ 301,369   $ 187,239   $ 166,368
"""

M = Decimal(1_000_000)


def _totals():
    return [
        Fact("public", "msft.pdf", metric, year, Decimal(v) * M, "USD", metric, 66, "row")
        for metric, values in {
            "revenue": {2024: 245122, 2023: 211915, 2022: 198270},
            "operating_income": {2024: 109433, 2023: 88523, 2022: 83383},
        }.items()
        for year, v in values.items()
    ]


def _parts(*rows):
    return lambda pages: [financials.PartMapping(**r) for r in rows]


def _segment_rows(breakdown, labels_values):
    return [dict(breakdown=breakdown, part=label, page=112, label=label, value=value,
                 fiscal_year=2024) for label, value in labels_values]


REVENUE_ROWS = [("Productivity and Business Processes", "77,728"),
                ("Intelligent Cloud", "105,362"), ("More Personal Computing", "62,032")]
INCOME_ROWS = [("Productivity and Business Processes", "40,540"),
               ("Intelligent Cloud", "49,584"), ("More Personal Computing", "19,309")]


def test_rejoin_rows_puts_split_rows_back_together():
    text = "Windows   23,244    21,507    24,732\nGaming   21,503\n \n  15,466\n \n  16,230\nLinkedIn   16,372"
    lines = financials.rejoin_rows(text).splitlines()
    assert "Gaming   21,503  15,466  16,230" in lines
    assert lines[0] == "Windows   23,244    21,507    24,732"
    assert lines[-1] == "LinkedIn   16,372"


def test_rejoin_rows_leaves_year_headers_alone():
    text = "Total  $ 245,122\n2024   2023   2022"
    assert financials.rejoin_rows(text).splitlines() == ["Total  $ 245,122", "2024   2023   2022"]


def test_breakdowns_that_add_up_to_the_totals_are_kept():
    pages = {112: SEGMENTS}
    facts = financials.extract_breakdowns(pages, "msft.pdf", "public", _totals(), mapper=_parts(
        *_segment_rows("segment_revenue", REVENUE_ROWS),
        *_segment_rows("segment_operating_income", INCOME_ROWS),
    ))
    by = {(f.metric, f.segment, f.fiscal_year): f for f in facts}
    assert len(facts) == 18          # 2 breakdowns x 3 segments x 3 years
    cloud = by[("segment_revenue", "Intelligent Cloud", 2022)]
    assert cloud.value == Decimal(74965) * M
    assert (cloud.page, cloud.line) == (112, "Intelligent Cloud   105,362    87,907    74,965")
    # The same label in the operating income block is read from that block
    assert by[("segment_operating_income", "Intelligent Cloud", 2024)].value == Decimal(49584) * M


def test_a_breakdown_that_does_not_add_up_is_dropped():
    # The long-lived assets table also lists the United States and other countries,
    # but its parts add up to 301,369, not revenue
    facts = financials.extract_breakdowns({112: SEGMENTS}, "msft.pdf", "public", _totals(),
                                          mapper=_parts(*_segment_rows("geographic_revenue", [
                                              ("United States", "186,106"),
                                              ("Other countries", "115,263")])))
    assert facts == []


def test_a_skipped_part_is_completed_by_the_one_row_that_fills_the_gap():
    # The model skipped More Personal Computing; only its revenue row (not its
    # operating income row) makes the parts add up in all three years
    facts = financials.extract_breakdowns({112: SEGMENTS}, "msft.pdf", "public", _totals(),
                                          mapper=_parts(*_segment_rows("segment_revenue",
                                                                       REVENUE_ROWS[:2])))
    by = {(f.segment, f.fiscal_year): f for f in facts}
    assert len(facts) == 9
    assert by[("More Personal Computing", 2022)].value == Decimal(59941) * M
    assert by[("More Personal Computing", 2024)].line == (
        "More Personal Computing   62,032    54,734    59,941")


def test_a_breakdown_missing_two_parts_is_dropped_whole():
    # No single row fills a gap made of two parts
    facts = financials.extract_breakdowns({112: SEGMENTS}, "msft.pdf", "public", _totals(),
                                          mapper=_parts(*_segment_rows("segment_revenue",
                                                                       REVENUE_ROWS[:1])))
    assert facts == []


def test_a_gap_two_rows_could_fill_is_not_guessed():
    page = SEGMENTS + "Copy of a part   62,032    54,734    59,941\n"
    facts = financials.extract_breakdowns({112: page}, "msft.pdf", "public", _totals(),
                                          mapper=_parts(*_segment_rows("segment_revenue",
                                                                       REVENUE_ROWS[:2])))
    assert facts == []


def test_a_part_mapped_to_the_wrong_block_breaks_the_sum():
    # Intelligent Cloud's operating income passed off as its revenue
    rows = [REVENUE_ROWS[0], ("Intelligent Cloud", "49,584"), REVENUE_ROWS[2]]
    facts = financials.extract_breakdowns({112: SEGMENTS}, "msft.pdf", "public", _totals(),
                                          mapper=_parts(*_segment_rows("segment_revenue", rows)))
    assert facts == []


def test_breakdowns_need_verified_totals():
    facts = financials.extract_breakdowns({112: SEGMENTS}, "msft.pdf", "public", [],
                                          mapper=_parts(*_segment_rows("segment_revenue",
                                                                       REVENUE_ROWS)))
    assert facts == []


def test_breakdown_figures_are_stored_alongside_totals():
    part = Fact("alice", "a.pdf", "segment_revenue", 2024, Decimal(105362) * M, "USD",
                "Intelligent Cloud", 112, "Intelligent Cloud   105,362", "Intelligent Cloud")
    financials.replace_facts("alice", "a.pdf", [_fact(), part])
    loaded = financials.load_facts(["alice"])
    assert sorted((f.metric, f.segment) for f in loaded) == [
        ("revenue", ""), ("segment_revenue", "Intelligent Cloud")]


def test_a_database_from_before_breakdowns_is_carried_over(tmp_path, monkeypatch):
    import sqlite3

    db = tmp_path / "old.sqlite3"
    monkeypatch.setattr(financials, "FACTS_DB", db)
    conn = sqlite3.connect(db)
    conn.execute("""CREATE TABLE facts (owner TEXT NOT NULL, source_file TEXT NOT NULL,
        metric TEXT NOT NULL, fiscal_year INTEGER NOT NULL, value TEXT NOT NULL,
        unit TEXT NOT NULL, label TEXT NOT NULL, page INTEGER NOT NULL, line TEXT NOT NULL,
        PRIMARY KEY (owner, source_file, metric, fiscal_year))""")
    conn.execute("INSERT INTO facts VALUES ('alice', 'a.pdf', 'revenue', 2024, '245122000000', "
                 "'USD', 'Total revenue', 66, 'Total revenue   245,122')")
    conn.commit()
    conn.close()
    assert financials.load_facts(["alice"]) == [_fact()]
    tables = sqlite3.connect(db).execute("SELECT name FROM sqlite_master").fetchall()
    assert ("facts",) not in tables


def test_verify_row_ignores_a_footnote_marker_in_the_label():
    page = APPLE_INCOME.replace("   Services  109,158", "   Services (1)  109,158")
    row = verify_row(page, "Services (1)", "109,158", 2025)
    assert row is not None
    assert row.values == [Decimal(109158), Decimal(96169), Decimal(85200)]


def test_breakdown_pages_include_net_sales_by_country():
    countries = ("The following tables show net sales for countries that individually "
                 "accounted for 10% or more of the respective totals (in millions):\n"
                 "2025 2024 2023\nU.S. $ 151,790 $ 142,196 $ 138,573\n"
                 "China (1)  64,377  66,952  72,559\n")
    assert list(financials.breakdown_pages({7: countries})) == [7]


def test_the_row_labelled_exactly_as_asked_breaks_a_tie():
    # J&J: both rows hold "Net earnings" and 26,804 for 2025, but differ in 2023
    page = ("(In millions)\n"
            "2025 2024 2023\n"
            "Net earnings from continuing operations 26,804 14,066 13,326\n"
            "Net earnings from discontinued operations, net of tax — — 21,827\n"
            "Net earnings $ 26,804 14,066 35,153\n")
    row = verify_row(page, "Net earnings", "26,804", 2025)
    assert row is not None
    assert row.values == [Decimal(26804), Decimal(14066), Decimal(35153)]


def test_rows_that_tie_and_disagree_are_still_ambiguous():
    # Neither row is labelled exactly "Net earnings", and they disagree on 2024
    page = ("2025 2024\n"
            "Net earnings from continuing operations 26,804 14,066\n"
            "Net earnings attributable to shareholders 26,804 13,000\n")
    assert verify_row(page, "Net earnings", "26,804", 2025) is None


def _statement_page(heading):
    figures = " ".join(f"{n},{n:03d}" for n in range(100, 115))
    return f"{heading}\n(In millions)\n2025 2024\nRow {figures}\n"


def test_statement_pages_keep_the_tightest_run_of_primary_statements():
    pages = {
        39: _statement_page("CONSOLIDATED STATEMENTS OF INCOME"),   # an MD&A table
        62: _statement_page("CONSOLIDATED STATEMENTS OF INCOME"),
        64: _statement_page("CONSOLIDATED BALANCE SHEETS"),
        65: _statement_page("CONSOLIDATED STATEMENTS OF CASH FLOWS"),
        78: _statement_page("CONSOLIDATED BALANCE SHEETS"),        # a note
        # A note naming all three statements doesn't anchor the run
        88: _statement_page("Statements of income, balance sheets and cash flows"),
    }
    assert list(financials.statement_pages(pages)) == [62, 64, 65]


def test_statement_pages_without_all_three_keep_every_match():
    pages = {10: _statement_page("CONSOLIDATED BALANCE SHEETS"),
             12: _statement_page("CONSOLIDATED STATEMENTS OF INCOME")}
    assert list(financials.statement_pages(pages)) == [10, 12]


def test_items_the_first_pass_skipped_are_asked_for_again():
    calls = []

    def mapper(pages, only=None):
        calls.append(only)
        if only is None:        # the first pass leaves net income out
            return [RowMapping(metric="revenue", page=66, label="Total revenue",
                               value="245,122", fiscal_year=2024)]
        return [RowMapping(metric="net_income", page=66, label="Net income",
                           value="88,136", fiscal_year=2024)]

    facts = extract_facts(PAGES, "msft.pdf", "public", mapper=mapper)
    assert calls[0] is None
    assert "revenue" not in calls[1] and "net_income" in calls[1]
    assert len(calls) == 2                       # one second pass, not a loop
    assert {f.metric for f in facts} == {"revenue", "net_income"}


def test_mappers_without_only_get_a_single_pass():
    calls = []

    def mapper(pages):
        calls.append(1)
        return []

    assert extract_facts(PAGES, "msft.pdf", "public", mapper=mapper) == []
    assert calls == [1]
