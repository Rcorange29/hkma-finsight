"""Tests for the XBRL cross-check (xbrl.py) and the EDGAR client (edgar.py).

The companyfacts document is built here in EDGAR's shape, with Apple's real
FY2025 values, so no network is needed.
"""

import json
import time
from decimal import Decimal

import pytest

from financial_rag import analysis, edgar, financials, xbrl
from financial_rag.financials import Fact

CIK = 320193
FY25 = {"start": "2024-09-29", "end": "2025-09-27", "form": "10-K", "fy": 2025, "fp": "FY"}
FY24 = {"start": "2023-10-01", "end": "2024-09-28", "form": "10-K", "fy": 2025, "fp": "FY"}
Q4 = {"start": "2025-06-29", "end": "2025-09-27", "form": "10-K", "fy": 2025, "fp": "FY"}
YEAR_END = {"end": "2025-09-27", "form": "10-K", "fy": 2025, "fp": "FY"}


def _rows(unit, *rows):
    return {"units": {unit: [dict(period, val=val) for period, val in rows]}}


COMPANYFACTS = {
    "cik": CIK, "entityName": "Apple Inc.",
    "facts": {"us-gaap": {
        "RevenueFromContractWithCustomerExcludingAssessedTax": _rows(
            "USD", (FY25, 416161000000), (FY24, 391035000000), (Q4, 102466000000)),
        "NetIncomeLoss": _rows("USD", (FY25, 112010000000)),
        "EarningsPerShareDiluted": _rows("USD/shares", (FY25, 7.46)),
        "WeightedAverageNumberOfDilutedSharesOutstanding": _rows("shares", (FY25, 15004697000)),
        "Assets": _rows("USD", (YEAR_END, 359241000000)),
        "PaymentsToAcquirePropertyPlantAndEquipment": _rows("USD", (FY25, 12715000000)),
        # A 10-Q value for the same period must never count
        "GrossProfit": _rows("USD", (dict(FY25, form="10-Q"), 1)),
    }},
}
TICKERS = {"0": {"cik_str": CIK, "ticker": "AAPL", "title": "Apple Inc."},
           "1": {"cik_str": 789019, "ticker": "MSFT", "title": "MICROSOFT CORP"},
           "2": {"cik_str": 1652044, "ticker": "GOOGL", "title": "Alphabet Inc."},
           "3": {"cik_str": 1652044, "ticker": "GOOG", "title": "Alphabet Inc."}}


def fact(metric, year, value, unit="USD", segment=""):
    return Fact("public", "apple.pdf", metric, year, Decimal(value), unit, metric, 31,
                f"{metric} row", segment)


@pytest.fixture
def sec(monkeypatch):
    """EDGAR serves the documents above; records which URLs were fetched."""
    fetched = []

    def get(url, agent):
        assert "@" in agent
        fetched.append(url)
        if url == edgar.TICKERS_URL:
            return TICKERS
        if url == edgar.COMPANYFACTS_URL.format(cik=CIK):
            return COMPANYFACTS
        raise edgar.EdgarUnavailable(f"404 {url}")

    monkeypatch.setattr(edgar, "_get", get)
    monkeypatch.setenv("SEC_USER_AGENT", "financial-rag-tests test@example.com")
    return fetched


# ---------------------------------------------------------------------------
# Matching one figure
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("f", "expected"),
    [
        (fact("revenue", 2025, 416161000000), xbrl.MATCH),
        (fact("revenue", 2024, 391035000000), xbrl.MATCH),
        (fact("revenue", 2025, 416161000001), xbrl.MISMATCH),
        # The quarter ending on the same day is not the year
        (fact("revenue", 2025, 102466000000), xbrl.MISMATCH),
        (fact("eps_diluted", 2025, "7.46", "USD/share"), xbrl.MATCH),
        (fact("diluted_shares", 2025, 15004697000, "shares"), xbrl.MATCH),
        # The same count read in thousands, as before the share-scale fix
        (fact("diluted_shares", 2025, 15004697000000, "shares"), xbrl.MISMATCH),
        (fact("total_assets", 2025, 359241000000), xbrl.MATCH),
        # Outflows are stored as magnitudes; XBRL reports payments as positive
        (fact("capital_expenditures", 2025, 12715000000), xbrl.MATCH),
        (fact("net_income", 2023, 96995000000), xbrl.ABSENT),     # no value for that year
        (fact("gross_profit", 2025, 1), xbrl.ABSENT),             # only a 10-Q has one
        (fact("product_revenue", 2025, 209586000000, segment="iPhone"), xbrl.ABSENT),
    ],
)
def test_check(f, expected):
    assert xbrl.check(f, COMPANYFACTS) == expected


def test_every_line_item_has_tags():
    assert set(xbrl.TAGS) == set(financials.LINE_ITEMS)


# ---------------------------------------------------------------------------
# Cross-checking a filing's figures
# ---------------------------------------------------------------------------

def test_crosscheck_marks_matches_and_drops_contradictions(sec):
    facts = [fact("revenue", 2025, 416161000000),
             fact("net_income", 2025, 112010000001),     # misread
             fact("net_income", 2023, 96995000000)]      # nothing to compare
    report = xbrl.crosscheck(facts, "Apple Inc.")
    assert report.cik == CIK
    assert [(f.metric, f.fiscal_year, f.xbrl) for f in report.facts] == [
        ("revenue", 2025, "match"), ("net_income", 2023, "")]
    assert report.counts == {"match": 1, "mismatch": 1, "absent": 1}
    summary = xbrl.summary(report, "apple.pdf")
    assert "1 match, 1 contradicted and dropped, 1 not in XBRL" in summary
    assert "dropped net_income FY2025: read 112010000001, XBRL 112010000000" in summary


def test_crosscheck_skips_unknown_companies(sec):
    facts = [fact("revenue", 2025, 1)]
    report = xbrl.crosscheck(facts, "Acme Widgets Inc.")
    assert report.facts == facts
    assert "no single SEC company named 'Acme Widgets Inc.'" in report.skipped


def test_crosscheck_without_a_contact_address_changes_nothing(sec, monkeypatch):
    monkeypatch.delenv("SEC_USER_AGENT")
    facts = [fact("revenue", 2025, 1)]
    report = xbrl.crosscheck(facts, "Apple Inc.")
    assert report.facts == facts
    assert "SEC_USER_AGENT isn't set" in report.skipped
    assert sec == []                                     # nothing was requested


def test_extraction_stores_the_xbrl_result(sec, monkeypatch):
    statement = [fact("revenue", 2025, 416161000000), fact("net_income", 2025, 5)]
    monkeypatch.setattr(financials, "extract_facts", lambda *a: list(statement))
    monkeypatch.setattr(financials, "extract_breakdowns", lambda *a: [])
    assert financials.extract_and_store({}, "apple.pdf", "public", "Apple Inc.") == 1
    [stored] = financials.load_facts(["public"])
    assert (stored.metric, stored.xbrl) == ("revenue", "match")


def test_confirmed_inputs_say_so_in_their_citation():
    confirmed = fact("revenue", 2025, 416161000000)
    assert analysis.cite(confirmed) == "(apple.pdf, p.32)"
    assert analysis.cite(confirmed.__class__(**{**confirmed.__dict__, "xbrl": "match"})) == (
        "(apple.pdf, p.32; matches SEC XBRL data)")


def test_databases_from_before_the_check_gain_its_column(tmp_path, monkeypatch):
    import sqlite3

    db = tmp_path / "old.sqlite3"
    monkeypatch.setattr(financials, "FACTS_DB", db)
    conn = sqlite3.connect(db)
    conn.execute("""CREATE TABLE figures (owner TEXT NOT NULL, source_file TEXT NOT NULL,
        metric TEXT NOT NULL, fiscal_year INTEGER NOT NULL, value TEXT NOT NULL,
        unit TEXT NOT NULL, label TEXT NOT NULL, page INTEGER NOT NULL, line TEXT NOT NULL,
        segment TEXT NOT NULL DEFAULT '',
        PRIMARY KEY (owner, source_file, metric, segment, fiscal_year))""")
    conn.execute("INSERT INTO figures VALUES ('public', 'a.pdf', 'revenue', 2024, '5', 'USD', "
                 "'Revenue', 1, 'Revenue 5', '')")
    conn.commit()
    conn.close()
    [old] = financials.load_facts(["public"])
    assert (old.metric, old.xbrl) == ("revenue", "")


# ---------------------------------------------------------------------------
# EDGAR client
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("name", "cik"),
    [("Apple Inc.", CIK), ("APPLE INC", CIK), ("Microsoft Corporation", 789019),
     ("Alphabet Inc.", 1652044),        # two share classes, one CIK
     ("Apple", CIK), ("Acme Corp", None), ("", None), (None, None)],
)
def test_cik_for_name(sec, name, cik):
    assert edgar.cik_for_name(name) == cik


def test_cik_for_ticker(sec):
    assert edgar.cik_for_ticker(" aapl ") == CIK
    assert edgar.cik_for_ticker("ZZZZ") is None


def test_responses_are_cached(sec):
    edgar.tickers()
    edgar.tickers()
    assert sec == [edgar.TICKERS_URL]


def test_stale_cache_is_refetched(sec, monkeypatch):
    edgar.tickers()
    path = edgar.CACHE_DIR / "tickers-all.json"
    old = time.time() - edgar.CACHE_SECONDS["tickers"] - 10
    import os

    os.utime(path, (old, old))
    edgar.tickers()
    assert sec == [edgar.TICKERS_URL, edgar.TICKERS_URL]
    assert json.loads(path.read_text(encoding="utf-8")) == TICKERS


@pytest.mark.parametrize("value", ["", "financial-rag", "   "])
def test_user_agent_needs_a_contact_address(monkeypatch, value):
    monkeypatch.setenv("SEC_USER_AGENT", value)
    with pytest.raises(edgar.EdgarUnavailable):
        edgar.user_agent()
