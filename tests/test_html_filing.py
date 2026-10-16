"""Tests for reading EDGAR's HTML 10-Ks (html_filing.py) and ingesting them."""

from decimal import Decimal

import pytest

from financial_rag import financials, html_filing, ingest

# The shape of an inline-XBRL 10-K: a hidden XBRL header, figures wrapped in
# ix: tags inside sentences, cells split into "$", value and ")" columns, and
# pages ended by page-break-after.
FILING = """<?xml version="1.0" encoding="utf-8"?>
<html xmlns:ix="http://www.xbrl.org/2013/inlineXBRL"><head><title>acme-10k</title></head>
<body>
<div style="display:none"><ix:header><ix:hidden>dei:EntityRegistrantName Acme</ix:hidden></ix:header></div>
<div><span>Commission File Number 001-00001</span></div>
<div><span>ACME CORPORATION</span></div>
<div><span>(Exact name of registrant as specified in its charter)</span></div>
<div>Revenue grew to $<ix:nonFraction name="us-gaap:Revenues">245,122</ix:nonFraction> million this year.</div>
<hr style="page-break-after:always"/>
<div><span>BALANCE SHEETS</span></div>
<div><span>(In millions)</span></div>
<table>
  <tr><td><span>June 30,</span></td><td colspan="3">2024</td><td colspan="3">2023</td></tr>
  <tr><td>Accounts payable</td><td>$</td><td>21,996</td><td></td><td>$</td><td>18,095</td><td></td></tr>
  <tr><td>Short-term unearned revenue</td><td></td><td>57,582</td><td></td><td></td><td>50,901</td><td></td></tr>
  <tr><td>Other income (expense), net</td><td></td><td>(1,646</td><td>)</td><td></td><td>788</td><td></td></tr>
  <tr><td></td><td></td><td></td></tr>
</table>
<div style="page-break-after:always"></div>
<div>   </div>
<div>Notes&#160;to&#160;financial statements</div>
</body></html>"""


@pytest.fixture
def pages():
    return html_filing.load_pages(FILING)


def test_pages_split_at_page_breaks_and_skip_empty_ones(pages):
    assert [p.number for p in pages] == [0, 1, 2]
    assert pages[2].text == "Notes to financial statements"


def test_hidden_xbrl_header_is_skipped(pages):
    assert "dei:" not in pages[0].text and "EntityRegistrantName" not in pages[0].text


def test_inline_xbrl_tags_stay_inside_their_sentence(pages):
    assert "Revenue grew to $245,122 million this year." in pages[0].text


def test_cover_page_names_the_registrant(pages):
    assert ingest.registrant_name(pages[0].text) == "Acme Corporation"


def test_table_rows_are_one_line_each(pages):
    lines = pages[1].text.split("\n")
    assert "June 30, 2024 2023" in lines
    assert "Accounts payable $ 21,996 $ 18,095" in lines
    assert "Short-term unearned revenue 57,582 50,901" in lines
    assert "Other income (expense), net (1,646 ) 788" in lines


def test_table_headings_are_the_rows_above_the_first_figure(pages):
    [table] = pages[1].tables
    assert table.headings == ["June 30, 2024 2023"]
    assert table.rows[0] == "Accounts payable $ 21,996 $ 18,095"
    assert pages[0].tables == [] and pages[2].tables == []


def test_rows_read_as_statement_rows(pages):
    row = financials.verify_row(pages[1].text, "Short-term unearned revenue", "57,582", 2024)
    assert row is not None and row.values == [Decimal(57582), Decimal(50901)]
    row = financials.verify_row(pages[1].text, "Other income (expense), net", "(1,646)", 2024)
    assert row is not None and row.values == [Decimal(-1646), Decimal(788)]


def test_chunks_that_lost_their_headings_get_them_back(pages):
    [table] = pages[1].tables
    cut = "Short-term unearned revenue 57,582 50,901\nOther text"
    assert html_filing.with_table_headings(cut, [table]) == (
        "[Table columns: June 30, 2024 2023]\n" + cut)
    whole = "June 30, 2024 2023\nShort-term unearned revenue 57,582 50,901"
    assert html_filing.with_table_headings(whole, [table]) == whole
    assert html_filing.with_table_headings("No table here", [table]) == "No table here"


# ---------------------------------------------------------------------------
# Ingesting HTML filings
# ---------------------------------------------------------------------------

@pytest.fixture
def filing_file(tmp_path):
    path = tmp_path / "acme-10k-2024.htm"
    path.write_text(FILING, encoding="utf-8")
    return path


def test_load_filing_reads_html_as_pages(filing_file):
    docs, tables = ingest.load_filing(filing_file, "alice")
    assert [d.metadata for d in docs] == [
        {"page": n, "source_file": "acme-10k-2024.htm", "owner": "alice"} for n in range(3)]
    assert len(tables[1]) == 1 and tables[0] == []


def test_financials_reads_html_pages(filing_file):
    pages = financials.load_pages(filing_file)
    assert sorted(pages) == [0, 1, 2]
    assert "Short-term unearned revenue 57,582 50,901" in pages[1]


def test_filing_jobs_include_html(tmp_path, monkeypatch):
    data = tmp_path / "data"
    (data / "users" / "alice").mkdir(parents=True)
    for name in ("msft.pdf", "aapl-10k.htm", "notes.txt"):
        (data / name).write_text("x")
    (data / "users" / "alice" / "acme.html").write_text("x")
    monkeypatch.setattr(ingest, "DATA_DIR", data)
    monkeypatch.setattr(ingest, "USERS_DIR", data / "users")
    assert [(p.name, owner) for p, owner in ingest.filing_jobs()] == [
        ("aapl-10k.htm", "public"), ("msft.pdf", "public"), ("acme.html", "alice")]


def test_ingesting_html_repeats_cut_off_headings(filing_file, monkeypatch):
    added = []

    class Store:
        def get(self, **kwargs):
            return {"ids": []}

        def add_documents(self, chunks):
            added.extend(chunks)

    monkeypatch.setattr(ingest, "open_vectorstore", Store)
    monkeypatch.setattr(ingest, "CHUNK_SIZE", 30)       # force the table across chunks
    monkeypatch.setattr(ingest, "CHUNK_OVERLAP", 0)
    stored = {}
    monkeypatch.setattr(financials, "extract_and_store",
                        lambda pages, name, owner, company, cik: stored.update(
                            name=name, company=company, cik=cik))
    ingest.ingest_file(filing_file, "alice", cik=42)
    with_row = [c.page_content for c in added if "57,582" in c.page_content]
    assert with_row and all("2024" in c for c in with_row)
    assert stored == {"name": "acme-10k-2024.htm", "company": "Acme Corporation", "cik": 42}
    assert {c.metadata["company"] for c in added} == {"Acme Corporation"}


def test_pdfs_still_load_as_pdfs(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "load_pdf", lambda path, owner: ["pdf pages"])
    assert ingest.load_filing(tmp_path / "msft.PDF", "alice") == (["pdf pages"], {})
