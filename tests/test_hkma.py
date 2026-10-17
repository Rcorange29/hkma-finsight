"""Tests for the HKMA annual report importer (financial_rag/hkma.py).

Everything is offline: fetch_text / fetch_bytes are replaced, and ingestion uses
fake embeddings and a fake PDF loader. No network, no API key, no model download.
"""

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from financial_rag import financials, hkma, ingest, rag

# ---------------------------------------------------------------------------
# Landing-page discovery
# ---------------------------------------------------------------------------

def _landing_html(year: int, files: list[str]) -> str:
    links = "".join(
        f'<a href="/media/eng/publication-and-research/annual-report/{year}/{f}">{f}</a>'
        for f in files
    )
    return f"<html><body>{links}</body></html>"


_CHAPTERS = [
    "01_Cover.pdf", "02_HKMAGlance.pdf", "03_Contents.pdf",
    "04_Chief_Executive_Statement.pdf", "19_The_Exchange_Fund.pdf",
    "20_Annex.pdf", "23_BackCover.pdf",
]


def test_discovery_picks_the_combined_report_2024():
    html = _landing_html(2024, _CHAPTERS + ["AR2024_E.pdf"])
    report = hkma.discover_annual_report(2024, fetch_text=lambda url: html)
    assert report.file_name == "AR2024_E.pdf"
    assert report.full_report_url == (
        "https://www.hkma.gov.hk/media/eng/publication-and-research/annual-report/2024/AR2024_E.pdf"
    )
    assert len(report.chapter_urls) == len(_CHAPTERS)


def test_discovery_handles_the_2025_underscore_filename():
    html = _landing_html(2025, _CHAPTERS + ["AR_2025E.pdf"])
    report = hkma.discover_annual_report(2025, fetch_text=lambda url: html)
    assert report.file_name == "AR_2025E.pdf"


def test_discovery_accepts_protocol_relative_and_absolute_links():
    html = (
        '<a href="//www.hkma.gov.hk/media/eng/publication-and-research/'
        'annual-report/2024/01_Cover.pdf">c</a>'
        '<a href="https://www.hkma.gov.hk/media/eng/publication-and-research/'
        'annual-report/2024/AR2024_E.pdf">full</a>'
    )
    report = hkma.discover_annual_report(2024, fetch_text=lambda url: html)
    assert report.full_report_url.startswith("https://www.hkma.gov.hk")
    assert report.chapter_urls[0].startswith("https://www.hkma.gov.hk")


def test_discovery_raises_when_the_page_has_no_pdfs():
    with pytest.raises(hkma.HKMAError, match="No Annual Report PDF"):
        hkma.discover_annual_report(1992, fetch_text=lambda url: "<html></html>")


def test_discovery_raises_when_there_is_no_single_combined_pdf():
    html = _landing_html(2024, _CHAPTERS)  # chapters only, no combined file
    with pytest.raises(hkma.HKMAError, match="one combined report"):
        hkma.discover_annual_report(2024, fetch_text=lambda url: html)


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

@pytest.fixture
def dest(tmp_path):
    return tmp_path / "data"


def test_download_writes_the_official_filename_and_bytes(dest):
    html = _landing_html(2024, _CHAPTERS + ["AR2024_E.pdf"])
    calls = {"bytes": 0}

    def fake_bytes(url):
        calls["bytes"] += 1
        return b"%PDF-1.7 fake"

    path, report = hkma.download_annual_report(
        2024, dest, fetch_text=lambda url: html, fetch_bytes=fake_bytes)
    assert path.name == "AR2024_E.pdf"
    assert path.read_bytes() == b"%PDF-1.7 fake"
    assert calls["bytes"] == 1


def test_download_skips_the_network_when_the_file_exists(dest):
    html = _landing_html(2024, _CHAPTERS + ["AR2024_E.pdf"])
    dest.mkdir(parents=True)
    (dest / "AR2024_E.pdf").write_bytes(b"already here")

    def no_bytes(url):
        raise AssertionError("must not re-download")

    path, _ = hkma.download_annual_report(
        2024, dest, fetch_text=lambda url: html, fetch_bytes=no_bytes)
    assert path.read_bytes() == b"already here"


def test_fetch_bytes_rejects_a_non_pdf_response(monkeypatch):
    import httpx

    def bad(url, **kw):
        return httpx.Response(
            200, headers={"content-type": "text/html"}, content=b"<html>",
            request=httpx.Request("GET", url),
        )

    monkeypatch.setattr(hkma.httpx, "get", bad)
    with pytest.raises(hkma.HKMAError, match="did not return a PDF"):
        hkma.fetch_bytes("https://example.com/x.pdf")


# ---------------------------------------------------------------------------
# End-to-end ingestion (fake PDF loader + fake embeddings), offline
# ---------------------------------------------------------------------------

def _fake_loader(monkeypatch, pages):
    def load(path, owner):
        return [
            Document(page_content=text,
                     metadata={"source_file": path.name, "owner": owner, "page": i})
            for i, text in enumerate(pages)
        ]
    monkeypatch.setattr(ingest, "load_pdf", load)


@pytest.fixture
def local_store(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "CHROMA_DIR", tmp_path / "chroma_db")
    monkeypatch.setattr(rag, "CHROMA_DIR", tmp_path / "chroma_db")
    monkeypatch.setattr(ingest, "get_embeddings", lambda: DeterministicFakeEmbedding(size=32))
    monkeypatch.setattr(rag, "get_embeddings", lambda: DeterministicFakeEmbedding(size=32))


def test_ingest_tags_hkma_title_and_skips_sec_figures(dest, local_store, monkeypatch):
    html = _landing_html(2024, _CHAPTERS + ["AR2024_E.pdf"])
    _fake_loader(monkeypatch, [
        "HONG KONG MONETARY AUTHORITY ANNUAL REPORT 2024",
        "The Exchange Fund total assets 4,071,732 million Hong Kong dollars",
        "Banking stability and currency board operations are described here.",
    ])

    chunks = hkma.ingest_annual_report(
        2024, dest, fetch_text=lambda url: html, fetch_bytes=lambda url: b"%PDF-1.7")

    assert chunks >= 1
    metas = ingest.open_vectorstore().get(include=["metadatas"])["metadatas"]
    assert {m["source_file"] for m in metas} == {"AR2024_E.pdf"}
    assert {m["company"] for m in metas} == {hkma.HKMA_TITLE}
    # The SEC/10-K figure engine is off for HKMA reports: no structured figures.
    assert financials.load_facts([ingest.PUBLIC_OWNER], "AR2024_E.pdf") == []


class _FakeReranker:
    def __init__(self, score=0.0):
        self._score = score

    def predict(self, pairs):
        return [self._score] * len(pairs)


@pytest.mark.filterwarnings("ignore:Relevance scores must be between 0 and 1")
def test_hkma_page_citation_is_verifiable_through_keyword_evidence(dest, local_store, monkeypatch):
    monkeypatch.setattr(rag, "_get_reranker", lambda: _FakeReranker(score=1.0))
    monkeypatch.setattr(rag, "HYBRID_SEARCH", True)
    monkeypatch.setattr(rag, "SCORE_THRESHOLD", 0.5)
    html = _landing_html(2024, _CHAPTERS + ["AR2024_E.pdf"])
    _fake_loader(monkeypatch, [
        "HONG KONG MONETARY AUTHORITY ANNUAL REPORT 2024",
        "Total assets of the Exchange Fund were 4071732 million Hong Kong dollars.",
    ])
    hkma.ingest_annual_report(
        2024, dest, fetch_text=lambda url: html, fetch_bytes=lambda url: b"%PDF-1.7")

    store = ingest.open_vectorstore()
    question = "Exchange Fund total assets 4071732"
    docs = rag.retrieve_and_rerank(
        question, store, owner=ingest.PUBLIC_OWNER,
        keyword=True, reranker="cross-encoder")
    assert docs, "the keyword evidence should be retrieved"
    sources = rag.build_sources(question, docs)
    assert any(s["source_file"] == "AR2024_E.pdf" and s["page"] == 2 for s in sources), sources


@pytest.mark.filterwarnings("ignore:Relevance scores must be between 0 and 1")
def test_hkma_query_refuses_when_evidence_is_below_threshold(dest, local_store, monkeypatch):
    monkeypatch.setattr(rag, "_get_reranker", lambda: _FakeReranker())
    html = _landing_html(2024, _CHAPTERS + ["AR2024_E.pdf"])
    _fake_loader(monkeypatch, ["unrelated content about the weather"])
    hkma.ingest_annual_report(
        2024, dest, fetch_text=lambda url: html, fetch_bytes=lambda url: b"%PDF-1.7")

    # No chunk reaches the threshold and keyword search has no match -> refusal,
    # returned without ever calling the language model.
    monkeypatch.setattr(rag, "SCORE_THRESHOLD", float("inf"))
    monkeypatch.setattr(rag, "HYBRID_SEARCH", False)
    result = rag.query("What was the total return of the Exchange Fund?")
    assert result.sources == []
    assert "No relevant context" in result.answer or "not contain" in result.answer


@pytest.mark.filterwarnings("ignore:Relevance scores must be between 0 and 1")
def test_cross_encoder_floor_refuses_irrelevant_but_keeps_exact_values(dest, local_store, monkeypatch):
    monkeypatch.setattr(rag, "RERANK_SCORE_FLOOR", -2.0)
    # Force every candidate through so the cross-encoder floor is the only gate.
    monkeypatch.setattr(rag, "SCORE_THRESHOLD", float("-inf"))
    monkeypatch.setattr(rag, "HYBRID_SEARCH", True)
    html = _landing_html(2024, _CHAPTERS + ["AR2024_E.pdf"])
    _fake_loader(monkeypatch, [
        "HONG KONG MONETARY AUTHORITY ANNUAL REPORT 2024",
        "Total assets of the Exchange Fund were 4071732 million Hong Kong dollars.",
        "The Authority will give a brief outlook on conditions today.",
    ])
    hkma.ingest_annual_report(
        2024, dest, fetch_text=lambda url: html, fetch_bytes=lambda url: b"%PDF-1.7")
    store = ingest.open_vectorstore()

    # Every passage scores far below the floor and no amount is quoted -> the
    # filler-only question is refused even though a common keyword ("today")
    # pulled candidates in.
    monkeypatch.setattr(rag, "_get_reranker", lambda: _FakeReranker(score=-9.0))
    assert rag.retrieve_and_rerank(
        "give me the weather today", store, owner=ingest.PUBLIC_OWNER, keyword=True) == []

    # The same low scores, but the question quotes an amount the page actually
    # contains: exact-value matches are exempt and the evidence is returned.
    docs = rag.retrieve_and_rerank(
        "Exchange Fund total assets 4071732", store,
        owner=ingest.PUBLIC_OWNER, keyword=True)
    assert docs
    assert any(d.metadata["retrieval"].get("exact_value_match") for d in docs)


class _ScoredReranker:
    """Per-passage cross-encoder scores, chosen by a content substring.

    Lets the calibrated floor be checked at its exact boundary without loading
    the real model; default scores are far below the floor.
    """

    def __init__(self, scores, default=-9.0):
        self._scores = scores  # list of (substring, score)
        self._default = default

    def predict(self, pairs):
        out = []
        for _, passage in pairs:
            score = self._default
            for needle, marked in self._scores:
                if needle in passage:
                    score = marked
                    break
            out.append(score)
        return out


@pytest.mark.filterwarnings("ignore:Relevance scores must be between 0 and 1")
def test_cross_encoder_floor_keeps_only_candidates_at_or_above_floor(
    dest, local_store, monkeypatch
):
    """The floor is inclusive (ce >= floor) and applies per candidate.

    Locked against the real-model calibration on AR2024 (floor = -2.0):
    genuine English answers and topical passages score >= -2.0 and survive;
    grossly irrelevant passages score well below and are dropped.
    """
    monkeypatch.setattr(rag, "RERANK_SCORE_FLOOR", -2.0)
    monkeypatch.setattr(rag, "SCORE_THRESHOLD", float("-inf"))
    monkeypatch.setattr(rag, "HYBRID_SEARCH", True)
    html = _landing_html(2024, _CHAPTERS + ["AR2024_E.pdf"])
    _fake_loader(monkeypatch, [
        "HONG KONG MONETARY AUTHORITY ANNUAL REPORT 2024",
        "The Hong Kong dollar traded within a band of 7.75 to 7.85 per US dollar.",
        "This sentence sits exactly at the boundary of the calibrated floor.",
        "General banking commentary that is mildly related but adds no figure.",
        "A weather forecast for the coming weekend with sunny skies.",
    ])
    hkma.ingest_annual_report(
        2024, dest, fetch_text=lambda url: html, fetch_bytes=lambda url: b"%PDF-1.7")
    store = ingest.open_vectorstore()

    monkeypatch.setattr(rag, "_get_reranker", lambda: _ScoredReranker([
        ("band of 7.75", 1.5),          # clearly above -> kept
        ("exactly at the boundary", -2.0),   # inclusive boundary -> kept
        ("mildly related", -1.5),       # above floor -> kept
        ("weather forecast", -3.0),     # below floor -> dropped
    ]))
    docs = rag.retrieve_and_rerank(
        "What is the Hong Kong dollar exchange rate band?",
        store, owner=ingest.PUBLIC_OWNER, keyword=True)
    assert docs, "passages at/above the floor must be returned"
    joined = " ".join(d.page_content for d in docs)
    assert "band of 7.75" in joined
    assert "exactly at the boundary" in joined
    assert "mildly related" in joined
    assert "weather forecast" not in joined
    # Nothing is falsely flagged as an exact-value match (question quotes no figure).
    assert not any(d.metadata["retrieval"].get("exact_value_match") for d in docs)
