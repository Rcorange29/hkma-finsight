"""Tests for context formatting in rag.py.

doc_to_context is shared between the LLM prompt and the RAGAs evaluation, so
these pin both the provenance line and the prompt layout. No API key needed:
nothing here touches the vector store or the model.
"""

import pytest
from langchain_core.documents import Document

from financial_rag import rag
from financial_rag.rag import _format_docs, doc_to_context


def _doc(content="Total revenue 245,122", page=66, source="microsoft-sec-10k.pdf"):
    return Document(page_content=content, metadata={"source_file": source, "page": page})


def test_context_carries_source_and_page():
    out = doc_to_context(_doc())
    assert out == "Source: microsoft-sec-10k.pdf, Page: 67\nTotal revenue 245,122"


def test_context_falls_back_to_the_source_metadata_key():
    doc = Document(page_content="body", metadata={"source": "other.pdf", "page": 3})
    assert doc_to_context(doc).startswith("Source: other.pdf, Page: 4")


def test_context_tolerates_missing_metadata():
    assert doc_to_context(Document(page_content="body")) == "Source: unknown, Page: ?\nbody"


def test_format_docs_numbers_entries_and_keeps_the_separator():
    """Pins the exact prompt layout - changing it silently changes the prompt."""
    out = _format_docs([_doc("first", page=1), _doc("second", page=2)])
    assert out == (
        "[1] Source: microsoft-sec-10k.pdf, Page: 2\nfirst"
        "\n\n---\n\n"
        "[2] Source: microsoft-sec-10k.pdf, Page: 3\nsecond"
    )


def test_format_docs_uses_the_same_provenance_line_as_the_evaluation():
    """The eval passes doc_to_context(doc); the prompt must contain that verbatim."""
    doc = _doc()
    assert doc_to_context(doc) in _format_docs([doc])


def test_context_names_the_company_when_known():
    doc = Document(page_content="body", metadata={
        "source_file": "acme.pdf", "page": 2, "company": "Acme Holdings, Inc."})
    assert doc_to_context(doc) == "Source: acme.pdf (Acme Holdings, Inc.), Page: 3\nbody"


# ---------------------------------------------------------------------------
# Checking answers against their excerpts
# ---------------------------------------------------------------------------

DOCS = [
    Document(page_content="Goodwill   50,969", metadata={
        "source_file": "msft.pdf", "page": 89, "page_label": "90"}),
    Document(page_content="Includes goodwill of $51.0 billion related to Activision.", metadata={
        "source_file": "msft.pdf", "page": 93, "page_label": "94"}),
    Document(page_content="Goodwill   $ 16,326 (Nuance)", metadata={
        "source_file": "msft.pdf", "page": 91, "page_label": "92"}),
]


def test_citations_use_the_page_number_a_reader_sees():
    # Index 89 is page 90 in a PDF viewer; the 0-based index isn't accepted
    assert rag.check_answer("Goodwill was $50,969 million (msft.pdf, p.89).", "q", DOCS) == [
        "(msft.pdf, p.89) isn't the source of any excerpt"]


def test_a_correct_answer_passes():
    assert rag.check_answer("Goodwill was $50,969 million (msft.pdf, p.90).", "q", DOCS) == []
    # Cited by the page's 1-based label works too
    assert rag.check_answer("It was $51.0 billion (msft.pdf, p.94).", "q", DOCS) == []


def test_a_blended_figure_is_caught():
    problems = rag.check_answer("Goodwill was $51,969 million (msft.pdf, p.90).", "q", DOCS)
    assert problems == ["51,969 isn't printed on msft.pdf p.90"]


def test_a_real_figure_cited_to_the_wrong_page_is_caught():
    # p.92 is a retrieved page (the Nuance note), but $51.0 billion isn't on it
    problems = rag.check_answer("It was $51.0 billion (msft.pdf, p.92).", "q", DOCS)
    assert problems == ["51.0 isn't printed on msft.pdf p.92"]


def test_a_citation_of_a_page_that_wasnt_retrieved_is_caught():
    problems = rag.check_answer("Goodwill was $50,969 million (msft.pdf, p.12).", "q", DOCS)
    assert problems == ["(msft.pdf, p.12) isn't the source of any excerpt"]


# The samples are EDGAR HTML 10-Ks: their citations must be checked the same way.
# Until this was caught, "(…htm, p.N)" wasn't recognised as a citation at all, so a
# wrong page or a page never retrieved went unnoticed.
HTML_DOCS = [
    Document(page_content="Short-term unearned revenue 57,582 50,901", metadata={
        "source_file": "microsoft-sec-10k.htm", "page": 57}),
    Document(page_content="Goodwill 50,969", metadata={
        "source_file": "microsoft-sec-10k.htm", "page": 75}),
]


@pytest.mark.parametrize(
    ("answer", "problems"),
    [
        ("It was $57,582 million (microsoft-sec-10k.htm, p. 58).", []),
        ("It was $57,582 million (microsoft-sec-10k.htm, p.76).",
         ["57,582 isn't printed on microsoft-sec-10k.htm p.76"]),
        ("It was $57,582 million (microsoft-sec-10k.htm, p.3).",
         ["(microsoft-sec-10k.htm, p.3) isn't the source of any excerpt"]),
        ("Goodwill was $50,969 million (MICROSOFT-SEC-10K.HTML, p.76).",
         ["(MICROSOFT-SEC-10K.HTML, p.76) isn't the source of any excerpt"]),
    ],
)
def test_html_filing_citations_are_checked(answer, problems):
    assert rag.check_answer(answer, "q", HTML_DOCS) == problems


def test_question_numbers_and_uncited_figures_are_not_evidence():
    assert rag.check_answer("Yes, 57,582 is not in these pages.", "Where is 57,582?", DOCS)
    assert rag.check_answer("Goodwill was 50,969.", "q", DOCS)
    assert "12,345 isn't printed in any excerpt" in rag.check_answer(
        "Goodwill was 12,345.", "q", DOCS)


def test_years_and_whole_percentages_require_support():
    problems = rag.check_answer("In fiscal 2024 the rate was 18%.", "q", DOCS)
    assert "2024 isn't printed in any excerpt" in problems
    assert "18 isn't printed in any excerpt" in problems


def test_a_failed_check_is_retried_with_the_problems(monkeypatch):
    calls = []

    def fake_answer(question, docs, problems=None):
        calls.append(problems)
        return ("Goodwill was $51,969 million (msft.pdf, p.90)." if problems is None
                else "Goodwill was $50,969 million (msft.pdf, p.90).")

    monkeypatch.setattr(rag, "_answer_with_docs", fake_answer)
    assert rag.answer_checked("q", DOCS) == "Goodwill was $50,969 million (msft.pdf, p.90)."
    assert calls == [None, ["51,969 isn't printed on msft.pdf p.90"]]


def test_an_answer_that_fails_twice_is_withheld(monkeypatch):
    monkeypatch.setattr(rag, "_answer_with_docs",
                        lambda q, docs, problems=None: "It was $1,234 million (msft.pdf, p.90).")
    answer = rag.answer_checked("q", DOCS)
    assert answer.startswith("Not found in context")
    assert "1,234" not in answer
    assert "Not verified against the excerpts" in answer


@pytest.mark.parametrize(("index", "shown"), [(0, 1), (66, 67), ("66", 67), (None, "?"), ("?", "?")])
def test_page_numbers_are_shown_the_way_pdf_viewers_count(index, shown):
    from financial_rag.financials import page_number
    assert page_number(index) == shown


def test_excerpts_drop_the_filings_own_page_footer():
    text = ("Greater China  64,377  (4) %  66,952\n"
            "Apple Inc. | 2025 Form 10-K | 22\n"
            "Japan  28,703")
    assert rag.strip_page_footers(text) == "Greater China  64,377  (4) %  66,952\n\nJapan  28,703"
    # Ordinary lines with a pipe or "Form 10-K" stay
    kept = "See Part II | Item 8 of this Form 10-K for details."
    assert rag.strip_page_footers(kept) == kept
