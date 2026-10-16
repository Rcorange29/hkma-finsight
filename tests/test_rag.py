"""Tests for context formatting in rag.py.

doc_to_context is shared between the LLM prompt and the RAGAs evaluation, so
these pin both the provenance line and the prompt layout. No API key needed:
nothing here touches the vector store or the model.
"""

from langchain_core.documents import Document

import rag
from rag import _format_docs, doc_to_context


def _doc(content="Total revenue 245,122", page=66, source="microsoft-sec-10k.pdf"):
    return Document(page_content=content, metadata={"source_file": source, "page": page})


def test_context_carries_source_and_page():
    out = doc_to_context(_doc())
    assert out == "Source: microsoft-sec-10k.pdf, Page: 66\nTotal revenue 245,122"


def test_context_falls_back_to_the_source_metadata_key():
    doc = Document(page_content="body", metadata={"source": "other.pdf", "page": 3})
    assert doc_to_context(doc).startswith("Source: other.pdf, Page: 3")


def test_context_tolerates_missing_metadata():
    assert doc_to_context(Document(page_content="body")) == "Source: unknown, Page: ?\nbody"


def test_format_docs_numbers_entries_and_keeps_the_separator():
    """Pins the exact prompt layout - changing it silently changes the prompt."""
    out = _format_docs([_doc("first", page=1), _doc("second", page=2)])
    assert out == (
        "[1] Source: microsoft-sec-10k.pdf, Page: 1\nfirst"
        "\n\n---\n\n"
        "[2] Source: microsoft-sec-10k.pdf, Page: 2\nsecond"
    )


def test_format_docs_uses_the_same_provenance_line_as_the_evaluation():
    """The eval passes doc_to_context(doc); the prompt must contain that verbatim."""
    doc = _doc()
    assert doc_to_context(doc) in _format_docs([doc])


def test_context_names_the_company_when_known():
    doc = Document(page_content="body", metadata={
        "source_file": "acme.pdf", "page": 2, "company": "Acme Holdings, Inc."})
    assert doc_to_context(doc) == "Source: acme.pdf (Acme Holdings, Inc.), Page: 2\nbody"


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


def test_a_correct_answer_passes():
    assert rag.check_answer("Goodwill was $50,969 million (msft.pdf, p.89).", "q", DOCS) == []
    # Cited by the page's 1-based label works too
    assert rag.check_answer("It was $51.0 billion (msft.pdf, p.94).", "q", DOCS) == []


def test_a_blended_figure_is_caught():
    problems = rag.check_answer("Goodwill was $51,969 million (msft.pdf, p.89).", "q", DOCS)
    assert problems == ["51,969 isn't printed on msft.pdf p.89"]


def test_a_real_figure_cited_to_the_wrong_page_is_caught():
    # p.92 is a retrieved page (the Nuance note), but $51.0 billion isn't on it
    problems = rag.check_answer("It was $51.0 billion (msft.pdf, p.92).", "q", DOCS)
    assert problems == ["51.0 isn't printed on msft.pdf p.92"]


def test_a_citation_of_a_page_that_wasnt_retrieved_is_caught():
    problems = rag.check_answer("Goodwill was $50,969 million (msft.pdf, p.12).", "q", DOCS)
    assert problems == ["(msft.pdf, p.12) isn't the source of any excerpt"]


def test_figures_from_the_question_and_uncited_figures_in_excerpts_pass():
    assert rag.check_answer("Yes, 57,582 is not in these pages.", "Where is 57,582?", DOCS) == []
    assert rag.check_answer("Goodwill was 50,969.", "q", DOCS) == []
    assert rag.check_answer("Goodwill was 12,345.", "q", DOCS) == [
        "12,345 isn't printed in any excerpt"]


def test_years_and_whole_percentages_are_not_checked():
    assert rag.check_answer("In fiscal 2024 the rate was 18%.", "q", DOCS) == []


def test_a_failed_check_is_retried_with_the_problems(monkeypatch):
    calls = []

    def fake_answer(question, docs, problems=None):
        calls.append(problems)
        return ("Goodwill was $51,969 million (msft.pdf, p.89)." if problems is None
                else "Goodwill was $50,969 million (msft.pdf, p.89).")

    monkeypatch.setattr(rag, "_answer_with_docs", fake_answer)
    assert rag.answer_checked("q", DOCS) == "Goodwill was $50,969 million (msft.pdf, p.89)."
    assert calls == [None, ["51,969 isn't printed on msft.pdf p.89"]]


def test_an_answer_that_fails_twice_carries_a_warning(monkeypatch):
    monkeypatch.setattr(rag, "_answer_with_docs",
                        lambda q, docs, problems=None: "It was $1,234 million (msft.pdf, p.89).")
    answer = rag.answer_checked("q", DOCS)
    assert answer.startswith("It was $1,234 million (msft.pdf, p.89).")
    assert "Not verified against the excerpts: 1,234 isn't printed on msft.pdf p.89" in answer
