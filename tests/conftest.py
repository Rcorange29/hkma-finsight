import pytest

import financials
import ingest
import rag


@pytest.fixture(autouse=True)
def isolated_facts_db(tmp_path, monkeypatch):
    """Every test gets its own figures database, never the real chroma_db/ one."""
    monkeypatch.setattr(financials, "FACTS_DB", tmp_path / "facts" / "financials.sqlite3")


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    """Nor the real vector store: the filing list and retrieval see an empty one."""
    monkeypatch.setattr(ingest, "CHROMA_DIR", tmp_path / "chroma_db")
    monkeypatch.setattr(rag, "CHROMA_DIR", tmp_path / "chroma_db")


@pytest.fixture(autouse=True)
def no_llm_reranking(monkeypatch):
    """Tests rerank with the (faked) cross-encoder only; LLM tests opt back in."""
    monkeypatch.setattr(rag, "RERANKER", "cross-encoder")
