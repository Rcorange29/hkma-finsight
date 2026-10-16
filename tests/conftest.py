import pytest

from financial_rag import edgar, financials, ingest, jobs, rag, usage


@pytest.fixture(autouse=True)
def trusted_user_header(monkeypatch):
    """Tests name users with X-User-Id (local mode); test_auth.py tests tokens."""
    monkeypatch.setenv("AUTH_DISABLED", "1")
    monkeypatch.delenv("COGNITO_ISSUER", raising=False)
    monkeypatch.delenv("COGNITO_CLIENT_ID", raising=False)


@pytest.fixture(autouse=True)
def no_edgar(tmp_path, monkeypatch):
    """Tests never reach the SEC; test_xbrl.py and test_edgar.py feed it data instead."""
    def offline(url, agent):
        raise edgar.EdgarUnavailable(f"tests are offline: {url}")

    def no_documents(url):
        raise edgar.EdgarUnavailable(f"tests are offline: {url}")

    monkeypatch.setattr(edgar, "_get", offline)
    monkeypatch.setattr(edgar, "fetch_document", no_documents)
    monkeypatch.setattr(edgar, "CACHE_DIR", tmp_path / "edgar_cache")


@pytest.fixture(autouse=True)
def isolated_jobs_db(tmp_path, monkeypatch):
    """Each test has its own job records, and its jobs finish before it ends.

    A job still queued at teardown would otherwise run during a later test, with
    that test's fakes, or none (it once fetched a real filing that way).
    """
    monkeypatch.setattr(jobs, "JOBS_DB", tmp_path / "jobs" / "jobs.sqlite3")
    yield
    jobs._executor.submit(lambda: None).result(timeout=30)


@pytest.fixture(autouse=True)
def isolated_facts_db(tmp_path, monkeypatch):
    """Every test gets its own figures database, never the real chroma_db/ one."""
    monkeypatch.setattr(financials, "FACTS_DB", tmp_path / "facts" / "financials.sqlite3")


@pytest.fixture(autouse=True)
def isolated_usage_db(tmp_path, monkeypatch):
    """Nor the real usage counts, so one test's questions never limit another's."""
    monkeypatch.setattr(usage, "USAGE_DB", tmp_path / "usage" / "usage.sqlite3")


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    """Nor the real vector store: the filing list and retrieval see an empty one."""
    monkeypatch.setattr(ingest, "CHROMA_DIR", tmp_path / "chroma_db")
    monkeypatch.setattr(rag, "CHROMA_DIR", tmp_path / "chroma_db")


@pytest.fixture(autouse=True)
def no_llm_reranking(monkeypatch):
    """Tests rerank with the (faked) cross-encoder only; LLM tests opt back in."""
    monkeypatch.setattr(rag, "RERANKER", "cross-encoder")
