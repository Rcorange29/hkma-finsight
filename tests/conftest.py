import pytest

import financials


@pytest.fixture(autouse=True)
def isolated_facts_db(tmp_path, monkeypatch):
    """Every test gets its own figures database, never the real chroma_db/ one."""
    monkeypatch.setattr(financials, "FACTS_DB", tmp_path / "facts" / "financials.sqlite3")
