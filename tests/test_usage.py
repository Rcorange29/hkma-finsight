"""Tests for usage limits: the counting in usage.py and how the API applies it."""

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from financial_rag import api, ingest, router, usage

ALICE = "0f8b1c2e-alice"
BOB = "7d3a9e41-bob"
LIMITS = usage.Limits(questions_per_month=3, uploads_per_month=1,
                      questions_per_minute=2, questions_per_day_all_users=5)


def _at(text: str) -> float:
    return datetime.fromisoformat(text).replace(tzinfo=timezone.utc).timestamp()


NOON = _at("2026-10-16T12:00:00")


def _ask(owner, n, start=NOON, gap=61.0, limits=LIMITS):
    """Record n questions spaced out so the per-minute limit stays clear."""
    for i in range(n):
        usage.record(owner, usage.QUESTION, limits, now=start + i * gap)


# ---------------------------------------------------------------------------
# Counting
# ---------------------------------------------------------------------------

def test_monthly_question_limit():
    _ask(ALICE, 3)
    with pytest.raises(usage.LimitReached) as exc:
        usage.record(ALICE, usage.QUESTION, LIMITS, now=NOON + 600)
    assert "3 questions this month" in str(exc.value)
    assert "November 1" in str(exc.value)
    # Other users have their own allowance
    usage.record(BOB, usage.QUESTION, LIMITS, now=NOON + 600)


def test_monthly_limit_resets_with_the_calendar_month():
    _ask(ALICE, 3)
    usage.record(ALICE, usage.QUESTION, LIMITS, now=_at("2026-11-01T00:00:01"))


def test_december_resets_in_january():
    december = _at("2026-12-31T23:00:00")
    _ask(ALICE, 3, start=december, gap=61)
    with pytest.raises(usage.LimitReached) as exc:
        usage.record(ALICE, usage.QUESTION, LIMITS, now=december + 600)
    assert "January 1" in str(exc.value)
    usage.record(ALICE, usage.QUESTION, LIMITS, now=_at("2027-01-01T00:00:01"))


def test_per_minute_limit_says_how_long_to_wait():
    _ask(ALICE, 2, gap=10)
    with pytest.raises(usage.LimitReached) as exc:
        usage.record(ALICE, usage.QUESTION, LIMITS, now=NOON + 20)
    assert exc.value.retry_after == 41     # the first question leaves the window at +60
    assert "41 seconds" in str(exc.value)
    usage.record(ALICE, usage.QUESTION, LIMITS, now=NOON + 61)


def test_daily_limit_across_all_users():
    limits = usage.Limits(questions_per_month=100, uploads_per_month=1,
                          questions_per_minute=0, questions_per_day_all_users=5)
    for i, owner in enumerate([ALICE, BOB, ALICE, BOB, ALICE]):
        usage.record(owner, usage.QUESTION, limits, now=NOON + i)
    with pytest.raises(usage.LimitReached) as exc:
        usage.record("someone-new", usage.QUESTION, limits, now=NOON + 10)
    assert "daily question limit" in str(exc.value)
    assert exc.value.retry_after == 12 * 3600 - 10 + 1
    usage.record("someone-new", usage.QUESTION, limits, now=_at("2026-10-17T00:00:01"))


def test_upload_limit_is_separate_from_questions():
    usage.record(ALICE, usage.UPLOAD, LIMITS, now=NOON)
    with pytest.raises(usage.LimitReached) as exc:
        usage.record(ALICE, usage.UPLOAD, LIMITS, now=NOON + 1)
    assert "1 filings this month" in str(exc.value)
    _ask(ALICE, 3, start=NOON + 2)


def test_zero_turns_a_limit_off():
    limits = usage.Limits(0, 0, 0, 0)
    for i in range(20):
        usage.record(ALICE, usage.QUESTION, limits, now=NOON + i)
        usage.record(ALICE, usage.UPLOAD, limits, now=NOON + i)


def test_refund_gives_the_allowance_back():
    _ask(ALICE, 2)
    event = usage.record(ALICE, usage.QUESTION, LIMITS, now=NOON + 200)
    usage.refund(event)
    usage.record(ALICE, usage.QUESTION, LIMITS, now=NOON + 300)


def test_local_user_is_never_limited_or_counted():
    for i in range(10):
        assert usage.record("local", usage.QUESTION, LIMITS, now=NOON + i) is None
    assert usage.usage("local", LIMITS, now=NOON).question_limit == 0
    usage.refund(None)


def test_usage_reports_this_months_counts():
    _ask(ALICE, 2, start=_at("2026-09-30T12:00:00"))     # last month: not counted
    _ask(ALICE, 2)
    usage.record(ALICE, usage.UPLOAD, LIMITS, now=NOON)
    assert usage.usage(ALICE, LIMITS, now=NOON + 1000) == usage.Usage(
        questions=2, question_limit=3, uploads=1, upload_limit=1, resets="2026-11-01",
    )


def test_limits_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("QUESTIONS_PER_MONTH", "7")
    monkeypatch.setenv("UPLOADS_PER_MONTH", "0")
    monkeypatch.setenv("QUESTIONS_PER_MINUTE", "not a number")
    monkeypatch.delenv("QUESTIONS_PER_DAY_ALL_USERS", raising=False)
    assert usage.Limits.from_env() == usage.Limits(7, 0, 5, 500)


# ---------------------------------------------------------------------------
# The API
# ---------------------------------------------------------------------------

@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(usage.Limits, "from_env", classmethod(lambda cls: LIMITS))
    return TestClient(api.app)


def _query(client, owner=ALICE):
    return client.post("/query", json={"question": "Revenue?"}, headers={"X-User-Id": owner})


def test_query_over_the_limit_gets_429(client, monkeypatch):
    monkeypatch.setattr(router, "answer",
                        lambda *a, **k: router.Answer(answer="ok", sources=[], route="text"))
    monkeypatch.setattr(usage, "_now", iter(NOON + 61 * i for i in range(10)).__next__)
    assert [_query(client).status_code for _ in range(3)] == [200, 200, 200]
    response = _query(client)
    assert response.status_code == 429
    assert "3 questions this month" in response.json()["detail"]
    assert int(response.headers["Retry-After"]) > 0
    assert client.get("/usage", headers={"X-User-Id": ALICE}).json()["questions"] == 3


def test_failed_query_is_not_counted(client, monkeypatch):
    def broken(*a, **k):
        raise RuntimeError("OpenAI is down")

    monkeypatch.setattr(router, "answer", broken)
    assert _query(client).status_code == 503
    assert client.get("/usage", headers={"X-User-Id": ALICE}).json() == {
        "questions": 0, "question_limit": 3, "uploads": 0, "upload_limit": 1,
        "resets": usage.usage(ALICE).resets,
    }


def _upload(client, owner=ALICE):
    return client.post(
        "/documents", files={"file": ("acme.pdf", b"%PDF-1.7 x", "application/pdf")},
        headers={"X-User-Id": owner},
    )


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(ingest, "USERS_DIR", tmp_path / "data" / "users")


def test_upload_over_the_limit_gets_429_before_ingesting(client, data_dir, monkeypatch):
    calls = []
    monkeypatch.setattr(ingest, "ingest_file",
                        lambda path, owner, max_pages=None: calls.append(max_pages) or 1)
    assert _upload(client).status_code == 201
    response = _upload(client)
    assert response.status_code == 429
    assert "1 filings this month" in response.json()["detail"]
    assert calls == [api.MAX_UPLOAD_PAGES]


def test_failed_upload_is_not_counted(client, data_dir, monkeypatch):
    def too_long(path, owner, max_pages=None):
        raise ValueError(f"{path.name} has 900 pages; the limit is {max_pages}.")

    monkeypatch.setattr(ingest, "ingest_file", too_long)
    response = _upload(client)
    assert response.status_code == 422
    assert "the limit is 500" in response.json()["detail"]
    assert client.get("/usage", headers={"X-User-Id": ALICE}).json()["uploads"] == 0


def test_ingest_rejects_files_over_the_page_limit(tmp_path, monkeypatch):
    pages =[Document(page_content="text", metadata={"page": i}) for i in range(3)]
    monkeypatch.setattr(ingest, "load_pdf", lambda path, owner: pages)
    monkeypatch.setattr(ingest, "open_vectorstore", lambda: pytest.fail("embedded"))
    with pytest.raises(ValueError, match="has 3 pages; the limit is 2"):
        ingest.ingest_file(tmp_path / "long.pdf", ALICE, max_pages=2)
