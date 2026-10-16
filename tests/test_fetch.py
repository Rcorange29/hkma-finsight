"""Tests for fetching a 10-K by ticker as a background job (POST /filings, GET /jobs)."""

import time

import pytest
from fastapi.testclient import TestClient

from financial_rag import api, edgar, financials, ingest, jobs, usage

ALICE = "0f8b1c2e-alice"
BOB = "7d3a9e41-bob"
FILING = edgar.FilingRef(cik=320193, company="Apple Inc.", form="10-K",
                         accession="0000320193-25-000079", report_date="2025-09-27",
                         filing_date="2025-10-31", document="aapl-20250927.htm")


def wait(job_id, owner=ALICE, client=None, timeout=10.0):
    """Poll a job until it finishes, as the UI does."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/jobs/{job_id}", headers={"X-User-Id": owner}).json()
        if body["state"] in (jobs.DONE, jobs.FAILED):
            return body
        time.sleep(0.02)
    raise AssertionError("job didn't finish")


@pytest.fixture
def client():
    return TestClient(api.app)


@pytest.fixture
def sec(tmp_path, monkeypatch):
    """EDGAR knows AAPL; ingestion is recorded instead of run."""
    monkeypatch.setattr(ingest, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(ingest, "USERS_DIR", tmp_path / "data" / "users")
    monkeypatch.setattr(edgar, "cik_for_ticker",
                        lambda t: {"AAPL": 320193, "BRK-B": 1067983}.get(t))
    monkeypatch.setattr(edgar, "latest_annual_report", lambda cik: FILING)
    monkeypatch.setattr(edgar, "fetch_document", lambda url: "<html>10-K</html>")
    calls = []

    def fake_ingest(path, owner, max_pages=None, cik=None):
        calls.append((path.name, owner, max_pages, cik, path.read_text(encoding="utf-8")))
        return 73

    monkeypatch.setattr(ingest, "ingest_file", fake_ingest)
    monkeypatch.setattr(ingest, "filings", lambda owners: [
        ingest.Filing("aapl-10k-2025-09-27.htm", ALICE, 73, "Apple Inc.")])
    monkeypatch.setattr(financials, "count_facts", lambda owner: {"aapl-10k-2025-09-27.htm": 83})
    return calls


def _fetch(client, ticker="AAPL", owner=ALICE):
    return client.post("/filings", json={"ticker": ticker}, headers={"X-User-Id": owner})


def test_fetch_runs_as_a_job_and_indexes_the_filing(client, sec, tmp_path):
    response = _fetch(client, " aapl ")
    assert response.status_code == 202
    queued = response.json()
    assert (queued["kind"], queued["subject"]) == ("fetch_10k", "AAPL")
    assert queued["state"] in (jobs.QUEUED, jobs.RUNNING, jobs.DONE)

    done = wait(queued["job_id"], client=client)
    assert done["state"] == jobs.DONE and done["error"] == ""
    assert done["document"] == {"source_file": "aapl-10k-2025-09-27.htm", "chunks": 73,
                                "figures": 83, "shared": False, "company": "Apple Inc."}
    # Ingested from the user's own folder under a readable name, with the CIK for XBRL
    assert sec == [("aapl-10k-2025-09-27.htm", ALICE, api.MAX_UPLOAD_PAGES, 320193,
                    "<html>10-K</html>")]
    saved = tmp_path / "data" / "users" / ALICE / "aapl-10k-2025-09-27.htm"
    assert saved.read_text(encoding="utf-8") == "<html>10-K</html>"
    assert [p.name for p in saved.parent.iterdir()] == [saved.name]   # staging cleaned up


def test_a_fetch_counts_as_an_upload(client, sec):
    wait(_fetch(client).json()["job_id"], client=client)
    assert usage.usage(ALICE).uploads == 1


def test_a_failed_fetch_is_refunded_and_says_why(client, sec, monkeypatch):
    monkeypatch.setattr(edgar, "latest_annual_report", lambda cik: None)
    done = wait(_fetch(client).json()["job_id"], client=client)
    assert done["state"] == jobs.FAILED
    assert done["error"] == "AAPL has no 10-K on SEC EDGAR."
    assert usage.usage(ALICE).uploads == 0
    assert sec == []


def test_unknown_tickers_are_refused_up_front(client, sec):
    response = _fetch(client, "ZZZZ")
    assert response.status_code == 404
    assert response.json()["detail"] == "No SEC filer has the ticker ZZZZ."
    assert usage.usage(ALICE).uploads == 0


@pytest.mark.parametrize("ticker", ["", "1ABC", "AAPL; rm", "../etc", "A" * 13])
def test_malformed_tickers_are_refused(client, sec, ticker):
    assert _fetch(client, ticker).status_code == 422


def test_class_share_tickers_use_edgars_spelling(client, sec):
    response = _fetch(client, "brk.b")
    assert response.status_code == 202
    assert response.json()["subject"] == "BRK-B"
    assert wait(response.json()["job_id"], client=client)["state"] == jobs.DONE
    assert sec[0][:4] == ("brk-b-10k-2025-09-27.htm", ALICE, api.MAX_UPLOAD_PAGES, 1067983)


def test_edgar_down_is_a_503(client, sec, monkeypatch):
    def down(ticker):
        raise edgar.EdgarUnavailable("SEC_USER_AGENT isn't set.")

    monkeypatch.setattr(edgar, "cik_for_ticker", down)
    response = _fetch(client)
    assert response.status_code == 503
    assert "SEC_USER_AGENT" in response.json()["detail"]


def test_over_the_upload_limit_no_job_starts(client, sec, monkeypatch):
    monkeypatch.setattr(usage.Limits, "from_env", classmethod(
        lambda cls: usage.Limits(50, 1, 5, 500)))
    wait(_fetch(client).json()["job_id"], client=client)
    response = _fetch(client)
    assert response.status_code == 429
    assert len(sec) == 1


def test_jobs_are_private_to_their_owner(client, sec):
    job_id = _fetch(client).json()["job_id"]
    wait(job_id, client=client)
    assert client.get(f"/jobs/{job_id}", headers={"X-User-Id": BOB}).status_code == 404
    assert client.get("/jobs/no-such-job", headers={"X-User-Id": ALICE}).status_code == 404


# ---------------------------------------------------------------------------
# The job runner
# ---------------------------------------------------------------------------

def _finish(job_id, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = jobs.get(job_id)
        if job.state in (jobs.DONE, jobs.FAILED):
            return job
        time.sleep(0.01)
    raise AssertionError("job didn't finish")


def test_a_job_reports_its_steps_and_result():
    seen = []

    def work(step):
        step("halfway")
        seen.append(jobs.get(job.id).step)
        return '{"ok": true}'

    job = jobs.submit("alice", "test", "x", work)
    finished = _finish(job.id)
    assert seen == ["halfway"]
    assert (finished.state, finished.result, finished.step) == (jobs.DONE, '{"ok": true}', "")


def test_jobs_run_one_at_a_time():
    running = []
    overlap = []

    def work(step):
        running.append(1)
        overlap.append(len(running))
        time.sleep(0.05)
        running.pop()
        return ""

    ids = [jobs.submit("alice", "test", str(i), work).id for i in range(3)]
    for job_id in ids:
        _finish(job_id)
    assert overlap == [1, 1, 1]


def test_a_crash_fails_the_job_and_runs_the_cleanup():
    cleaned = []

    def work(step):
        raise RuntimeError("disk full")

    job = jobs.submit("alice", "test", "x", work, on_failure=lambda: cleaned.append(1))
    failed = _finish(job.id)
    assert (failed.state, failed.error) == (jobs.FAILED, "disk full")
    assert cleaned == [1]


def test_jobs_left_running_by_a_restart_are_failed():
    jobs._write("INSERT INTO jobs (id, owner, kind, subject, state, created, updated) "
                "VALUES ('old', 'alice', 'fetch_10k', 'AAPL', 'running', 0, 0)", ())
    assert jobs.recover() == 1
    old = jobs.get("old")
    assert old.state == jobs.FAILED
    assert "restart" in old.error
