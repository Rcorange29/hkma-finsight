"""Tests for streamed answers (POST /query/stream) and the progress they report.

router.answer is replaced by a stand-in that opens the same spans and emits the
same tokens a real answer would, so nothing is called.
"""

import json

import pytest
from fastapi.testclient import TestClient

from financial_rag import api, router, tracing, usage

ALICE = "0f8b1c2e-alice"


@pytest.fixture
def client():
    tracing.setup()     # as api.py does at import: the progress processor
    return TestClient(api.app)


def _fake_answer(steps=("plan", "retrieve", "rerank.cross_encoder", "rerank.llm", "generate",
                        "check"), tokens=("Revenue was ", "$245,122 million."), retry=False,
                 error=None):
    def answer(question, owner, source_file=None, history=None, **kw):
        for step in steps:
            with tracing.tracer.start_as_current_span(step):
                if step == "generate":
                    for token in tokens:
                        tracing.emit("token", token)
                if step == "check" and retry:
                    tracing.emit("retry")
        if error:
            raise error
        return router.Answer("Revenue was $245,122 million (msft.htm, p.56).",
                             [{"source_file": "msft.htm", "page": 56, "excerpt": "Total revenue"}],
                             "text", question=question)
    return answer


def _stream(client, question="What was revenue?", owner=ALICE):
    response = client.post("/query/stream", json={"question": question},
                           headers={"X-User-Id": owner})
    return response, [json.loads(line) for line in response.text.splitlines() if line]


def test_a_streamed_answer_reports_steps_then_tokens_then_the_checked_answer(
        client, monkeypatch):
    monkeypatch.setattr(router, "answer", _fake_answer())
    response, events = _stream(client)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    assert [e for e in events if e["type"] == "step"] == [
        {"type": "step", "step": "Reading the question"},
        {"type": "step", "step": "Searching the filings"},
        {"type": "step", "step": "Ranking the passages"},     # not the cross-encoder
        {"type": "step", "step": "Writing the answer"},
        {"type": "step", "step": "Checking its figures against the filing"},
    ]
    assert "".join(e["text"] for e in events if e["type"] == "token") == \
        "Revenue was $245,122 million."
    done = events[-1]
    assert done["type"] == "done"
    assert done["response"]["answer"] == "Revenue was $245,122 million (msft.htm, p.56)."
    assert done["response"]["sources"][0]["page"] == 56
    assert usage.usage(ALICE).questions == 1


def test_the_same_step_twice_in_a_row_is_reported_once(client, monkeypatch):
    # A comparison searches each filing in turn
    monkeypatch.setattr(router, "answer", _fake_answer(steps=("plan", "retrieve", "retrieve")))
    _, events = _stream(client)
    assert [e["step"] for e in events if e["type"] == "step"] == [
        "Reading the question", "Searching the filings"]


def test_a_failed_check_withdraws_the_streamed_text(client, monkeypatch):
    monkeypatch.setattr(router, "answer", _fake_answer(retry=True))
    _, events = _stream(client)
    kinds = [e["type"] for e in events]
    assert kinds.index("retry") > kinds.index("token")
    assert kinds[-1] == "done"


def test_an_error_after_streaming_starts_is_an_event_and_refunded(client, monkeypatch):
    monkeypatch.setattr(router, "answer",
                        _fake_answer(error=router.UnknownFiling("No filing named x.pdf.")))
    response, events = _stream(client)
    assert response.status_code == 200
    assert events[-1] == {"type": "error", "status": 404, "detail": "No filing named x.pdf."}
    assert usage.usage(ALICE).questions == 0


def test_an_empty_question_is_refused_before_streaming(client, monkeypatch):
    monkeypatch.setattr(router, "answer", _fake_answer())
    response = client.post("/query/stream", json={"question": "  "},
                           headers={"X-User-Id": ALICE})
    assert response.status_code == 422


def test_a_usage_limit_is_refused_before_streaming(client, monkeypatch):
    monkeypatch.setenv("QUESTIONS_PER_MONTH", "1")
    monkeypatch.setattr(router, "answer", _fake_answer())
    assert _stream(client)[0].status_code == 200
    response, _ = _stream(client)
    assert response.status_code == 429


def test_progress_reaches_only_the_listener_in_its_own_context():
    heard = []
    tracing.setup()
    with tracing.listening(lambda kind, value: heard.append((kind, value))):
        assert tracing.listened()
        with tracing.tracer.start_as_current_span("plan"):
            tracing.emit("token", "x")
    with tracing.tracer.start_as_current_span("plan"):
        tracing.emit("token", "y")          # nobody listening: dropped
    assert not tracing.listened()
    assert heard == [("step", "plan"), ("token", "x")]
