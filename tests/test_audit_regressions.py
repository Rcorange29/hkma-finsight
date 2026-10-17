"""Deterministic audit regressions; synthetic fixtures are not HKMA quotations."""
import json
import os
import subprocess
import sys

import pytest
from langchain_core.documents import Document

from financial_rag import rag


@pytest.fixture
def excerpt():
    return Document(page_content="Investment income 218.8 billion.", metadata={
        "source_file": "report.pdf", "page": 183, "page_label": "182"})


def test_printed_page_label_is_not_a_pdf_page_alias(excerpt):
    assert rag.check_answer("Income 218.8 billion (report.pdf, p.182).", "q", [excerpt])


@pytest.mark.parametrize("claim", ["Salary 987654", "Employees 999", "Rate 18%"])
def test_integer_claims_require_evidence(excerpt, claim):
    assert rag.check_answer(f"{claim} (report.pdf, p.184).", "q", [excerpt])


def test_question_is_not_evidence(excerpt):
    assert rag.check_answer("Salary 123.45 (report.pdf, p.184).",
                            "Is salary 123.45?", [excerpt])


def test_uncited_answer_is_not_verified(excerpt):
    assert rag.check_answer("Income 218.8 billion.", "q", [excerpt])


def test_empty_answer_is_not_success(excerpt):
    assert rag.check_answer("", "q", [excerpt])


def test_repeated_failure_withholds_bad_answer(monkeypatch, excerpt):
    monkeypatch.setattr(rag, "_answer_with_docs", lambda *a, **k:
                        "Salary 987.65 (report.pdf, p.184).")
    answer = rag.answer_checked("Salary?", [excerpt])
    assert "987.65" not in answer
    assert "not found in context" in answer.lower()


def test_config_reads_dotenv_before_resolving_settings(tmp_path):
    (tmp_path / ".env").write_text("LLM_PROVIDER=doubao\nARK_API_KEY=offline-dummy\n")
    env = {k: v for k, v in os.environ.items()
           if k not in {"ARK_API_KEY", "OPENAI_API_KEY", "LLM_PROVIDER", "PYTHON_DOTENV_DISABLED"}}
    env["PYTHONPATH"] = os.getcwd()
    code = "from financial_rag import config; assert config.LLM_API_KEY == 'offline-dummy'"
    subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env, check=True)


def test_ark_does_not_inherit_openai_credentials_or_url():
    env = dict(os.environ, LLM_PROVIDER="doubao", ARK_API_KEY="", ARK_BASE_URL="",
               OPENAI_API_KEY="offline-openai-dummy", OPENAI_BASE_URL="https://other.invalid/v1")
    code = ("import json; from financial_rag import config; "
            "print(json.dumps([bool(config.LLM_API_KEY), config.LLM_BASE_URL]))")
    result = subprocess.run([sys.executable, "-c", code], env=env, check=True,
                            capture_output=True, text=True)
    assert json.loads(result.stdout) == [False, "https://ark.cn-beijing.volces.com/api/v3"]


def test_supported_integer_and_exact_refusal_pass():
    doc = Document(page_content="Staff 125. Rate 18% in 2024.",
                   metadata={"source_file": "fixture.pdf", "page": 0})
    assert not rag.check_answer("Staff 125 (fixture.pdf, p.1).", "q", [doc])
    assert not rag.check_answer("Rate 18% in 2024 (fixture.pdf, p.1).", "q", [doc])
    assert not rag.check_answer("not found in context", "q", [doc])


def test_no_docs_never_calls_generation(monkeypatch):
    def unexpected(*a, **k):
        pytest.fail("No evidence must not call the LLM")
    monkeypatch.setattr(rag, "_answer_with_docs", unexpected)
    assert rag.answer_checked("q", []) == rag.ABSTENTION


def test_stream_never_publishes_rejected_draft(monkeypatch, excerpt):
    from langchain_core.runnables import RunnableLambda

    from financial_rag import tracing
    events = []
    monkeypatch.setattr(tracing, "emit", lambda *a: events.append(a))
    monkeypatch.setattr(tracing, "listened", lambda: True)
    monkeypatch.setattr(rag, "get_chat_llm", lambda: RunnableLambda(
        lambda _: "Salary 987.65 (report.pdf, p.184)."))
    answer = rag.answer_checked("Salary?", [excerpt])
    published = "".join(e[1] for e in events if e[0] == "token")
    assert published == answer
    assert "987.65" not in published


@pytest.mark.parametrize("status", [200, 401, 429, 500, "timeout", "empty"])
def test_chat_transport_offline(monkeypatch, status):
    import httpx
    import openai

    from financial_rag import config, llm

    monkeypatch.setattr(config, "LLM_API_KEY", "offline-dummy")
    monkeypatch.setattr(config, "LLM_BASE_URL", "https://ark.example.invalid/api/v3")
    requests = []

    def handler(request):
        requests.append(request)
        assert request.url.path == "/api/v3/chat/completions"
        payload = json.loads(request.content)
        assert payload["model"] == config.CHAT_MODEL
        assert payload["messages"][0]["content"] == "offline prompt"
        assert request.headers["authorization"] == "Bearer offline-dummy"
        if status == "timeout":
            raise httpx.ReadTimeout("offline timeout", request=request)
        if status == "empty":
            return httpx.Response(200, json={"choices": []})
        if status != 200:
            return httpx.Response(status, json={"error": {"message": "offline error"}})
        return httpx.Response(200, json={"id": "offline", "object": "chat.completion",
            "created": 0, "model": config.CHAT_MODEL,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                         "finish_reason": "stop"}]})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        model = llm.get_chat_llm(http_client=client, max_retries=0)
        assert model.request_timeout == 30
        if status == 200:
            assert model.invoke("offline prompt").content == "ok"
        elif status == "empty":
            with pytest.raises((ValueError, IndexError)):
                model.invoke("offline prompt")
        else:
            error = openai.APITimeoutError if status == "timeout" else openai.APIStatusError
            with pytest.raises(error):
                model.invoke("offline prompt")
    assert len(requests) == 1


def test_same_filename_owned_by_different_users_is_ambiguous():
    docs = [Document(page_content=f"Amount {n}", metadata={
        "source_file": "same.pdf", "page": 0, "owner": owner})
        for n, owner in [(100, "public"), (200, "alice")]]
    assert rag.check_answer("Amount 200 (same.pdf, p.1).", "q", docs)
