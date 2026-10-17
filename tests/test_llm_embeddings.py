"""Tests for model wiring: Doubao/Ark chat config and local embeddings.

No network and no model download: the OpenAI/LangChain client classes are
replaced by a recorder that captures the constructor arguments.
"""

import langchain_openai
import pytest

from financial_rag import config, embeddings, llm, rag


class _Recorder:
    """Captures the kwargs a client class was built with."""

    last_kwargs: dict | None = None

    def __init__(self, **kwargs):
        type(self).last_kwargs = kwargs

    def __getattr__(self, name):
        return lambda *a, **k: None


def test_get_chat_llm_points_at_doubao_ark_by_default(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "doubao")
    monkeypatch.setattr(config, "CHAT_MODEL", config.DOUBAO_CHAT_MODEL_DEFAULT)
    monkeypatch.setattr(config, "LLM_BASE_URL", config.ARK_BASE_URL_DEFAULT)
    monkeypatch.setattr(config, "LLM_API_KEY", "ark-test-key")
    monkeypatch.setattr(langchain_openai, "ChatOpenAI", _Recorder)

    llm.get_chat_llm(timeout=12)

    kw = _Recorder.last_kwargs
    assert kw["model"] == "doubao-seed-2-1-pro-260628"
    assert kw["base_url"] == "https://ark.cn-beijing.volces.com/api/v3"
    assert kw["api_key"] == "ark-test-key"
    assert kw["temperature"] == 0
    assert kw["timeout"] == 12


def test_get_chat_llm_openai_provider_has_no_custom_base_url(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(config, "CHAT_MODEL", config.OPENAI_CHAT_MODEL_DEFAULT)
    monkeypatch.setattr(config, "LLM_BASE_URL", None)
    monkeypatch.setattr(config, "LLM_API_KEY", "sk-test")
    monkeypatch.setattr(langchain_openai, "ChatOpenAI", _Recorder)

    llm.get_chat_llm()

    kw = _Recorder.last_kwargs
    assert kw["model"] == "gpt-4o-mini"
    assert kw["base_url"] is None
    assert kw["api_key"] == "sk-test"


def test_get_chat_llm_raises_without_a_key(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "doubao")
    monkeypatch.setattr(config, "LLM_API_KEY", None)
    with pytest.raises(llm.LLMConfigError, match="ARK_API_KEY"):
        llm.get_chat_llm()


def test_get_embeddings_uses_local_sentence_transformer_by_default(monkeypatch):
    monkeypatch.setattr(config, "EMBEDDING_PROVIDER", "local")
    captured = {}

    class FakeHF:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import langchain_community.embeddings as lce
    monkeypatch.setattr(lce, "HuggingFaceEmbeddings", FakeHF)

    embeddings.get_embeddings()
    assert captured["model_name"] == "BAAI/bge-small-en-v1.5"


def test_get_embeddings_uses_openai_when_configured(monkeypatch):
    monkeypatch.setattr(config, "EMBEDDING_PROVIDER", "openai")
    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(langchain_openai, "OpenAIEmbeddings", FakeOpenAI)

    embeddings.get_embeddings()
    assert captured["model"] == config.EMBEDDING_MODEL


def test_local_embeddings_default_to_a_strict_relevance_gate():
    # BGE normalized cosine scores on-topic ~0.5+ and off-topic ~0.1; the gate
    # must be high enough that an off-topic question is refused (calibrated on
    # a real HKMA report: 0.51-0.63 relevant vs 0.08-0.10 irrelevant).
    assert config.EMBEDDING_PROVIDER == "local"
    assert rag.SCORE_THRESHOLD == 0.35
