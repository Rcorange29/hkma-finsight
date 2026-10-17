"""
llm.py — One place that builds the chat model for every language-model call.

The engine talks to an OpenAI-compatible API, so the same ChatOpenAI client
reaches OpenAI or Volcengine Ark (Doubao); only the base URL, API key and model
name differ, and those all come from config (see config.py).

The client class is looked up on the ``langchain_openai`` module at call time
rather than imported as a name, so tests can replace
``langchain_openai.ChatOpenAI`` with a fake and have every caller use it.
"""

from __future__ import annotations

import langchain_openai

from financial_rag import config


class LLMConfigError(RuntimeError):
    """Raised when no API key is configured for a language-model call."""


def require_api_key() -> str:
    """The configured API key, or a clear error telling the user how to set it."""
    if not config.LLM_API_KEY or not config.LLM_API_KEY.strip():
        provider = "ARK_API_KEY" if config.LLM_PROVIDER == "doubao" else "OPENAI_API_KEY"
        raise LLMConfigError(
            "No language-model API key set. Put "
            f"{provider}=<your key> in your .env (or export it) for "
            f"the {config.LLM_PROVIDER!r} provider, then retry. "
            "Ingestion and retrieval run locally without it; only answering needs the model."
        )
    return config.LLM_API_KEY


def get_chat_llm(**overrides):
    """
    A ChatOpenAI client pointed at the configured provider (Doubao/Ark by
    default). Any keyword overrides (timeout, max_retries, temperature, …) win
    over the defaults and are passed straight through.
    """
    kwargs = {
        "model": config.CHAT_MODEL,
        "temperature": 0,
        "timeout": 30,
        "max_retries": 1,
        "base_url": config.LLM_BASE_URL,
        "api_key": require_api_key(),
    }
    kwargs.update(overrides)
    # Look the class up at call time so monkeypatching langchain_openai.ChatOpenAI
    # in tests replaces the client every caller gets.
    return langchain_openai.ChatOpenAI(**kwargs)
