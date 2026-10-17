"""
embeddings.py — One place that builds the embedding model for the vector store.

By default this is a **local** sentence-transformer
(``BAAI/bge-small-en-v1.5``): free, no API key, no per-token cost, and it runs
offline after the model is downloaded once. Set EMBEDDING_PROVIDER=openai to use
OpenAI's hosted ``text-embedding-3-small`` instead (paid).

The vector store binds vectors from whichever model produced them, so changing
EMBEDDING_PROVIDER or LOCAL_EMBEDDING_MODEL means rebuilding the store
(``python start.py --reingest``).

The concrete classes are looked up at call time so tests can monkeypatch them
without downloading a model.
"""

from __future__ import annotations

from financial_rag import config


def get_embeddings():
    """A LangChain embeddings object: local sentence-transformer by default."""
    if config.EMBEDDING_PROVIDER == "openai":
        import langchain_openai

        return langchain_openai.OpenAIEmbeddings(model=config.EMBEDDING_MODEL)

    # Local sentence-transformer wrapped with the LangChain embeddings interface.
    from langchain_community.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=config.LOCAL_EMBEDDING_MODEL,
        # bge models work best when the query is embedded with this instruction;
        # it is a no-op for other model families that ignore it.
        encode_kwargs={"normalize_embeddings": True},
    )
