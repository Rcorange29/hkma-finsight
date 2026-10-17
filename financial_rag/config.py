"""
config.py — The models the pipeline uses, named in one place.

Every language-model call (planning, follow-up rewriting, figure extraction,
LLM reranking, answering, the answer retry) uses CHAT_MODEL. Set the
CHAT_MODEL environment variable to try another model, and measure the change
with eval_answers.py rather than guessing:

    CHAT_MODEL=doubao-seed-2-1-pro-260628 python -m financial_rag.eval_answers

LLM providers
-------------
The chat model is reached through an OpenAI-compatible API, so the same code
works against OpenAI or Volcengine Ark (Doubao). Select it with LLM_PROVIDER:

  LLM_PROVIDER=doubao   (default)
      base URL : ARK_BASE_URL, else the Ark endpoint
      api key  : ARK_API_KEY
      model    : CHAT_MODEL, else doubao-seed-2-1-pro-260628
  LLM_PROVIDER=openai
      base URL : OPENAI_BASE_URL (unset → OpenAI's default)
      api key  : OPENAI_API_KEY
      model    : CHAT_MODEL, else gpt-4o-mini

Embeddings
----------
Embeddings default to a **local** sentence-transformer model (free, offline, no
paid API calls). Set EMBEDDING_PROVIDER=openai to use OpenAI's hosted model
instead. The vector store holds vectors from whichever model was used, so
changing it means re-ingesting (python start.py --reingest).
"""

import os

from dotenv import load_dotenv

# Resolve the local file before freezing settings, regardless of import order.
load_dotenv()

# ---------------------------------------------------------------------------
# Chat / language model (OpenAI-compatible: OpenAI or Volcengine Ark/Doubao)
# ---------------------------------------------------------------------------

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "doubao").strip().lower()

# Volcengine Ark's OpenAI-compatible endpoint.
ARK_BASE_URL_DEFAULT = "https://ark.cn-beijing.volces.com/api/v3"
# A current Doubao model id on Ark (overridable; an endpoint id "ep-..." works too).
DOUBAO_CHAT_MODEL_DEFAULT = "doubao-seed-2-1-pro-260628"
OPENAI_CHAT_MODEL_DEFAULT = "gpt-4o-mini"

if LLM_PROVIDER == "openai":
    CHAT_MODEL = os.getenv("CHAT_MODEL", OPENAI_CHAT_MODEL_DEFAULT)
    LLM_BASE_URL = os.getenv("OPENAI_BASE_URL") or None
    LLM_API_KEY = os.getenv("OPENAI_API_KEY")
else:  # doubao (Ark), or any OpenAI-compatible provider
    CHAT_MODEL = os.getenv("CHAT_MODEL", DOUBAO_CHAT_MODEL_DEFAULT)
    LLM_BASE_URL = (
        os.getenv("ARK_BASE_URL")
        or ARK_BASE_URL_DEFAULT
    )
    LLM_API_KEY = os.getenv("ARK_API_KEY")

# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

# "local" runs a free sentence-transformer on this machine; "openai" uses the
# hosted model (paid, needs OPENAI_API_KEY).
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local").strip().lower()

# Used when EMBEDDING_PROVIDER=openai. Not overridable: the vector store holds
# vectors from this model, so changing it means re-ingesting.
EMBEDDING_MODEL = "text-embedding-3-small"

# Used when EMBEDDING_PROVIDER=local. English financial text; 384 dims, ~130 MB.
# Override with LOCAL_EMBEDDING_MODEL (changing it means re-ingesting).
LOCAL_EMBEDDING_MODEL = os.getenv(
    "LOCAL_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"
)

# The Dockerfile bakes this model into the image; keep its RUN line in sync.
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
