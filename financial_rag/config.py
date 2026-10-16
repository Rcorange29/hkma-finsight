"""
config.py — The models the pipeline uses, named in one place.

Every language-model call (planning, follow-up rewriting, figure extraction,
LLM reranking, answering, the answer retry) uses CHAT_MODEL. Set the
CHAT_MODEL environment variable to try another model, and measure the change
with eval_answers.py rather than guessing:

    CHAT_MODEL=gpt-5-mini python -m financial_rag.eval_answers
"""

import os

CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

# Not overridable: the vector store holds vectors from this model, so changing
# it means re-ingesting everything (python start.py --reingest).
EMBEDDING_MODEL = "text-embedding-3-small"

# The Dockerfile bakes this model into the image; keep its RUN line in sync.
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
