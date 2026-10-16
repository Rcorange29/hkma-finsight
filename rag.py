"""
rag.py — Core RAG logic: hybrid (semantic + BM25) retrieval and cross-encoder reranking.

Pipeline:
  1. Semantic: embed the question, take the top RETRIEVE_K chunks from ChromaDB,
     and drop any whose relevance score is below SCORE_THRESHOLD.
  2. Keyword: BM25 over the same chunks (the caller's own filings plus the public
     samples) gives its top KEYWORD_K. It catches exact fiscal years, line-item
     labels and values that embeddings blur together.
  3. Fuse the two rankings (weighted reciprocal rank fusion; the keyword weight
     depends on the question, see hybrid.profile_query) and keep RERANK_CANDIDATES.
  4. Rerank those with cross-encoder/ms-marco-MiniLM-L-6-v2. When the question
     quotes exact values ("57,582"), chunks containing all of them go first.
  5. Send the top TOP_K (5) to GPT-4o-mini. Every returned chunk records how it
     was found under metadata["retrieval"], and the API passes that on.

HYBRID_SEARCH=0 turns step 2 off, leaving the semantic-only pipeline.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import hybrid
from ingest import PUBLIC_OWNER

load_dotenv()

CHROMA_DIR = Path("chroma_db")
COLLECTION_NAME = "financial_reports"
RETRIEVE_K = 20           # semantic candidates pulled from ChromaDB before filtering
SCORE_THRESHOLD = 0.10    # minimum relevance score (Chroma L2-normalised scores top ~0.16)
KEYWORD_K = 20            # BM25 candidates
RERANK_CANDIDATES = 30    # fused candidates the cross-encoder scores
TOP_K = 5                 # final chunks sent to the LLM after reranking
HYBRID_SEARCH = os.getenv("HYBRID_SEARCH", "1") != "0"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

SYSTEM_PROMPT = """You are a financial analyst assistant. Answer questions using ONLY \
the retrieved 10-K excerpts provided below.

STRICT RULES — follow these exactly:
1. Only report figures that appear VERBATIM in the excerpts. Do NOT compute, infer, \
   derive, or combine values from multiple rows unless the document explicitly states \
   the combined figure.
2. If a figure is not explicitly stated in the excerpts, respond with \
   "not found in context" for that specific item. Never guess or estimate.
3. Do not add commentary, context, or assumptions beyond what is directly stated.
4. After each claim, cite the source document and page in parentheses, \
   e.g. (microsoft-sec-10k.pdf, p.66).
5. The excerpts may come from more than one filing, each named in its Source line. \
   If more than one filing answers the question, answer for each one separately \
   under its company or file name; never merge, combine or compare figures across \
   filings unless the question asks you to.

Context excerpts:
{context}
"""

# ---------------------------------------------------------------------------
# Lazy cross-encoder singleton (loaded once per process)
# ---------------------------------------------------------------------------
_reranker = None


def _get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        print(f"Loading reranker model {RERANKER_MODEL} …")
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class RAGResponse:
    answer: str
    sources: list[dict] = field(default_factory=list)
    retrieved_docs: list = field(default_factory=list)   # used by evaluate.py


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def doc_to_context(doc) -> str:
    """Render one retrieved chunk the way the LLM sees it: provenance + content.

    evaluate.py uses this too, so RAGAs judges the answer against exactly the
    context the model was given. Passing bare page_content instead makes the page
    citation the system prompt mandates look like an unsupported claim, which
    caps faithfulness at roughly 0.5 for every correctly-cited answer.
    """
    meta = doc.metadata
    source = meta.get("source_file", meta.get("source", "unknown"))
    if meta.get("company"):
        source = f"{source} ({meta['company']})"
    page = meta.get("page", "?")
    return f"Source: {source}, Page: {page}\n{doc.page_content}"


def _format_docs(docs) -> str:
    parts = [f"[{i}] {doc_to_context(doc)}" for i, doc in enumerate(docs, 1)]
    return "\n\n---\n\n".join(parts)


def _answer_with_docs(question: str, docs: list) -> str:
    """Format context from pre-fetched docs and call GPT-4o-mini."""
    context = _format_docs(docs)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{question}"),
        ]
    )
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = (
        {"context": RunnableLambda(lambda _: context), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain.invoke(question)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def _load_vectorstore() -> Chroma:
    if not CHROMA_DIR.exists():
        raise RuntimeError(
            f"ChromaDB not found at {CHROMA_DIR.resolve()}. "
            "Run `python ingest.py` first."
        )
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )


def load_retriever():
    """Backward-compatible retriever for evaluate.py (bypasses reranker)."""
    return _load_vectorstore().as_retriever(search_kwargs={"k": TOP_K})


def owner_filter(owner: str | None) -> dict | None:
    """Restrict retrieval to the owner's own filings plus the public samples.

    None searches every filing; only the CLI and evaluate.py do that.
    """
    if owner is None:
        return None
    return {"owner": {"$in": [owner, PUBLIC_OWNER]}}


# Keyword corpus: chunk id -> (chunk, BM25 term counts), per Chroma collection.
# Filled lazily from the store and kept in step with it, so a query only
# re-reads chunks added since the last one.
_corpora: dict[str, dict[str, tuple[Document, hybrid.Indexed]]] = {}
_corpora_lock = threading.Lock()
_FETCH_BATCH = 500


def _chunk_id(doc) -> str:
    """Chroma's id for the chunk, or a stand-in built from its provenance."""
    if doc.id:
        return doc.id
    meta = doc.metadata
    return f"{meta.get('owner')}/{meta.get('source_file')}/{meta.get('page')}/{hash(doc.page_content)}"


def _visible(meta: dict, owner: str | None) -> bool:
    return owner is None or meta.get("owner") in (owner, PUBLIC_OWNER)


@dataclass(frozen=True)
class Scope:
    """
    The chunks one search may see: the owner's filings plus the public samples,
    narrowed to a single filing when `filing` is set. A filing is (its owner,
    its source_file), and must itself be visible to the owner.
    """
    owner: str | None
    filing: tuple[str, str] | None = None

    def __post_init__(self):
        if self.filing and not _visible({"owner": self.filing[0]}, self.owner):
            raise ValueError("That filing belongs to another user.")

    def where(self) -> dict | None:
        if self.filing is None:
            return owner_filter(self.owner)
        file_owner, name = self.filing
        return {"$and": [{"owner": file_owner}, {"source_file": name}]}

    def contains(self, meta: dict) -> bool:
        if self.filing is not None:
            return (meta.get("owner"), meta.get("source_file")) == self.filing
        return _visible(meta, self.owner)


def _keyword_corpus(vectorstore: Chroma, scope: Scope) -> list[tuple[Document, hybrid.Indexed]]:
    """
    Every chunk in scope, with its BM25 term counts.

    Scope is applied by Chroma with the same filter as the semantic search, so
    the keyword path can't surface another user's chunks either.
    """
    ids = vectorstore.get(where=scope.where(), include=[])["ids"]
    with _corpora_lock:
        cache = _corpora.setdefault(vectorstore._collection.id, {})
        missing = [i for i in ids if i not in cache]
        for start in range(0, len(missing), _FETCH_BATCH):
            got = vectorstore.get(
                ids=missing[start:start + _FETCH_BATCH], include=["documents", "metadatas"]
            )
            for cid, text, meta in zip(got["ids"], got["documents"], got["metadatas"], strict=True):
                text = text or ""
                cache[cid] = (
                    Document(id=cid, page_content=text, metadata=meta or {}),
                    hybrid.index_text(text),
                )
        # Forget chunks in this scope that the store no longer has (replaced filings)
        present = set(ids)
        for cid in [c for c, (doc, _) in cache.items()
                    if c not in present and scope.contains(doc.metadata)]:
            del cache[cid]
        return [cache[i] for i in ids if i in cache]


def _keyword_search(profile: hybrid.QueryProfile, vectorstore: Chroma, scope: Scope):
    """Top KEYWORD_K chunks by BM25 as (chunk, score, matched terms); no-match chunks dropped."""
    corpus = _keyword_corpus(vectorstore, scope)
    scores = hybrid.bm25_scores(profile.terms, [indexed for _, indexed in corpus])
    ranked = sorted(
        ((score, doc, indexed) for score, (doc, indexed) in zip(scores, corpus, strict=True)
         if score > 0),
        key=lambda x: x[0],
        reverse=True,
    )[:KEYWORD_K]
    return [
        (doc, score, [t for t in profile.terms if t in indexed.tf])
        for score, doc, indexed in ranked
    ]


def retrieve_and_rerank(
    question: str,
    vectorstore: Chroma,
    owner: str | None = None,
    keyword: bool | None = None,
    filing: tuple[str, str] | None = None,
) -> list:
    """
    Hybrid retrieval for the chunks the owner may see; see the module docstring.

    Returns up to TOP_K chunks, best first. Each is a copy of the stored chunk
    whose metadata["retrieval"] records how it was found: chunk_id, matched_by
    ("semantic" and/or "keyword"), each search's rank and score, the query terms
    it matched, and the cross-encoder score. `keyword` overrides HYBRID_SEARCH.
    `filing` = (owner, source_file) searches that one filing only.
    """
    keyword = HYBRID_SEARCH if keyword is None else keyword
    profile = hybrid.profile_query(question)
    scope = Scope(owner, filing)

    # Semantic retrieval with relevance scores (0 = dissimilar, 1 = identical)
    scored = vectorstore.similarity_search_with_relevance_scores(
        question, k=RETRIEVE_K, filter=scope.where()
    )
    semantic = [(doc, score) for doc, score in scored if score >= SCORE_THRESHOLD]
    lexical = _keyword_search(profile, vectorstore, scope) if keyword else []

    docs: dict[str, Document] = {}
    trace: dict[str, dict] = {}

    def note(doc, via: str, **fields) -> str:
        cid = _chunk_id(doc)
        docs.setdefault(cid, doc)
        entry = trace.setdefault(cid, {"chunk_id": cid, "matched_by": []})
        entry["matched_by"].append(via)
        entry.update(fields)
        return cid

    semantic_ids = [
        note(doc, "semantic", semantic_rank=rank, semantic_score=round(score, 4))
        for rank, (doc, score) in enumerate(semantic, 1)
    ]
    keyword_ids = [
        note(doc, "keyword", keyword_rank=rank, keyword_score=round(score, 4),
             matched_terms=terms)
        for rank, (doc, score, terms) in enumerate(lexical, 1)
    ]
    if not docs:
        return []

    fused = hybrid.rrf(
        [(semantic_ids, hybrid.SEMANTIC_WEIGHT), (keyword_ids, profile.keyword_weight)]
    )[:RERANK_CANDIDATES]

    # Cross-encoder reranking of the fused candidates; a chunk containing every
    # value the question quotes outranks one that doesn't, whatever the model says
    reranker = _get_reranker()
    candidates = [cid for cid, _ in fused]
    ce_scores = reranker.predict([(question, docs[cid].page_content) for cid in candidates])
    exact = {
        cid: keyword and hybrid.has_amounts(profile, hybrid.index_text(docs[cid].page_content))
        for cid in candidates
    }
    ranked = sorted(
        zip(ce_scores, candidates, strict=True),
        key=lambda x: (exact[x[1]], x[0]),
        reverse=True,
    )

    out = []
    for score, cid in ranked[:TOP_K]:
        doc = docs[cid]
        info = {**trace[cid], "rerank_score": round(float(score), 4)}
        if exact[cid]:
            info["exact_value_match"] = True
        out.append(Document(id=doc.id, page_content=doc.page_content,
                            metadata={**doc.metadata, "retrieval": info}))
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def query(
    question: str, owner: str | None = None, filing: tuple[str, str] | None = None
) -> RAGResponse:
    """Answer from the owner's filings and the samples, or from one `filing` only."""
    vectorstore = _load_vectorstore()
    top_docs = retrieve_and_rerank(question, vectorstore, owner, filing=filing)

    if not top_docs:
        return RAGResponse(
            answer=(
                "No relevant context found above the confidence threshold. "
                "The documents may not contain information about this question."
            ),
            sources=[],
            retrieved_docs=[],
        )

    answer = _answer_with_docs(question, top_docs)
    return RAGResponse(
        answer=answer, sources=build_sources(question, top_docs), retrieved_docs=top_docs
    )


def build_sources(question: str, docs: list) -> list[dict]:
    """
    One citation per (file, page), in rank order, with how its chunk was found.

    The excerpt is the stretch of the chunk that matches the question, so a
    figure that was found by keyword is visible in the citation, not cut off.
    """
    terms = hybrid.profile_query(question).terms
    sources = []
    seen: set = set()
    for doc in docs:
        meta = doc.metadata
        source = meta.get("source_file", meta.get("source", "unknown"))
        page = meta.get("page", "?")
        key = (source, page)
        if key not in seen:
            seen.add(key)
            sources.append(
                {
                    "source_file": source,
                    "page": page,
                    "excerpt": hybrid.focus(doc.page_content, terms)[:300],
                    "retrieval": meta.get("retrieval"),
                }
            )
    return sources

if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What was the total revenue?"
    result = query(q)
    print(f"\nAnswer:\n{result.answer}\n")
    print("Sources:")
    for s in result.sources:
        found = s["retrieval"] or {}
        how = " + ".join(found.get("matched_by", []))
        print(f"  - {s['source_file']} p.{s['page']} [{how}]: {s['excerpt'][:120]} …")
