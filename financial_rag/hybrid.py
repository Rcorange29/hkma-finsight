"""
hybrid.py — Keyword (BM25) scoring and rank fusion for rag.py's hybrid retrieval.

Semantic search finds passages that mean the same thing as the question, but it
is unreliable on exact tokens: fiscal years, line-item labels ("Additions to
property and equipment") and precise values ("57,582"). BM25 matches those
literally. rag.py runs both searches, fuses the two rankings with weighted
reciprocal rank fusion, and reranks the union with the cross-encoder.

The search is query-aware: profile_query() reads the question's exact anchors.
"FY24" is searched as "fiscal 2024", the way filings print it; the keyword
ranking gets more fusion weight when the question names a year or quotes a
value, and less when it's purely conceptual; and chunks containing every value
the question quotes are ranked first after reranking (has_amounts).

Everything here is pure (no vector store or model), so it is all unit-tested.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

# BM25 (Okapi) parameters: the usual defaults
K1 = 1.5
B = 0.75

# Reciprocal rank fusion constant: score = weight / (RRF_K + rank)
RRF_K = 60

# Fusion weight of the semantic ranking; the keyword weight depends on the query
SEMANTIC_WEIGHT = 1.0
KEYWORD_WEIGHT_AMOUNT = 2.0    # question quotes a value: "57,582", "$11.80", "15.7%"
KEYWORD_WEIGHT_YEAR = 1.0      # names a fiscal year: "fiscal 2024", "FY24"
KEYWORD_WEIGHT_DEFAULT = 0.75  # conceptual: "What are the main risk factors?"

STOPWORDS = frozenset(
    """
    a about above after again all also am an and any are as at be been before being below
    between both but by can could did do does doing down during each few for from further
    had has have having he her here hers him his how i if in into is it its itself just me
    more most much many my no nor not now of off on once only or other our ours out over
    own same she should so some such than that the their theirs them then there these they
    this those through to too under until up very was we were what when where which while
    who whom why will with would you your
    """.split()
)
# Conversational/request verbs and politeness words. They are never a
# discriminating financial term but occur frequently in the corpus, so a query
# that contains only an out-of-corpus subject plus one of these ("give me a
# recipe for chocolate cake") would otherwise get BM25 hits on the filler and
# slip past the no-evidence refusal gate. Financial homographs such as "state",
# "use", "list", "share", "return", "rate" or "report" are deliberately kept
# out of this set.
QUESTION_FILLER = frozenset(
    """
    please thanks thank give gives gave giving get gets got getting let lets allow
    tell tells told ask asks asked show shows shown showing find finds found
    make makes made take takes took want wants wanted need needs needed know knew
    knows mention mentions describe describes explain explains provide provides
    outline outlines summarise summarize detail details clarify may might must shall
    """.split()
)
STOPWORDS = STOPWORDS | QUESTION_FILLER

_FISCAL_YEAR = re.compile(r"\bfy\s?'?(\d{4}|\d{2})\b")
# Comma-grouped numbers first, so "245,122" stays one token rather than "245" + "122"
_TOKEN = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|[a-z]+(?:'[a-z]+)?")
_YEAR = re.compile(r"(?:19|20)\d{2}")


def _fiscal_year(match: re.Match) -> str:
    year = match.group(1)
    return f"fiscal {'20' + year if len(year) == 2 else year}"


def _stem(word: str) -> str:
    """Fold plurals so "revenues" matches "revenue" and "liabilities" "liability"."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith(("sses", "xes", "ches", "shes")):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def tokenize(text: str) -> list[str]:
    """
    Lowercase word and number tokens for BM25.

    Numbers drop their grouping commas, so "245,122", "$245,122" and "245122"
    are the same token, and "(1,646)" matches "1,646". "FY24" and "FY2024"
    become "fiscal 2024", the way filings write it. Stopwords and one-letter
    words are dropped, and plurals are folded.
    """
    text = _FISCAL_YEAR.sub(_fiscal_year, text.lower())
    tokens = []
    for tok in _TOKEN.findall(text):
        if tok[0].isdigit():
            tokens.append(tok.replace(",", ""))
            continue
        tok = tok.split("'")[0]   # company's -> company
        if len(tok) > 1 and tok not in STOPWORDS:
            tokens.append(_stem(tok))
    return tokens


# ---------------------------------------------------------------------------
# Query profile
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QueryProfile:
    terms: tuple[str, ...]     # distinct query tokens, in order
    years: tuple[str, ...]     # fiscal or calendar years named
    amounts: tuple[str, ...]   # other exact values: amounts, EPS, percentages
    keyword_weight: float      # fusion weight of the keyword ranking


def profile_query(question: str) -> QueryProfile:
    terms = tuple(dict.fromkeys(tokenize(question)))
    numbers = [t for t in terms if t[0].isdigit()]
    years = tuple(t for t in numbers if _YEAR.fullmatch(t))
    # One- and two-digit numbers ("top 3 risks", "June 30") don't pin a value
    amounts = tuple(
        t for t in numbers if not _YEAR.fullmatch(t) and ("." in t or len(t) >= 3)
    )
    if amounts:
        weight = KEYWORD_WEIGHT_AMOUNT
    elif years:
        weight = KEYWORD_WEIGHT_YEAR
    else:
        weight = KEYWORD_WEIGHT_DEFAULT
    return QueryProfile(terms, years, amounts, weight)


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Indexed:
    """A chunk's term frequencies, computed once and cached by rag.py."""
    tf: Counter
    length: int


def index_text(text: str) -> Indexed:
    tokens = tokenize(text)
    return Indexed(Counter(tokens), len(tokens))


def bm25_scores(terms: tuple[str, ...] | list[str], docs: list[Indexed]) -> list[float]:
    """
    Okapi BM25 score of each doc for the query terms.

    Document frequencies come from `docs` itself: the chunks the caller may see,
    so one user's filings never shape another user's term weights.
    """
    n = len(docs)
    scores = [0.0] * n
    if not n or not terms:
        return scores
    avgdl = sum(d.length for d in docs) / n or 1.0
    for term in terms:
        df = sum(1 for d in docs if term in d.tf)
        if not df:
            continue
        # Lucene's non-negative idf: a term in every chunk still counts a little
        idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
        for i, d in enumerate(docs):
            f = d.tf.get(term)
            if f:
                scores[i] += idf * f * (K1 + 1) / (f + K1 * (1 - B + B * d.length / avgdl))
    return scores


# ---------------------------------------------------------------------------
# Fusion and reranker input
# ---------------------------------------------------------------------------

def rrf(rankings: list[tuple[list[str], float]]) -> list[tuple[str, float]]:
    """
    Weighted reciprocal rank fusion of (ids best-first, weight) rankings.

    Ranks, not raw scores, are combined: cosine similarities and BM25 scores
    aren't on comparable scales. Returns (id, fused score), best first; ties
    keep the order ids were first seen in.
    """
    fused: dict[str, float] = {}
    for ids, weight in rankings:
        for rank, doc_id in enumerate(ids, 1):
            fused[doc_id] = fused.get(doc_id, 0.0) + weight / (RRF_K + rank)
    return sorted(fused.items(), key=lambda kv: kv[1], reverse=True)


def has_amounts(profile: QueryProfile, indexed: Indexed) -> bool:
    """Whether the chunk contains every exact value the question quotes."""
    return bool(profile.amounts) and all(t in indexed.tf for t in profile.amounts)


def focus(text: str, terms: tuple[str, ...] | list[str], max_words: int = 50) -> str:
    """
    The max_words-long window of `text` with the most query-term hits.

    Used for citation excerpts, so a figure found deep in a chunk is shown
    rather than cut off. (Not for the reranker: feeding it the window instead
    of the whole chunk lost the statement heading and made it worse.) Short
    texts, and texts with no better window than the start, come back as is.
    """
    words = text.split()
    if len(words) <= max_words or not terms:
        return text
    wanted = set(terms)
    hits = [sum(1 for t in tokenize(w) if t in wanted) for w in words]
    best_start, best = 0, sum(hits[:max_words])
    window = best
    for start in range(1, len(words) - max_words + 1):
        window += hits[start + max_words - 1] - hits[start - 1]
        if window > best:
            best_start, best = start, window
    if best_start == 0:
        return text
    return " ".join(words[best_start:best_start + max_words])
