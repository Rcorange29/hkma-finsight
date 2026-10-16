"""
eval_retrieval.py — Retrieval accuracy of semantic-only vs hybrid search.

A question scores a hit when a retrieved chunk contains its evidence: the figure
from the ground truth, verbatim as the filing prints it. Reports hit@1,
hit@TOP_K and mean reciprocal rank for both pipelines over two sets:

  - the evaluate.py questions (plain-English, "fiscal year 2024")
  - exact-term questions: quoted values, line-item labels, "FY24"

No LLM calls: only question embeddings (well under a cent) and the local
cross-encoder. Needs the ingested store (python ingest.py).

Usage:
    python eval_retrieval.py
"""

from __future__ import annotations

import re

import rag
from evaluate import EVAL_QUESTIONS
from ingest import PUBLIC_OWNER

# Evidence for ground truths without a figure in them; None = not scored
EVIDENCE_OVERRIDES = {
    "Who is the company's independent registered public accounting firm?": "Deloitte & Touche",
    "What are the company's reportable segments?": None,
    "What are the main risk factors disclosed in the filing?": None,
}

# Figures from the evaluate.py ground truths, asked the way an analyst types them
EXACT_QUESTIONS = [
    ("Which balance sheet line was $57,582 million at June 30, 2024?", "57,582"),
    ("Where does the filing report 245,122?", "245,122"),
    ("FY24 additions to property and equipment", "44,477"),
    ("Diluted EPS FY24", "11.80"),
    ("What was \"Net cash from operations\" in fiscal 2024?", "118,548"),
    ("Common stock cash dividends paid FY2024", "21,771"),
    ("Intelligent Cloud revenue FY24", "105,362"),
    ("Stock-based compensation expense FY24", "10,734"),
    ("Goodwill at June 30, 2024", "119,220"),
    ("Short-term unearned revenue FY24", "57,582"),
]

_FIGURE = re.compile(r"\d{1,3}(?:,\d{3})+|\d+\.\d+")


def _evidence(item: dict) -> str | None:
    if item["question"] in EVIDENCE_OVERRIDES:
        return EVIDENCE_OVERRIDES[item["question"]]
    match = _FIGURE.search(item["ground_truth"])
    return match.group(0) if match else None


def _rank(docs: list, evidence: str) -> int | None:
    for i, doc in enumerate(docs, 1):
        if evidence in doc.page_content:
            return i
    return None


def _score(cases: list[tuple[str, str]], vectorstore, keyword: bool) -> list[int | None]:
    return [
        _rank(rag.retrieve_and_rerank(q, vectorstore, PUBLIC_OWNER, keyword=keyword), ev)
        for q, ev in cases
    ]


def _summary(ranks: list[int | None]) -> str:
    n = len(ranks)
    hit1 = sum(1 for r in ranks if r == 1)
    hitk = sum(1 for r in ranks if r is not None)
    mrr = sum(1 / r for r in ranks if r) / n
    return f"hit@1 {hit1:>2}/{n}   hit@{rag.TOP_K} {hitk:>2}/{n}   MRR {mrr:.3f}"


def main() -> None:
    vectorstore = rag._load_vectorstore()
    sets = {
        "evaluate.py questions": [
            (item["question"], ev) for item in EVAL_QUESTIONS if (ev := _evidence(item))
        ],
        "exact-term questions": EXACT_QUESTIONS,
    }
    for name, cases in sets.items():
        semantic = _score(cases, vectorstore, keyword=False)
        hybrid = _score(cases, vectorstore, keyword=True)
        print(f"\n{name} ({len(cases)})")
        print(f"  semantic only  {_summary(semantic)}")
        print(f"  hybrid         {_summary(hybrid)}")
        for (q, ev), s, h in zip(cases, semantic, hybrid, strict=True):
            if s != h:
                print(f"    {s or '-'!s:>2} -> {h or '-'!s:<2}  {q[:70]}  [{ev}]")


if __name__ == "__main__":
    main()
