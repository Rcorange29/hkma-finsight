"""
eval_answers.py — End-to-end scores for final answers, across every route.

eval_retrieval.py measures search; evaluate.py has an LLM judge text answers.
This runs whole questions through router.answer, exactly as the API does
(follow-up rewriting, planning, filing choice, calculation or retrieval, LLM
reranking, the answer check), and scores the final answers against figures
checked by hand in the two sample filings, Microsoft's FY2024 10-K and Apple's
FY2025 10-K. Questions that name no company are answered with both loaded:

  - correct:  every expected figure is in the answer, and nothing that must not
              be (a refusal must not quote another company's numbers)
  - route:    the engine that answered is the one expected (calculation / text)
  - verified: no "Not verified" warning from the answer check
  - cited:    the answer cites a page (refusals excepted)

Also reports latency and gpt-4o-mini tokens per question, with a cost estimate
at list prices. A run costs about $0.05. Needs both samples in data/
(microsoft-sec-10k.pdf, apple-sec-10k.pdf), ingested (python ingest.py) with
their figures (python financials.py).
Results go to answer_eval_results.json.

Usage:
    python eval_answers.py
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.callbacks import get_usage_metadata_callback

import router

OWNER = "answer-eval"       # sees the public samples only
RESULTS_PATH = Path("answer_eval_results.json")
# gpt-4o-mini list prices, USD per million tokens
PRICE_INPUT, PRICE_CACHED_INPUT, PRICE_OUTPUT = 0.15, 0.075, 0.60

# Phrases that mark an answer as a refusal ("not found in context", ...)
REFUSAL_MARKERS = (
    "not found", "couldn't", "could not", "not available", "isn't available",
    "no filing", "does not contain", "doesn't contain", "not in the", "no information",
    "not provided", "unable",
)

NET_INCOME_2024 = [
    ("user", "What was Microsoft's net income in fiscal 2024?"),
    ("assistant", "**Net income** · microsoft-sec-10k.pdf\n- **FY2024: $88,136 million** "
                  "(microsoft-sec-10k.pdf, p.67)"),
]
REVENUE_2024 = [
    ("user", "What was total revenue in fiscal 2024?"),
    ("assistant", "**Total revenue** · microsoft-sec-10k.pdf\n- **FY2024: $245,122 million** "
                  "(microsoft-sec-10k.pdf, p.67)"),
]
LINKEDIN_2024 = [
    ("user", "What was LinkedIn revenue in fiscal 2024?"),
    ("assistant", "LinkedIn revenue was $16,372 million in fiscal 2024 (microsoft-sec-10k.pdf, p.114)."),
]
APPLE_NET_INCOME_2025 = [
    ("user", "What was Apple's net income in fiscal 2025?"),
    ("assistant", "**Net income** · apple-sec-10k.pdf\n- **FY2025: $112,010 million** "
                  "(apple-sec-10k.pdf, p.32)"),
]
EMPLOYEES = [
    ("user", "How many people did Microsoft employ as of June 30, 2024?"),
    ("assistant", "As of June 30, 2024, Microsoft employed approximately 228,000 people on a "
                  "full-time basis (microsoft-sec-10k.pdf, p.12)."),
]


@dataclass
class Case:
    kind: str
    question: str
    # Each entry must be in the answer; a tuple means any one of its strings
    expect: list[str | tuple[str, ...]] = field(default_factory=list)
    route: str | None = None
    history: list[tuple[str, str]] = field(default_factory=list)
    absent: list[str] = field(default_factory=list)   # must not be in the answer
    refusal: bool = False                             # the right answer is "not available"


CASES = [
    # Calculations: expected values as the analysis engine renders them
    Case("calculation", "What was the operating margin in fiscal 2024?", ["44.6%"], "calculation"),
    Case("calculation", "What was free cash flow in fiscal 2024?", ["$74,071 million"], "calculation"),
    Case("calculation", "What is the current ratio?", ["1.27x"], "calculation"),
    Case("calculation", "What was return on equity in fiscal 2024?", ["32.8%"], "calculation"),
    Case("calculation", "What was the effective tax rate in fiscal 2024?", ["18.2%"], "calculation"),
    Case("calculation", "How much did revenue grow per year from 2022 to 2024?", ["11.2%"],
         "calculation"),
    Case("calculation", "What was net income in fiscal 2023?", ["$72,361 million"], "calculation"),
    Case("calculation", "What was diluted earnings per share in fiscal 2024?", ["$11.80"],
         "calculation"),
    Case("calculation",
         "What was the free cash flow margin in 2024 and how did it change from 2023?",
         ["30.2%", "2.2 percentage points"], "calculation"),
    Case("calculation", "What was the gross margin percentage in fiscal 2024?", ["69.8%"],
         "calculation"),
    # Text: figures and facts outside the primary statements
    Case("text", "How many people did the company employ as of June 30, 2024?", ["228,000"], "text"),
    Case("text", "How much was left in the share repurchase program at June 30, 2024?",
         ["$10.3 billion"], "text"),
    Case("text", "How much did Microsoft pay for Activision Blizzard?",
         [("$75.4 billion", "75,408")], "text"),
    Case("text", "How much goodwill was recorded for the Activision Blizzard acquisition?",
         [("50,969", "$51.0 billion")], "text"),
    Case("text", "What was short-term unearned revenue at June 30, 2024?", ["57,582"], "text"),
    # Parts of the company the filing breaks out: calculated from the verified breakdowns
    Case("breakdown", "What was Intelligent Cloud revenue in fiscal 2024?", ["105,362"],
         "calculation"),
    Case("breakdown", "What was LinkedIn revenue in fiscal 2024?", ["16,372"], "calculation"),
    Case("breakdown", "How much revenue came from the United States in fiscal 2024?",
         ["124,704"], "calculation"),
    Case("breakdown", "What was the Intelligent Cloud operating margin in fiscal 2024?",
         ["47.1%"], "calculation"),
    Case("breakdown", "How did Gaming revenue change from fiscal 2023 to 2024?", ["39.0%"],
         "calculation"),
    # Not a part the filing breaks out: from the text
    Case("text", "What was Microsoft Cloud revenue in fiscal 2024?", ["$137.4 billion"], "text"),
    Case("text", "Who is Microsoft's independent auditor?", ["Deloitte"], "text"),
    Case("text", "How fast did Azure and other cloud services revenue grow in fiscal 2024?",
         ["30%"], "text"),
    Case("text", "What are the main risks the company describes related to AI?", [], "text"),
    # Declared, not paid: the engine only has dividends paid ($21,771 million)
    Case("text", "What were total dividends declared in fiscal 2024?",
         [("$22.3 billion", "22,293")], absent=["21,771"]),
    # Follow-ups, read in the context of the conversation
    Case("follow-up", "And in 2023?", ["$72,361 million"], "calculation", NET_INCOME_2024),
    Case("follow-up", "What about the operating margin?", ["44.6%"], "calculation", REVENUE_2024),
    Case("follow-up", "And Dynamics?", ["6,481"], "calculation", LINKEDIN_2024),
    Case("follow-up", "How many of them were in the U.S.?", ["126,000"], "text", EMPLOYEES),
    # Apple: a second company, its fiscal year ending in September, shares in thousands
    Case("apple", "What was Apple's operating margin in fiscal 2025?", ["32.0%"], "calculation"),
    Case("apple", "What was Apple's free cash flow in fiscal 2025?", ["$98,767 million"],
         "calculation"),
    Case("apple", "What was Apple's gross margin percentage in fiscal 2025?", ["46.9%"],
         "calculation"),
    Case("apple", "What was Apple's effective tax rate in fiscal 2025?", ["15.6%"],
         "calculation"),
    Case("apple", "What is Apple's current ratio?", ["0.89x"], "calculation"),
    Case("apple", "What was Apple's diluted earnings per share in fiscal 2025?", ["$7.46"],
         "calculation"),
    Case("apple", "How many diluted shares did Apple have in fiscal 2025?",
         ["15,004.7 million shares"], "calculation"),
    Case("apple", "How did Apple's revenue change from fiscal 2024 to 2025?", ["6.4%"],
         "calculation"),
    Case("apple", "What was Apple's revenue in fiscal 2024?", ["$391,035 million"],
         "calculation", absent=["245,122"]),
    Case("apple", "What was Apple's iPhone revenue in fiscal 2025?", ["209,586"], "calculation"),
    Case("apple", "How did Apple's Services revenue change from fiscal 2024 to 2025?",
         ["13.5%"], "calculation"),
    Case("apple", "How much of Apple's revenue came from the United States in fiscal 2025?",
         ["151,790"], "calculation"),
    Case("apple", "What were Apple's Greater China net sales in fiscal 2025?", ["64,377"]),
    Case("apple", "How many full-time equivalent employees did Apple have at the end of "
         "fiscal 2025?", ["166,000"], "text"),
    Case("apple", "Who is Apple's independent auditor?", ["Ernst & Young"], "text"),
    Case("apple", "And in 2024?", ["$93,736 million"], "calculation", APPLE_NET_INCOME_2025),
    # Nothing to answer from: must say so, not answer with another filing's figures
    Case("refusal", "What was Tesla's revenue in fiscal 2024?", absent=["245,122", "391,035"],
         refusal=True),
    Case("refusal", "What was Microsoft's revenue in fiscal 2030?", absent=["245,122"],
         refusal=True),
]


def _has(answer: str, wanted: str | tuple[str, ...]) -> bool:
    options = wanted if isinstance(wanted, tuple) else (wanted,)
    return any(option in answer for option in options)


def _cost(usage: dict) -> tuple[int, int, float]:
    tokens_in = tokens_out = 0
    dollars = 0.0
    for model in usage.values():
        cached = (model.get("input_token_details") or {}).get("cache_read", 0) or 0
        tokens_in += model.get("input_tokens", 0)
        tokens_out += model.get("output_tokens", 0)
        dollars += ((model.get("input_tokens", 0) - cached) * PRICE_INPUT
                    + cached * PRICE_CACHED_INPUT
                    + model.get("output_tokens", 0) * PRICE_OUTPUT) / 1e6
    return tokens_in, tokens_out, dollars


def run(case: Case) -> dict:
    start = time.perf_counter()
    with get_usage_metadata_callback() as usage:
        result = router.answer(case.question, OWNER, history=list(case.history))
    seconds = time.perf_counter() - start
    text = result.answer
    tokens_in, tokens_out, dollars = _cost(usage.usage_metadata)
    correct = all(_has(text, w) for w in case.expect) and not any(a in text for a in case.absent)
    if case.refusal:
        correct = correct and any(m in text.lower() for m in REFUSAL_MARKERS)
    return {
        "kind": case.kind,
        "question": case.question,
        "asked_as": result.question,
        "answer": text,
        "route": result.route,
        "correct": correct,
        "route_ok": case.route is None or result.route == case.route,
        "verified": "Not verified" not in text,
        "cited": case.refusal or ".pdf, p." in text,
        "seconds": round(seconds, 2),
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "usd": round(dollars, 5),
    }


def _summary(rows: list[dict]) -> str:
    n = len(rows)
    count = lambda key: sum(1 for r in rows if r[key])  # noqa: E731
    return (f"correct {count('correct'):>2}/{n}   route {count('route_ok'):>2}/{n}   "
            f"verified {count('verified'):>2}/{n}   cited {count('cited'):>2}/{n}")


def main() -> None:
    rows = []
    for case in CASES:
        row = run(case)
        rows.append(row)
        mark = "ok  " if row["correct"] and row["route_ok"] else "FAIL"
        print(f"{mark} [{row['route']:<11}] {case.question[:70]}", flush=True)
    print()
    for kind in dict.fromkeys(r["kind"] for r in rows):
        group = [r for r in rows if r["kind"] == kind]
        print(f"{kind:<12} ({len(group):>2})  {_summary(group)}")
    print(f"{'all':<12} ({len(rows):>2})  {_summary(rows)}")
    print(f"\nper question: median {statistics.median(r['seconds'] for r in rows):.1f} s, "
          f"median {statistics.median(r['tokens_in'] for r in rows):,.0f} input tokens, "
          f"mean ${statistics.mean(r['usd'] for r in rows):.4f} "
          f"(total ${sum(r['usd'] for r in rows):.3f}; embeddings not counted)")
    failures = [r for r in rows if not (r["correct"] and r["route_ok"] and r["verified"]
                                        and r["cited"])]
    for r in failures:
        print(f"\n--- {r['question']}  (read as: {r['asked_as']})\n{r['answer'][:600]}")
    RESULTS_PATH.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nResults written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
