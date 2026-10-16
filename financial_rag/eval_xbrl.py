"""
eval_xbrl.py — Benchmark questions generated from the SEC's XBRL data, for any company.

eval_answers.py scores 47 questions checked by hand in two filings. This scores
as many companies as you name, with no hand-checking: the expected answers are
the company's own XBRL values (EDGAR's companyfacts API) for the fiscal year of
its latest 10-K.

For each ticker it:
  1. fetches the latest 10-K from EDGAR and indexes it into a separate store
     (chroma_db/eval_xbrl/), once; later runs reuse it;
  2. asks for reported line items ("What was NVIDIA Corp's net income in fiscal
     2026?") and derived metrics (operating and net margin, free cash flow,
     current ratio, computed with the engine's formulas from XBRL inputs);
  3. scores each answer: the expected figure, formatted the way the engine
     formats it, must be in the answer, from the calculation route, with a
     page citation.

All companies share the store, so naming the company in the question also
tests choosing the right filing. A question is only generated when XBRL has
every value it needs.

Cost: indexing a 10-K is about $0.01-0.03 (embeddings and figure extraction),
a question about $0.001. Needs SEC_USER_AGENT and OPENAI_API_KEY in .env.

Usage:
    python -m financial_rag.eval_xbrl                    # the default tickers
    python -m financial_rag.eval_xbrl NVDA KO WMT        # any others
    python -m financial_rag.eval_xbrl --refresh AAPL     # re-index first
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from dotenv import load_dotenv

from financial_rag import analysis, edgar, financials, ingest, rag, router, xbrl

load_dotenv()

DEFAULT_TICKERS = ["AAPL", "MSFT", "NVDA", "GOOGL", "KO", "JNJ"]
REPORTED = ["revenue", "net_income", "operating_income", "eps_diluted", "total_assets",
            "operating_cash_flow"]
DERIVED = ["operating_margin", "net_margin", "free_cash_flow", "current_ratio"]
STORE = Path("chroma_db") / "eval_xbrl"
OWNER = "xbrl-eval"
_CITATION = re.compile(r"\.(?:pdf|html?), p\.")


def use_store(path: Path) -> None:
    """Point indexing, retrieval and figures at a store of the benchmark's own."""
    ingest.CHROMA_DIR = path / "chroma"
    rag.CHROMA_DIR = path / "chroma"
    financials.FACTS_DB = path / "figures.sqlite3"


@dataclass(frozen=True)
class Case:
    ticker: str
    metric: str
    fiscal_year: int
    question: str
    expected: str           # as the engine would print it: "$416,161 million", "32.0%"


def company_name(name: str) -> str:
    """EDGAR's "MICROSOFT CORP" as a question would say it: "Microsoft Corp"."""
    return name.title() if name == name.upper() else name


def xbrl_value(facts_json: dict, metric: str, fiscal_year: int,
               accession: str) -> Decimal | None:
    """The value the filing itself tagged, or the only one reported for that year."""
    values = xbrl.annual_values(facts_json, metric, fiscal_year, accession)
    if len(values) == 1:
        return values.pop()
    values = xbrl.annual_values(facts_json, metric, fiscal_year)
    return values.pop() if len(values) == 1 else None


def make_cases(ticker: str, company: str, fiscal_year: int, facts_json: dict,
               accession: str) -> list[Case]:
    who = company_name(company)
    cases = []
    for key in REPORTED:
        value = xbrl_value(facts_json, key, fiscal_year, accession)
        if value is None:
            continue
        item = financials.LINE_ITEMS[key]
        if item.magnitude:
            value = abs(value)
        metric = analysis.METRICS[key]
        cases.append(Case(ticker, key, fiscal_year,
                          f"What was {who}'s {metric.name.lower()} in fiscal {fiscal_year}?",
                          analysis.format_value(value, metric.unit)))
    for key in DERIVED:
        metric = analysis.METRICS[key]
        formula = metric.formulas[0]
        inputs = []
        for name in formula.inputs:
            value = xbrl_value(facts_json, name, fiscal_year, accession)
            if value is None:
                break
            inputs.append(abs(value) if financials.LINE_ITEMS[name].magnitude else value)
        else:
            value = formula.compute(*inputs)
            cases.append(Case(ticker, key, fiscal_year,
                              f"What was {who}'s {metric.name.lower()} in fiscal {fiscal_year}?",
                              analysis.format_value(value, metric.unit)))
    return cases


def prepare(ticker: str, refresh: bool = False) -> tuple[str, list[Case]]:
    """Index the ticker's latest 10-K into the store (once) and generate its cases."""
    cik = edgar.cik_for_ticker(ticker)
    if cik is None:
        raise SystemExit(f"No SEC filer has the ticker {ticker}.")
    filing = edgar.latest_annual_report(cik)
    if filing is None:
        raise SystemExit(f"{ticker} has no 10-K on EDGAR.")
    name = f"{ticker.lower()}-10k-{filing.report_date}.htm"
    path = STORE / "filings" / name
    if refresh or not path.exists() or not financials.count_facts(OWNER).get(name):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(edgar.fetch_document(filing.url), encoding="utf-8")
        ingest.ingest_file(path, OWNER, cik=cik)
    fiscal_year = int(filing.report_date[:4])
    cases = make_cases(ticker, filing.company, fiscal_year, edgar.companyfacts(cik),
                       filing.accession)
    return name, cases


def run(case: Case) -> dict:
    start = time.perf_counter()
    result = router.answer(case.question, OWNER)
    text = result.answer
    return {
        "ticker": case.ticker, "metric": case.metric, "question": case.question,
        "expected": case.expected, "answer": text, "route": result.route,
        "filing": result.filing,
        "correct": case.expected in text,
        "route_ok": result.route == "calculation",
        "cited": bool(_CITATION.search(text)),
        "seconds": round(time.perf_counter() - start, 2),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("tickers", nargs="*", default=DEFAULT_TICKERS)
    parser.add_argument("--refresh", action="store_true", help="re-fetch and re-index")
    args = parser.parse_args(argv)
    use_store(STORE)

    rows = []
    for ticker in (t.upper() for t in args.tickers):
        name, cases = prepare(ticker, args.refresh)
        print(f"{ticker}: {name}, {len(cases)} questions", flush=True)
        for case in cases:
            row = run(case)
            rows.append(row)
            mark = "ok  " if row["correct"] and row["route_ok"] else "FAIL"
            print(f"  {mark} {case.metric:<20} expected {case.expected}", flush=True)

    print()
    for ticker in dict.fromkeys(r["ticker"] for r in rows):
        group = [r for r in rows if r["ticker"] == ticker]
        print(f"{ticker:<6} {_summary(group)}")
    print(f"{'all':<6} {_summary(rows)}")
    for r in rows:
        if not (r["correct"] and r["route_ok"] and r["cited"]):
            print(f"\n--- {r['question']}  (expected {r['expected']}, route {r['route']})\n"
                  f"{r['answer'][:500]}")
    out = STORE / "results.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nResults written to {out}")


def _summary(rows: list[dict]) -> str:
    n = len(rows)
    count = lambda key: sum(1 for r in rows if r[key])  # noqa: E731
    return f"correct {count('correct'):>3}/{n}   route {count('route_ok'):>3}/{n}   " \
           f"cited {count('cited'):>3}/{n}"


if __name__ == "__main__":
    main()
