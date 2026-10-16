"""
router.py — Send each question to the engine that can answer it properly.

Questions about a reported amount, margin, ratio, growth rate or year-over-year
comparison go to the deterministic analysis engine (analysis.py), which
computes from structured figures and cites every input. Everything else goes
to text retrieval (rag.py).

The model's only job here is to read the question into a plan: which metrics
from a fixed catalog, which fiscal years, which filing. The numbers and the
answer text are produced by code, so the model never writes a figure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel

import analysis
import financials
import rag
from ingest import PUBLIC_OWNER

PLANNER_MODEL = "gpt-4o-mini"

MetricKey = Literal[tuple(analysis.METRICS)]  # type: ignore[valid-type]


class QuestionPlan(BaseModel):
    route: Literal["calculation", "text"]
    metrics: list[MetricKey]
    fiscal_years: list[int]
    year_over_year: bool
    source_file: str | None


@dataclass
class Answer:
    answer: str
    sources: list[dict]
    route: str                                   # calculation | text
    series: list[analysis.Series] = field(default_factory=list)


PLANNER_PROMPT = """You route questions about company financial filings.

Choose route "calculation" whenever the question asks for the value of a metric listed \
below: a reported amount (such as revenue or net income), a margin, ratio or return, its \
growth, or a comparison between years. This holds in any wording or tense and whether or \
not a year is named: with no year, the latest year in the filing is used.
Choose "text" when the question asks why or how something happened, or about anything \
not in the list: risks, segments, products, strategy, accounting policies, other amounts.

For "calculation":
- metrics: the keys of the metrics asked about ("operating margin" -> operating_margin; \
"revenue growth" -> revenue with year_over_year true).
- fiscal_years: only the fiscal years named in the question, as integers \
("fiscal 2024" -> 2024). Empty if none is named. For a range ("from 2022 to 2024") list \
every year in it. Never add a year just to compare with: set year_over_year instead.
- year_over_year: true if the question asks about growth, change, an increase or decrease, \
or a comparison with the prior year.
- source_file: the filing the question is about, from the list below, or null if the \
question doesn't single one out.

For "text": metrics and fiscal_years empty, year_over_year false, source_file null.

Examples:
- "What is the current ratio?" -> calculation, [current_ratio], [], false
- "Return on equity?" -> calculation, [return_on_equity], [], false
- "What was the operating margin in 2024 and how did it change?" -> calculation, \
[operating_margin], [2024], true
- "How did revenue grow from 2022 to 2024?" -> calculation, [revenue], [2022, 2023, 2024], true
- "Gross and net margin for fiscal 2023" -> calculation, [gross_margin, net_margin], [2023], false
- "Why did operating margin improve?" -> text
- "How fast did the cloud segment grow?" -> text (segment revenue is not in the list)

Metrics:
{metrics}

Filings with extracted figures:
{documents}
"""


def _describe_documents(facts: list[financials.Fact], owner: str) -> dict[str, str]:
    years: dict[str, set[int]] = {}
    shared: dict[str, bool] = {}
    for f in facts:
        years.setdefault(f.source_file, set()).add(f.fiscal_year)
        shared[f.source_file] = f.owner != owner
    return {
        name: f"- {name}{' (sample filing)' if shared[name] else ''}: "
              f"fiscal years {min(ys)}-{max(ys)}"
        for name, ys in years.items()
    }


def plan_question(question: str, documents: dict[str, str]) -> QuestionPlan:
    from langchain_openai import ChatOpenAI

    metrics = "\n".join(
        f"- {m.key}: {m.name}. {m.description}" for m in analysis.METRICS.values()
    )
    llm = ChatOpenAI(model=PLANNER_MODEL, temperature=0).with_structured_output(QuestionPlan)
    return llm.invoke([
        ("system", PLANNER_PROMPT.format(metrics=metrics, documents="\n".join(documents.values()))),
        ("human", question),
    ])


def _text_answer(question: str, owner: str, preface: str = "") -> Answer:
    result = rag.query(question, owner)
    return Answer(preface + result.answer, result.sources, "text")


def visible_facts(owner: str) -> list[financials.Fact]:
    """The caller's figures plus the samples'. A user's own file shadows a sample
    of the same name, so the two are never mixed into one set of figures."""
    facts = financials.load_facts([owner, PUBLIC_OWNER])
    own_files = {f.source_file for f in facts if f.owner == owner}
    return [f for f in facts if f.owner == owner or f.source_file not in own_files]


def answer(question: str, owner: str, planner=plan_question) -> Answer:
    facts = visible_facts(owner)
    if not facts:
        return _text_answer(question, owner)

    documents = _describe_documents(facts, owner)
    try:
        plan = planner(question, documents)
    except Exception as exc:
        print(f"Planner failed, answering from text: {exc}")
        return _text_answer(question, owner)
    if plan.route != "calculation" or not plan.metrics:
        return _text_answer(question, owner)

    if plan.source_file in documents:
        facts = [f for f in facts if f.source_file == plan.source_file]
    metrics = list(dict.fromkeys(plan.metrics))     # dedupe, keep order
    series = analysis.analyze(facts, metrics, plan.fiscal_years, plan.year_over_year)

    if not any(r.value is not None for s in series for r in s.results):
        # Nothing computable: say why, then answer from the text without arithmetic.
        problems = sorted({
            f"{s.metric.name} FY{r.fiscal_year}: {r.problem}"
            for s in series for r in s.results if r.problem
        })
        preface = (
            "I couldn't calculate this from the structured figures"
            + (":\n" + "\n".join(f"- {p}" for p in problems) if problems else ".")
            + "\n\nFrom the filing text instead:\n\n"
        )
        return _text_answer(question, owner, preface)

    markdown, sources = analysis.render(series)
    return Answer(markdown, sources, "calculation", series)
