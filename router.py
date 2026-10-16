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

import re
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

import analysis
import financials
import followup
import ingest
import rag
from ingest import PUBLIC_OWNER

PLANNER_MODEL = "gpt-4o-mini"

MetricKey = Literal[tuple(analysis.METRICS)]  # type: ignore[valid-type]


class QuestionPlan(BaseModel):
    route: Literal["calculation", "text"]
    metrics: list[MetricKey]
    fiscal_years: list[int]
    year_over_year: bool
    # Copied from the question, not chosen from the filing list: matching it to a
    # filing is done in code (match_filing), which the model was unreliable at.
    company: str | None = Field(
        description="The company whose own filing the question asks about, as the question "
                    "names it (or via its product or ticker); not an acquisition, competitor, "
                    "customer or partner it merely mentions. null if it doesn't name one."
    )
    # Also copied, not decided: any qualifier sends the question to text retrieval,
    # because the engine only holds company-wide annual figures. Left to the model,
    # "Intelligent Cloud revenue" was answered with total revenue.
    qualifier: str | None = Field(
        description="Anything in the question that narrows or changes what a listed metric "
                    "measures: a segment, product, geography, quarter, adjusted or non-GAAP "
                    "measure, or a different version of the item. null if none."
    )


class UnknownFiling(Exception):
    """The caller asked about a filing they can't see (or that doesn't exist)."""


@dataclass
class Answer:
    answer: str
    sources: list[dict]
    route: str                                   # calculation | text | none
    series: list[analysis.Series] = field(default_factory=list)
    filing: str | None = None                    # the one filing searched, if narrowed
    question: str = ""                           # as answered: follow-ups made standalone


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

For both routes:
- company: the company whose own filing the question asks about, when the question names \
it ("Microsoft's" -> "Microsoft") or names its product or ticker ("Azure" -> "Microsoft", \
"MSFT" -> "Microsoft"). Not a company that is only mentioned as an acquisition, \
competitor, customer, supplier or partner of the company being asked about. null when \
the question doesn't name the company it asks about.
- qualifier: anything in the question that narrows or changes what a listed metric \
measures, copied from the question: a segment or product ("Intelligent Cloud", \
"LinkedIn"), a geography ("United States"), a quarter ("Q4"), an adjusted or non-GAAP \
measure ("adjusted operating income"), or a different version of the item ("dividends \
declared", when the list has dividends paid). null when the question asks for a listed \
company-wide annual metric as it is, in its own words or the list's. A company name or a \
fiscal year is not a qualifier.

For "text": metrics and fiscal_years empty, year_over_year false.

Examples:
- "What is the current ratio?" -> calculation, [current_ratio], [], false
- "Return on equity?" -> calculation, [return_on_equity], [], false
- "What was the operating margin in 2024 and how did it change?" -> calculation, \
[operating_margin], [2024], true
- "How did revenue grow from 2022 to 2024?" -> calculation, [revenue], [2022, 2023, 2024], true
- "Gross and net margin for fiscal 2023" -> calculation, [gross_margin, net_margin], [2023], false
- "Why did operating margin improve?" -> text
- "How fast did the cloud segment grow?" -> text (segment revenue is not in the list)
- "What risks does Acme describe?" -> text, company "Acme"
- "What was Microsoft's net margin?" -> calculation, [net_margin], company "Microsoft"
- "What was revenue?" -> calculation, [revenue], company null
- "How much goodwill came from the Activision Blizzard acquisition?" -> text, company null \
(Activision Blizzard is what was acquired, not the company asked about)
- "How does Microsoft compete with Google?" -> text, company "Microsoft"
- "What was Intelligent Cloud revenue in 2024?" -> qualifier "Intelligent Cloud"
- "What were dividends declared in fiscal 2024?" -> qualifier "declared"
- "What were dividends paid in fiscal 2024?" -> qualifier null (that is the listed metric)
- "What was the operating margin in fiscal 2024?" -> qualifier null

Metrics:
{metrics}
"""


def visible_filings(owner: str) -> dict[str, ingest.Filing]:
    """The caller's filings plus the samples, by file name. As with figures, a
    user's own file shadows a sample of the same name."""
    out: dict[str, ingest.Filing] = {}
    for f in ingest.filings([owner, PUBLIC_OWNER]):
        if f.source_file not in out or f.owner == owner:
            out[f.source_file] = f
    return out


# Words that don't tell one company's filing from another's
_GENERIC_NAME_WORDS = frozenset(
    "inc incorporated corp corporation co company companies holdings holding group ltd "
    "limited plc llc lp sa ag nv the and of sec form annual report filing pdf 10k".split()
)


def _name_words(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if len(w) > 1 and w not in _GENERIC_NAME_WORDS}


def filings_named(company: str | None, filings: dict[str, ingest.Filing]) -> list[str]:
    """
    The filings whose company or file name shares a distinctive word with the
    company the question names ("Microsoft's" matches "Microsoft Corporation" and
    microsoft-sec-10k.pdf). Empty when the question names no company, or one
    with no filing here.
    """
    wanted = _name_words(company or "")
    if not wanted:
        return []
    return [
        name for name, filing in filings.items()
        if wanted & (_name_words(filing.company or "") | _name_words(name.rsplit(".", 1)[0]))
    ]


def match_filing(company: str | None, filings: dict[str, ingest.Filing]) -> str | None:
    """The one filing the named company matches; None for no match or several."""
    hits = filings_named(company, filings)
    return hits[0] if len(hits) == 1 else None


def _no_filing_for(company: str, filings: dict[str, ingest.Filing]) -> Answer:
    available = "; ".join(
        f"{f.company} ({name})" if f.company else name for name, f in filings.items()
    )
    return Answer(
        f"No filing for {company} is available, so I can't answer that from a filing. "
        f"Filings you can ask about: {available or 'none yet'}. Upload {company}'s 10-K "
        "to ask about it.",
        [], "none",
    )


def plan_question(question: str) -> QuestionPlan:
    """
    Read a question into a plan. The model isn't shown the user's filings: given
    a list, it tended to fill in a company the question never named.
    """
    from langchain_openai import ChatOpenAI

    metrics = "\n".join(
        f"- {m.key}: {m.name}. {m.description}" for m in analysis.METRICS.values()
    )
    llm = ChatOpenAI(model=PLANNER_MODEL, temperature=0).with_structured_output(QuestionPlan)
    return llm.invoke([
        ("system", PLANNER_PROMPT.format(metrics=metrics)),
        ("human", question),
    ])


def _text_answer(
    question: str, owner: str, filing: ingest.Filing | None = None, preface: str = ""
) -> Answer:
    scope = (filing.owner, filing.source_file) if filing else None
    result = rag.query(question, owner, scope)
    return Answer(preface + result.answer, result.sources, "text",
                  filing=filing.source_file if filing else None)


def visible_facts(owner: str) -> list[financials.Fact]:
    """The caller's figures plus the samples'. A user's own file shadows a sample
    of the same name, so the two are never mixed into one set of figures."""
    facts = financials.load_facts([owner, PUBLIC_OWNER])
    own_files = {f.source_file for f in facts if f.owner == owner}
    return [f for f in facts if f.owner == owner or f.source_file not in own_files]


def answer(
    question: str,
    owner: str,
    source_file: str | None = None,
    planner=plan_question,
    history: list[tuple[str, str]] | None = None,
    rewriter=followup.standalone_question,
) -> Answer:
    """
    Answer from the caller's filings and the samples.

    `history` (earlier (role, text) turns, oldest first) lets a follow-up such as
    "and in 2023?" be rewritten into a standalone question first; the answer
    records the question it actually answered.

    `source_file` (picked in the UI) narrows both routes to that one filing.
    Otherwise the planner narrows them when the question names a company or
    filing; a question that names none searches every filing, and the prompt
    makes the model say which filing each figure comes from.
    """
    asked = rewriter(question, history) if history else question
    result = _answer(asked, owner, source_file, planner)
    result.question = asked
    return result


def _answer(question: str, owner: str, source_file: str | None, planner) -> Answer:
    filings = visible_filings(owner)
    if source_file is not None and source_file not in filings:
        raise UnknownFiling(f"No filing named {source_file}.")
    facts = visible_facts(owner)

    plan = None
    if facts or (source_file is None and len(filings) > 1):
        try:
            plan = planner(question)
        except Exception as exc:
            print(f"Planner failed, answering from text: {exc}")

    chosen = source_file
    if chosen is None and plan and plan.company:
        named = filings_named(plan.company, filings)
        if not named:
            # Answering with another company's figures would be wrong, not partial
            return _no_filing_for(plan.company, filings)
        chosen = named[0] if len(named) == 1 else None
    filing = filings[chosen] if chosen else None
    if filing:
        facts = [f for f in facts if f.source_file == filing.source_file]
    if (plan is None or plan.route != "calculation" or not plan.metrics or not facts
            or plan.qualifier):
        # A qualifier (a segment, a quarter, "declared" vs paid, ...) means the
        # company-wide annual figures in the engine aren't what was asked for
        return _text_answer(question, owner, filing)

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
        return _text_answer(question, owner, filing, preface)

    markdown, sources = analysis.render(series)
    return Answer(markdown, sources, "calculation", series,
                  filing=filing.source_file if filing else None)
