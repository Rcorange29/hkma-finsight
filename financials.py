"""
financials.py — Structured financial records extracted from 10-K statements.

Text retrieval is the wrong tool for "what was the operating margin?": the
answer is a calculation over specific line items. This module turns the three
primary statements (income, balance sheet, cash flows) into typed records that
analysis.py computes on.

Extraction is split so that no number ever comes from a language model:

  1. Code picks the statement pages (heading near the top, dense with figures).
  2. The model maps each canonical line item to the row that reports it: the
     page, the row label as printed, and one sample value with its fiscal year.
  3. Code verifies the mapping against the page text: the row must exist, its
     label must match, and the sample value must sit in the column whose header
     year matches. Only then are the values read, for every year column, from
     the PDF text itself and scaled by the page's "(In millions ...)" header.

Every stored figure keeps the file, page and row it was read from, so any
calculation can cite each of its inputs.

Breakdowns of the totals are extracted the same way: revenue and operating
income by reportable segment, revenue by geography, and revenue by product or
service line, from the notes. A breakdown is stored only if its parts add up
exactly to the verified company total for every year, so a mis-mapped table
(say, long-lived assets by country) can't pass for revenue.

Records live in a SQLite file inside the vector store directory, so rebuilding
the store (start.py --reingest) rebuilds both together.

Usage:
    python financials.py              # (re)extract figures for every PDF in data/
    python financials.py FILE.pdf     # print the figures extracted from one PDF
"""

from __future__ import annotations

import re
import sqlite3
import sys
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel

import config

load_dotenv()

FACTS_DB = Path("chroma_db") / "financials.sqlite3"
MAX_STATEMENT_PAGES = 8


# ---------------------------------------------------------------------------
# Canonical line items
# ---------------------------------------------------------------------------


def page_number(index) -> int | str:
    """
    The page number a reader sees for a stored page index.

    Chunks and figures store PyPDF's 0-based page index; PDF viewers and the
    pages' own printed footers ("67/139") count from 1. Every citation shows
    this number, so "p.67" opens the right page. Unknown pages show as "?".
    """
    try:
        return int(index) + 1
    except (TypeError, ValueError):
        return "?"

@dataclass(frozen=True)
class LineItem:
    key: str
    name: str
    statement: str      # income | balance | cash
    unit: str           # USD | USD/share | shares
    magnitude: bool     # store the absolute value: outflows print in parentheses
    hint: str           # guidance for mapping the row; never used for values


LINE_ITEMS: dict[str, LineItem] = {
    item.key: item
    for item in [
        LineItem("revenue", "Total revenue", "income", "USD", False,
                 "Total revenue / total net sales / net revenues (the total row, not a segment)."),
        LineItem("cost_of_revenue", "Cost of revenue", "income", "USD", True,
                 "Total cost of revenue / cost of sales (the total row)."),
        LineItem("gross_profit", "Gross profit", "income", "USD", False,
                 "Gross profit, or the dollar row labelled 'Gross margin'. Only if printed."),
        LineItem("research_and_development", "Research and development", "income", "USD", True,
                 "Research and development expense."),
        LineItem("operating_income", "Operating income", "income", "USD", False,
                 "Operating income / income from operations / operating loss."),
        LineItem("income_before_taxes", "Income before income taxes", "income", "USD", False,
                 "Income before income taxes / pre-tax income."),
        LineItem("income_tax_expense", "Provision for income taxes", "income", "USD", False,
                 "Provision for (benefit from) income taxes / income tax expense."),
        LineItem("net_income", "Net income", "income", "USD", False,
                 "Net income (loss) for the company, from the income statement."),
        LineItem("eps_basic", "Basic earnings per share", "income", "USD/share", False,
                 "Basic earnings (loss) per share."),
        LineItem("eps_diluted", "Diluted earnings per share", "income", "USD/share", False,
                 "Diluted earnings (loss) per share."),
        LineItem("diluted_shares", "Diluted weighted average shares", "income", "shares", False,
                 "Diluted weighted average shares outstanding."),
        LineItem("cash_and_equivalents", "Cash and cash equivalents", "balance", "USD", False,
                 "Cash and cash equivalents on the balance sheet (not including investments)."),
        LineItem("total_current_assets", "Total current assets", "balance", "USD", False,
                 "Total current assets."),
        LineItem("total_assets", "Total assets", "balance", "USD", False,
                 "Total assets."),
        LineItem("total_current_liabilities", "Total current liabilities", "balance", "USD", False,
                 "Total current liabilities."),
        LineItem("total_liabilities", "Total liabilities", "balance", "USD", False,
                 "Total liabilities. Only if printed as its own row."),
        LineItem("long_term_debt", "Long-term debt", "balance", "USD", False,
                 "Long-term debt, the non-current row (not the current portion)."),
        LineItem("total_equity", "Total stockholders' equity", "balance", "USD", False,
                 "Total stockholders'/shareholders' equity (not 'total liabilities and equity')."),
        LineItem("operating_cash_flow", "Net cash from operating activities", "cash", "USD", False,
                 "Net cash provided by (used in) operating activities / net cash from operations."),
        LineItem("capital_expenditures", "Capital expenditures", "cash", "USD", True,
                 "Capital expenditures / purchases of or additions to property and equipment."),
        LineItem("dividends_paid", "Dividends paid", "cash", "USD", True,
                 "Cash dividends paid to common stockholders."),
        LineItem("share_repurchases", "Share repurchases", "cash", "USD", True,
                 "Common stock repurchased / repurchases of common stock."),
    ]
}

LineItemKey = Literal[tuple(LINE_ITEMS)]  # type: ignore[valid-type]


@dataclass(frozen=True)
class Breakdown:
    """A total split into parts: a segment, a geographic area, a product line."""
    key: str
    name: str
    dimension: str      # segment | geography | product
    total: str          # the LINE_ITEMS key the parts must add up to


BREAKDOWNS: dict[str, Breakdown] = {
    b.key: b
    for b in [
        Breakdown("segment_revenue", "Segment revenue", "segment", "revenue"),
        Breakdown("segment_operating_income", "Segment operating income", "segment",
                  "operating_income"),
        Breakdown("geographic_revenue", "Revenue by geography", "geography", "revenue"),
        Breakdown("product_revenue", "Revenue by product or service", "product", "revenue"),
    ]
}

BreakdownKey = Literal[tuple(BREAKDOWNS)]  # type: ignore[valid-type]

# Display names for every stored metric key
ITEM_NAMES: dict[str, str] = {
    **{k: item.name for k, item in LINE_ITEMS.items()},
    **{k: b.name for k, b in BREAKDOWNS.items()},
}


@dataclass(frozen=True)
class Fact:
    """One verified figure: a line item's value for one fiscal year."""

    owner: str
    source_file: str
    metric: str
    fiscal_year: int
    value: Decimal      # normalised: dollars, dollars per share, or shares
    unit: str
    label: str          # row label as printed in the filing
    page: int
    line: str           # the row as printed: the citation excerpt
    segment: str = ""   # for breakdowns: the part ("Intelligent Cloud"); "" for totals


# ---------------------------------------------------------------------------
# Reading numbers from statement text
# ---------------------------------------------------------------------------

_DASHES = {"—", "–", "-"}   # some filings print zero as a dash
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_GROUPED = re.compile(r"\b\d{1,3}(?:,\d{3})+\b")
_SCALE = re.compile(r"\bin\s+(thousands|millions|billions)\b", re.IGNORECASE)
# Within a scale note: shares with their own scale ("shares ... in thousands"), or
# shares excepted from it ("except share data"). "per share" amounts are neither.
_SHARE_SCALE = re.compile(
    r"(?<!per )\bshares?\b[^)]*?\bin\s+(thousands|millions|billions)\b", re.IGNORECASE)
_SHARES_UNSCALED = re.compile(r"\bexcept\b[^)]*?(?<!per )\bshares?\b", re.IGNORECASE)
# "September 27," / "June 30" / "Dec. 31,": the date above a year in a column header
_DATE_FRAGMENT = re.compile(r"[A-Z][a-z]{2,8}\.?\s+\d{1,2},?")
_SCALES = {"thousands": Decimal(1_000), "millions": Decimal(1_000_000),
           "billions": Decimal(1_000_000_000)}
_WORD = re.compile(r"[a-z0-9]+")
_STOPWORDS = {"a", "and", "of", "the", "for", "in", "to"}


def parse_number(token: str) -> Decimal | None:
    """Parse one printed figure: '245,122', '$ 11.80', '(1,646)', '-'. None if not a number."""
    s = token.strip().replace("$", "").replace(" ", "")
    if s in _DASHES:
        return Decimal(0)
    negative = (s.startswith("(") and s.endswith(")")) or s.startswith("-")
    s = s.strip("()").lstrip("-")
    if not re.fullmatch(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?", s):
        return None
    try:
        value = Decimal(s.replace(",", ""))
    except InvalidOperation:
        return None
    return -value if negative else value


def split_row(line: str) -> tuple[str, list[Decimal]]:
    """Split a statement row into its label and the figures at the end of the line.

    Figures are read right to left until the first non-number, so numbers inside
    the label ("allowance of $830 and $650") only join the run if they touch it;
    callers take just as many trailing figures as there are year columns.
    """
    s = re.sub(r"\(\s+", "(", line)
    s = re.sub(r"\s+\)", ")", s)
    tokens = s.split()
    values: list[Decimal] = []
    i = len(tokens)
    while i > 0:
        token = tokens[i - 1]
        if token == "$":
            i -= 1
            continue
        value = parse_number(token)
        if value is None:
            break
        values.append(value)
        i -= 1
    values.reverse()
    label = " ".join(t for t in tokens[:i] if t != "$")
    return label, values


def header_years(lines: list[str], row: int) -> list[int] | None:
    """
    Column years from the nearest header above `row`: one line ('... 2024 2023 2022'),
    or one year per line, as some PDFs extract a header of dated columns
    ('September 27,' / '2025' / 'September 28,' / '2024').
    """
    stacked: list[int] = []
    for j in range(row - 1, -1, -1):
        line = lines[j].strip()
        if _GROUPED.search(line):
            if stacked:
                break
            continue    # a data row, not a header
        if _YEAR.fullmatch(line):
            stacked.insert(0, int(line))
            continue
        if stacked:
            if _DATE_FRAGMENT.fullmatch(line):
                continue
            break
        years = [int(y) for y in _YEAR.findall(line)]
        if len(years) >= 2 and len(set(years)) == len(years):
            return years
    if len(stacked) >= 2 and len(set(stacked)) == len(stacked):
        return stacked
    return None


def page_scale(text: str, unit: str = "USD") -> Decimal | None:
    """
    Multiplier from the statement's '(In millions, except per share amounts)' note.

    Share counts can be scaled differently from dollars: '(In millions, except
    number of shares, which are reflected in thousands)' puts them in thousands,
    and '(In thousands, except share data)' leaves them unscaled.
    """
    match = _SCALE.search(text)
    if not match:
        return None
    if unit == "shares":
        end = text.find(")", match.start())
        note = text[match.start(): end if end != -1 else match.end() + 200]
        if own := _SHARE_SCALE.search(note):
            return _SCALES[own.group(1).lower()]
        if _SHARES_UNSCALED.search(note):
            return Decimal(1)
    return _SCALES[match.group(1).lower()]


def _words(text: str) -> set[str]:
    return set(_WORD.findall(text.lower())) - _STOPWORDS


def same_words(a: str, b: str) -> bool:
    """True if two labels differ only in case, punctuation or mis-encoded characters."""
    return _words(a) == _words(b)


def label_score(expected: str, printed: str) -> float:
    """Share of the expected label's words that appear in the printed label."""
    want = _words(expected)
    return len(want & _words(printed)) / len(want) if want else 0.0


LABEL_THRESHOLD = 0.75


@dataclass(frozen=True)
class VerifiedRow:
    label: str
    excerpt: str
    years: list[int]
    values: list[Decimal]


def verify_row(page_text: str, label: str, value_text: str, fiscal_year: int) -> VerifiedRow | None:
    """
    Find the row the model pointed at and confirm it, or return None.

    The row must carry `value_text` in the column whose header year is
    `fiscal_year`, and its label (allowing two wrapped lines above) must match
    `label`. If two different rows qualify equally, the mapping is ambiguous and
    rejected rather than guessed.
    """
    target = parse_number(value_text)
    if target is None:
        return None
    label = _FOOTNOTE.sub("", label)    # the model may copy a footnote marker
    lines = [ln.rstrip() for ln in page_text.splitlines()]
    candidates: list[tuple[float, VerifiedRow]] = []
    for i, line in enumerate(lines):
        row_label, values = split_row(line)
        if not values:
            continue
        years = header_years(lines, i)
        if not years or len(values) < len(years) or fiscal_year not in years:
            continue
        columns = values[-len(years):]
        if abs(columns[years.index(fiscal_year)]) != abs(target):
            continue
        # Labels can wrap: try the row alone, then with up to two lines above.
        best_score, best_label, best_excerpt = -1.0, row_label, line.strip()
        parts = [row_label]
        excerpt = [line.strip()]
        for k in range(i, max(i - 3, -1), -1):
            if k < i:
                above_label, above_values = split_row(lines[k])
                if above_values:
                    break   # the line above is another data row
                parts.insert(0, above_label)
                excerpt.insert(0, lines[k].strip())
            joined = " ".join(p for p in parts if p)
            score = label_score(label, joined)
            if score > best_score:
                best_score, best_label, best_excerpt = score, joined, " ".join(excerpt)
        if best_score >= LABEL_THRESHOLD:
            candidates.append(
                (best_score, VerifiedRow(best_label, best_excerpt, years, columns))
            )
    if not candidates:
        return None
    top = max(score for score, _ in candidates)
    best = [row for score, row in candidates if score == top]
    if any(row.values != best[0].values for row in best[1:]):
        return None     # equally good rows disagree: ambiguous
    return best[0]


# ---------------------------------------------------------------------------
# Choosing the statement pages
# ---------------------------------------------------------------------------

_HEADINGS = {
    "income": re.compile(
        r"income\s+statements?|statements?\s+of\s+(?:consolidated\s+)?(?:operations|income|earnings)",
        re.IGNORECASE),
    "balance": re.compile(
        r"balance\s+sheets?|statements?\s+of\s+financial\s+(?:position|condition)", re.IGNORECASE),
    "cash": re.compile(
        r"cash\s+flows?\s+statements?|statements?\s+of\s+(?:consolidated\s+)?cash\s+flows?",
        re.IGNORECASE),
}
_HEADING_LINES = 8          # the heading must sit near the top of the page
_MIN_FIGURES = 12           # comma-grouped numbers; a TOC page has almost none


def statement_pages(pages: dict[int, str]) -> dict[int, str]:
    """
    Pages that hold a primary financial statement, by page number.

    A page qualifies when a statement heading appears in its first few lines
    (which excludes notes that merely mention "balance sheets") and it is dense
    with figures (which excludes the table of contents).
    """
    chosen: dict[int, str] = {}
    for number in sorted(pages):
        text = pages[number]
        top = [ln for ln in text.splitlines() if ln.strip()][:_HEADING_LINES]
        heading = "\n".join(ln for ln in top if "comprehensive" not in ln.lower())
        if len(_GROUPED.findall(text)) < _MIN_FIGURES:
            continue
        if any(rx.search(heading) for rx in _HEADINGS.values()):
            chosen[number] = text
    return dict(list(chosen.items())[:MAX_STATEMENT_PAGES])


# ---------------------------------------------------------------------------
# Row mapping (the model) and extraction (code)
# ---------------------------------------------------------------------------

class RowMapping(BaseModel):
    metric: LineItemKey
    page: int
    label: str
    value: str
    fiscal_year: int


class RowMappings(BaseModel):
    rows: list[RowMapping]


_MAPPING_PROMPT = """You map line items in a company's financial statements to the rows \
that report them. You will get the text of the statement pages, each starting with \
"=== Page N ===".

For each line item below that the statements print as its own row, return:
- metric: the line item key
- page: the page number N the row is on
- label: the row label exactly as printed
- value: one value from that row exactly as printed (e.g. "245,122" or "(44,477 )")
- fiscal_year: the fiscal year of the column that value is in

Rules:
- Use rows from the primary statements only. Never compute, add up or derive a value.
- Skip a line item if no single row reports it. Do not substitute a different row.
- Each metric at most once.

Line items:
{items}
"""


def _map_rows(pages: dict[int, str]) -> list[RowMapping]:
    """Ask the model which row reports each line item. Its values are only hints."""
    from langchain_openai import ChatOpenAI

    items = "\n".join(f"- {item.key}: {item.hint}" for item in LINE_ITEMS.values())
    text = "\n\n".join(f"=== Page {n} ===\n{t}" for n, t in pages.items())
    llm = ChatOpenAI(model=config.CHAT_MODEL, temperature=0).with_structured_output(RowMappings)
    result = llm.invoke(
        [("system", _MAPPING_PROMPT.format(items=items)), ("human", text)]
    )
    return result.rows


def extract_facts(
    pages: dict[int, str], source_file: str, owner: str, mapper=_map_rows
) -> list[Fact]:
    """
    Extract verified figures from a filing's pages ({page number: text}).

    Mappings that fail verification are dropped. If two mappings for the same
    metric produce different values for a year, that metric is dropped entirely
    rather than picking one.
    """
    statements = statement_pages(pages)
    if not statements:
        print(f"No financial statement pages found in {source_file}.")
        return []

    found: dict[tuple[str, int], Fact] = {}
    conflicted: set[str] = set()
    rejected = 0
    for mapping in mapper(statements):
        item = LINE_ITEMS.get(mapping.metric)
        text = statements.get(mapping.page)
        if item is None or text is None:
            rejected += 1
            continue
        row = verify_row(text, mapping.label, mapping.value, mapping.fiscal_year)
        if row is None:
            rejected += 1
            continue
        if item.unit == "USD/share":
            multiplier = Decimal(1)
        else:
            multiplier = page_scale(text, item.unit)
            if multiplier is None:
                rejected += 1   # no "(In millions ...)" note: scale unknown
                continue
        for year, raw in zip(row.years, row.values, strict=True):
            value = (abs(raw) if item.magnitude else raw) * multiplier
            fact = Fact(owner, source_file, item.key, year, value, item.unit,
                        row.label, mapping.page, row.excerpt)
            previous = found.get((item.key, year))
            if previous is not None and previous.value != value:
                conflicted.add(item.key)
            found.setdefault((item.key, year), fact)

    facts = [f for (metric, _), f in sorted(found.items()) if metric not in conflicted]
    print(
        f"Extracted {len(facts)} figures from {source_file} "
        f"({rejected} unverified mappings dropped"
        + (f", conflicting rows for {sorted(conflicted)}" if conflicted else "")
        + ")."
    )
    return facts


# ---------------------------------------------------------------------------
# Breakdowns: segments, geography, product lines
# ---------------------------------------------------------------------------

_BREAKDOWN_PAGE = re.compile(
    r"segment|geographic|product and service|by product|offerings|\bcountries\b",
    re.IGNORECASE)
MAX_BREAKDOWN_PAGES = 6
_NUMBERS_ONLY = re.compile(r"^[\s$]*(?:\(?\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*\)?[\s$]*)+$")
_FOOTNOTE = re.compile(r"\s*\([a-z0-9]\)\s*$")     # "Services (1)", "Other (a)"


def rejoin_rows(text: str) -> str:
    """
    Put back rows that PDF text extraction split across lines, e.g.
    "Gaming 21,503" / "" / "15,466" / "" / "16,230" -> "Gaming 21,503 15,466 16,230".
    Number-only lines are appended to the data row above them.
    """
    out: list[str] = []
    pending_blank = 0
    for line in text.splitlines():
        if not line.strip():
            pending_blank += 1
            continue
        years_only = all(_YEAR.fullmatch(t) for t in line.split())
        if (_NUMBERS_ONLY.match(line) and not years_only and out
                and split_row(out[-1])[1] and split_row(out[-1])[0]):
            out[-1] = f"{out[-1].rstrip()}  {line.strip()}"
        else:
            out.extend([""] * pending_blank)
            out.append(line)
        pending_blank = 0
    return "\n".join(out)


def breakdown_pages(pages: dict[int, str]) -> dict[int, str]:
    """Note and MD&A pages that may hold breakdown tables, densest first, rejoined."""
    candidates = [
        (len(_GROUPED.findall(text)), number, text)
        for number, text in pages.items()
        if _BREAKDOWN_PAGE.search(text) and len(_GROUPED.findall(text)) >= 6
    ]
    candidates.sort(key=lambda c: (-c[0], c[1]))
    return {number: rejoin_rows(text) for _, number, text in candidates[:MAX_BREAKDOWN_PAGES]}


class PartMapping(BaseModel):
    breakdown: BreakdownKey
    part: str
    page: int
    label: str
    value: str
    fiscal_year: int


class PartMappings(BaseModel):
    rows: list[PartMapping]


_BREAKDOWN_PROMPT = """You find tables in a company's 10-K that split a company total into \
its parts, and map each part's row. You will get the text of some pages, each starting \
with "=== Page N ===".

The breakdowns to look for:
{breakdowns}

For every row that is one part of such a table (never the "Total" row), return:
- breakdown: the breakdown key
- part: the part's name, e.g. "Intelligent Cloud", "United States", "LinkedIn"
- page: the page number N the row is on
- label: the row label exactly as printed
- value: one value from that row exactly as printed
- fiscal_year: the fiscal year of the column that value is in

Rules:
- Only tables whose parts add up to the company's total revenue or total operating income \
for the year. Skip other tables that list segments or countries (assets, goodwill, \
headcount, unearned revenue, ...).
- Never compute, add up or derive a value.
- Each (breakdown, part) at most once.
"""


def _map_parts(pages: dict[int, str]) -> list[PartMapping]:
    """Ask the model which rows are the parts of each breakdown. Values are only hints."""
    from langchain_openai import ChatOpenAI

    breakdowns = "\n".join(
        f"- {b.key}: {ITEM_NAMES[b.total].lower()} by {b.dimension}" for b in BREAKDOWNS.values()
    )
    text = "\n\n".join(f"=== Page {n} ===\n{t}" for n, t in pages.items())
    llm = ChatOpenAI(model=config.CHAT_MODEL, temperature=0).with_structured_output(PartMappings)
    return llm.invoke(
        [("system", _BREAKDOWN_PROMPT.format(breakdowns=breakdowns)), ("human", text)]
    ).rows


def extract_breakdowns(
    pages: dict[int, str], source_file: str, owner: str, totals: list[Fact], mapper=_map_parts
) -> list[Fact]:
    """
    Verified parts of the company totals in `totals` (revenue, operating income).

    Each mapped row is verified like a statement row. Then each breakdown is
    kept only if its parts, for every year they report, add up exactly to the
    company total for that year. Anything else (a missing part, a row from
    another table, a wrong year) fails that check, and the whole breakdown is
    dropped rather than stored incomplete.
    """
    candidates = breakdown_pages(pages)
    if not candidates:
        return []
    total = {(f.metric, f.fiscal_year): f.value for f in totals}
    parts: dict[str, dict[str, tuple[VerifiedRow, int, Decimal]]] = {}
    for mapping in mapper(candidates):
        text = candidates.get(mapping.page)
        breakdown = BREAKDOWNS.get(mapping.breakdown)
        if text is None or breakdown is None:
            continue
        row = verify_row(text, mapping.label, mapping.value, mapping.fiscal_year)
        scale = page_scale(text)
        if row is None or scale is None:
            continue
        name = _FOOTNOTE.sub("", row.label).strip()
        parts.setdefault(breakdown.key, {}).setdefault(name, (row, mapping.page, scale))

    facts: list[Fact] = []
    for key, members in parts.items():
        breakdown = BREAKDOWNS[key]
        years = set.intersection(*(set(row.years) for row, _, _ in members.values()))
        sums_match = bool(years) and len(members) >= 2 and all(
            (breakdown.total, year) in total
            and sum(row.values[row.years.index(year)] * scale
                    for row, _, scale in members.values()) == total[(breakdown.total, year)]
            for year in years
        )
        if not sums_match:
            print(f"Dropped {key} from {source_file}: its parts don't add up to the total.")
            continue
        for name, (row, page, scale) in members.items():
            for year in sorted(years):
                facts.append(Fact(owner, source_file, key, year,
                                  row.values[row.years.index(year)] * scale, "USD",
                                  row.label, page, row.excerpt, name))
    if facts:
        kinds = sorted({f.metric for f in facts})
        print(f"Extracted {len(facts)} breakdown figures from {source_file} ({', '.join(kinds)}).")
    return facts


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS figures (
    owner TEXT NOT NULL,
    source_file TEXT NOT NULL,
    metric TEXT NOT NULL,
    fiscal_year INTEGER NOT NULL,
    value TEXT NOT NULL,
    unit TEXT NOT NULL,
    label TEXT NOT NULL,
    page INTEGER NOT NULL,
    line TEXT NOT NULL,
    segment TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (owner, source_file, metric, segment, fiscal_year)
)
"""


def _connect() -> sqlite3.Connection:
    FACTS_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(FACTS_DB)
    conn.execute(_SCHEMA)
    # Databases from before breakdowns have a "facts" table keyed without a
    # segment: carry its rows over once, as company totals.
    legacy = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'facts'"
    ).fetchone()
    if legacy:
        with conn:
            conn.execute("INSERT OR IGNORE INTO figures SELECT *, '' FROM facts")
            conn.execute("DROP TABLE facts")
    return conn


def replace_facts(owner: str, source_file: str, facts: list[Fact]) -> None:
    """Replace everything stored for one owner's file with `facts`."""
    with _connect() as conn:
        conn.execute(
            "DELETE FROM figures WHERE owner = ? AND source_file = ?", (owner, source_file)
        )
        conn.executemany(
            "INSERT INTO figures VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (f.owner, f.source_file, f.metric, f.fiscal_year, str(f.value), f.unit,
                 f.label, f.page, f.line, f.segment)
                for f in facts
            ],
        )
    conn.close()


def load_facts(owners: list[str], source_file: str | None = None) -> list[Fact]:
    """Facts belonging to any of `owners`, optionally for one file."""
    if not FACTS_DB.exists() or not owners:
        return []
    query = f"SELECT * FROM figures WHERE owner IN ({','.join('?' * len(owners))})"
    params: list = list(owners)
    if source_file is not None:
        query += " AND source_file = ?"
        params.append(source_file)
    with _connect() as conn:
        rows = conn.execute(
            query + " ORDER BY source_file, metric, segment, fiscal_year", params
        ).fetchall()
    conn.close()
    return [
        Fact(owner, source, metric, year, Decimal(value), unit, label, page, line, segment)
        for owner, source, metric, year, value, unit, label, page, line, segment in rows
    ]


def count_facts(owner: str) -> dict[str, int]:
    """Number of stored figures per source file for one owner."""
    if not FACTS_DB.exists():
        return {}
    with _connect() as conn:
        rows = conn.execute(
            "SELECT source_file, COUNT(*) FROM figures WHERE owner = ? GROUP BY source_file",
            (owner,),
        ).fetchall()
    conn.close()
    return dict(rows)


def extract_and_store(pages: dict[int, str], source_file: str, owner: str) -> int:
    """Re-extract one file's figures, replacing any stored before. Returns the count.

    Old figures are cleared even if extraction fails, so a replaced filing never
    keeps the previous version's numbers.
    """
    replace_facts(owner, source_file, [])
    facts = extract_facts(pages, source_file, owner)
    if facts:
        try:
            facts += extract_breakdowns(pages, source_file, owner, facts)
        except Exception as exc:
            # Breakdowns are an extra: the statement figures still stand
            print(f"Warning: breakdown extraction failed for {source_file}: {exc}")
    replace_facts(owner, source_file, facts)
    return len(facts)


def load_pages(pdf_path: Path) -> dict[int, str]:
    from langchain_community.document_loaders import PyPDFLoader

    return {
        doc.metadata["page"]: doc.page_content for doc in PyPDFLoader(str(pdf_path)).load()
    }


def main(argv: list[str]) -> None:
    if argv:
        for path in map(Path, argv):
            pages = load_pages(path)
            facts = extract_facts(pages, path.name, "local")
            facts += extract_breakdowns(pages, path.name, "local", facts)
            for fact in facts:
                print(f"{fact.metric:28} {fact.segment[:24]:24} FY{fact.fiscal_year}  "
                      f"{fact.value:>20}  p.{page_number(fact.page)}  {fact.line}")
        return

    import ingest  # local import: ingest imports this module

    for pdf_path, owner in ingest.pdf_jobs():
        extract_and_store(load_pages(pdf_path), pdf_path.name, owner)


if __name__ == "__main__":
    main(sys.argv[1:])
