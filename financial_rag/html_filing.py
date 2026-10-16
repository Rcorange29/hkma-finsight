"""
html_filing.py — Read a 10-K as EDGAR serves it: HTML with inline XBRL.

A PDF's text loses table structure: cells become runs of spaces, and a table
split across chunks loses the column headings that say which column is which
year. The HTML keeps both, so this renders it to plain text the rest of the
pipeline already reads:

  - pages split where the filing breaks pages (page-break-after/-before), so
    citations keep page numbers (the filing's own pagination, not a PDF's);
  - each table row on one line, its non-empty cells separated by spaces
    ("Total net sales $ 416,161 $ 391,035"), the shape financials.py parses;
  - hidden content (the inline-XBRL header) skipped;
  - each table's column headings (the rows above its first figure) recorded,
    so ingest can repeat them in any chunk holding that table's rows but not
    its headings (see with_table_headings).
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field

from bs4 import BeautifulSoup, NavigableString, Tag, XMLParsedAsHTMLWarning

_BLOCK = {"p", "div", "br", "tr", "li", "ul", "ol", "table", "h1", "h2", "h3", "h4", "h5",
          "h6", "hr", "section", "article", "center", "blockquote", "pre", "body"}
_SKIP = {"script", "style", "head", "title", "ix:header"}
_HIDDEN = re.compile(r"display\s*:\s*none", re.IGNORECASE)
_BREAK_AFTER = re.compile(r"page-break-after\s*:\s*always|break-after\s*:\s*page", re.I)
_BREAK_BEFORE = re.compile(r"page-break-before\s*:\s*always|break-before\s*:\s*page", re.I)
_FIGURE = re.compile(r"\d{1,3}(?:,\d{3})+|\d+\.\d+")
_PAGE = "\f"


@dataclass
class Table:
    headings: list[str]             # lines above the first row with a figure
    rows: list[str]                 # every other line


@dataclass
class Page:
    number: int                     # 0-based, like PDF pages in metadata
    text: str
    tables: list[Table] = field(default_factory=list)


def render_table(table: Tag) -> Table:
    lines = []
    for tr in table.find_all("tr"):
        cells = [" ".join(td.get_text(" ", strip=True).split())
                 for td in tr.find_all(["td", "th"])]
        # Spaced exactly as the page text is, so rows can be found in chunks
        line = " ".join(c for c in cells if c)
        if line:
            lines.append(line)
    first = next((i for i, line in enumerate(lines) if _FIGURE.search(line)), len(lines))
    return Table(headings=lines[:first], rows=lines[first:])


class _Text:
    """Text built up in document order, with each table's position in it."""

    def __init__(self):
        self.parts: list[str] = []
        self.length = 0
        self.tables: list[tuple[int, Table]] = []

    def add(self, s: str) -> None:
        self.parts.append(s)
        self.length += len(s)

    def walk(self, node) -> None:
        if isinstance(node, NavigableString):
            if type(node) is NavigableString:     # not a comment, CDATA or doctype
                self.add(str(node))
            return
        if not isinstance(node, Tag) or node.name in _SKIP:
            return
        style = node.get("style") or ""
        if _HIDDEN.search(style):
            return
        if _BREAK_BEFORE.search(style):
            self.add(_PAGE)
        if node.name == "table":
            table = render_table(node)
            if table.headings or table.rows:
                self.add("\n")
                self.tables.append((self.length, table))
                self.add("\n".join(table.headings + table.rows) + "\n")
        else:
            block = node.name in _BLOCK
            if block:
                self.add("\n")
            for child in node.children:
                self.walk(child)
            if block:
                self.add("\n")
        if _BREAK_AFTER.search(style):
            self.add(_PAGE)


def _tidy(text: str) -> str:
    lines = [" ".join(line.replace("\xa0", " ").split()) for line in text.split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def load_pages(html: str) -> list[Page]:
    """The filing's pages as text, each with the tables on it. Empty pages are dropped."""
    with warnings.catch_warnings():
        # Inline XBRL is XHTML; parsed as HTML on purpose (lenient, keeps the layout)
        warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
        soup = BeautifulSoup(html, "lxml")
    text = _Text()
    text.walk(soup.body or soup)
    full = "".join(text.parts)

    pages: list[Page] = []
    start = 0
    for raw in full.split(_PAGE):
        end = start + len(raw)
        body = _tidy(raw)
        if body:
            tables = [t for at, t in text.tables if start <= at < end]
            pages.append(Page(len(pages), body, tables))
        start = end + len(_PAGE)
    return pages


def with_table_headings(chunk: str, tables: list[Table]) -> str:
    """Repeat a table's column headings at the top of a chunk that cut them off.

    A balance-sheet chunk starting at "Short-term unearned revenue 57,582 50,901"
    doesn't say which column is 2024; prefixed with the table's headings it does.
    """
    missing = [
        " / ".join(table.headings) for table in tables
        if table.headings
        and any(row in chunk for row in table.rows)
        and not any(heading in chunk for heading in table.headings)
    ]
    if not missing:
        return chunk
    return "[Table columns: " + " | ".join(missing) + "]\n" + chunk
