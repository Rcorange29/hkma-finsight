"""
api.py — FastAPI server for the RAG engine.

  POST /query       ask a question about your filings and the public samples
  POST /documents   upload a PDF; it is saved under data/users/<id>/ and ingested
  GET  /documents   list your filings and the public samples
  GET  /financials  the structured figures extracted from those filings
  GET  /usage       this month's questions and uploads, against your limits

Calculation questions (margins, ratios, growth, year-over-year changes) are
answered by the deterministic analysis engine and return each input figure
with its citation in `calculations`; other questions use text retrieval.

Questions and uploads count against usage limits (see usage.py); one over a
limit gets 429 with a Retry-After header.

Every request except /health carries an X-User-Id header naming the signed-in
user. The API trusts it without checking a token, because only the Streamlit
UI can reach this service: it must never be exposed to the internet directly.

Usage:
    uvicorn financial_rag.api:app --reload --port 8000
"""

from __future__ import annotations

import re
import tempfile
import threading
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from pypdf.errors import PdfReadError

from financial_rag import analysis, financials, ingest, router, usage

MAX_UPLOAD_BYTES = 50 * 1024 * 1024   # a 10-K PDF is typically 1-10 MB
MAX_UPLOAD_PAGES = 500                # most 10-Ks run 80-300 pages

# Cognito subs are UUIDs; "local" is the single user when login is disabled.
# The id becomes a directory name, so anything else is rejected outright.
_OWNER_RE = re.compile(r"[A-Za-z0-9-]{1,64}")

# Serialises store writes: two uploads of the same filename must not interleave
# their delete-then-add, or both sets of chunks survive.
_ingest_lock = threading.Lock()

app = FastAPI(
    title="Financial Report Q&A Engine",
    description="Ask questions about 10-K filings via RAG.",
    version="1.0.0",
)


@app.exception_handler(usage.LimitReached)
def limit_reached(request: Request, exc: usage.LimitReached) -> JSONResponse:
    return JSONResponse(
        status_code=429, content={"detail": str(exc)},
        headers={"Retry-After": str(exc.retry_after)},
    )


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=20_000)


class QueryRequest(BaseModel):
    question: str
    # Ask about this one filing only (a name from GET /documents). Without it, the
    # question's own wording decides: naming a company narrows to its filing.
    source_file: str | None = None
    # Earlier turns of the chat, oldest first, so a follow-up ("and in 2023?") can
    # be read in context. Only the last few are used.
    history: list[Turn] = Field(default=[], max_length=50)


class RetrievalTrace(BaseModel):
    """How a cited chunk was found (see rag.retrieve_and_rerank)."""
    chunk_id: str
    matched_by: list[str]                 # "semantic" and/or "keyword"
    semantic_rank: int | None = None
    semantic_score: float | None = None
    keyword_rank: int | None = None
    keyword_score: float | None = None
    matched_terms: list[str] = []         # query terms the chunk contains
    rerank_score: float | None = None    # cross-encoder score
    reranker: str | None = None          # "llm" or "cross-encoder": which set the order
    llm_rank: int | None = None          # the LLM reranker's position for this chunk
    exact_value_match: bool = False       # contains every value the question quotes


class SourceCitation(BaseModel):
    source_file: str
    page: int | str
    excerpt: str
    retrieval: RetrievalTrace | None = None


class Figure(BaseModel):
    """One extracted figure and where it was read from."""

    source_file: str
    page: int
    line: str               # the statement row as printed
    label: str
    metric: str
    name: str
    fiscal_year: int
    value: str              # exact decimal, normalised to dollars / shares
    unit: str
    display: str
    shared: bool = False
    segment: str = ""       # for breakdowns: the part ("Intelligent Cloud"); "" for totals


class Calculation(BaseModel):
    metric: str
    name: str
    source_file: str
    fiscal_year: int
    value: str | None       # exact decimal; percentages and multiples as fractions
    display: str | None
    formula: str | None     # None for a reported line item
    problem: str | None     # why value is None
    inputs: list[Figure]
    segment: str = ""       # for breakdowns: the part calculated for


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceCitation]
    route: str = "text"     # "calculation", "text", or "none" (no filing to answer from)
    calculations: list[Calculation] = []
    filing: str | None = None   # the one filing the answer was drawn from, if narrowed
    question: str = ""          # the question as answered: a follow-up made standalone


class DocumentInfo(BaseModel):
    source_file: str
    chunks: int
    figures: int = 0        # structured figures extracted for the analysis engine
    shared: bool = False    # True for public sample filings every user sees
    company: str | None = None   # registrant named on the cover page


class FigureList(BaseModel):
    figures: list[Figure]


class UsageInfo(BaseModel):
    questions: int
    question_limit: int     # 0 = unlimited
    uploads: int
    upload_limit: int
    resets: str             # date the monthly counts start again (UTC)


def _figure(fact: financials.Fact, shared: bool = False) -> Figure:
    return Figure(
        source_file=fact.source_file, page=financials.page_number(fact.page), line=fact.line,
        label=fact.label,
        metric=fact.metric, name=financials.ITEM_NAMES[fact.metric],
        fiscal_year=fact.fiscal_year, value=str(fact.value), unit=fact.unit,
        display=analysis.format_value(fact.value, fact.unit), shared=shared,
        segment=fact.segment,
    )


class DocumentList(BaseModel):
    documents: list[DocumentInfo]


def safe_pdf_name(filename: str | None) -> str:
    """
    Reduce a client-supplied filename to a safe basename in data/.

    Strips any directory part (both separators, since the client may be on
    Windows), and replaces anything outside [A-Za-z0-9._-] so the name is safe
    on disk and readable in citations.
    """
    name = re.split(r"[\\/]", filename or "")[-1]
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name).lstrip(".")
    if not name.lower().endswith(".pdf") or len(name) <= len(".pdf"):
        raise HTTPException(status_code=415, detail="Only .pdf files are accepted.")
    return name


def current_owner(x_user_id: str | None = Header(default=None)) -> str:
    if (
        not x_user_id
        or not _OWNER_RE.fullmatch(x_user_id)
        or x_user_id == ingest.PUBLIC_OWNER
    ):
        raise HTTPException(status_code=401, detail="Missing or invalid user id.")
    return x_user_id


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest, owner: str = Depends(current_owner)):
    if not request.question.strip():
        raise HTTPException(status_code=422, detail="Question must not be empty.")
    event = usage.record(owner, usage.QUESTION)
    try:
        result = router.answer(
            request.question, owner, request.source_file,
            history=[(turn.role, turn.content) for turn in request.history],
        )
    except router.UnknownFiling as exc:
        usage.refund(event)
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        usage.refund(event)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        usage.refund(event)
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}") from exc

    calculations = [
        Calculation(
            metric=s.metric.key, name=s.metric.name, source_file=r.source_file,
            fiscal_year=r.fiscal_year,
            value=None if r.value is None else str(r.value),
            display=None if r.value is None else analysis.format_value(r.value, s.metric.unit),
            formula=r.formula, problem=r.problem,
            inputs=[_figure(f, shared=f.owner != owner) for f in r.inputs],
            segment=s.part,
        )
        for s in result.series
        for r in s.results
    ]
    return QueryResponse(
        answer=result.answer,
        sources=[SourceCitation(**s) for s in result.sources],
        route=result.route,
        calculations=calculations,
        filing=result.filing,
        question=result.question or request.question,
    )


@app.get("/documents", response_model=DocumentList)
def list_documents(owner: str = Depends(current_owner)):
    figures = {
        who: financials.count_facts(who) for who in (owner, ingest.PUBLIC_OWNER)
    }
    filings = ingest.filings([owner, ingest.PUBLIC_OWNER])
    # The caller's own filings first, then the shared samples
    filings.sort(key=lambda f: f.owner != owner)
    return DocumentList(documents=[
        DocumentInfo(source_file=f.source_file, chunks=f.chunks,
                     figures=figures[f.owner].get(f.source_file, 0),
                     shared=f.owner != owner, company=f.company)
        for f in filings
    ])


@app.get("/financials", response_model=FigureList)
def list_financials(source_file: str | None = None, owner: str = Depends(current_owner)):
    facts = router.visible_facts(owner)
    if source_file is not None:
        facts = [f for f in facts if f.source_file == source_file]
    return FigureList(figures=[_figure(f, shared=f.owner != owner) for f in facts])


@app.get("/usage", response_model=UsageInfo)
def get_usage(owner: str = Depends(current_owner)):
    counts = usage.usage(owner)
    return UsageInfo(
        questions=counts.questions, question_limit=counts.question_limit,
        uploads=counts.uploads, upload_limit=counts.upload_limit, resets=counts.resets,
    )


@app.post("/documents", response_model=DocumentInfo, status_code=201)
def upload_document(file: UploadFile, owner: str = Depends(current_owner)):
    name = safe_pdf_name(file.filename)
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.",
        )
    if not data.startswith(b"%PDF-"):
        raise HTTPException(status_code=415, detail="File is not a valid PDF.")
    # Counted once the file looks like a PDF; refunded below if it won't ingest
    event = usage.record(owner, usage.UPLOAD)

    user_dir = ingest.USERS_DIR / owner
    user_dir.mkdir(parents=True, exist_ok=True)
    # Stage inside the user's dir (same filesystem, so the final move is an
    # atomic replace) under the real name, which ingest records as source_file
    # for citations. The globs in ingest.main() are non-recursive, so they never
    # see the staging dir. A file only lands in data/ once it ingests, so a
    # failed upload can't break a later `python -m financial_rag.ingest` run.
    with tempfile.TemporaryDirectory(dir=user_dir) as staging:
        staged = Path(staging) / name
        staged.write_bytes(data)
        try:
            with _ingest_lock:
                chunks = ingest.ingest_file(staged, owner, max_pages=MAX_UPLOAD_PAGES)
                staged.replace(user_dir / name)
        except (ValueError, PdfReadError) as exc:
            usage.refund(event)
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            usage.refund(event)
            raise HTTPException(status_code=500, detail=f"Ingestion error: {exc}") from exc

    company = next(
        (f.company for f in ingest.filings([owner]) if f.source_file == name), None
    )
    return DocumentInfo(
        source_file=name, chunks=chunks, figures=financials.count_facts(owner).get(name, 0),
        company=company,
    )
