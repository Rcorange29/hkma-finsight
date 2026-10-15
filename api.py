"""
api.py — FastAPI server for the RAG engine.

  POST /query       ask a question
  POST /documents   upload a PDF; it is saved to data/ and ingested immediately
  GET  /documents   list ingested files and their chunk counts

Usage:
    uvicorn api:app --reload --port 8000
"""

from __future__ import annotations

import re
import tempfile
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel
from pypdf.errors import PdfReadError

import ingest
import rag

MAX_UPLOAD_BYTES = 50 * 1024 * 1024   # a 10-K PDF is typically 1-10 MB

# Serialises store writes: two uploads of the same filename must not interleave
# their delete-then-add, or both sets of chunks survive.
_ingest_lock = threading.Lock()

app = FastAPI(
    title="Financial Report Q&A Engine",
    description="Ask questions about 10-K filings via RAG.",
    version="1.0.0",
)


class QueryRequest(BaseModel):
    question: str


class SourceCitation(BaseModel):
    source_file: str
    page: int | str
    excerpt: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceCitation]


class DocumentInfo(BaseModel):
    source_file: str
    chunks: int


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


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=422, detail="Question must not be empty.")
    try:
        result = rag.query(request.question)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}") from exc

    return QueryResponse(
        answer=result.answer,
        sources=[SourceCitation(**s) for s in result.sources],
    )


@app.get("/documents", response_model=DocumentList)
def list_documents():
    docs = ingest.list_documents()
    return DocumentList(
        documents=[DocumentInfo(source_file=n, chunks=c) for n, c in docs.items()]
    )


@app.post("/documents", response_model=DocumentInfo, status_code=201)
def upload_document(file: UploadFile):
    name = safe_pdf_name(file.filename)
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.",
        )
    if not data.startswith(b"%PDF-"):
        raise HTTPException(status_code=415, detail="File is not a valid PDF.")

    ingest.DATA_DIR.mkdir(parents=True, exist_ok=True)
    # Stage inside data/ (same filesystem, so the final move is an atomic
    # replace) under the real name, which ingest records as source_file for
    # citations. The glob in ingest.main() is non-recursive, so it never sees
    # the staging dir. A file only lands in data/ once it ingests, so a failed
    # upload can't break a later `python ingest.py` run.
    with tempfile.TemporaryDirectory(dir=ingest.DATA_DIR) as staging:
        staged = Path(staging) / name
        staged.write_bytes(data)
        try:
            with _ingest_lock:
                chunks = ingest.ingest_file(staged)
                staged.replace(ingest.DATA_DIR / name)
        except (ValueError, PdfReadError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Ingestion error: {exc}") from exc

    return DocumentInfo(source_file=name, chunks=chunks)
