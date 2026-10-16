"""
ingest.py — Load 10-K PDFs from ./data, chunk them, embed with OpenAI,
and persist to ChromaDB. Re-running is idempotent: each file's chunks
replace any previously ingested from a file of the same name and owner.

Every chunk carries an `owner`:
  - data/*.pdf               owner "public": sample filings every user can query
  - data/users/<id>/*.pdf    owner <id>: one user's private uploads

api.py reuses ingest_file() to add user-uploaded PDFs one at a time.

Each file's financial statements are also extracted into structured figures
(see financials.py) for the deterministic analysis engine.

Chunking strategy:
  - Token-aware splitting via tiktoken (cl100k_base), 1024 tokens / 128 overlap.
  - Table boundary protection: consecutive lines that look like financial table
    rows are temporarily joined so the splitter never cuts across them.

Usage:
    python -m financial_rag.ingest
"""

import re
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from financial_rag import config, financials

load_dotenv()

DATA_DIR = Path("data")
USERS_DIR = DATA_DIR / "users"
PUBLIC_OWNER = "public"
CHROMA_DIR = Path("chroma_db")
COLLECTION_NAME = "financial_reports"

CHUNK_SIZE = 1024    # tokens
CHUNK_OVERLAP = 128  # tokens

# Placeholder used to hide intra-table newlines from the splitter
_TABLE_NL = "\u2060NL\u2060"   # word-joiner + NL + word-joiner (invisible in embedding)


# ---------------------------------------------------------------------------
# Table boundary protection
# ---------------------------------------------------------------------------

def _is_table_line(line: str) -> bool:
    """
    Return True if the line looks like part of a financial table row:
      - Contains a $ sign plus at least one number, or
      - Contains 3+ comma-formatted numbers (e.g. 245,122), or
      - Is a row of dashes/spaces used as a separator between table rows.
    """
    s = line.strip()
    if not s:
        return False
    # Dash-separator rows  (e.g. "  --------   --------   --------")
    if re.fullmatch(r'[\s\-\$=]+', s) and len(s) > 4:
        return True
    # Count large financial numbers:  1,234  or  1,234,567
    big_numbers = re.findall(r'\b\d{1,3}(?:,\d{3})+\b', s)
    dollars = s.count("$")
    if dollars >= 1 and len(big_numbers) >= 1:
        return True
    if len(big_numbers) >= 3:
        return True
    return False


def _protect_table_blocks(docs: list) -> list:
    """
    Within each document's page_content, replace newlines that fall *inside*
    a contiguous block of table-like lines with _TABLE_NL so the splitter
    treats the whole block as one logical unit.
    """
    for doc in docs:
        lines = doc.page_content.split("\n")
        out: list[str] = []
        in_table = False
        for line in lines:
            if _is_table_line(line):
                if not in_table:
                    in_table = True
                    # Make sure we don't stick the table onto the previous para
                    if out and out[-1] != "":
                        out.append("")
                # Store the line with its trailing newline replaced by placeholder
                out.append(line + _TABLE_NL)
            else:
                if in_table:
                    in_table = False
                    # Blank line after the table block to signal a paragraph break.
                    # Skipped when the terminating line is itself blank, since it
                    # already provides the break.
                    if line != "":
                        out.append("")
                out.append(line)
        doc.page_content = "\n".join(out)
    return docs


def _restore_table_blocks(chunks: list) -> list:
    """Restore the placeholder newlines in each chunk after splitting."""
    for chunk in chunks:
        chunk.page_content = chunk.page_content.replace(_TABLE_NL, "\n")
    return chunks


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def load_pdf(pdf_path: Path, owner: str) -> list:
    print(f"Loading {pdf_path.name} ...")
    pages = PyPDFLoader(str(pdf_path)).load()
    for page in pages:
        page.metadata["source_file"] = pdf_path.name
        page.metadata["owner"] = owner
    return pages


def split_documents(docs: list) -> list:
    # Protect table rows before splitting
    docs = _protect_table_blocks(docs)

    # Token-aware splitter using the same tokeniser as text-embedding-3-small
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    # Restore placeholder newlines inside table blocks
    chunks = _restore_table_blocks(chunks)
    return chunks


def open_vectorstore() -> Chroma:
    """Open the persisted store, creating it on first write if it doesn't exist."""
    return Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=OpenAIEmbeddings(model=config.EMBEDDING_MODEL),
        collection_name=COLLECTION_NAME,
    )


def ingest_file(pdf_path: Path, owner: str = PUBLIC_OWNER, max_pages: int | None = None) -> int:
    """
    Chunk and embed one PDF into the store, returning the number of chunks added,
    then extract its financial statements into structured figures.

    A file longer than max_pages is rejected before anything is embedded: the
    cost of ingesting grows with its length.

    Chunks and figures previously ingested from a file with the same name and
    owner are replaced, so re-ingesting (or re-uploading) a filing never
    duplicates passages or keeps stale numbers. Other owners' files of the same
    name are untouched.
    """
    docs = load_pdf(pdf_path, owner)
    if max_pages is not None and len(docs) > max_pages:
        raise ValueError(
            f"{pdf_path.name} has {len(docs)} pages; the limit is {max_pages}."
        )
    if not any(doc.page_content.strip() for doc in docs):
        raise ValueError(
            f"{pdf_path.name} has no extractable text (scanned PDFs need OCR first)."
        )
    # Copy the page text first: split_documents rewrites table newlines in place.
    pages = {doc.metadata["page"]: doc.page_content for doc in docs}
    chunks = split_documents(docs)
    company = registrant_name(pages.get(0, ""))
    if company:
        for chunk in chunks:
            chunk.metadata["company"] = company

    vectorstore = open_vectorstore()
    stale = vectorstore.get(
        where={"$and": [{"source_file": pdf_path.name}, {"owner": owner}]}, include=[]
    )["ids"]
    if stale:
        print(f"Replacing {len(stale)} existing chunks from {pdf_path.name}.")
        vectorstore.delete(ids=stale)
    print(f"Embedding {len(chunks)} chunks from {pdf_path.name} …")
    vectorstore.add_documents(chunks)

    # Figures are an extra: if extraction fails, text Q&A still works, and the
    # analysis engine reports the figures as missing rather than guessing.
    try:
        financials.extract_and_store(pages, pdf_path.name, owner, company)
    except Exception as exc:
        print(f"Warning: financial figure extraction failed for {pdf_path.name}: {exc}")
    return len(chunks)


# ---------------------------------------------------------------------------
# Filings and their companies
# ---------------------------------------------------------------------------

_REGISTRANT_NOTE = re.compile(r"exact name of registrant", re.IGNORECASE)
_FILE_NUMBER = re.compile(r"commission file number", re.IGNORECASE)


def registrant_name(cover: str) -> str | None:
    """
    The company a 10-K belongs to, read from its cover page.

    Most covers print the name above "(Exact name of registrant as specified in
    its charter)"; some, like Microsoft's, print it straight after the
    "Commission File Number" line instead. All-capital names are title-cased.
    """
    lines = [" ".join(line.split()) for line in cover.split("\n")]
    lines = [line for line in lines if line]
    name = None
    for i, line in enumerate(lines):
        if _REGISTRANT_NOTE.search(line):
            before = _REGISTRANT_NOTE.split(line)[0].strip(" (")
            name = before or (lines[i - 1] if i else None)
            break
    else:
        for i, line in enumerate(lines[:-1]):
            if _FILE_NUMBER.search(line):
                name = lines[i + 1]
                break
    if not name or len(name) > 80 or not re.search(r"[A-Za-z]{2}", name):
        return None
    return name.title() if name == name.upper() else name


@dataclass(frozen=True)
class Filing:
    source_file: str
    owner: str
    chunks: int
    company: str | None


_companies: dict[tuple[str, str, int], str | None] = {}


def filings(owners: list[str]) -> list[Filing]:
    """
    Every filing the given owners have ingested, with its chunk count and company.

    The company comes from the chunks' "company" metadata (set at ingest), or for
    filings ingested before that, from the cover page text, cached per filing.
    """
    if not CHROMA_DIR.exists():
        return []
    store = open_vectorstore()
    got = store.get(where={"owner": {"$in": owners}}, include=["metadatas"])
    counts: dict[tuple[str, str], int] = {}
    companies: dict[tuple[str, str], str] = {}
    for meta in got["metadatas"]:
        key = (meta.get("owner", ""), meta.get("source_file", "unknown"))
        counts[key] = counts.get(key, 0) + 1
        if meta.get("company"):
            companies[key] = meta["company"]
    out = []
    for (owner, name), n in sorted(counts.items(), key=lambda kv: kv[0][1]):
        company = companies.get((owner, name))
        if company is None:
            cache_key = (owner, name, n)
            if cache_key not in _companies:
                cover = store.get(
                    where={"$and": [{"owner": owner}, {"source_file": name}, {"page": 0}]},
                    include=["documents"],
                )["documents"]
                _companies[cache_key] = registrant_name("\n".join(cover))
            company = _companies[cache_key]
        out.append(Filing(name, owner, n, company))
    return out


def list_documents(owner: str) -> dict[str, int]:
    """Map each source file the owner has ingested to its chunk count."""
    if not CHROMA_DIR.exists():
        return {}
    metadatas = open_vectorstore().get(where={"owner": owner}, include=["metadatas"])[
        "metadatas"
    ]
    counts: dict[str, int] = {}
    for meta in metadatas:
        name = meta.get("source_file", "unknown")
        counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items()))


def pdf_jobs() -> list[tuple[Path, str]]:
    """Every PDF under data/ with its owner: public samples, then each user's uploads."""
    jobs = [(p, PUBLIC_OWNER) for p in sorted(DATA_DIR.glob("*.pdf"))]
    if USERS_DIR.exists():
        for user_dir in sorted(p for p in USERS_DIR.iterdir() if p.is_dir()):
            jobs += [(p, user_dir.name) for p in sorted(user_dir.glob("*.pdf"))]
    return jobs


def main():
    if not DATA_DIR.exists():
        DATA_DIR.mkdir(parents=True)
        print(f"Created {DATA_DIR.resolve()} — place your 10-K PDF files there and re-run.")
        return

    # Rebuilding the store from scratch restores everything with its access.
    jobs = pdf_jobs()
    if not jobs:
        raise FileNotFoundError(f"No PDF files found in {DATA_DIR.resolve()}")
    for pdf_path, owner in jobs:
        ingest_file(pdf_path, owner)
    print(f"Vectorstore persisted to {CHROMA_DIR.resolve()}")
    print("Ingestion complete.")


if __name__ == "__main__":
    main()
