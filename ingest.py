"""
ingest.py — Load 10-K PDFs from ./data, chunk them, embed with OpenAI,
and persist to ChromaDB. Re-running is idempotent: each file's chunks
replace any previously ingested from a file of the same name.

api.py reuses ingest_file() to add user-uploaded PDFs one at a time.

Chunking strategy:
  - Token-aware splitting via tiktoken (cl100k_base), 1024 tokens / 128 overlap.
  - Table boundary protection: consecutive lines that look like financial table
    rows are temporarily joined so the splitter never cuts across them.

Usage:
    python ingest.py
"""

import re
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

DATA_DIR = Path("data")
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

def load_pdf(pdf_path: Path) -> list:
    print(f"Loading {pdf_path.name} ...")
    pages = PyPDFLoader(str(pdf_path)).load()
    for page in pages:
        page.metadata["source_file"] = pdf_path.name
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
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
        collection_name=COLLECTION_NAME,
    )


def ingest_file(pdf_path: Path) -> int:
    """
    Chunk and embed one PDF into the store, returning the number of chunks added.

    Chunks previously ingested from a file with the same name are deleted first,
    so re-ingesting (or re-uploading) a filing replaces it instead of duplicating
    every passage in retrieval results.
    """
    docs = load_pdf(pdf_path)
    if not any(doc.page_content.strip() for doc in docs):
        raise ValueError(
            f"{pdf_path.name} has no extractable text (scanned PDFs need OCR first)."
        )
    chunks = split_documents(docs)

    vectorstore = open_vectorstore()
    stale = vectorstore.get(where={"source_file": pdf_path.name}, include=[])["ids"]
    if stale:
        print(f"Replacing {len(stale)} existing chunks from {pdf_path.name}.")
        vectorstore.delete(ids=stale)
    print(f"Embedding {len(chunks)} chunks from {pdf_path.name} …")
    vectorstore.add_documents(chunks)
    return len(chunks)


def list_documents() -> dict[str, int]:
    """Map each ingested source file to its chunk count."""
    if not CHROMA_DIR.exists():
        return {}
    metadatas = open_vectorstore().get(include=["metadatas"])["metadatas"]
    counts: dict[str, int] = {}
    for meta in metadatas:
        name = meta.get("source_file", "unknown")
        counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items()))


def main():
    if not DATA_DIR.exists():
        DATA_DIR.mkdir(parents=True)
        print(f"Created {DATA_DIR.resolve()} — place your 10-K PDF files there and re-run.")
        return

    pdf_files = sorted(DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {DATA_DIR.resolve()}")
    for pdf_path in pdf_files:
        ingest_file(pdf_path)
    print(f"Vectorstore persisted to {CHROMA_DIR.resolve()}")
    print("Ingestion complete.")


if __name__ == "__main__":
    main()
