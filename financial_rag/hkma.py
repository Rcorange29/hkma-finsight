"""
hkma.py — Import the Hong Kong Monetary Authority (HKMA) official Annual Report.

The HKMA publishes each year's Annual Report on its website as one combined
PDF plus per-chapter PDFs. Rather than hard-coding file names (they change by
year, e.g. 2024's ``AR2024_E.pdf`` vs 2025's ``AR_2025E.pdf``), this module
reads the year's landing page and discovers the combined report there:

    landing page : .../eng/data-publications-and-research/publications/annual-report/<year>/
    combined PDF : the one PDF under that year's media folder that is NOT a
                   numbered chapter (chapters are named ``01_Cover.pdf`` …
                   ``23_BackCover.pdf``).

The combined PDF is a single continuous document, so the 0-based page index
PyPDF stores maps straight to the page a PDF viewer shows (page + 1), which is
exactly what the citations in rag.py use.

Ingestion reuses ingest.ingest_file unchanged (table-aware chunking, local
embeddings, ChromaDB). The SEC/10-K deterministic figure engine is switched off
for these documents — it models US statements, not the Exchange Fund accounts —
so text Q&A with verifiable page citations is what the HKMA report provides.

Usage:
    python -m financial_rag.hkma --year 2024            # download + ingest
    python -m financial_rag.hkma --year 2024 --max-pages 40   # first pages only (a try-out)
    python -m financial_rag.hkma --year 2024 --download-only   # just save the PDF under data/
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path

import httpx

from financial_rag import ingest

HKMA_BASE = "https://www.hkma.gov.hk"
LANDING_URL = (
    HKMA_BASE
    + "/eng/data-publications-and-research/publications/annual-report/{year}/"
)
# The media folder that holds a year's PDFs, as linked from its landing page.
MEDIA_PREFIX = "/media/eng/publication-and-research/annual-report/{year}/"

# A polite, identifiable User-Agent. Reuse the SEC_USER_AGENT contact setting
# (a "<name> <email>" string) when it is set, else a generic project UA.
USER_AGENT = os.getenv("SEC_USER_AGENT", "financial-rag/hkma-annual-report (research)")
TIMEOUT = 60.0

# The institution name stamped on every chunk's "company" metadata, so citations
# and the document list identify the source.
HKMA_TITLE = "Hong Kong Monetary Authority"

# Numbered chapter files: "01_Cover.pdf", "20_Annex.pdf", …
_CHAPTER = re.compile(r"^\d{2}_")
# Any href to a PDF under the year's media folder (absolute, root- or protocol-relative).
_HREF = re.compile(r"href=[\"']([^\"']+?\.pdf)[\"']", re.IGNORECASE)


class HKMAError(RuntimeError):
    """The HKMA report could not be discovered or downloaded."""


# ---------------------------------------------------------------------------
# HTTP (thin wrappers tests replace)
# ---------------------------------------------------------------------------

def fetch_text(url: str, timeout: float = TIMEOUT) -> str:
    """GET a page as text. Tests replace this to stay offline."""
    try:
        resp = httpx.get(url, headers={"User-Agent": USER_AGENT},
                         timeout=timeout, follow_redirects=True)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise HKMAError(f"Request failed for {url}: {exc}") from exc
    return resp.text


def fetch_bytes(url: str, timeout: float = TIMEOUT) -> bytes:
    """GET a PDF as bytes. Tests replace this to stay offline."""
    try:
        resp = httpx.get(url, headers={"User-Agent": USER_AGENT},
                         timeout=timeout, follow_redirects=True)
        resp.raise_for_status()
        if "pdf" not in resp.headers.get("content-type", "").lower():
            raise HKMAError(f"{url} did not return a PDF (content-type "
                            f"{resp.headers.get('content-type')!r}).")
    except httpx.HTTPError as exc:
        raise HKMAError(f"Request failed for {url}: {exc}") from exc
    return resp.content


# ---------------------------------------------------------------------------
# Discovering the year's combined report
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AnnualReport:
    year: int
    full_report_url: str
    chapter_urls: list[str]

    @property
    def file_name(self) -> str:
        """The official PDF basename (AR2024_E.pdf / AR_2025E.pdf / …)."""
        return self.full_report_url.rsplit("/", 1)[-1]


def _absolute(url_or_path: str) -> str:
    """Normalise an href to an absolute https URL."""
    if url_or_path.startswith("//"):
        return "https:" + url_or_path
    if url_or_path.startswith("http://") or url_or_path.startswith("https://"):
        return url_or_path
    if url_or_path.startswith("/"):
        return HKMA_BASE + url_or_path
    return HKMA_BASE + "/" + url_or_path


def parse_pdf_links(html: str, year: int) -> list[str]:
    """Every PDF URL under the year's media folder linked from its landing page."""
    prefix = MEDIA_PREFIX.format(year=year)
    out: list[str] = []
    for href in _HREF.findall(html):
        url = _absolute(href.strip())
        if prefix in url and url not in out:
            out.append(url)
    return out


def discover_annual_report(
    year: int, fetch_text=fetch_text
) -> AnnualReport:
    """
    Find the year's combined annual report PDF from its landing page.

    Raises HKMAError if the year page or its combined report cannot be found.
    `fetch_text` is injected in tests so this never needs the network.
    """
    landing = LANDING_URL.format(year=year)
    html = fetch_text(landing)
    urls = parse_pdf_links(html, year)
    if not urls:
        raise HKMAError(
            f"No Annual Report PDF links were found on {landing}. "
            "Check the year; the HKMA publishes reports from 1993 onward."
        )
    chapters = sorted(u for u in urls if _CHAPTER.match(u.rsplit("/", 1)[-1]))
    full = [u for u in urls if not _CHAPTER.match(u.rsplit("/", 1)[-1])]
    if len(full) != 1:
        raise HKMAError(
            f"Expected one combined report for {year} on {landing}, found "
            f"{len(full)}: {[u.rsplit('/', 1)[-1] for u in full]}. The numbered "
            "chapter PDFs were found but a single combined PDF is required for "
            "continuous page numbers."
        )
    return AnnualReport(year, full[0], chapters)


# ---------------------------------------------------------------------------
# Download + ingest
# ---------------------------------------------------------------------------

def download_annual_report(
    year: int,
    dest_dir: Path | None = None,
    *,
    fetch_text=fetch_text,
    fetch_bytes=fetch_bytes,
    force: bool = False,
) -> tuple[Path, AnnualReport]:
    """
    Download the year's combined report into dest_dir (default ingest.DATA_DIR).

    Skips the download when the file already exists unless force=True. Returns
    the saved path and the discovered report.
    """
    report = discover_annual_report(year, fetch_text=fetch_text)
    dest = dest_dir or ingest.DATA_DIR
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / report.file_name
    if path.exists() and not force:
        print(f"{path.name} already exists in {dest} — skipping download.")
        return path, report
    print(f"Downloading HKMA Annual Report {year}: {report.full_report_url} …")
    data = fetch_bytes(report.full_report_url)
    path.write_bytes(data)
    print(f"Saved {len(data):,} bytes to {path}.")
    return path, report


def ingest_annual_report(
    year: int,
    dest_dir: Path | None = None,
    owner: str = ingest.PUBLIC_OWNER,
    max_pages: int | None = None,
    *,
    fetch_text=fetch_text,
    fetch_bytes=fetch_bytes,
    force_download: bool = False,
) -> int:
    """Download (if needed) and ingest the HKMA annual report; returns chunk count."""
    path, _ = download_annual_report(
        year, dest_dir, fetch_text=fetch_text, fetch_bytes=fetch_bytes,
        force=force_download,
    )
    return ingest.ingest_file(
        path, owner, max_pages=max_pages,
        title=HKMA_TITLE, extract_figures=False,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download and ingest an HKMA Annual Report.")
    parser.add_argument("--year", type=int, required=True, help="Report year, e.g. 2024")
    parser.add_argument("--max-pages", type=int, default=None,
                        help="ingest only the first N pages (a quick try-out)")
    parser.add_argument("--download-only", action="store_true",
                        help="save the PDF under data/ but do not ingest")
    parser.add_argument("--force", action="store_true",
                        help="re-download even if the PDF already exists")
    args = parser.parse_args(argv)

    if args.download_only:
        path, report = download_annual_report(
            args.year, force=args.force)
        print(f"Chapters available: {len(report.chapter_urls)}")
        print(f"Downloaded {path.name}.")
    else:
        chunks = ingest_annual_report(
            args.year, max_pages=args.max_pages, force_download=args.force)
        print(f"Ingested {chunks} chunks from the HKMA Annual Report {args.year}.")


if __name__ == "__main__":
    main()
