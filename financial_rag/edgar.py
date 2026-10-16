"""
edgar.py — A small client for SEC EDGAR's JSON APIs.

  company_tickers   every listed company: ticker, CIK and registered name
  submissions       a company's filings (form, accession number, documents)
  companyfacts      every XBRL figure a company has tagged in its filings

The SEC requires each automated request to identify who is making it, with a
contact address, in the User-Agent header. It comes from the SEC_USER_AGENT
environment variable ("<app name> <email>"), kept in .env, never in code. Without
it, nothing is fetched and the features that need EDGAR are skipped.

Requests stay under the SEC's limit of 10 a second, and responses are cached on
disk (next to the vector store) so re-ingesting doesn't refetch them.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

CACHE_DIR = Path("chroma_db") / "edgar_cache"
CACHE_SECONDS = {"tickers": 7 * 86_400, "submissions": 86_400, "companyfacts": 86_400}
MIN_INTERVAL = 0.12          # seconds between requests: under 10 a second
TIMEOUT = 30.0

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

_lock = threading.Lock()
_last_request = 0.0


class EdgarUnavailable(Exception):
    """EDGAR can't be used: no contact address configured, or the request failed."""


def user_agent() -> str:
    value = os.getenv("SEC_USER_AGENT", "").strip()
    if "@" not in value:
        raise EdgarUnavailable(
            "SEC_USER_AGENT isn't set. EDGAR requires a contact address: set it in .env "
            'as "<app name> <email>".')
    return value


def _get(url: str, agent: str) -> dict:
    """GET a JSON document, politely: identified, rate-limited, one at a time."""
    global _last_request
    headers = {"User-Agent": agent, "Accept-Encoding": "gzip, deflate"}
    with _lock:
        wait = _last_request + MIN_INTERVAL - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        try:
            response = httpx.get(url, headers=headers, timeout=TIMEOUT, follow_redirects=True)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise EdgarUnavailable(f"EDGAR request failed for {url}: {exc}") from exc
        finally:
            _last_request = time.monotonic()


def fetch_json(kind: str, url: str, key: str) -> dict:
    """A cached EDGAR document: from disk while fresh, else fetched and saved.

    Without a contact address EDGAR is off entirely, cached documents included,
    so behaviour doesn't depend on what happens to be on disk.
    """
    agent = user_agent()
    path = CACHE_DIR / f"{kind}-{key}.json"
    try:
        if time.time() - path.stat().st_mtime < CACHE_SECONDS[kind]:
            return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    data = _get(url, agent)
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")
    except OSError:
        pass    # the cache is an optimisation; a read-only disk shouldn't fail the fetch
    return data


# ---------------------------------------------------------------------------
# Companies
# ---------------------------------------------------------------------------

# Words that don't tell companies apart: "Microsoft Corporation" = "MICROSOFT CORP"
_NAME_NOISE = {"inc", "incorporated", "corp", "corporation", "co", "company", "ltd",
               "limited", "plc", "llc", "lp", "the", "holdings", "group", "sa", "nv", "ag"}


def name_key(name: str) -> str:
    words = re.findall(r"[a-z0-9]+", name.lower().replace("&", " and "))
    return " ".join(w for w in words if w not in _NAME_NOISE)


def tickers() -> list[dict]:
    """[{"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}, ...]"""
    return list(fetch_json("tickers", TICKERS_URL, "all").values())


def cik_for_ticker(ticker: str) -> int | None:
    wanted = ticker.strip().upper()
    return next((int(t["cik_str"]) for t in tickers() if t["ticker"] == wanted), None)


def cik_for_name(name: str | None) -> int | None:
    """The CIK registered under this company name, or None if none or several match.

    Several tickers (share classes) can share one CIK; that still counts as one.
    """
    key = name_key(name or "")
    if not key:
        return None
    ciks = {int(t["cik_str"]) for t in tickers() if name_key(t["title"]) == key}
    return ciks.pop() if len(ciks) == 1 else None


def companyfacts(cik: int) -> dict:
    return fetch_json("companyfacts", COMPANYFACTS_URL.format(cik=cik), str(cik))


def submissions(cik: int) -> dict:
    return fetch_json("submissions", SUBMISSIONS_URL.format(cik=cik), str(cik))


# ---------------------------------------------------------------------------
# Filings
# ---------------------------------------------------------------------------

ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"
MAX_DOCUMENT_BYTES = 40 * 1024 * 1024     # the largest 10-K HTML runs to ~15 MB


@dataclass(frozen=True)
class FilingRef:
    cik: int
    company: str
    form: str
    accession: str          # 0000320193-25-000079
    report_date: str        # the fiscal year end it reports: 2025-09-27
    filing_date: str
    document: str           # the primary document: aapl-20250927.htm

    @property
    def url(self) -> str:
        return ARCHIVE_URL.format(cik=self.cik, accession=self.accession.replace("-", ""),
                                  document=self.document)


def latest_annual_report(cik: int) -> FilingRef | None:
    """The company's most recent 10-K (amendments aside), or None if it has none."""
    data = submissions(cik)
    recent = data.get("filings", {}).get("recent", {})
    for i, form in enumerate(recent.get("form", [])):
        if form == "10-K" and recent["primaryDocument"][i].lower().endswith((".htm", ".html")):
            return FilingRef(
                cik=cik, company=data.get("name", ""), form=form,
                accession=recent["accessionNumber"][i], report_date=recent["reportDate"][i],
                filing_date=recent["filingDate"][i], document=recent["primaryDocument"][i])
    return None


def fetch_document(url: str) -> str:
    """A filing document's text, politely, refusing anything implausibly large."""
    global _last_request
    agent = user_agent()
    with _lock:
        wait = _last_request + MIN_INTERVAL - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        try:
            with httpx.stream("GET", url, headers={"User-Agent": agent}, timeout=120.0,
                              follow_redirects=True) as response:
                response.raise_for_status()
                body = bytearray()
                for chunk in response.iter_bytes():
                    body += chunk
                    if len(body) > MAX_DOCUMENT_BYTES:
                        raise EdgarUnavailable(f"{url} is larger than "
                                               f"{MAX_DOCUMENT_BYTES // 2**20} MB.")
        except httpx.HTTPError as exc:
            raise EdgarUnavailable(f"EDGAR request failed for {url}: {exc}") from exc
        finally:
            _last_request = time.monotonic()
    return bytes(body).decode("utf-8", errors="replace")
