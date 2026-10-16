"""
usage.py — Usage limits, so a public deployment can't run up an OpenAI bill.

Every question makes several gpt-4o-mini calls and every upload embeds a whole
filing, so each is counted before it runs:

  - per user, per calendar month (UTC): questions and uploads
  - per user, per minute: questions, so nobody can fire off a burst
  - across all users, per day (UTC): questions, because sign-up is free and
    per-user caps alone don't bound the total

Limits come from the environment (0 turns one off). The "local" user, who only
exists when login is disabled, is never limited. Counts live in SQLite next to
the vector store, so they survive restarts and rebuilds.

A request that fails (an API error, a PDF that won't ingest) is refunded: only
work that was done counts against the user.
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

USAGE_DB = Path("chroma_db") / "usage.sqlite3"
UNLIMITED_OWNER = "local"

QUESTION = "question"
UPLOAD = "upload"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id    INTEGER PRIMARY KEY AUTOINCREMENT,
    owner TEXT NOT NULL,
    kind  TEXT NOT NULL,
    at    REAL NOT NULL
)
"""
_INDEX = "CREATE INDEX IF NOT EXISTS events_by_owner ON events (owner, kind, at)"

# Check-then-record must be atomic, or two simultaneous requests could both
# pass a limit with one slot left. One process serves the API, so a lock does it.
_lock = threading.Lock()
_now = time.time     # the clock; tests replace it


def _limit(name: str, default: int) -> int:
    try:
        return max(0, int(os.getenv(name, default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class Limits:
    questions_per_month: int
    uploads_per_month: int
    questions_per_minute: int
    questions_per_day_all_users: int

    @classmethod
    def from_env(cls) -> Limits:
        return cls(
            questions_per_month=_limit("QUESTIONS_PER_MONTH", 50),
            uploads_per_month=_limit("UPLOADS_PER_MONTH", 3),
            questions_per_minute=_limit("QUESTIONS_PER_MINUTE", 5),
            questions_per_day_all_users=_limit("QUESTIONS_PER_DAY_ALL_USERS", 500),
        )


class LimitReached(Exception):
    """The request would go over a limit. The message is shown to the user."""

    def __init__(self, message: str, retry_after: int):
        super().__init__(message)
        self.retry_after = retry_after   # seconds until the limit resets


@dataclass(frozen=True)
class Usage:
    questions: int
    question_limit: int     # 0 = unlimited
    uploads: int
    upload_limit: int
    resets: str             # ISO date the monthly counts start again


def _connect() -> sqlite3.Connection:
    USAGE_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(USAGE_DB)
    conn.execute(_SCHEMA)
    conn.execute(_INDEX)
    return conn


def _month_start(now: float) -> datetime:
    day = datetime.fromtimestamp(now, timezone.utc)
    return day.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def _next_month(now: float) -> datetime:
    start = _month_start(now)
    return start.replace(year=start.year + start.month // 12, month=start.month % 12 + 1)


def _day_start(now: float) -> float:
    day = datetime.fromtimestamp(now, timezone.utc)
    return day.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()


def _count(conn: sqlite3.Connection, kind: str, since: float, owner: str | None = None) -> int:
    if owner is None:
        sql, args = "SELECT COUNT(*) FROM events WHERE kind = ? AND at >= ?", (kind, since)
    else:
        sql = "SELECT COUNT(*) FROM events WHERE owner = ? AND kind = ? AND at >= ?"
        args = (owner, kind, since)
    return conn.execute(sql, args).fetchone()[0]


def _check(conn: sqlite3.Connection, owner: str, kind: str, now: float, limits: Limits) -> None:
    month = _month_start(now).timestamp()
    until_next_month = int(_next_month(now).timestamp() - now) + 1
    next_month = _next_month(now)
    resets = f"{next_month:%B} {next_month.day}"

    if kind == UPLOAD:
        cap = limits.uploads_per_month
        if cap and _count(conn, UPLOAD, month, owner) >= cap:
            raise LimitReached(
                f"You've uploaded {cap} filings this month, the most allowed. "
                f"You can upload more from {resets}.", until_next_month)
        return

    cap = limits.questions_per_month
    if cap and _count(conn, QUESTION, month, owner) >= cap:
        raise LimitReached(
            f"You've asked {cap} questions this month, the most allowed. "
            f"You can ask more from {resets}.", until_next_month)
    cap = limits.questions_per_minute
    if cap:
        recent = conn.execute(
            "SELECT at FROM events WHERE owner = ? AND kind = ? AND at > ? ORDER BY at",
            (owner, QUESTION, now - 60),
        ).fetchall()
        if len(recent) >= cap:
            wait = int(recent[0][0] + 60 - now) + 1
            raise LimitReached(
                f"You're asking questions too quickly. Try again in {wait} seconds.", wait)
    cap = limits.questions_per_day_all_users
    if cap and _count(conn, QUESTION, _day_start(now)) >= cap:
        wait = int(_day_start(now) + 86_400 - now) + 1
        raise LimitReached(
            "The app has reached its daily question limit. Please try again tomorrow "
            "(the limit resets at midnight UTC).", wait)


def record(owner: str, kind: str, limits: Limits | None = None,
           now: float | None = None) -> int | None:
    """Count one question or upload, or raise LimitReached if it would go over.

    Returns an id to pass to refund() if the work then fails, or None for an
    unlimited user (nothing was recorded).
    """
    if owner == UNLIMITED_OWNER:
        return None
    limits = limits or Limits.from_env()
    now = _now() if now is None else now
    with _lock:
        conn = _connect()
        try:
            with conn:
                _check(conn, owner, kind, now, limits)
                cursor = conn.execute(
                    "INSERT INTO events (owner, kind, at) VALUES (?, ?, ?)", (owner, kind, now)
                )
            return cursor.lastrowid
        finally:
            conn.close()


def refund(event_id: int | None) -> None:
    """Un-count a recorded question or upload whose work failed."""
    if event_id is None:
        return
    with _lock:
        conn = _connect()
        try:
            with conn:
                conn.execute("DELETE FROM events WHERE id = ?", (event_id,))
        finally:
            conn.close()


def usage(owner: str, limits: Limits | None = None, now: float | None = None) -> Usage:
    """This month's counts for one user, with the limits they count against."""
    limits = limits or Limits.from_env()
    now = _now() if now is None else now
    unlimited = owner == UNLIMITED_OWNER
    conn = _connect()
    try:
        month = _month_start(now).timestamp()
        return Usage(
            questions=_count(conn, QUESTION, month, owner),
            question_limit=0 if unlimited else limits.questions_per_month,
            uploads=_count(conn, UPLOAD, month, owner),
            upload_limit=0 if unlimited else limits.uploads_per_month,
            resets=_next_month(now).date().isoformat(),
        )
    finally:
        conn.close()
