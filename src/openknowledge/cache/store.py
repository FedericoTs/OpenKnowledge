"""SQLite-backed answer store, pin registry, and cost ledger.

SQLite because the whole product has to survive `docker compose up` on an IT
department's spare VM. Postgres is a swap-in later (see ADR 0003); nothing here
depends on SQLite beyond the connection.

Three tables, three jobs:

``pinned_answers``
    Human-authored answers to the questions that actually repeat. This is the
    only tier that is deterministic *by construction* rather than by caching -
    an admin wrote the text, so it is exactly right and it never drifts.

``answer_cache``
    Answers a model produced, keyed by :func:`~openknowledge.cache.keys.answer_key`.

``ledger``
    One row per answered question, including the free ones. Without the free
    ones you cannot compute a blended cost per question, which is the number
    this project lives or dies on.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from ..costs import Usage
from ..types import Answer, Citation, Tier

_SCHEMA = """
CREATE TABLE IF NOT EXISTS pinned_answers (
    canonical_query TEXT PRIMARY KEY,
    answer          TEXT NOT NULL,
    citations       TEXT NOT NULL DEFAULT '[]',
    author          TEXT,
    updated_at      REAL NOT NULL,
    enabled         INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS answer_cache (
    cache_key       TEXT PRIMARY KEY,
    canonical_query TEXT NOT NULL,
    answer          TEXT NOT NULL,
    citations       TEXT NOT NULL DEFAULT '[]',
    tier            TEXT NOT NULL,
    model_id        TEXT NOT NULL,
    corpus_version  TEXT NOT NULL,
    usage           TEXT NOT NULL DEFAULT '{}',
    cost_usd        REAL NOT NULL DEFAULT 0.0,
    created_at      REAL NOT NULL,
    last_hit_at     REAL,
    hits            INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_cache_corpus ON answer_cache(corpus_version);
CREATE INDEX IF NOT EXISTS idx_cache_query  ON answer_cache(canonical_query);

CREATE TABLE IF NOT EXISTS ledger (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              REAL NOT NULL,
    canonical_query TEXT NOT NULL,
    tier            TEXT NOT NULL,
    model_id        TEXT NOT NULL,
    cost_usd        REAL NOT NULL DEFAULT 0.0,
    usage           TEXT NOT NULL DEFAULT '{}',
    channel         TEXT
);
CREATE INDEX IF NOT EXISTS idx_ledger_ts ON ledger(ts);
"""


@dataclass(frozen=True, slots=True)
class PinnedAnswer:
    canonical_query: str
    answer: str
    citations: tuple[Citation, ...] = ()
    author: str | None = None
    updated_at: float = 0.0
    enabled: bool = True


@dataclass(frozen=True, slots=True)
class LedgerEntry:
    ts: float
    canonical_query: str
    tier: Tier
    model_id: str
    cost_usd: float
    channel: str | None


@dataclass(frozen=True, slots=True)
class CacheEntry:
    cache_key: str
    canonical_query: str
    answer: str
    citations: tuple[Citation, ...]
    tier: Tier
    model_id: str
    corpus_version: str
    usage: Usage
    cost_usd: float
    hits: int


def _dump_citations(citations: tuple[Citation, ...]) -> str:
    return json.dumps(
        [
            {
                "document_id": c.document_id,
                "document_title": c.document_title,
                "snippet": c.snippet,
                "locator": c.locator,
                "url": c.url,
            }
            for c in citations
        ],
        sort_keys=True,
    )


def _load_citations(blob: str) -> tuple[Citation, ...]:
    return tuple(Citation(**item) for item in json.loads(blob or "[]"))


def _dump_usage(usage: Usage) -> str:
    return json.dumps(
        {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cache_read_tokens": usage.cache_read_tokens,
            "cache_write_tokens": usage.cache_write_tokens,
            "batch": usage.batch,
        },
        sort_keys=True,
    )


def _load_usage(blob: str) -> Usage:
    return Usage(**json.loads(blob or "{}"))


class AnswerStore:
    """Pins, cached answers, and the cost ledger over one SQLite file."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False + an explicit lock: FastAPI serves requests on
        # a thread pool, and SQLite is happy to be shared as long as we
        # serialise writes ourselves.
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> AnswerStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- pins -------------------------------------------------------------
    def pin(
        self,
        canonical_query: str,
        answer: str,
        *,
        citations: tuple[Citation, ...] = (),
        author: str | None = None,
    ) -> PinnedAnswer:
        """Record a human-authored answer. Overwrites any existing pin."""
        now = time.time()
        with self._lock:
            self._conn.execute(
                "INSERT INTO pinned_answers"
                " (canonical_query, answer, citations, author, updated_at, enabled)"
                " VALUES (?, ?, ?, ?, ?, 1)"
                " ON CONFLICT(canonical_query) DO UPDATE SET"
                "   answer=excluded.answer, citations=excluded.citations,"
                "   author=excluded.author, updated_at=excluded.updated_at, enabled=1",
                (canonical_query, answer, _dump_citations(citations), author, now),
            )
            self._conn.commit()
        return PinnedAnswer(canonical_query, answer, citations, author, now, True)

    def unpin(self, canonical_query: str) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM pinned_answers WHERE canonical_query = ?", (canonical_query,)
            )
            self._conn.commit()
        return cur.rowcount > 0

    def get_pin(self, canonical_query: str) -> PinnedAnswer | None:
        row = self._conn.execute(
            "SELECT * FROM pinned_answers WHERE canonical_query = ? AND enabled = 1",
            (canonical_query,),
        ).fetchone()
        if row is None:
            return None
        return PinnedAnswer(
            canonical_query=row["canonical_query"],
            answer=row["answer"],
            citations=_load_citations(row["citations"]),
            author=row["author"],
            updated_at=row["updated_at"],
            enabled=bool(row["enabled"]),
        )

    def list_pins(self) -> list[PinnedAnswer]:
        rows = self._conn.execute(
            "SELECT * FROM pinned_answers ORDER BY canonical_query"
        ).fetchall()
        return [
            PinnedAnswer(
                canonical_query=r["canonical_query"],
                answer=r["answer"],
                citations=_load_citations(r["citations"]),
                author=r["author"],
                updated_at=r["updated_at"],
                enabled=bool(r["enabled"]),
            )
            for r in rows
        ]

    # -- cached answers ---------------------------------------------------
    def get(self, cache_key: str) -> CacheEntry | None:
        """Fetch a cached answer and count the hit."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM answer_cache WHERE cache_key = ?", (cache_key,)
            ).fetchone()
            if row is None:
                return None
            self._conn.execute(
                "UPDATE answer_cache SET hits = hits + 1, last_hit_at = ? WHERE cache_key = ?",
                (time.time(), cache_key),
            )
            self._conn.commit()
        return CacheEntry(
            cache_key=row["cache_key"],
            canonical_query=row["canonical_query"],
            answer=row["answer"],
            citations=_load_citations(row["citations"]),
            tier=Tier(row["tier"]),
            model_id=row["model_id"],
            corpus_version=row["corpus_version"],
            usage=_load_usage(row["usage"]),
            cost_usd=row["cost_usd"],
            hits=row["hits"] + 1,
        )

    def put(
        self, cache_key: str, canonical_query: str, answer: Answer, corpus_version: str
    ) -> None:
        """Store an answer. Refused answers are never cached."""
        if answer.tier is Tier.REFUSED:
            return
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO answer_cache"
                " (cache_key, canonical_query, answer, citations, tier, model_id,"
                "  corpus_version, usage, cost_usd, created_at, last_hit_at, hits)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, 0)",
                (
                    cache_key,
                    canonical_query,
                    answer.text,
                    _dump_citations(answer.citations),
                    answer.tier.value,
                    answer.model_id,
                    corpus_version,
                    _dump_usage(answer.usage),
                    answer.cost_usd,
                    time.time(),
                ),
            )
            self._conn.commit()

    def evict_other_corpus_versions(self, current: str) -> int:
        """Drop answers derived from a superseded corpus.

        Not required for correctness - those entries are already unreachable,
        because ``corpus_version`` is part of the key - but it keeps the file
        from growing without bound after every document re-sync.
        """
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM answer_cache WHERE corpus_version != ?", (current,)
            )
            self._conn.commit()
        return cur.rowcount

    # -- ledger -----------------------------------------------------------
    def record(self, canonical_query: str, answer: Answer, *, channel: str | None = None) -> None:
        """Append one answered question to the ledger, free or not."""
        with self._lock:
            self._conn.execute(
                "INSERT INTO ledger (ts, canonical_query, tier, model_id, cost_usd, usage, channel)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    time.time(),
                    canonical_query,
                    answer.tier.value,
                    answer.model_id,
                    answer.cost_usd,
                    _dump_usage(answer.usage),
                    channel,
                ),
            )
            self._conn.commit()

    def cost_report(self, since: float | None = None) -> dict[str, object]:
        """Blended cost per question, broken down by tier."""
        where, params = ("WHERE ts >= ?", (since,)) if since is not None else ("", ())
        rows = self._conn.execute(
            f"SELECT tier, COUNT(*) AS n, SUM(cost_usd) AS spend FROM ledger {where} GROUP BY tier",
            params,
        ).fetchall()

        by_tier = {
            r["tier"]: {"questions": r["n"], "spend_usd": round(r["spend"] or 0.0, 6)} for r in rows
        }
        total_questions = sum(v["questions"] for v in by_tier.values())
        total_spend = sum(v["spend_usd"] for v in by_tier.values())
        return {
            "questions": total_questions,
            "spend_usd": round(total_spend, 6),
            "cost_per_question_usd": round(total_spend / total_questions, 6)
            if total_questions
            else 0.0,
            "by_tier": by_tier,
        }

    def recent_questions(self, limit: int = 50) -> list[LedgerEntry]:
        """Most recent answered questions, newest first."""
        rows = self._conn.execute(
            "SELECT ts, canonical_query, tier, model_id, cost_usd, channel"
            " FROM ledger ORDER BY ts DESC, id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            LedgerEntry(
                ts=r["ts"],
                canonical_query=r["canonical_query"],
                tier=Tier(r["tier"]),
                model_id=r["model_id"],
                cost_usd=r["cost_usd"],
                channel=r["channel"],
            )
            for r in rows
        ]

    def top_questions(self, limit: int = 20) -> list[tuple[str, int]]:
        """Most-asked questions - the shortlist an admin should pin."""
        rows = self._conn.execute(
            "SELECT canonical_query, COUNT(*) AS n FROM ledger"
            " GROUP BY canonical_query ORDER BY n DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [(r["canonical_query"], r["n"]) for r in rows]
