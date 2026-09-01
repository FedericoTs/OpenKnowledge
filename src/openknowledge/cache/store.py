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

from ..clock import ordered_now
from ..costs import Usage
from ..types import Answer, Citation, Tier

#: How much of a source document to keep as a pin's citation snippet.
_PIN_SNIPPET_CHARS = 280

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
    channel         TEXT,
    -- The answer covered part of the question and said the documents did not
    -- cover the rest. Not a refusal, and still a gap worth reporting.
    partial         INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_ledger_ts ON ledger(ts);
"""


def _add_missing_columns(conn: sqlite3.Connection) -> None:
    """Columns added after a database was first created.

    CREATE TABLE IF NOT EXISTS does nothing to a table that already exists, so
    a column added later never reaches an install that has been running. Every
    one is nullable or defaulted, so an old row keeps its meaning: a ledger
    entry written before `partial` existed is simply not known to be partial,
    which is exactly what it was.
    """
    known = {row[1] for row in conn.execute("PRAGMA table_info(ledger)")}
    if "partial" not in known:
        conn.execute("ALTER TABLE ledger ADD COLUMN partial INTEGER NOT NULL DEFAULT 0")
        conn.commit()


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
                "section": c.section,
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
            _add_missing_columns(self._conn)
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
        # Ordered, not merely current: this timestamp is compared against
        # conflict detection times to decide whether the pinner saw them.
        now = ordered_now()
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
                "INSERT INTO ledger"
                " (ts, canonical_query, tier, model_id, cost_usd, usage, channel, partial)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    time.time(),
                    canonical_query,
                    answer.tier.value,
                    answer.model_id,
                    answer.cost_usd,
                    _dump_usage(answer.usage),
                    channel,
                    int(answer.declined_in_part),
                ),
            )
            self._conn.commit()

    @staticmethod
    def now() -> float:
        """Wall clock, in one place so tests can move it."""
        return time.time()

    def spend_since(self, since: float) -> tuple[float, int]:
        """Dollars spent and questions answered since ``since``.

        One indexed aggregate, called on every question that reaches a model, so
        it has to stay a single row read rather than a scan. Counts free answers
        in the question total on purpose: the budget governor is pacing spend
        across expected *traffic*, and cache hits are traffic.
        """
        row = self._conn.execute(
            "SELECT COALESCE(SUM(cost_usd), 0.0) AS spend, COUNT(*) AS n FROM ledger WHERE ts >= ?",
            (since,),
        ).fetchone()
        return float(row["spend"] or 0.0), int(row["n"] or 0)

    def knowledge_gaps(
        self, *, since: float | None = None, limit: int = 50
    ) -> list[dict[str, object]]:
        """Questions the documents could not answer, most asked first.

        The refusal is this product's most useful output and until now it was
        the only one nobody kept. Every other tier leaves something behind - an
        answer, a cached entry, a line in the cost report - while "I don't know
        - that isn't covered by the documents I have" was said once and
        forgotten, so the person who owns the corpus never learned that eleven
        colleagues had asked about contractor notice periods that month.

        Read from the ledger, which records every final answer with its tier
        and carries no identity at all: this can say a question was asked
        forty times and can never say by whom. That is a property worth
        keeping rather than a limitation to work around - a knowledge base
        that reports what its people are looking for should not also be a log
        of who looked.

        Grouped by the canonical query, so "how much parental leave" and "How
        much parental leave?" are one gap rather than two.
        """
        # What makes a row a gap: the ask left something unanswered. A refusal
        # leaves everything unanswered; a partial answer leaves the part it
        # named. Both belong here, and until this they did not - when a partial
        # decline stopped being a refusal (v0.2.17) the gap it names went with
        # it, silently, which is the worst way for a report like this to be
        # wrong.
        unanswered = "(l.tier = 'refused' OR l.partial = 1)"
        where = f"WHERE {unanswered}"
        params: tuple[object, ...] = ()
        if since is not None:
            where += " AND l.ts >= ?"
            params = (since,)

        # A gap that has been dealt with has to leave the list, or the person
        # working through it does the work and watches nothing happen. Two
        # ways it can be dealt with, and both are read from what already
        # happened rather than asked of a model:
        #
        # * somebody pinned an answer - immediate, because that is the click
        #   this report exists to prompt;
        # * somebody wrote the document, and the question has since been
        #   answered - so the most recent time it was asked was not a refusal.
        #
        # The count stays the count of refusals: the question is how much
        # this gap cost, not how often it has been asked since.
        rows = self._conn.execute(
            "SELECT l.canonical_query, COUNT(*) AS asked,"
            "       SUM(l.partial) AS in_part, MAX(l.ts) AS last_asked,"
            "       (SELECT later.partial FROM ledger later"
            "         WHERE later.canonical_query = l.canonical_query"
            "         ORDER BY later.ts DESC, later.id DESC LIMIT 1) AS latest_partial"
            f" FROM ledger l {where}"
            "   AND NOT EXISTS ("
            "     SELECT 1 FROM pinned_answers p"
            "     WHERE p.canonical_query = l.canonical_query AND p.enabled = 1)"
            "   AND ("
            "     SELECT (later.tier = 'refused' OR later.partial = 1)"
            "     FROM ledger later"
            "     WHERE later.canonical_query = l.canonical_query"
            "     ORDER BY later.ts DESC, later.id DESC LIMIT 1) = 1"
            " GROUP BY l.canonical_query"
            " ORDER BY asked DESC, last_asked DESC LIMIT ?",
            (*params, limit),
        ).fetchall()
        return [
            {
                "question": r["canonical_query"],
                "asked": int(r["asked"]),
                # How many of those asks got half an answer rather than none,
                # and which of the two the last one was. "We answer this
                # halfway every time" is a different job from "we cannot
                # answer it at all", and the two should not look alike.
                "answered_in_part": int(r["in_part"] or 0),
                "kind": "partial" if r["latest_partial"] else "refused",
                "last_asked": float(r["last_asked"]),
            }
            for r in rows
        ]

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
            "by_model": self._spend_by_model(where, params),
        }

    def _spend_by_model(self, where: str, params: tuple[float, ...]) -> dict[str, object]:
        """Per-model breakdown, which is how an escalation ladder is read.

        Tiers answer "was it free"; models answer "which rung answered, and what
        did that rung cost" - the question a ladder exists to let you tune.
        """
        rows = self._conn.execute(
            f"SELECT model_id, COUNT(*) AS n, SUM(cost_usd) AS spend FROM ledger {where} "
            "GROUP BY model_id ORDER BY spend DESC",
            params,
        ).fetchall()
        return {
            r["model_id"]: {"questions": r["n"], "spend_usd": round(r["spend"] or 0.0, 6)}
            for r in rows
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


def citations_for(retriever: object, document_ids: tuple[str, ...]) -> tuple[Citation, ...]:
    """Build citations for ``document_ids`` using an indexed retriever.

    An id that is not in the corpus still produces a citation, with the id as its
    title. Dropping it silently would let an admin pin an answer that claims a
    source which does not exist - and the grounding rules that apply to models
    should apply to people too.
    """
    describe = getattr(retriever, "describe_document", None)
    citations: list[Citation] = []
    for doc_id in document_ids:
        found = describe(doc_id) if callable(describe) else None
        if found is None:
            citations.append(
                Citation(
                    document_id=doc_id,
                    document_title=doc_id,
                    snippet="(not currently in the indexed corpus)",
                )
            )
            continue
        title, text = found
        snippet = text[:_PIN_SNIPPET_CHARS] + ("..." if len(text) > _PIN_SNIPPET_CHARS else "")
        citations.append(Citation(document_id=doc_id, document_title=title, snippet=snippet))
    return tuple(citations)
