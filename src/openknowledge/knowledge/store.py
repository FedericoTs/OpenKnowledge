"""Storage for drafted FAQ answers and detected conflicts.

Two review queues, both designed around one assumption: **nobody reviews three
thousand items**. A feature that dumps every machine-generated Q&A pair into a
list and calls it "validation" gets rubber-stamped, and rubber-stamped machine
output sitting in the highest-trust tier is worse than no feature at all.

So proposals carry a ``priority`` - what approving this is actually worth - and
the queue is ranked by it. Reviewing the top fifty is a morning's work and
captures most of the value. The rest can wait, or never happen, and the system
still works because unreviewed drafts are held to the grounding gate rather than
to a human's attention.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from ..clock import ordered_now
from ..retrieval.base import Document
from ..types import Citation
from .claims import Conflict

_SCHEMA = """
CREATE TABLE IF NOT EXISTS proposals (
    id               TEXT PRIMARY KEY,
    canonical_query  TEXT NOT NULL,
    question         TEXT NOT NULL,
    answer           TEXT NOT NULL,
    citations        TEXT NOT NULL DEFAULT '[]',
    status           TEXT NOT NULL DEFAULT 'draft',
    source           TEXT NOT NULL DEFAULT 'ingest',
    origin_documents TEXT NOT NULL DEFAULT '[]',
    corpus_version   TEXT NOT NULL,
    support_ratio    REAL NOT NULL DEFAULT 0.0,
    priority         REAL NOT NULL DEFAULT 0.0,
    created_at       REAL NOT NULL,
    reviewed_at      REAL,
    reviewer         TEXT,
    review_note      TEXT,
    supersedes       TEXT
);
CREATE INDEX IF NOT EXISTS idx_prop_status ON proposals(status, priority DESC);
CREATE INDEX IF NOT EXISTS idx_prop_query  ON proposals(canonical_query, status);

CREATE TABLE IF NOT EXISTS conflicts (
    key             TEXT PRIMARY KEY,
    left_document   TEXT NOT NULL,
    left_raw        TEXT NOT NULL,
    left_sentence   TEXT NOT NULL,
    right_document  TEXT NOT NULL,
    right_raw       TEXT NOT NULL,
    right_sentence  TEXT NOT NULL,
    unit            TEXT NOT NULL DEFAULT '',
    kind            TEXT NOT NULL DEFAULT 'numeric',
    overlap         REAL NOT NULL DEFAULT 0.0,
    context         TEXT NOT NULL DEFAULT '[]',
    status          TEXT NOT NULL DEFAULT 'open',
    resolution      TEXT,
    detected_at     REAL NOT NULL,
    resolved_at     REAL,
    resolver        TEXT
);
CREATE INDEX IF NOT EXISTS idx_conflict_status ON conflicts(status);

CREATE TABLE IF NOT EXISTS document_versions (
    document_id  TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,
    title        TEXT NOT NULL DEFAULT '',
    first_seen   REAL NOT NULL,
    updated_at   REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS folder_access (
    folder     TEXT PRIMARY KEY,
    principals TEXT NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS answer_reports (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical_query TEXT NOT NULL,
    question        TEXT NOT NULL,
    answer          TEXT NOT NULL,
    answer_hash     TEXT NOT NULL,
    tier            TEXT NOT NULL DEFAULT '',
    citations       TEXT NOT NULL DEFAULT '[]',
    corpus_version  TEXT NOT NULL DEFAULT '',
    notes           TEXT NOT NULL DEFAULT '[]',
    reports         INTEGER NOT NULL DEFAULT 1,
    first_at        REAL NOT NULL,
    last_at         REAL NOT NULL,
    status          TEXT NOT NULL DEFAULT 'open',
    resolved_at     REAL,
    resolution      TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_report_answer
    ON answer_reports(canonical_query, answer_hash);
CREATE INDEX IF NOT EXISTS idx_report_status ON answer_reports(status);

CREATE TABLE IF NOT EXISTS admin_actions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    at         REAL NOT NULL,
    actor      TEXT NOT NULL,
    actor_name TEXT NOT NULL,
    actor_kind TEXT NOT NULL,
    action     TEXT NOT NULL,
    target     TEXT NOT NULL DEFAULT '',
    detail     TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_admin_actions_at ON admin_actions(at);
"""


class ProposalStatus(StrEnum):
    DRAFT = "draft"  # gate-passed, no human has looked at it
    APPROVED = "approved"  # a human signed off; also written as a pin
    REJECTED = "rejected"  # a human said no; never offered again
    SUPERSEDED = "superseded"  # the source documents changed underneath it


@dataclass(frozen=True, slots=True)
class Proposal:
    id: str
    canonical_query: str
    question: str
    answer: str
    citations: tuple[Citation, ...]
    status: ProposalStatus
    source: str
    origin_documents: tuple[str, ...]
    corpus_version: str
    support_ratio: float
    priority: float
    created_at: float
    reviewed_at: float | None = None
    reviewer: str | None = None
    review_note: str | None = None
    #: Id of an approved proposal this draft would replace, if any.
    supersedes: str | None = None


@dataclass(frozen=True, slots=True)
class StoredConflict:
    key: str
    left_document: str
    left_raw: str
    left_sentence: str
    right_document: str
    right_raw: str
    right_sentence: str
    unit: str
    kind: str
    overlap: float
    context: frozenset[str]
    status: str
    resolution: str | None = None
    detected_at: float = 0.0
    resolved_at: float | None = None
    resolver: str | None = None

    @property
    def documents(self) -> frozenset[str]:
        return frozenset({self.left_document, self.right_document})

    def describe(self) -> str:
        return (
            f"[{self.left_document}] says {self.left_raw}, "
            f"[{self.right_document}] says {self.right_raw}"
        )


#: How many notes one report keeps. A wrong answer somebody bothered to
#: explain is worth reading; the twenty-first person saying "still wrong"
#: adds a count, not a paragraph.
MAX_REPORT_NOTES = 20


@dataclass(frozen=True, slots=True)
class AnswerReport:
    """A reader said an answer was wrong.

    The one signal this product could not previously collect. A refusal
    leaves a trace in the gaps report; an answer that was confidently wrong
    left nothing at all, because it looked exactly like an answer that was
    right. Somebody noticed, told a colleague, and the corpus never heard.

    Carries no identity, deliberately - the same rule the gaps report
    follows. What is useful is *which answer* is wrong and *why somebody
    thinks so*, and a knowledge base that reports what its people got wrong
    should not also be a record of who complained.
    """

    id: int
    canonical_query: str
    question: str
    answer: str
    tier: str
    citations: tuple[Citation, ...]
    corpus_version: str
    notes: tuple[str, ...]
    reports: int
    first_at: float
    last_at: float
    status: str
    resolved_at: float | None = None
    resolution: str | None = None


@dataclass(frozen=True, slots=True)
class Actor:
    """Who performed an admin action, as far as the server can honestly tell.

    ``kind`` is the part that matters when reading the log back. A ``person``
    signed in through the directory and is named by their stable subject id,
    so the row survives a rename and points at an account someone can
    disable. A ``token`` is the shared admin token: it says an authorised
    caller did this and nothing more, because a shared secret cannot
    distinguish the people holding it. The log does not pretend otherwise -
    it records ``token`` and lets the reader draw the obvious conclusion
    about turning sign-in on.
    """

    id: str
    name: str
    kind: str  # 'person' | 'token' | 'console'

    @staticmethod
    def token() -> Actor:
        return Actor(id="admin-token", name="shared admin token", kind="token")

    @staticmethod
    def console() -> Actor:
        """Someone at the server's own command line. Attributed no further:
        reaching the CLI already means holding the machine."""
        return Actor(id="console", name="the server console", kind="console")


@dataclass(frozen=True, slots=True)
class AdminAction:
    """One entry in the admin log."""

    id: int
    at: float
    actor: Actor
    action: str
    target: str
    detail: dict[str, object]


def proposal_id(canonical_query: str, origin_documents: tuple[str, ...], variant: str = "") -> str:
    """Stable id, so re-running a draft updates rather than duplicates.

    ``variant`` distinguishes a re-verified answer from the approved one it
    would replace: without it the new draft would collide with the approval and
    be discarded, which is exactly the update an admin needs to see.
    """
    payload = "\x1f".join([canonical_query, *sorted(origin_documents), variant])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


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


class KnowledgeStore:
    """Proposals and conflicts over one SQLite file."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> KnowledgeStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- proposals --------------------------------------------------------
    def propose(
        self,
        *,
        canonical_query: str,
        question: str,
        answer: str,
        citations: tuple[Citation, ...],
        origin_documents: tuple[str, ...],
        corpus_version: str,
        support_ratio: float = 0.0,
        priority: float = 0.0,
        source: str = "ingest",
        variant: str = "",
        supersedes: str | None = None,
    ) -> Proposal | None:
        """Record a drafted answer.

        Returns ``None`` if this question was already reviewed - a rejected
        proposal must not reappear on the next ingest run, or the queue becomes
        a treadmill and the reviewer stops trusting it.
        """
        pid = proposal_id(canonical_query, origin_documents, variant)
        now = time.time()

        with self._lock:
            existing = self._conn.execute(
                "SELECT status FROM proposals WHERE id = ?", (pid,)
            ).fetchone()
            if existing is not None and existing["status"] in (
                ProposalStatus.APPROVED,
                ProposalStatus.REJECTED,
            ):
                return None

            self._conn.execute(
                "INSERT OR REPLACE INTO proposals"
                " (id, canonical_query, question, answer, citations, status, source,"
                "  origin_documents, corpus_version, support_ratio, priority, created_at,"
                "  supersedes)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    pid,
                    canonical_query,
                    question,
                    answer,
                    _dump_citations(citations),
                    ProposalStatus.DRAFT,
                    source,
                    json.dumps(sorted(origin_documents)),
                    corpus_version,
                    support_ratio,
                    priority,
                    now,
                    supersedes,
                ),
            )
            self._conn.commit()

        return Proposal(
            id=pid,
            canonical_query=canonical_query,
            question=question,
            answer=answer,
            citations=citations,
            status=ProposalStatus.DRAFT,
            source=source,
            origin_documents=tuple(sorted(origin_documents)),
            corpus_version=corpus_version,
            support_ratio=support_ratio,
            priority=priority,
            created_at=now,
            supersedes=supersedes,
        )

    def _row_to_proposal(self, row: sqlite3.Row) -> Proposal:
        return Proposal(
            id=row["id"],
            canonical_query=row["canonical_query"],
            question=row["question"],
            answer=row["answer"],
            citations=_load_citations(row["citations"]),
            status=ProposalStatus(row["status"]),
            source=row["source"],
            origin_documents=tuple(json.loads(row["origin_documents"])),
            corpus_version=row["corpus_version"],
            support_ratio=row["support_ratio"],
            priority=row["priority"],
            created_at=row["created_at"],
            reviewed_at=row["reviewed_at"],
            reviewer=row["reviewer"],
            review_note=row["review_note"],
            supersedes=row["supersedes"],
        )

    def get(self, proposal_id_: str) -> Proposal | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM proposals WHERE id = ?", (proposal_id_,)
            ).fetchone()
            return self._row_to_proposal(row) if row else None

    def draft_for(self, canonical_query: str) -> Proposal | None:
        """The servable draft for a question, if there is one."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM proposals WHERE canonical_query = ? AND status = ?"
                " ORDER BY support_ratio DESC LIMIT 1",
                (canonical_query, ProposalStatus.DRAFT),
            ).fetchone()
            return self._row_to_proposal(row) if row else None

    def pending(self, limit: int = 50) -> list[Proposal]:
        """Drafts awaiting review, most valuable first."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM proposals WHERE status = ?"
                " ORDER BY priority DESC, support_ratio DESC, created_at ASC LIMIT ?",
                (ProposalStatus.DRAFT, limit),
            ).fetchall()
            return [self._row_to_proposal(r) for r in rows]

    def _review(
        self, proposal_id_: str, status: ProposalStatus, reviewer: str | None, note: str | None
    ) -> Proposal | None:
        with self._lock:
            cur = self._conn.execute(
                "UPDATE proposals SET status = ?, reviewed_at = ?, reviewer = ?, review_note = ?"
                " WHERE id = ? AND status = ?",
                (status, time.time(), reviewer, note, proposal_id_, ProposalStatus.DRAFT),
            )
            self._conn.commit()
        return self.get(proposal_id_) if cur.rowcount else None

    def approve(
        self, proposal_id_: str, *, reviewer: str | None = None, note: str | None = None
    ) -> Proposal | None:
        return self._review(proposal_id_, ProposalStatus.APPROVED, reviewer, note)

    def reject(
        self, proposal_id_: str, *, reviewer: str | None = None, note: str | None = None
    ) -> Proposal | None:
        return self._review(proposal_id_, ProposalStatus.REJECTED, reviewer, note)

    def supersede_for_documents(self, document_ids: frozenset[str]) -> list[Proposal]:
        """Mark drafts stale when a document they came from has changed.

        Approved proposals are left alone here - those went through a human, and
        silently retiring them would erase a decision. They are handled by
        re-verification instead, which asks the reviewer about the specific
        claim that moved.
        """
        stale: list[Proposal] = []
        for proposal in self._all(ProposalStatus.DRAFT):
            if document_ids & set(proposal.origin_documents):
                stale.append(proposal)
        with self._lock:
            for proposal in stale:
                self._conn.execute(
                    "UPDATE proposals SET status = ? WHERE id = ?",
                    (ProposalStatus.SUPERSEDED, proposal.id),
                )
            self._conn.commit()
        return stale

    def _all(self, status: ProposalStatus | None = None) -> list[Proposal]:
        with self._lock:
            if status is None:
                rows = self._conn.execute("SELECT * FROM proposals").fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM proposals WHERE status = ?", (status,)
                ).fetchall()
            return [self._row_to_proposal(r) for r in rows]

    def all_proposals(self, status: ProposalStatus | None = None) -> list[Proposal]:
        """Every proposal, optionally filtered by status."""
        return self._all(status)

    def approved_citing(self, document_id: str) -> list[Proposal]:
        """Approved answers that cite a document - the ones to re-check when it changes.

        This is what keeps contradiction detection affordable: on a document
        update we revisit only the answers that actually depend on it, rather
        than comparing the new document against the whole corpus.
        """
        return [
            p
            for p in self._all(ProposalStatus.APPROVED)
            if document_id in {c.document_id for c in p.citations}
            or document_id in p.origin_documents
        ]

    def counts(self) -> dict[str, int]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT status, COUNT(*) AS n FROM proposals GROUP BY status"
            ).fetchall()
            return {r["status"]: r["n"] for r in rows}

    # -- conflicts --------------------------------------------------------
    def record_conflict(self, conflict: Conflict) -> StoredConflict:
        """Store a detected conflict, preserving any existing resolution.

        Re-detecting an already-resolved conflict must not reopen it, or every
        re-index would undo the admin's decisions.
        """
        shared = sorted(conflict.left.context & conflict.right.context)
        # Ordered, not merely current: detected_at is compared against pin
        # update times to decide whether a pin predates this disagreement.
        now = ordered_now()
        with self._lock:
            existing = self._conn.execute(
                "SELECT * FROM conflicts WHERE key = ?", (conflict.key,)
            ).fetchone()
            if existing is not None:
                return self._row_to_conflict(existing)

            self._conn.execute(
                "INSERT INTO conflicts"
                " (key, left_document, left_raw, left_sentence, right_document, right_raw,"
                "  right_sentence, unit, kind, overlap, context, status, detected_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', ?)",
                (
                    conflict.key,
                    conflict.left.document_id,
                    conflict.left.raw,
                    conflict.left.sentence,
                    conflict.right.document_id,
                    conflict.right.raw,
                    conflict.right.sentence,
                    conflict.left.unit,
                    conflict.kind,
                    conflict.overlap,
                    json.dumps(shared),
                    now,
                ),
            )
            self._conn.commit()
            row = self._conn.execute(
                "SELECT * FROM conflicts WHERE key = ?", (conflict.key,)
            ).fetchone()
        return self._row_to_conflict(row)

    def _row_to_conflict(self, row: sqlite3.Row) -> StoredConflict:
        return StoredConflict(
            key=row["key"],
            left_document=row["left_document"],
            left_raw=row["left_raw"],
            left_sentence=row["left_sentence"],
            right_document=row["right_document"],
            right_raw=row["right_raw"],
            right_sentence=row["right_sentence"],
            unit=row["unit"],
            kind=row["kind"],
            overlap=row["overlap"],
            context=frozenset(json.loads(row["context"])),
            status=row["status"],
            resolution=row["resolution"],
            detected_at=row["detected_at"],
            resolved_at=row["resolved_at"],
            resolver=row["resolver"],
        )

    def open_conflicts(self) -> list[StoredConflict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM conflicts WHERE status = 'open' ORDER BY overlap DESC, key"
            ).fetchall()
            return [self._row_to_conflict(r) for r in rows]

    def resolve_conflict(
        self, key: str, *, resolution: str, resolver: str | None = None
    ) -> StoredConflict | None:
        with self._lock:
            cur = self._conn.execute(
                "UPDATE conflicts SET status = 'resolved', resolution = ?, resolved_at = ?,"
                " resolver = ? WHERE key = ? AND status = 'open'",
                (resolution, time.time(), resolver, key),
            )
            self._conn.commit()
            if not cur.rowcount:
                return None
            row = self._conn.execute("SELECT * FROM conflicts WHERE key = ?", (key,)).fetchone()
        return self._row_to_conflict(row)

    def drop_conflicts_for_documents(self, present: frozenset[str]) -> int:
        """Forget conflicts whose documents are no longer both in the corpus.

        Deleting one side of a disagreement resolves it, and leaving the flag up
        would block answers for a contradiction that no longer exists.
        """
        removed = 0
        with self._lock:
            for row in self._conn.execute(
                "SELECT key, left_document, right_document FROM conflicts"
            ).fetchall():
                if not {row["left_document"], row["right_document"]} <= present:
                    self._conn.execute("DELETE FROM conflicts WHERE key = ?", (row["key"],))
                    removed += 1
            self._conn.commit()
        return removed

    def drop_open_conflicts_absent_from(self, detected: frozenset[str]) -> int:
        """Close the flag on disagreements this corpus no longer contains.

        A contradiction is resolved by deleting one of the documents, and
        equally by *correcting the text* - and only the first was ever handled.
        Fixing the figure left the flag up for good, so with
        ``block_on_conflict`` on, every question it gated stayed refused after
        the corpus was already right, and the only way out was an admin
        resolving something that no longer existed.

        Only open rows. A resolved one is the record of a decision somebody
        made, it gates nothing, and deleting it to tidy up would erase the
        history the admin log points at.

        The caller passes what a scan of the **whole** corpus detected; giving
        it a subset would clear flags for documents it never looked at.
        """
        with self._lock:
            rows = self._conn.execute("SELECT key FROM conflicts WHERE status = 'open'").fetchall()
            stale = [row["key"] for row in rows if row["key"] not in detected]
            for key in stale:
                self._conn.execute("DELETE FROM conflicts WHERE key = ?", (key,))
            self._conn.commit()
        return len(stale)

    # -- document versions ------------------------------------------------
    def sync_documents(
        self, documents: list[Document]
    ) -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
        """Record the corpus and report ``(added, changed, removed)``.

        This is what keeps ingest-time work one-off. Without it, every re-index
        would re-draft the whole corpus, and the cost argument for moving work
        to upload time would quietly reverse.
        """
        now = time.time()
        incoming = {d.document_id: d for d in documents}

        with self._lock:
            known = {
                row["document_id"]: row["content_hash"]
                for row in self._conn.execute(
                    "SELECT document_id, content_hash FROM document_versions"
                ).fetchall()
            }

        added = frozenset(incoming) - frozenset(known)
        removed = frozenset(known) - frozenset(incoming)
        changed = frozenset(
            doc_id
            for doc_id, doc in incoming.items()
            if doc_id in known and known[doc_id] != doc.content_hash
        )

        with self._lock:
            for doc_id, doc in incoming.items():
                self._conn.execute(
                    "INSERT INTO document_versions"
                    " (document_id, content_hash, title, first_seen, updated_at)"
                    " VALUES (?, ?, ?, ?, ?)"
                    " ON CONFLICT(document_id) DO UPDATE SET"
                    "   content_hash=excluded.content_hash, title=excluded.title,"
                    "   updated_at=excluded.updated_at",
                    (doc_id, doc.content_hash, doc.title, now, now),
                )
            for doc_id in removed:
                self._conn.execute("DELETE FROM document_versions WHERE document_id = ?", (doc_id,))
            self._conn.commit()

        return added, changed, removed

    # -- folder access -----------------------------------------------------
    # Who may read which folder. Kept here with the other human decisions
    # (approvals, resolutions): an access rule must survive every re-index,
    # because the index is disposable and the decision is not.

    def folder_rules(self) -> dict[str, frozenset[str]]:
        with self._lock:
            return {
                row["folder"]: frozenset(json.loads(row["principals"]))
                for row in self._conn.execute(
                    "SELECT folder, principals FROM folder_access"
                ).fetchall()
            }

    def set_folder_access(self, folder: str, principals: frozenset[str]) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO folder_access (folder, principals, updated_at)"
                " VALUES (?, ?, ?)"
                " ON CONFLICT(folder) DO UPDATE SET"
                "   principals=excluded.principals, updated_at=excluded.updated_at",
                (folder, json.dumps(sorted(principals)), time.time()),
            )
            self._conn.commit()

    def clear_folder_access(self, folder: str) -> bool:
        """Remove a rule; True if one existed. The folder becomes open again."""
        with self._lock:
            cursor = self._conn.execute("DELETE FROM folder_access WHERE folder = ?", (folder,))
            self._conn.commit()
        return cursor.rowcount > 0

    # -- reported answers ----------------------------------------------------
    # The loop the product was missing. A refusal leaves a trace in the gaps
    # report; an answer that was confidently wrong left nothing, because it
    # looked exactly like one that was right. This is where a reader says so.

    def report_answer(
        self,
        canonical_query: str,
        question: str,
        answer: str,
        *,
        tier: str = "",
        citations: tuple[Citation, ...] = (),
        corpus_version: str = "",
        note: str = "",
    ) -> AnswerReport:
        """Record that somebody said this answer was wrong.

        The same wrong answer to the same question is one report with a
        count, not a hundred rows: what an admin needs is "this one, and
        eleven people agree", ranked against everything else. A note is kept
        with it because "the figure changed in April" is the whole fix, and
        the person who typed it is not.

        Re-reporting an answer that was marked fixed re-opens it - if it is
        still wrong, the fix did not work, and a closed row would hide that.
        """
        note = note.strip()
        now = time.time()
        digest = hashlib.sha256(answer.encode("utf-8")).hexdigest()[:16]
        with self._lock:
            self._conn.execute(
                "INSERT INTO answer_reports"
                " (canonical_query, question, answer, answer_hash, tier, citations,"
                "  corpus_version, notes, reports, first_at, last_at, status)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, 'open')"
                " ON CONFLICT(canonical_query, answer_hash) DO UPDATE SET"
                "   reports = reports + 1,"
                "   last_at = excluded.last_at,"
                "   status = 'open',"
                "   resolved_at = NULL,"
                "   resolution = NULL",
                (
                    canonical_query,
                    question,
                    answer,
                    digest,
                    tier,
                    _dump_citations(citations),
                    corpus_version,
                    json.dumps([note] if note else []),
                    now,
                    now,
                ),
            )
            if note:
                row = self._conn.execute(
                    "SELECT notes FROM answer_reports"
                    " WHERE canonical_query = ? AND answer_hash = ?",
                    (canonical_query, digest),
                ).fetchone()
                kept = _names_list(row["notes"])
                if note not in kept:
                    kept = [*kept, note][-MAX_REPORT_NOTES:]
                    self._conn.execute(
                        "UPDATE answer_reports SET notes = ?"
                        " WHERE canonical_query = ? AND answer_hash = ?",
                        (json.dumps(kept), canonical_query, digest),
                    )
            self._conn.commit()
        stored = self._report_by(canonical_query, digest)
        assert stored is not None  # just written, under the same lock
        return stored

    def _report_by(self, canonical_query: str, digest: str) -> AnswerReport | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM answer_reports WHERE canonical_query = ? AND answer_hash = ?",
                (canonical_query, digest),
            ).fetchone()
            return _as_report(row) if row is not None else None

    def answer_reports(self, *, status: str = "open", limit: int = 50) -> tuple[AnswerReport, ...]:
        """Reported answers, most-reported first. ``status=''`` for all."""
        with self._lock:
            sql = "SELECT * FROM answer_reports"
            args: list[object] = []
            if status:
                sql += " WHERE status = ?"
                args.append(status)
            sql += " ORDER BY reports DESC, last_at DESC LIMIT ?"
            args.append(max(1, limit))
            return tuple(_as_report(row) for row in self._conn.execute(sql, args).fetchall())

    def resolve_report(self, report_id: int, *, status: str, resolution: str = "") -> bool:
        """Close a report. ``fixed`` or ``dismissed``; True if one changed."""
        if status not in {"fixed", "dismissed"}:
            raise ValueError(f"a report is fixed or dismissed, not {status!r}")
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE answer_reports SET status = ?, resolution = ?, resolved_at = ?"
                " WHERE id = ? AND status = 'open'",
                (status, resolution, time.time(), report_id),
            )
            self._conn.commit()
        return cursor.rowcount > 0

    # -- the admin log -------------------------------------------------------
    # Every change an admin makes, in order, with who made it. Kept with the
    # other human decisions because that is what it records: not what the
    # machine inferred, but what a person chose.
    #
    # Append-only through the API - nothing exposes a delete, and nothing
    # should. That is a property of the surface, not of the file: anyone with
    # the server's disk can edit this table like any other. The honest claim
    # is "an admin cannot quietly undo their own change through the app",
    # which is the threat model a company server actually has.

    def record_action(
        self,
        actor: Actor,
        action: str,
        target: str = "",
        detail: dict[str, object] | None = None,
    ) -> None:
        """Write one entry. Never raises into the caller's request.

        A failed log write must not fail the admin action itself: losing the
        record of a change is bad, refusing the change because the record
        would not write is worse - it turns an audit trail into an outage.
        The write is small and local, so this is close to unreachable; it is
        here so that "close to" is not load-bearing.
        """
        try:
            with self._lock:
                self._conn.execute(
                    "INSERT INTO admin_actions"
                    " (at, actor, actor_name, actor_kind, action, target, detail)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        ordered_now(),
                        actor.id,
                        actor.name,
                        actor.kind,
                        action,
                        target,
                        json.dumps(detail or {}, default=str, sort_keys=True),
                    ),
                )
                self._conn.commit()
        except sqlite3.Error:
            return

    def admin_actions(self, limit: int = 100, since: float = 0.0) -> tuple[AdminAction, ...]:
        """The most recent entries, newest first."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, at, actor, actor_name, actor_kind, action, target, detail"
                " FROM admin_actions WHERE at >= ?"
                " ORDER BY at DESC, id DESC LIMIT ?",
                (since, max(1, limit)),
            ).fetchall()
            return tuple(
                AdminAction(
                    id=int(row["id"]),
                    at=float(row["at"]),
                    actor=Actor(id=row["actor"], name=row["actor_name"], kind=row["actor_kind"]),
                    action=row["action"],
                    target=row["target"],
                    detail=_loaded_detail(row["detail"]),
                )
                for row in rows
            )

    def admin_action_count(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) AS n FROM admin_actions").fetchone()
            return int(row["n"])


def _loaded_detail(raw: str) -> dict[str, object]:
    """A detail blob as a dict, whatever the column actually holds.

    A restored or hand-edited database can put anything here; a log viewer
    that raises on one malformed row hides every row after it.
    """
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _names_list(raw: str) -> list[str]:
    """A JSON list of strings out of a column, whatever it actually holds."""
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return []
    return [str(v) for v in value] if isinstance(value, list) else []


def _as_report(row: sqlite3.Row) -> AnswerReport:
    return AnswerReport(
        id=int(row["id"]),
        canonical_query=row["canonical_query"],
        question=row["question"],
        answer=row["answer"],
        tier=row["tier"],
        citations=_load_citations(row["citations"]),
        corpus_version=row["corpus_version"],
        notes=tuple(_names_list(row["notes"])),
        reports=int(row["reports"]),
        first_at=float(row["first_at"]),
        last_at=float(row["last_at"]),
        status=row["status"],
        resolved_at=row["resolved_at"],
        resolution=row["resolution"],
    )
