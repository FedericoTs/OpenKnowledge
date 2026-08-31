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
        row = self._conn.execute("SELECT * FROM proposals WHERE id = ?", (proposal_id_,)).fetchone()
        return self._row_to_proposal(row) if row else None

    def draft_for(self, canonical_query: str) -> Proposal | None:
        """The servable draft for a question, if there is one."""
        row = self._conn.execute(
            "SELECT * FROM proposals WHERE canonical_query = ? AND status = ?"
            " ORDER BY support_ratio DESC LIMIT 1",
            (canonical_query, ProposalStatus.DRAFT),
        ).fetchone()
        return self._row_to_proposal(row) if row else None

    def pending(self, limit: int = 50) -> list[Proposal]:
        """Drafts awaiting review, most valuable first."""
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
        return {
            row["folder"]: frozenset(json.loads(row["principals"]))
            for row in self._conn.execute("SELECT folder, principals FROM folder_access").fetchall()
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
