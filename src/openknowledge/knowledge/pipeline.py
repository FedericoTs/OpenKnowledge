"""What happens when documents are uploaded or change.

The user-facing promise is that maintenance is a one-off at upload rather than a
standing chore, so this runs once per corpus change and does four things:

1. **Detect numeric conflicts** between documents - free, no model, deterministic.
2. **Retire stale drafts** whose source documents moved underneath them.
3. **Re-verify approved answers** that cite a changed document, and raise the
   ones whose figures moved. This is the contradiction flag the whole feature is
   for, and it is affordable precisely because it is anchored to answers a human
   already cared about rather than to document pairs.
4. **Draft new FAQ entries** for documents that are new or changed.

Step 4 is the only one that costs money, and it is charged per changed document,
not per question asked. That is the trade the design is making: pay once at
upload so the answer is free for as long as the document stands.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace

from ..providers.base import ChatProvider
from ..retrieval.base import Document, Retriever
from .claims import compare_documents
from .crosscheck import crosscheck_answers
from .generate import draft_from_document
from .store import KnowledgeStore, Proposal
from .variants import group_by_document_pair

log = logging.getLogger(__name__)


@dataclass
class IngestReport:
    """What one ingest run did, and what it cost."""

    documents: int = 0
    added: tuple[str, ...] = ()
    changed: tuple[str, ...] = ()
    removed: tuple[str, ...] = ()
    conflicts_detected: int = 0
    #: Stored answers a newly arrived document disagrees with. Found for free.
    answers_contradicted: int = 0
    conflicts_open: int = 0
    conflicts_cleared: int = 0
    drafts_created: int = 0
    drafts_rejected: int = 0
    drafts_superseded: int = 0
    revisions_raised: int = 0
    documents_drafted: int = 0
    cost_usd: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def needs_review(self) -> int:
        return self.drafts_created + self.revisions_raised + self.conflicts_open

    def summary(self) -> str:
        return (
            f"{self.documents} documents "
            f"({len(self.added)} new, {len(self.changed)} changed, {len(self.removed)} removed); "
            f"{self.drafts_created} answers drafted, {self.revisions_raised} revisions raised, "
            f"{self.answers_contradicted} answers contradicted, "
            f"{self.conflicts_open} conflicts open; ${self.cost_usd:.4f} spent"
        )


def scan_documents(
    documents: list[Document],
    *,
    store: KnowledgeStore,
    retriever: Retriever | None = None,
    min_conflict_overlap: float = 0.34,
    deontic_strictness: float = 1.0,
) -> IngestReport:
    """The free half of ingest: version tracking, conflicts, stale drafts.

    Separate from drafting so that re-indexing can never spend money by
    surprise. An operator who runs `index` gets conflict detection for nothing;
    spending is a second, explicit step.
    """
    report = IngestReport(documents=len(documents))

    added, changed, removed = store.sync_documents(documents)
    report.added = tuple(sorted(added))
    report.changed = tuple(sorted(changed))
    report.removed = tuple(sorted(removed))

    # Conflicts are free, so re-run over the whole corpus every time: a
    # disagreement between two untouched documents is still a disagreement, and
    # re-running beats reasoning about which pairs might have been affected.
    present = frozenset(d.document_id for d in documents)
    report.conflicts_cleared = store.drop_conflicts_for_documents(present)
    conflicts, agreements = compare_documents(
        documents,
        min_overlap=min_conflict_overlap,
        deontic_strictness=deontic_strictness,
    )
    for pair in group_by_document_pair(conflicts, agreements):
        # Two documents disagreeing on two dozen shared figures are not two
        # dozen contradictions - they are two versions of one document, and
        # the decision a human owes is which copy stands, said once. The
        # audit has said this for a while; the server used to open every
        # figure as its own blocking conflict, which turned a stale archive
        # copy into a corpus-wide refusal. One recorded "versions" conflict
        # keeps the decision visible in /manage without gating answers -
        # relevance excludes the kind.
        if pair.is_variant:
            store.record_conflict(replace(pair.conflicts[0], kind="versions"))
            report.conflicts_detected += 1
            report.notes.append(pair.describe())
            continue
        for conflict in pair.conflicts:
            store.record_conflict(conflict)
            report.conflicts_detected += 1

    # A new document contradicting an existing *answer* is the case document
    # pairing cannot see - no stored answer cites a file that did not exist. The
    # FAQ already knows what the corpus claims, so checking against it is one
    # lexical search plus regex: free, and it runs on every ingest.
    if retriever is not None and (added or changed):
        for finding in crosscheck_answers(
            store=store,
            retriever=retriever,
            document_ids=added | changed,
            deontic_strictness=deontic_strictness,
        ):
            report.answers_contradicted += 1
            report.notes.append(f"answer contradicted - {finding.describe()}")
            for conflict in finding.conflicts:
                store.record_conflict(conflict)

    report.conflicts_open = len(store.open_conflicts())

    # Drafts built from text that has since moved were verified against
    # something that no longer exists, so they stop being servable.
    if changed or removed:
        report.drafts_superseded = len(store.supersede_for_documents(changed | removed))

    return report


async def draft_for_documents(
    documents: list[Document],
    *,
    store: KnowledgeStore,
    provider: ChatProvider,
    corpus_version: str,
    document_ids: frozenset[str],
    report: IngestReport | None = None,
    min_support_ratio: float = 0.45,
    min_support_ratio_cited: float = 0.30,
    max_documents: int | None = None,
) -> IngestReport:
    """The paid half: draft FAQ answers for the named documents.

    Charged once per document rather than once per question, which is the whole
    argument for doing this at upload time.
    """
    report = report or IngestReport(documents=len(documents))

    to_draft = [d for d in documents if d.document_id in document_ids]
    if max_documents is not None and len(to_draft) > max_documents:
        skipped = len(to_draft) - max_documents
        report.notes.append(
            f"drafted {max_documents} of {len(to_draft)} changed documents; "
            f"{skipped} skipped by the per-run cap - run again to continue"
        )
        to_draft = to_draft[:max_documents]

    for document in to_draft:
        result = await draft_from_document(
            provider,
            document,
            min_support_ratio=min_support_ratio,
            min_support_ratio_cited=min_support_ratio_cited,
        )
        report.documents_drafted += 1
        report.cost_usd += result.cost_usd
        report.drafts_rejected += len(result.rejected)

        for draft in result.drafted:
            proposal = store.propose(
                canonical_query=draft.canonical_query,
                question=draft.question,
                answer=draft.answer,
                citations=draft.citations,
                origin_documents=draft.origin_documents,
                corpus_version=corpus_version,
                support_ratio=draft.support_ratio,
                source="ingest",
            )
            if proposal is not None:
                report.drafts_created += 1

    return report


async def ingest_documents(
    documents: list[Document],
    *,
    store: KnowledgeStore,
    corpus_version: str,
    provider: ChatProvider | None = None,
    min_support_ratio: float = 0.45,
    max_documents_to_draft: int | None = None,
    min_conflict_overlap: float = 0.34,
) -> IngestReport:
    """Run both halves in order. Convenience wrapper for the CLI and tests."""
    report = scan_documents(documents, store=store, min_conflict_overlap=min_conflict_overlap)

    touched = frozenset(report.added) | frozenset(report.changed)
    if not touched:
        report.notes.append("no documents changed; nothing to draft")
        return report

    if provider is None:
        report.notes.append(
            "no model configured, so no answers were drafted; conflict detection still ran"
        )
        return report

    return await draft_for_documents(
        documents,
        store=store,
        provider=provider,
        corpus_version=corpus_version,
        document_ids=touched,
        report=report,
        min_support_ratio=min_support_ratio,
        max_documents=max_documents_to_draft,
    )


def rank_by_demand(
    proposals: list[Proposal],
    *,
    demand: dict[str, int],
    cost_per_answer_usd: float,
) -> list[tuple[Proposal, float]]:
    """Score drafts by what approving them is actually worth.

    A queue ordered by nothing in particular gets abandoned. Ordered by the
    money each entry saves, the first fifty are worth someone's morning and the
    tail can wait indefinitely without the system degrading.

    ``demand`` is the ledger's question counts, so a draft answering something
    nobody asks scores zero however well written it is.
    """
    scored = [(p, demand.get(p.canonical_query, 0) * cost_per_answer_usd) for p in proposals]
    scored.sort(key=lambda pair: (-pair[1], -pair[0].support_ratio, pair[0].created_at))
    return scored
