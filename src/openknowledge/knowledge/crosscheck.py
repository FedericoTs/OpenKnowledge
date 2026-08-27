"""Checking stored answers against newly arrived documents - for free.

Re-verification has a structural blind spot. It re-asks the approved answers
that *cite* a changed document, which cannot cover the most common way a
contradiction enters a corpus: somebody uploads a **new** document. No existing
answer cites it, because it did not exist, so nothing is re-asked and the
disagreement sits there until an employee finds it.

Closing that at the document level means comparing the new file against every
other file - O(N^2), and a model call per pair.

Closing it at the FAQ level is free. A stored answer is short text with
extractable claims, and BM25 already knows, at no cost, which questions a new
document has an opinion about: the ones it ranks highly for. So:

1. For every stored answer, ask whether the new document appears in the top
   results for its question. Free - it is one lexical search.
2. If it does, compare the claims in the stored answer against the claims in the
   document passage that matched. Free - regex and set arithmetic.
3. Anything that disagrees is a contradiction, already expressed as "your
   answer says X, this document says Y".

No model call anywhere in that path. The model is only needed when an answer has
no extractable claims at all, which falls through to re-verification.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ..retrieval.base import Document
from ..retrieval.bm25 import BM25Retriever
from .claims import Conflict, find_conflicts
from .store import KnowledgeStore, Proposal, ProposalStatus

log = logging.getLogger(__name__)

#: A document has to actually rank for a question before we treat it as having
#: an opinion about it. Without this every answer would be compared against
#: every new document, which is the O(N^2) we are avoiding.
DEFAULT_TOP_K = 4


@dataclass(frozen=True, slots=True)
class CrossCheckFinding:
    """A stored answer that a newly arrived document disagrees with."""

    proposal: Proposal
    document_id: str
    document_title: str
    conflicts: tuple[Conflict, ...]

    @property
    def is_approved(self) -> bool:
        return self.proposal.status is ProposalStatus.APPROVED

    def describe(self) -> str:
        top = self.conflicts[0]
        return (
            f"{self.proposal.question!r}: your "
            f"{'approved' if self.is_approved else 'drafted'} answer says "
            f"{top.left.raw}, but [{self.document_id}] says {top.right.raw}"
        )


def _answer_as_document(proposal: Proposal) -> Document:
    """Wrap a stored answer so the claim extractors can read it.

    Given the id of the document the answer was approved from, so the resulting
    conflict reads as a disagreement between two sources rather than between a
    document and an opaque record id.
    """
    source = proposal.citations[0].document_id if proposal.citations else "approved-answer"
    title = proposal.citations[0].document_title if proposal.citations else "Your answer"
    return Document(document_id=source, title=title, text=proposal.answer)


def crosscheck_answers(
    *,
    store: KnowledgeStore,
    retriever: BM25Retriever,
    document_ids: frozenset[str],
    top_k: int = DEFAULT_TOP_K,
    deontic_strictness: float = 1.0,
) -> list[CrossCheckFinding]:
    """Compare every stored answer against the documents that just arrived.

    ``document_ids`` are the documents added or changed in this ingest run.
    Costs nothing: one lexical search and some regex per stored answer.
    """
    if not document_ids:
        return []

    candidates = [
        p
        for p in store.all_proposals(ProposalStatus.APPROVED)
        + store.all_proposals(ProposalStatus.DRAFT)
        # An answer already derived from this document cannot disagree with it.
        if not set(p.origin_documents) & document_ids
    ]
    if not candidates:
        return []

    findings: list[CrossCheckFinding] = []
    for proposal in candidates:
        hits = retriever.search(proposal.question, k=top_k)
        touched = [h.chunk for h in hits if h.chunk.document_id in document_ids]
        if not touched:
            continue  # the new document has no opinion about this question

        answer_doc = _answer_as_document(proposal)
        for chunk in touched:
            if chunk.document_id == answer_doc.document_id:
                continue  # the answer's own source, restated
            passage = Document(chunk.document_id, chunk.document_title, chunk.text)
            conflicts = find_conflicts([answer_doc, passage], deontic_strictness=deontic_strictness)
            if not conflicts:
                continue
            findings.append(
                CrossCheckFinding(
                    proposal=proposal,
                    document_id=chunk.document_id,
                    document_title=chunk.document_title,
                    conflicts=tuple(conflicts),
                )
            )
            break  # one finding per answer is enough to get a human looking

    # Approved answers first: those represent a decision somebody made, so a
    # document that disagrees with one is more urgent than one that disagrees
    # with a draft nobody has looked at.
    findings.sort(key=lambda f: (not f.is_approved, f.proposal.question))
    return findings
