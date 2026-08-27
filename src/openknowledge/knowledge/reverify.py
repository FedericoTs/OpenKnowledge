"""Re-checking approved answers when the documents under them change.

This is the contradiction flag, and its whole design is about affordability.

The obvious implementation compares each uploaded document against every other
document in the corpus. That is O(N^2) model calls, and at 500 documents it
costs about $5.00 every time somebody uploads a file - a price that quietly
discourages keeping the corpus current, which is the opposite of the goal.

The cheap implementation notices that approved answers already carry citations.
When a document changes, only the answers that *cite it* can be affected, so
those are the only ones worth re-asking - typically a handful. That is around
$0.08 per upload, roughly sixty times cheaper, and it produces a better artefact
for the reviewer: instead of "these two documents differ somewhere", they get
"you approved EUR 500 for this question and the document now says EUR 1,000".

The comparison itself is figure-first. A reworded answer with identical numbers
is not worth interrupting anyone about; a changed threshold always is.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ..costs import PricingError, cost_usd, get_price
from ..prompts import SYSTEM_PROMPT, format_context
from ..providers.base import ChatProvider, ProviderError
from ..retrieval.base import Document, Retriever
from ..retrieval.grounding import check_grounding
from .claims import extract_claims
from .store import KnowledgeStore, Proposal

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Revision:
    """An approved answer whose supporting documents have moved."""

    approved: Proposal
    new_answer: str
    #: ``(was, now)`` pairs for figures that changed. Empty means the wording
    #: moved but every number held.
    figure_changes: tuple[tuple[str, str], ...]
    proposal: Proposal | None
    cost_usd: float = 0.0

    @property
    def is_material(self) -> bool:
        return bool(self.figure_changes)

    def describe(self) -> str:
        if not self.figure_changes:
            return f"{self.approved.question!r}: wording changed, all figures unchanged"
        moves = ", ".join(f"{was} -> {now}" for was, now in self.figure_changes)
        return f"{self.approved.question!r}: {moves}"


def figure_changes(old_text: str, new_text: str) -> tuple[tuple[str, str], ...]:
    """Figures present in ``old_text`` that ``new_text`` contradicts.

    Compares by unit: a figure is "changed" when the new answer gives a
    different value for the same unit. Numbers that simply disappear are not
    reported here - that is a scope change for a human to read, not a
    contradiction we can name precisely.
    """
    old_claims = extract_claims(Document("old", "old", old_text))
    new_claims = extract_claims(Document("new", "new", new_text))

    new_by_unit: dict[str, list[tuple[float, str]]] = {}
    for claim in new_claims:
        new_by_unit.setdefault(claim.unit, []).append((claim.value, claim.raw))

    changes: list[tuple[str, str]] = []
    for claim in old_claims:
        candidates = new_by_unit.get(claim.unit, [])
        if not candidates:
            continue
        if any(value == claim.value for value, _ in candidates):
            continue  # the figure survived
        changes.append((claim.raw, candidates[0][1]))
    return tuple(changes)


async def _answer_freshly(
    provider: ChatProvider,
    retriever: Retriever,
    question: str,
    *,
    k: int,
    min_support_ratio: float,
    max_tokens: int,
) -> tuple[str, float, float] | None:
    """Answer a question against the current corpus, bypassing every cache.

    Returns ``(text, support_ratio, cost)``, or None if nothing grounded came
    back - which is itself worth knowing, but is handled by the caller.
    """
    hits = retriever.search(question, k=k)
    chunks = [h.chunk for h in hits]
    if not chunks:
        return None

    try:
        completion = await provider.complete(
            system=SYSTEM_PROMPT,
            context=format_context(chunks),
            question=question,
            max_tokens=max_tokens,
        )
    except ProviderError as exc:
        log.warning("re-verification call failed for %r: %s", question, exc)
        return None

    model_id = "local" if getattr(provider, "tier", "") == "local" else completion.model_id
    try:
        cost = cost_usd(completion.usage, get_price(model_id))
    except PricingError:
        cost = 0.0

    report = check_grounding(completion.text, chunks, min_support_ratio=min_support_ratio)
    if not report.passed:
        log.info("re-verified answer failed the gate for %r", question)
        return None
    return completion.text, report.support_ratio, cost


async def reverify_changed_documents(
    changed_documents: frozenset[str],
    *,
    store: KnowledgeStore,
    retriever: Retriever,
    provider: ChatProvider,
    corpus_version: str,
    k: int = 6,
    min_support_ratio: float = 0.45,
    max_tokens: int = 1500,
) -> list[Revision]:
    """Re-ask every approved question that depends on a changed document."""
    if not changed_documents:
        return []

    affected: dict[str, Proposal] = {}
    for document_id in sorted(changed_documents):
        for proposal in store.approved_citing(document_id):
            affected[proposal.id] = proposal

    if not affected:
        log.info("no approved answers cite the changed documents; nothing to re-verify")
        return []

    revisions: list[Revision] = []
    for proposal in affected.values():
        fresh = await _answer_freshly(
            provider,
            retriever,
            proposal.question,
            k=k,
            min_support_ratio=min_support_ratio,
            max_tokens=max_tokens,
        )
        if fresh is None:
            continue
        new_answer, support_ratio, cost = fresh

        changes = figure_changes(proposal.answer, new_answer)
        if not changes and new_answer.strip() == proposal.answer.strip():
            continue  # nothing moved at all

        # Only raise a review item when a figure actually moved. Interrupting a
        # reviewer for a paraphrase teaches them to dismiss the queue.
        draft = None
        if changes:
            draft = store.propose(
                canonical_query=proposal.canonical_query,
                question=proposal.question,
                answer=new_answer,
                citations=proposal.citations,
                origin_documents=proposal.origin_documents,
                corpus_version=corpus_version,
                support_ratio=support_ratio,
                source="reverify",
                variant=corpus_version,
                supersedes=proposal.id,
            )

        revisions.append(
            Revision(
                approved=proposal,
                new_answer=new_answer,
                figure_changes=changes,
                proposal=draft,
                cost_usd=cost,
            )
        )

    return revisions
