"""Which unanswered questions are still unanswered.

The gap report reads the ledger, and the ledger records what happened when a
question was asked. That is the right source for "this was not answered" and
the wrong one for "this is not answerable", because the two drift apart the
moment somebody fixes something: the row clears only when the question is asked
again, and nobody re-asks a question that failed them once.

Measured on a real install. Sixteen questions were listed; four of them were
answered free at the time of reading and had been since a fix landed a week
earlier. They sat at the top of the report competing for attention with the
eleven that were genuinely open, and there was no way to tell which was which.

So each row is re-tried here against the tiers that cost nothing - the corpus
recogniser, the assistant safety net, and a document's own structure. No model
is called and no answer is produced: the question is only whether one of those
paths would now take the question, which is a retrieval search and some string
work. A row that would be is reported as answered rather than as a gap.

Pins are not checked because the ledger query already excludes them: a pinned
question stops being a gap the moment it is pinned, which is the one case that
was already right.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .cascade.corpus import (
    asks_about_the_assistant,
    corpus_has_nothing_to_say,
    evidence_text,
    recognise,
)
from .cascade.scope import outline, recognise_scope

if TYPE_CHECKING:
    from .retrieval.base import Retriever


def free_tier_for(question: str, retriever: Retriever) -> str | None:
    """The free tier that would answer ``question`` today, or None.

    Named tiers rather than a bare bool because the report says *how* a row
    was closed, and "answered from the index" and "answered from the
    document's structure" send a curator to different places.
    """
    if recognise(question) is not None:
        return "corpus"

    # Both remaining paths need the passages this question retrieves. One
    # search, reused, and no model is reachable from here at all.
    hits = retriever.search(question, k=6)
    if asks_about_the_assistant(question) and corpus_has_nothing_to_say(
        question, evidence_text([h.chunk for h in hits])
    ):
        return "corpus"

    scope = recognise_scope(question, hits)
    if scope is not None and scope.kind == "enumerate":
        blocks = retriever.blocks_of(scope.document_id)
        if outline(blocks, scope.wants, question, noun=scope.noun) is not None:
            return "outline"
    return None


def mark_answerable(rows: list[dict[str, object]], retriever: Retriever) -> list[dict[str, object]]:
    """Annotate gap rows with the free tier that would now answer them.

    Every row keeps its place and its counts - the report still says this
    question went unanswered and how often, because it did. What is added is
    ``answered_now``: the tier, or None. Sorting and presentation are the
    caller's business; this only knows which rows are stale.
    """
    marked = []
    for row in rows:
        question = str(row.get("question", ""))
        marked.append({**row, "answered_now": free_tier_for(question, retriever)})
    return marked
