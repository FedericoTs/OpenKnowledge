"""Choosing which retrieved chunks are worth a slot.

BM25 ranks every chunk independently, which means the top *k* can be six views
of one paragraph and still score beautifully. Three failures follow from that,
and all three are recall failures: the answer was retrievable and did not make
the cut.

**One document takes every slot.** A question whose answer spans the expenses
policy *and* the travel guidelines gets six chunks of expenses policy, because
that document happened to use the query's words more often. The grounding gate
then rejects the answer for a claim it could not support, and the question
escalates - paying frontier prices for a retrieval problem.

**Overlapping chunks are near-duplicates by construction.** The chunker overlaps
windows by design, so adjacent chunks share sixty words. Two of them in the top
six is one slot of evidence occupying two.

**Heading matches are scored as ordinary text.** ADR 0007 went to some trouble to
recover document structure; a chunk whose *heading trail* matches the question is
far more likely to be the right section than one that mentions the words in
passing, and BM25 cannot tell the difference.

None of these need a model. The default reranker here is free, deterministic and
runs on the retrieved candidates in microseconds, and it exists because
escalation - not context size - is what a tuned deployment actually pays for: a
gate failure costs orders of magnitude more than the chunks that caused it.

It is **not** a cross-encoder and does not claim a cross-encoder's gains. It
fixes three identifiable failures with structure already in hand. A real
cross-encoder is a heavier dependency and a separate decision; the protocol here
is what it would plug into.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .base import ScoredChunk, tokenize

#: The heading trail the chunker writes ahead of a block's text.
_HEADING_SEPARATOR = " > "


@runtime_checkable
class Reranker(Protocol):
    """Reorders retrieved candidates and keeps the best ``k``."""

    def rerank(self, query: str, hits: list[ScoredChunk], *, k: int) -> list[ScoredChunk]: ...


@dataclass(frozen=True, slots=True)
class StructuralReranker:
    """Diversity, redundancy and heading awareness. No model, no dependencies."""

    #: Slots any one document may take. Two is enough for a rule and its
    #: exception; more than that and a single verbose file crowds out the
    #: document that actually answers the second half of the question.
    max_per_document: int = 2
    #: Added to a chunk's score in proportion to how much of the question its
    #: heading trail accounts for. Deliberately modest: a heading is evidence
    #: about topic, not about whether the answer is in the body.
    heading_boost: float = 0.35
    #: Token overlap above which a candidate is treated as already covered.
    #: The chunker's own overlap puts adjacent windows around 0.2-0.4, so this
    #: catches genuine restatement rather than ordinary neighbours.
    redundancy_threshold: float = 0.75

    def rerank(self, query: str, hits: list[ScoredChunk], *, k: int) -> list[ScoredChunk]:
        if k <= 0 or not hits:
            return []

        terms = frozenset(tokenize(query))
        # Score first, then select greedily. Sorting on the adjusted score with
        # chunk_id as the tie-break keeps this deterministic: identical
        # questions must produce identical context or the cache key lies.
        adjusted = sorted(
            ((self._adjust(hit, terms), hit) for hit in hits),
            key=lambda pair: (-pair[0], pair[1].chunk.chunk_id),
        )

        kept: list[ScoredChunk] = []
        kept_tokens: list[frozenset[str]] = []
        per_document: dict[str, int] = {}
        deferred: list[tuple[float, ScoredChunk]] = []

        for score, hit in adjusted:
            if len(kept) >= k:
                break
            document = hit.chunk.document_id
            if per_document.get(document, 0) >= self.max_per_document:
                deferred.append((score, hit))
                continue
            body = frozenset(tokenize(hit.chunk.text))
            if any(_jaccard(body, seen) >= self.redundancy_threshold for seen in kept_tokens):
                continue  # already covered; dropping it frees a slot for new evidence
            kept.append(ScoredChunk(chunk=hit.chunk, score=score))
            kept_tokens.append(body)
            per_document[document] = per_document.get(document, 0) + 1

        # A cap is a preference, not a rule. Sending four chunks when six were
        # asked for, because only one document is relevant, would be a worse
        # answer for the sake of a tidier spread.
        for score, hit in deferred:
            if len(kept) >= k:
                break
            kept.append(ScoredChunk(chunk=hit.chunk, score=score))

        return kept

    def _adjust(self, hit: ScoredChunk, terms: frozenset[str]) -> float:
        if not terms or self.heading_boost <= 0:
            return hit.score
        heading = frozenset(tokenize(_heading_of(hit.chunk.text)))
        if not heading:
            return hit.score
        covered = len(terms & heading) / len(terms)
        return hit.score * (1.0 + self.heading_boost * covered)


def _heading_of(text: str) -> str:
    """The heading trail the chunker prefixed, if this chunk carries one."""
    first, _, _ = text.partition("\n")
    return first.split(":", 1)[0] if _HEADING_SEPARATOR in first else ""


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)
