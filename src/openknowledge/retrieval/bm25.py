"""BM25 retrieval, no dependencies.

Lexical search is not a placeholder for "real" vector search - for policy and
procedure questions it is often the stronger half. People ask using the exact
nouns their company uses ("the T&E policy", "form RA-14"), and a keyword index
matches those precisely where an embedding blurs them into neighbours.

The plan is hybrid: this, plus a local embedding model, fused and reranked
(ADR 0004). Keeping BM25 dependency-free means the default install stays small
and OpenKnowledge answers usefully before anyone downloads a model.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from dataclasses import dataclass, field

from .base import Chunk, Document, ScoredChunk, chunk_document, demote_superseded, tokenize
from .tags import corpus_document_frequency, derive_tags, fold_tags, guarantee_routed, route_by_tags

_K1 = 1.5
_B = 0.75


@dataclass(frozen=True, slots=True)
class _Index:
    """Everything a search reads, so it can be replaced in one assignment.

    The arrays run in parallel - ``chunks[i]`` is scored with ``term_freqs[i]``
    and ``lengths[i]`` - and rebuilding them in place meant a search could
    arrive between two of them. A short read gives a wrong "not covered"; a
    read that lands after chunks has grown but before term_freqs has scores
    one chunk with another's statistics, which is a citation naming a document
    the text never came from. That is the worst answer this product can
    produce, and it is worse than the staleness a rebuild exists to fix.

    Holding the state in one frozen object makes the swap a single attribute
    assignment, which the interpreter cannot interleave. A reader takes the
    snapshot once and finishes against it: either wholly the old corpus or
    wholly the new one, never a seam between them.
    """

    chunks: tuple[Chunk, ...] = ()
    term_freqs: tuple[Counter[str], ...] = ()
    lengths: tuple[int, ...] = ()
    doc_freq: Counter[str] = field(default_factory=Counter)
    avg_length: float = 0.0
    corpus_version: str = "empty"
    doc_principals: dict[str, frozenset[str]] = field(default_factory=dict)
    doc_tags: dict[str, tuple[str, ...]] = field(default_factory=dict)
    doc_tags_folded: dict[str, frozenset[str]] = field(default_factory=dict)


class BM25Retriever:
    """In-memory BM25 over chunked documents."""

    def __init__(
        self, *, target_words: int = 350, overlap_words: int = 60, tag_routing: bool = True
    ) -> None:
        self._target_words = target_words
        self._overlap_words = overlap_words
        self.tag_routing = tag_routing
        self._index = _Index()

    @property
    def chunks(self) -> tuple[Chunk, ...]:
        """The indexed chunks, as one consistent snapshot."""
        return self._index.chunks

    @property
    def corpus_version(self) -> str:
        return self._index.corpus_version

    def __len__(self) -> int:
        return len(self._index.chunks)

    def documents_visible_to(self, principals: frozenset[str] | None) -> tuple[list[str], int]:
        """Titles this asker may see, and how many are being withheld.

        Access control applies here exactly as it does to retrieval. A list of
        document titles is not nothing: "Project Northstar - Redundancy Plan"
        tells you what it is without opening it, so answering "what do you have"
        without filtering would route around the ACL that search respects.
        """
        seen: dict[str, str] = {}
        hidden: set[str] = set()
        for chunk in self._index.chunks:
            if chunk.document_id in seen or chunk.document_id in hidden:
                continue
            allowed = chunk.allowed_principals
            if principals is not None and allowed and not (allowed & principals):
                hidden.add(chunk.document_id)
                continue
            seen[chunk.document_id] = chunk.document_title
        return sorted(seen.values()), len(hidden)

    @property
    def document_count(self) -> int:
        """How many documents, not how many chunks.

        These differ by roughly an order of magnitude and get confused: a health
        endpoint reporting the chunk count under the name "documents_indexed"
        tells an operator with four files that it has six of them.
        """
        return len({chunk.document_id for chunk in self._index.chunks})

    def index(self, documents: list[Document]) -> None:
        """Rebuild the index from scratch.

        A full rebuild is the honest operation here: it makes ``corpus_version``
        a true function of the current corpus, so a deleted document really does
        disappear from every future answer instead of lingering in the index.
        """
        chunks: list[Chunk] = []
        term_freqs: list[Counter[str]] = []
        lengths: list[int] = []
        doc_freq: Counter[str] = Counter()
        doc_principals: dict[str, frozenset[str]] = {}
        doc_tags: dict[str, tuple[str, ...]] = {}
        doc_tags_folded: dict[str, frozenset[str]] = {}

        # Tags are derived here rather than in the connector because tf-idf
        # needs the whole corpus: "expenses" is distinctive only against the
        # documents that never mention it.
        corpus_df = corpus_document_frequency(documents)
        for doc in documents:
            doc_principals[doc.document_id] = doc.allowed_principals
            tags = derive_tags(doc, corpus_df, len(documents))
            doc_tags[doc.document_id] = tags
            doc_tags_folded[doc.document_id] = fold_tags(tags)
            for chunk in chunk_document(
                doc, target_words=self._target_words, overlap_words=self._overlap_words
            ):
                tokens = tokenize(chunk.text)
                chunks.append(chunk)
                term_freqs.append(Counter(tokens))
                lengths.append(len(tokens))
                doc_freq.update(set(tokens))

        avg_length = (sum(lengths) / len(lengths)) if lengths else 0.0

        digest = hashlib.sha256()
        for doc in sorted(documents, key=lambda d: d.document_id):
            digest.update(doc.document_id.encode("utf-8"))
            digest.update(b"\x1f")
            digest.update(doc.content_hash.encode("utf-8"))
            digest.update(b"\x1f")
        # The one assignment. Everything above built a private copy; nothing
        # a reader can see has changed until this line runs, and after it
        # every reader sees the whole new corpus.
        self._index = _Index(
            chunks=tuple(chunks),
            term_freqs=tuple(term_freqs),
            lengths=tuple(lengths),
            doc_freq=doc_freq,
            avg_length=avg_length,
            corpus_version=digest.hexdigest()[:16] if documents else "empty",
            doc_principals=doc_principals,
            doc_tags=doc_tags,
            doc_tags_folded=doc_tags_folded,
        )

    def search(
        self, query: str, *, k: int = 6, principals: frozenset[str] | None = None
    ) -> list[ScoredChunk]:
        # One read of the snapshot, used for the whole search. Taking it
        # again mid-scoring would let a rebuild land between two reads and
        # score a chunk against another chunk's statistics.
        index = self._index
        if not index.chunks:
            return []

        query_terms = tokenize(query)
        if not query_terms:
            return []

        # Tag routing guarantees the named documents a place among the
        # results when the question names them decisively; None - the
        # common case - changes nothing.
        within = route_by_tags(query, index.doc_tags_folded) if self.tag_routing else None

        n = len(index.chunks)
        scored: list[ScoredChunk] = []

        for i, chunk in enumerate(index.chunks):
            # Access control is applied during scoring, not after: filtering a
            # top-k list would silently shrink results for restricted users.
            if (
                principals is not None
                and chunk.allowed_principals
                and not (chunk.allowed_principals & principals)
            ):
                continue

            tf = index.term_freqs[i]
            length = index.lengths[i]
            score = 0.0
            for term in query_terms:
                freq = tf.get(term, 0)
                if not freq:
                    continue
                df = index.doc_freq[term]
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                denom = freq + _K1 * (1 - _B + _B * length / (index.avg_length or 1.0))
                score += idf * (freq * (_K1 + 1)) / denom
            if score > 0:
                scored.append(ScoredChunk(chunk=chunk, score=score))

        # Sort by score, then chunk_id: ties must break the same way on every
        # run or identical questions would retrieve different context.
        # Demotion sees the full ranking so it can backfill; the route then
        # rescues any named document still below the cut.
        scored.sort(key=lambda s: (-s.score, s.chunk.chunk_id))
        return guarantee_routed(demote_superseded(scored, len(scored)), within, k)

    def visible_to(self, document_ids: set[str], principals: frozenset[str] | None) -> bool:
        """Whether ``principals`` may see every document in ``document_ids``.

        Used to re-check a *cached* answer before serving it. The cache key
        deliberately excludes the asker's identity - keying on it would give every
        employee a private cache and destroy the hit rate the cost model depends
        on. So access is enforced at read time instead: a cached answer is only
        served if the person asking could have retrieved each of its sources
        themselves. Otherwise it is treated as a miss and the question is answered
        again over the documents they *can* see.

        An unknown document id is treated as not visible. Failing closed matters
        here: an id goes missing exactly when a document has been removed from the
        corpus, which is often the moment its contents became restricted.
        """
        if principals is None:
            return True
        index = self._index
        for doc_id in document_ids:
            if doc_id not in index.doc_principals:
                return False
            allowed = index.doc_principals[doc_id]
            if allowed and not (allowed & principals):
                return False
        return True

    def document_tags(self) -> dict[str, tuple[str, ...]]:
        """Each indexed document's derived tags, as readable words for
        listings. Routing uses the folded form via :meth:`routing_tags`."""
        return dict(self._index.doc_tags)

    def routing_tags(self) -> dict[str, frozenset[str]]:
        """The folded tag sets routing matches against."""
        return self._index.doc_tags_folded

    def visible_document_tags(
        self, principals: frozenset[str] | None
    ) -> dict[str, tuple[str, ...]]:
        """Each visible document's tags, keyed by title, for the corpus tier.

        Filtered exactly as the listing is: tags derive from a document's own
        vocabulary, so showing a walled document's tags would leak what the
        asker may not read."""
        index = self._index
        visible: dict[str, tuple[str, ...]] = {}
        seen: set[str] = set()
        for chunk in index.chunks:
            if chunk.document_id in seen:
                continue
            seen.add(chunk.document_id)
            allowed = chunk.allowed_principals
            if principals is not None and allowed and not (allowed & principals):
                continue
            visible[chunk.document_title] = index.doc_tags.get(chunk.document_id, ())
        return visible

    def document_ids(self) -> frozenset[str]:
        """Every document currently indexed.

        Used to tell "this case cites a document that does not exist" apart from
        "this case cites a real document retrieval did not rank" - a typo and a
        ranking problem look identical in a report and need opposite fixes.
        """
        return frozenset(c.document_id for c in self._index.chunks)

    def describe_document(self, document_id: str) -> tuple[str, str] | None:
        """Return ``(title, snippet)`` for an indexed document, or None.

        Used when an admin pins an answer and names its sources: the pin then
        carries the same provenance a model-generated answer would, so a reader
        can check it rather than taking it on trust.
        """
        for chunk in self._index.chunks:
            if chunk.document_id == document_id:
                return chunk.document_title, chunk.text
        return None
