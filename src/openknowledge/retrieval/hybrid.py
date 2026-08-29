"""BM25 and dense retrieval, fused.

Two rankings have to become one. The obvious way - normalise both scores and
add them - is the wrong way: BM25 scores are unbounded and corpus-dependent
while cosine similarities sit in a narrow band near the top, so any weighting
that works on one corpus is wrong on the next, and there is no principled place
to get the weight from.

Reciprocal rank fusion uses only the *positions*. A chunk ranked r by a
retriever contributes 1/(k + r), summed across retrievers. It has no scale to
calibrate, no weight to tune per corpus, and it is exactly as deterministic as
its inputs - which the answer cache depends on. A chunk both halves like beats
one that either half loves, which is the behaviour worth having: agreement
between a keyword match and a meaning match is the strongest signal available
here.

Falling back is not a failure mode, it is the normal one. Most installs will
have no embedding endpoint on first run, and BM25 alone is what they get.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .base import Chunk, Document, ScoredChunk, demote_superseded
from .bm25 import BM25Retriever
from .embed import Embedder, EmbeddingError, normalise, text_key
from .tags import guarantee_routed, route_by_tags

log = logging.getLogger(__name__)

# Measured on ten real policy documents plus three carrying rare literals -
# form RA-14, a system codename, a policy number - over nineteen questions
# split between casual paraphrases and exact terms:
#
#     BM25 only        9/13 paraphrase+exact,  6/6 literals
#     dense only      13/13,                   6/6
#     RRF equal       11/13,                   -
#     RRF dense x3    12/13,                   -
#     interleave      12/13,                   6/6
#
# Two things follow, and the second is the uncomfortable one.
#
# Reciprocal rank fusion loses. It counts every retriever's ranking as votes,
# and on a paraphrased question BM25's ranking is close to noise - twenty
# chunks all voting, drowning the one the dense half put first. Weighting the
# dense side x3 recovers most of it, but that 3 is a constant fitted to
# thirteen questions on one corpus, which is exactly the kind of number this
# project should not ship.
#
# And dense retrieval alone beat every fusion, including on the identifiers
# BM25 is supposed to protect. That is not enough to remove BM25: nineteen
# questions over thirteen documents, each literal unique to one file, is a
# sample structurally kind to embeddings, and BM25 is what runs when nothing
# has been downloaded. It is enough to stop pretending the ranks should be
# averaged.
#
# So: interleave. Each retriever's Nth choice is taken before anyone's N+1th,
# which guarantees the dense half's best hit is in the top two and the lexical
# half's best hit is too. No weight, no constant, nothing fitted.


@dataclass
class VectorIndex:
    """Chunk vectors, unit length, in memory. No vector database.

    Brute force is the right answer at this scale and by a long way. A corpus of
    ten thousand chunks at 768 dimensions is a 30 MB matrix; one query is a
    single matrix-vector product, which numpy does in about a millisecond. An
    approximate index would add a dependency, a build step, and a source of
    non-determinism, to save time that is not being spent.
    """

    vectors: list[list[float]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.vectors)

    def search(self, query: list[float], k: int) -> list[tuple[int, float]]:
        """Indices of the ``k`` nearest chunks, nearest first."""
        if not self.vectors:
            return []
        import numpy as np

        matrix = np.asarray(self.vectors, dtype=np.float32)
        scores = matrix @ np.asarray(query, dtype=np.float32)
        # Ties broken by index, so the same corpus and question always produce
        # the same order - the property the cache is built on.
        order = np.lexsort((np.arange(len(scores)), -scores))[:k]
        return [(int(i), float(scores[i])) for i in order]


@dataclass
class HybridRetriever:
    """BM25 plus dense retrieval over the same chunks, fused by rank.

    Wraps rather than replaces :class:`BM25Retriever`: lexical search stays the
    thing that runs when nothing else can, and the dense half is an addition
    that is allowed to be absent.
    """

    lexical: BM25Retriever
    embedder: Embedder | None = None
    #: Cached vectors by text hash, so re-indexing does not re-embed unchanged
    #: paragraphs. Supplied by the store; a plain dict in tests.
    cache: dict[str, list[float]] = field(default_factory=dict)
    #: Where new vectors are written back to, when there is somewhere to write.
    #: Duck-typed on purpose so tests can pass a dict and skip the disk.
    store: object | None = None
    vectors: VectorIndex = field(default_factory=VectorIndex)
    #: Set when the dense half could not be built, so callers can say why.
    degraded: str = ""

    # -- the BM25Retriever surface, delegated ------------------------------

    def __len__(self) -> int:
        return len(self.lexical)

    @property
    def corpus_version(self) -> str:
        """Fingerprinted with the embedding model.

        Vectors from two models are not comparable, so a model change is a
        corpus change as far as cached answers are concerned.
        """
        base = self.lexical.corpus_version
        if self.embedder is None or not self.vectors:
            return base
        import hashlib

        digest = hashlib.sha256(f"{base}|{self.embedder.fingerprint}".encode())
        return digest.hexdigest()[:16]

    @property
    def document_count(self) -> int:
        return self.lexical.document_count

    def documents_visible_to(self, principals: frozenset[str] | None) -> tuple[list[str], int]:
        return self.lexical.documents_visible_to(principals)

    def visible_to(self, document_ids: set[str], principals: frozenset[str] | None) -> bool:
        return self.lexical.visible_to(document_ids, principals)

    def describe_document(self, document_id: str) -> tuple[str, str] | None:
        return self.lexical.describe_document(document_id)

    def document_ids(self) -> frozenset[str]:
        return self.lexical.document_ids()

    def document_tags(self) -> dict[str, tuple[str, ...]]:
        return self.lexical.document_tags()

    def visible_document_tags(
        self, principals: frozenset[str] | None
    ) -> dict[str, tuple[str, ...]]:
        return self.lexical.visible_document_tags(principals)

    @property
    def chunks(self) -> list[Chunk]:
        return self.lexical._chunks  # noqa: SLF001 - one object, split for clarity

    # -- indexing ----------------------------------------------------------

    def index(self, documents: list[Document]) -> None:
        self.lexical.index(documents)
        self.vectors = VectorIndex()
        self.degraded = ""
        if self.embedder is None:
            return

        chunks = self.chunks
        fingerprint = self.embedder.fingerprint
        keys = [text_key(chunk.text, fingerprint) for chunk in chunks]
        missing = [i for i, key in enumerate(keys) if key not in self.cache]

        if missing:
            log.info(
                "embedding %d of %d chunks (%d cached)",
                len(missing),
                len(chunks),
                len(chunks) - len(missing),
            )
            try:
                fresh = self.embedder.embed_documents([chunks[i].text for i in missing])
            except EmbeddingError as exc:
                # The corpus is still fully searchable lexically. Losing dense
                # retrieval is a quality cost, not an outage, and pretending
                # otherwise would take the whole system down with the endpoint.
                self.degraded = str(exc)
                log.warning("dense retrieval unavailable, using BM25 alone: %s", exc)
                return
            written = {keys[i]: normalise(vector) for i, vector in zip(missing, fresh, strict=True)}
            self.cache.update(written)
            put_many = getattr(self.store, "put_many", None)
            if put_many is not None:
                put_many(written)

        self.vectors = VectorIndex([self.cache[key] for key in keys])

    # -- searching ---------------------------------------------------------

    def search(
        self, query: str, *, k: int = 6, principals: frozenset[str] | None = None
    ) -> list[ScoredChunk]:
        """The top ``k`` chunks by fused rank, or by BM25 alone when dense is off."""
        # Ask each half for more than k: fusion is only interesting where the
        # two rankings disagree, and that happens below either one's top k.
        depth = max(k * 4, 20)
        lexical = self.lexical.search(query, k=depth, principals=principals)
        if self.embedder is None or not self.vectors:
            return lexical[:k]

        try:
            query_vector = normalise(self.embedder.embed_query(query))
        except (EmbeddingError, IndexError) as exc:
            log.warning("could not embed the question, using BM25 alone: %s", exc)
            return lexical[:k]

        chunks = self.chunks
        dense: list[ScoredChunk] = []
        for index, score in self.vectors.search(query_vector, depth * 2):
            chunk = chunks[index]
            if (
                principals is not None
                and chunk.allowed_principals
                and not (chunk.allowed_principals & principals)
            ):
                continue
            dense.append(ScoredChunk(chunk=chunk, score=score))
            if len(dense) >= depth:
                break

        # Dense first, so on a tie for the same position its choice leads. It is
        # the half that handles the phrasing people actually use, and the half
        # a fused ranking was most likely to bury.
        #
        # The route and the demotion both run on the *fused* list, after both
        # halves have voted: the lexical half already applied them internally,
        # but the dense half scores every chunk, so the combined view is the
        # one that decides. The route is the same pure function of question
        # and index the lexical half computed; recomputing is cheaper than
        # threading it through the call.
        within = (
            route_by_tags(query, self.lexical.routing_tags()) if self.lexical.tag_routing else None
        )
        fused = _interleave(dense, lexical)
        fused = demote_superseded(fused, len(fused))
        return guarantee_routed(fused, within, k)


def _chunk_key(chunk: Chunk) -> str:
    return f"{chunk.document_id}#{chunk.locator}#{hash(chunk.text)}"


def _interleave(*rankings: list[ScoredChunk]) -> list[ScoredChunk]:
    """Round robin: every retriever's Nth choice before anyone's N+1th.

    Parameter-free, and deterministic given deterministic inputs - both
    properties the answer cache depends on.

    Scores are rewritten to the fused position, and that is not cosmetic. The
    first version of this kept each chunk's original score, on the stated
    reasoning that nothing downstream compares them. That was simply false:
    StructuralReranker sorts by ``hit.score``, so it was handed BM25 scores
    (unbounded, 2 to 15 here) and cosine similarities (0.5 to 0.8) in one list
    and re-sorted the careful interleaved order by the mixture. Accuracy on the
    golden set went from 100% to 23.5% - thirteen answerable questions refused,
    because the right chunk was ranked below noise from the other scale.

    So the fused score is 1/position: comparable by construction, descending,
    and it leaves the reranker's multiplicative boost doing what it was written
    to do.
    """
    out: list[ScoredChunk] = []
    seen: set[str] = set()
    longest = max((len(ranking) for ranking in rankings), default=0)
    for position in range(longest):
        for ranking in rankings:
            if position >= len(ranking):
                continue
            scored = ranking[position]
            key = _chunk_key(scored.chunk)
            if key in seen:
                continue
            seen.add(key)
            out.append(ScoredChunk(chunk=scored.chunk, score=1.0 / (len(out) + 1)))
    return out
