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

from .base import Chunk, Document, ScoredChunk, chunk_document, tokenize

_K1 = 1.5
_B = 0.75


class BM25Retriever:
    """In-memory BM25 over chunked documents."""

    def __init__(self, *, target_words: int = 350, overlap_words: int = 60) -> None:
        self._target_words = target_words
        self._overlap_words = overlap_words
        self._chunks: list[Chunk] = []
        self._term_freqs: list[Counter[str]] = []
        self._lengths: list[int] = []
        self._doc_freq: Counter[str] = Counter()
        self._avg_length = 0.0
        self._corpus_version = "empty"
        self._doc_principals: dict[str, frozenset[str]] = {}

    @property
    def corpus_version(self) -> str:
        return self._corpus_version

    def __len__(self) -> int:
        return len(self._chunks)

    def index(self, documents: list[Document]) -> None:
        """Rebuild the index from scratch.

        A full rebuild is the honest operation here: it makes ``corpus_version``
        a true function of the current corpus, so a deleted document really does
        disappear from every future answer instead of lingering in the index.
        """
        self._chunks = []
        self._term_freqs = []
        self._lengths = []
        self._doc_freq = Counter()
        self._doc_principals = {}

        for doc in documents:
            self._doc_principals[doc.document_id] = doc.allowed_principals
            for chunk in chunk_document(
                doc, target_words=self._target_words, overlap_words=self._overlap_words
            ):
                tokens = tokenize(chunk.text)
                self._chunks.append(chunk)
                self._term_freqs.append(Counter(tokens))
                self._lengths.append(len(tokens))
                self._doc_freq.update(set(tokens))

        self._avg_length = (sum(self._lengths) / len(self._lengths)) if self._lengths else 0.0

        digest = hashlib.sha256()
        for doc in sorted(documents, key=lambda d: d.document_id):
            digest.update(doc.document_id.encode("utf-8"))
            digest.update(b"\x1f")
            digest.update(doc.content_hash.encode("utf-8"))
            digest.update(b"\x1f")
        self._corpus_version = digest.hexdigest()[:16] if documents else "empty"

    def search(
        self, query: str, *, k: int = 6, principals: frozenset[str] | None = None
    ) -> list[ScoredChunk]:
        if not self._chunks:
            return []

        query_terms = tokenize(query)
        if not query_terms:
            return []

        n = len(self._chunks)
        scored: list[ScoredChunk] = []

        for i, chunk in enumerate(self._chunks):
            # Access control is applied during scoring, not after: filtering a
            # top-k list would silently shrink results for restricted users.
            if (
                principals is not None
                and chunk.allowed_principals
                and not (chunk.allowed_principals & principals)
            ):
                continue

            tf = self._term_freqs[i]
            length = self._lengths[i]
            score = 0.0
            for term in query_terms:
                freq = tf.get(term, 0)
                if not freq:
                    continue
                df = self._doc_freq[term]
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                denom = freq + _K1 * (1 - _B + _B * length / (self._avg_length or 1.0))
                score += idf * (freq * (_K1 + 1)) / denom
            if score > 0:
                scored.append(ScoredChunk(chunk=chunk, score=score))

        # Sort by score, then chunk_id: ties must break the same way on every
        # run or identical questions would retrieve different context.
        scored.sort(key=lambda s: (-s.score, s.chunk.chunk_id))
        return scored[:k]

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
        for doc_id in document_ids:
            if doc_id not in self._doc_principals:
                return False
            allowed = self._doc_principals[doc_id]
            if allowed and not (allowed & principals):
                return False
        return True
