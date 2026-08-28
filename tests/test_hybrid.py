"""BM25 and dense retrieval, fused.

The case that motivated this: "how much can I spend on dinner?" against a
document that says "meals are reimbursed up to EUR 45 per day". They share
almost no vocabulary, BM25 returns nothing useful, and the system looks like it
does not know something it plainly does.

The embedder here is a stub with hand-placed vectors rather than a real model.
That is deliberate: what needs testing is the fusion, the fallback and the
determinism, and a real model would make those tests slow, non-hermetic and no
more truthful.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pytest

from openknowledge.retrieval import BM25Retriever
from openknowledge.retrieval.base import Document
from openknowledge.retrieval.embed import EmbeddingError, normalise, text_key
from openknowledge.retrieval.hybrid import HybridRetriever

DINNER = Document(
    "expenses",
    "Expenses Policy",
    "Meals and subsistence. Meals are reimbursed up to EUR 45 per day when "
    "travelling on company business. Alcohol is not reimbursable.",
)
VPN = Document(
    "security",
    "Information Security Policy",
    "Remote access. Staff must connect through GlobalProtect before reaching "
    "internal systems. Split tunnelling is prohibited.",
)
PARKING = Document(
    "facilities",
    "Facilities Guide",
    "Parking. Spaces are allocated by the office manager on request.",
)


@dataclass
class StubEmbedder:
    """Vectors placed by hand, so 'dinner' sits near 'meals' and nowhere else."""

    model: str = "stub-embed"
    base_url: str = "http://stub/v1"
    fail: bool = False
    calls: list[list[str]] = field(default_factory=list)

    @property
    def fingerprint(self) -> str:
        return f"{self.model}@stub"

    #: The prefixes a real embedder applies. Recorded so a test can check they
    #: are asymmetric - a question and the passage answering it go to different
    #: places, and swapping the two is worse than using neither.
    document_prefix: str = "search_document: "
    query_prefix: str = "search_query: "

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self.fail:
            raise EmbeddingError("stub-embed: endpoint is down")
        self.calls.append(texts)
        return [self._vector(text) for text in texts]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed([f"{self.document_prefix}{text}" for text in texts])

    def embed_query(self, text: str) -> list[float]:
        return self.embed([f"{self.query_prefix}{text}"])[0]

    @staticmethod
    def _vector(text: str) -> list[float]:
        low = text.lower()
        # Three crude topics. Dinner/food, network, and everything else.
        food = any(w in low for w in ("meal", "dinner", "food", "subsistence", "eat"))
        net = any(w in low for w in ("vpn", "globalprotect", "remote access", "tunnel"))
        return [1.0 if food else 0.0, 1.0 if net else 0.0, 0.2]


def build(*, embedder: object | None) -> HybridRetriever:
    retriever = HybridRetriever(lexical=BM25Retriever(), embedder=embedder)  # type: ignore[arg-type]
    retriever.index([DINNER, VPN, PARKING])
    return retriever


def test_bm25_alone_misses_the_paraphrase() -> None:
    """The gap, stated as a test, so the fix has something to be measured against."""
    lexical = BM25Retriever()
    lexical.index([DINNER, VPN, PARKING])

    hits = lexical.search("how much can I spend on dinner?", k=1)
    assert not hits or hits[0].chunk.document_id != "expenses"


def test_the_fused_retriever_finds_it() -> None:
    hits = build(embedder=StubEmbedder()).search("how much can I spend on dinner?", k=1)
    assert hits, "found nothing at all"
    assert hits[0].chunk.document_id == "expenses"


def test_exact_terms_still_win_where_bm25_is_strongest() -> None:
    """Embeddings blur identifiers toward their neighbours; BM25 does not.

    Keeping lexical search in the fusion is the whole reason the paraphrase fix
    does not cost accuracy on the terms this workload is full of.
    """
    hits = build(embedder=StubEmbedder()).search("GlobalProtect", k=1)
    assert hits[0].chunk.document_id == "security"


def test_an_unreachable_embedder_degrades_to_bm25_rather_than_failing() -> None:
    """Most installs have no embedding model on first run. That is not an outage."""
    retriever = build(embedder=StubEmbedder(fail=True))

    assert retriever.degraded, "did not record why the dense half is missing"
    hits = retriever.search("GlobalProtect", k=1)
    assert hits[0].chunk.document_id == "security"


def test_the_same_question_retrieves_the_same_context() -> None:
    """The answer cache is unsound without this."""
    retriever = build(embedder=StubEmbedder())
    first = [h.chunk.text for h in retriever.search("how much for dinner", k=3)]
    second = [h.chunk.text for h in retriever.search("how much for dinner", k=3)]
    assert first == second


def test_unchanged_text_is_never_embedded_twice() -> None:
    """Re-indexing a corpus where one file changed must not re-embed the rest."""
    embedder = StubEmbedder()
    retriever = HybridRetriever(lexical=BM25Retriever(), embedder=embedder)  # type: ignore[arg-type]
    retriever.index([DINNER, VPN])
    first_pass = sum(len(batch) for batch in embedder.calls)
    embedder.calls.clear()

    retriever.index([DINNER, VPN, PARKING])  # one new document
    second_pass = sum(len(batch) for batch in embedder.calls)

    assert first_pass >= 2
    assert second_pass == 1, "re-embedded text that had not changed"


def test_changing_the_embedding_model_changes_the_corpus_version() -> None:
    """Vectors from two models are not comparable, so cached answers must go."""
    one = build(embedder=StubEmbedder(model="a"))
    two = build(embedder=StubEmbedder(model="b"))
    assert one.corpus_version != two.corpus_version


def test_dense_hits_respect_access_control() -> None:
    secret = Document(
        "secret",
        "Redundancy Plan",
        "Meals for the offsite dinner are covered.",
        allowed_principals=frozenset({"hr"}),
    )
    retriever = HybridRetriever(lexical=BM25Retriever(), embedder=StubEmbedder())  # type: ignore[arg-type]
    retriever.index([DINNER, secret])

    hits = retriever.search("dinner", k=5, principals=frozenset({"all-staff"}))
    assert {h.chunk.document_id for h in hits} == {"expenses"}


def test_a_vector_is_stored_once_per_model_and_text() -> None:
    assert text_key("hello", "a@x") != text_key("hello", "b@x")
    assert text_key("hello", "a@x") == text_key("hello", "a@x")


def test_normalising_makes_cosine_a_dot_product() -> None:
    vector = normalise([3.0, 4.0])
    assert math.isclose(sum(x * x for x in vector), 1.0, rel_tol=1e-6)
    assert normalise([0.0, 0.0]) == [0.0, 0.0]  # no division by zero


@pytest.mark.parametrize("k", [1, 3, 6])
def test_it_never_returns_more_than_asked_for(k: int) -> None:
    assert len(build(embedder=StubEmbedder()).search("meals", k=k)) <= k


# --- vectors on disk ---------------------------------------------------------


def test_vectors_survive_a_restart(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Embedding is the one genuinely slow part of indexing.

    Re-doing it on every start would make a ten-thousand-chunk corpus take
    minutes to come up, for work that was already done.
    """
    from openknowledge.retrieval.vectorstore import VectorCache

    path = tmp_path / "vectors.db"
    first = VectorCache(path)
    first.put_many({"a": [1.0, 0.0], "b": [0.0, 1.0]})
    assert len(first) == 2
    first.close()

    second = VectorCache(path)
    assert second.load() == {"a": [1.0, 0.0], "b": [0.0, 1.0]}
    assert second.evict_all() == 2
    assert second.load() == {}
    second.close()


def test_a_cached_vector_is_not_recomputed_across_restarts() -> None:
    """The cache is keyed on text and model, so it survives re-chunking."""
    embedder = StubEmbedder()
    shared: dict[str, list[float]] = {}

    first = HybridRetriever(lexical=BM25Retriever(), embedder=embedder, cache=shared)  # type: ignore[arg-type]
    first.index([DINNER, VPN])
    embedder.calls.clear()

    second = HybridRetriever(lexical=BM25Retriever(), embedder=embedder, cache=shared)  # type: ignore[arg-type]
    second.index([DINNER, VPN])
    assert embedder.calls == [], "re-embedded a corpus it had already seen"
    assert second.search("dinner", k=1)[0].chunk.document_id == "expenses"


# --- how the two rankings combine -------------------------------------------


def test_neither_half_can_be_drowned_out_by_the_other() -> None:
    """The reason this is interleaved rather than rank-fused.

    Measured on real documents, reciprocal rank fusion scored 11/13 where dense
    retrieval alone scored 13/13: on a paraphrased question BM25's ranking is
    close to noise, and RRF counts all twenty of its chunks as votes, drowning
    the one the dense half put first. Interleaving cannot do that - each
    retriever's first choice is in the top two by construction.
    """
    from openknowledge.retrieval.base import ScoredChunk
    from openknowledge.retrieval.hybrid import _interleave

    def scored(name: str) -> ScoredChunk:
        return ScoredChunk(chunk=Chunk_(name), score=1.0)

    dense = [scored("right"), *[scored(f"d{i}") for i in range(9)]]
    lexical = [scored(f"noise{i}") for i in range(20)]

    order = [s.chunk.document_id for s in _interleave(dense, lexical)]
    assert order[0] == "right", "the dense half's best hit was buried"
    assert order[1] == "noise0", "the lexical half's best hit was buried"


def test_interleaving_drops_duplicates_without_reordering() -> None:
    from openknowledge.retrieval.base import ScoredChunk
    from openknowledge.retrieval.hybrid import _interleave

    shared = Chunk_("both")
    dense = [ScoredChunk(chunk=shared, score=1.0), ScoredChunk(chunk=Chunk_("d"), score=0.9)]
    lexical = [ScoredChunk(chunk=shared, score=5.0), ScoredChunk(chunk=Chunk_("l"), score=4.0)]

    order = [s.chunk.document_id for s in _interleave(dense, lexical)]
    assert order == ["both", "d", "l"]


def Chunk_(name: str):  # noqa: N802 - a builder, named to read at the call site
    from openknowledge.retrieval.base import Chunk

    return Chunk(
        chunk_id=f"{name}#1",
        document_id=name,
        document_title=name.title(),
        text=f"text of {name}",
        locator="chunk 1",
    )


def test_the_question_and_the_passage_get_different_prefixes() -> None:
    """Asymmetric by design; swapping them is worse than using neither."""
    from openknowledge.retrieval.embed import prefixes_for

    document, query = prefixes_for("nomic-embed-text")
    assert document and query and document != query
    assert prefixes_for("some-unknown-model") == ("", "")


def test_changing_the_prefix_changes_the_fingerprint() -> None:
    """Vectors embedded under different prefixes are not comparable."""
    from openknowledge.retrieval.embed import Embedder

    plain = Embedder(model="x", document_prefix="", query_prefix="")
    prefixed = Embedder(model="x", document_prefix="passage: ", query_prefix="query: ")
    assert plain.fingerprint != prefixed.fingerprint


def test_the_fused_score_is_comparable_across_retrievers() -> None:
    """The regression that took accuracy from 100% to 23.5%.

    The reranker downstream sorts by score. Handed BM25 scores (unbounded) and
    cosine similarities (0 to 1) in one list, it re-sorted the interleaved order
    by a mixture of two scales and buried the right chunk under noise from the
    other one. Thirteen answerable questions were refused.

    The fused score has to describe the fused position and nothing else.
    """
    from openknowledge.retrieval.base import ScoredChunk
    from openknowledge.retrieval.hybrid import _interleave

    dense = [ScoredChunk(chunk=Chunk_("d1"), score=0.61)]  # cosine
    lexical = [ScoredChunk(chunk=Chunk_("l1"), score=14.2)]  # BM25

    fused = _interleave(dense, lexical)
    assert [s.chunk.document_id for s in fused] == ["d1", "l1"]
    assert [s.score for s in fused] == [1.0, 0.5], "kept an incomparable score"
    assert fused[0].score > fused[1].score, "fused score must fall with position"


def test_document_ids_are_delegated_like_the_rest_of_the_surface() -> None:
    """The eval preflight asks the retriever for document_ids; a hybrid
    configuration crashed it because only the lexical half answered."""
    retriever = build(embedder=StubEmbedder())
    assert retriever.document_ids() == {"expenses", "security", "facilities"}
