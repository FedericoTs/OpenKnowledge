"""Choosing which retrieved chunks are worth a slot.

Each of these is a recall failure BM25 cannot see, because it scores every chunk
on its own: one document taking every slot, overlapping windows occupying two
slots with one fact, and a matching heading counted as ordinary prose. They
matter because a retrieval miss becomes a gate failure, and a gate failure
escalates - which is where a tuned deployment's money actually goes.
"""

from __future__ import annotations

from openknowledge.retrieval.base import Chunk, ScoredChunk
from openknowledge.retrieval.rerank import StructuralReranker


def hit(chunk_id: str, document: str, text: str, score: float) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(chunk_id=chunk_id, document_id=document, document_title=document, text=text),
        score=score,
    )


def ids(hits: list[ScoredChunk]) -> list[str]:
    return [h.chunk.chunk_id for h in hits]


def test_one_document_cannot_take_every_slot() -> None:
    hits = [hit(f"a{n}", "expenses", f"expense rule {n}", 10.0 - n) for n in range(5)]
    hits.append(hit("b0", "travel", "travel approval rule", 1.0))

    kept = StructuralReranker(max_per_document=2).rerank("expense approval", hits, k=3)

    assert {h.chunk.document_id for h in kept} == {"expenses", "travel"}
    assert sum(h.chunk.document_id == "expenses" for h in kept) == 2


def test_the_cap_yields_when_only_one_document_is_relevant() -> None:
    """A tidier spread is not worth sending less evidence than was asked for."""
    hits = [hit(f"a{n}", "expenses", f"expense rule {n}", 10.0 - n) for n in range(5)]

    kept = StructuralReranker(max_per_document=2).rerank("expense", hits, k=4)

    assert len(kept) == 4, "capping must not shrink the context when there is nothing else"


def test_a_near_duplicate_does_not_occupy_a_second_slot() -> None:
    """The chunker overlaps windows by design, so this is not a rare case."""
    shared = "employees may claim up to five hundred euro per trip without prior approval"
    hits = [
        hit("a0", "expenses", shared, 10.0),
        hit("a1", "expenses", shared + " from a line manager", 9.0),
        hit("c0", "travel", "trips must be booked through the travel desk", 1.0),
    ]

    kept = StructuralReranker(max_per_document=3).rerank("expense claim", hits, k=3)

    assert "c0" in ids(kept)
    assert len(kept) < 3 or "a1" not in ids(kept)


def test_a_matching_heading_outranks_an_incidental_mention() -> None:
    heading_match = hit(
        "a0",
        "handbook",
        "PARENTAL LEAVE > 4.2 Entitlement: Employees get twenty weeks.",
        5.0,
    )
    passing_mention = hit(
        "b0",
        "handbook",
        "BENEFITS > 9.1 Other: See also parental leave elsewhere in this handbook.",
        5.4,
    )

    kept = StructuralReranker().rerank(
        "parental leave entitlement", [passing_mention, heading_match], k=2
    )

    assert ids(kept)[0] == "a0"


def test_a_chunk_with_no_heading_trail_is_not_penalised() -> None:
    plain = hit("a0", "d", "Employees get twenty weeks of parental leave.", 5.0)
    kept = StructuralReranker().rerank("parental leave", [plain], k=1)
    assert kept[0].score == 5.0


def test_ties_break_the_same_way_every_run() -> None:
    """Identical questions must produce identical context, or the cache key lies."""
    hits = [hit(f"c{n}", f"doc{n}", f"same words here {n}", 1.0) for n in range(6)]
    reranker = StructuralReranker()

    first = ids(reranker.rerank("same words", list(hits), k=3))
    second = ids(reranker.rerank("same words", list(reversed(hits)), k=3))
    assert first == second


def test_it_never_returns_more_than_asked_for() -> None:
    hits = [hit(f"c{n}", f"doc{n}", f"text {n}", float(10 - n)) for n in range(10)]
    assert len(StructuralReranker().rerank("text", hits, k=4)) == 4
    assert StructuralReranker().rerank("text", hits, k=0) == []
    assert StructuralReranker().rerank("text", [], k=4) == []


def test_reranking_is_part_of_the_cache_key() -> None:
    """Change what goes into the prompt and answers built on the old context
    must not survive."""
    from openknowledge.cache import AnswerStore
    from openknowledge.cascade import Cascade
    from openknowledge.config import Settings
    from openknowledge.retrieval import BM25Retriever

    def policy(**changes: object) -> str:
        settings = Settings().model_copy(update=changes)
        cascade = Cascade(
            store=AnswerStore(":memory:"), retriever=BM25Retriever(), settings=settings
        )
        return cascade._key_context().policy_version

    assert policy() != policy(rerank_candidates=0)
    assert policy() != policy(rerank_max_per_document=4)
