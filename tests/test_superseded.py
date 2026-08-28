"""Documents that declare themselves superseded stop competing with the
current copy.

Measured on the Aveline corpus before this existed: the archived 2023
expenses policy - which opens with "SUPERSEDED by Expenses Policy v4.1.
Retained for audit only." - was retrieved alongside the current one, and the
local model led with the archived CFO limit and disclosed the archived meal
rate. Both figures are on the golden set's must-not-say lists. The document
had already said "do not use me"; retrieval just wasn't listening.

The demotion is a drop, not a downrank: a downranked stale figure still lands
in the model's context, and the model is then asked to adjudicate a
versioning question retrieval already knows the answer to. It fails open when
nothing current matches, so a corpus whose only document on a topic was
retired without replacement still answers from it, cited as what it is.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from openknowledge.documents import declares_superseded
from openknowledge.knowledge import KnowledgeStore, scan_documents
from openknowledge.knowledge.pipeline import draft_for_documents
from openknowledge.retrieval import BM25Retriever, Document
from openknowledge.retrieval.base import ScoredChunk, chunk_document, demote_superseded
from openknowledge.retrieval.hybrid import HybridRetriever
from openknowledge.types import Citation

# -- detection ---------------------------------------------------------------


@pytest.mark.parametrize(
    "head",
    [
        "# Expenses Policy (2023)\n**Status:** SUPERSEDED by Expenses Policy v4.1. "
        "Retained for audit only.",
        "Travel Policy\nStatus: superseded\nEffective until 2024.",
        "Old procedure. This document has been replaced by the 2026 edition.",
        "Data handling rules. This policy is no longer in force.",
        "Retention schedule, retained for reference purposes after the 2025 merge.",
    ],
)
def test_documents_that_declare_themselves_retired_are_detected(head: str) -> None:
    assert declares_superseded(head)


@pytest.mark.parametrize(
    "head",
    [
        # The *current* copy names what it replaced. "Supersedes:" is the new
        # document speaking and must not mark it stale.
        "# Expenses Policy\n**Version:** 4.1\n**Supersedes:** Expenses Policy v3.0 (2023)",
        "# Travel Policy\nThis policy supersedes all previous travel guidance.",
        "# Onboarding\nWelcome to the company. Nothing here about versions.",
    ],
)
def test_current_documents_are_not_mistaken_for_retired_ones(head: str) -> None:
    assert not declares_superseded(head)


def test_a_body_mention_of_supersession_does_not_retire_the_document() -> None:
    """Only the head is examined: a current policy discussing its superseded
    predecessor three sections in is talking about another document."""
    body = (
        "# Current Policy\n\n"
        + ("All figures below are current. " * 40)
        + ("For history: the 2023 edition was superseded by this document.")
    )
    assert not declares_superseded(body)


# -- stamping ----------------------------------------------------------------

OLD = Document(
    "expenses-2023",
    "Expenses Policy (2023)",
    "Status: SUPERSEDED by Expenses Policy v4.1. Retained for audit only. "
    "The meal allowance is EUR 35 per day for domestic travel. "
    "The Chief Financial Officer may approve up to EUR 15,000.",
    superseded=True,
)
NEW = Document(
    "expenses",
    "Expenses Policy",
    "Supersedes: Expenses Policy v3.0. "
    "The meal allowance is EUR 45 per day for domestic travel. "
    "The Chief Financial Officer may approve up to EUR 25,000.",
)
PARKING = Document(
    "facilities",
    "Facilities Guide",
    "Parking spaces are allocated by the office manager on request.",
)


def test_chunks_inherit_the_superseded_flag() -> None:
    assert all(c.superseded for c in chunk_document(OLD))
    assert not any(c.superseded for c in chunk_document(NEW))


# -- the demotion itself -----------------------------------------------------


def _scored(*chunks_flags: tuple[str, bool]) -> list[ScoredChunk]:
    out = []
    for i, (doc_id, stale) in enumerate(chunks_flags):
        chunk = chunk_document(Document(doc_id, doc_id, "words " * 30, superseded=stale))[0]
        out.append(ScoredChunk(chunk=chunk, score=1.0 / (i + 1)))
    return out


def test_superseded_chunks_are_dropped_when_anything_current_matched() -> None:
    ranked = _scored(("old", True), ("new", False), ("old2", True), ("other", False))
    kept = demote_superseded(ranked, k=3)
    assert [s.chunk.document_id for s in kept] == ["new", "other"]


def test_demotion_fails_open_when_only_superseded_documents_match() -> None:
    ranked = _scored(("old", True), ("old2", True))
    kept = demote_superseded(ranked, k=2)
    assert [s.chunk.document_id for s in kept] == ["old", "old2"]


def test_bm25_no_longer_serves_the_archived_figure() -> None:
    retriever = BM25Retriever()
    retriever.index([OLD, NEW, PARKING])
    hits = retriever.search("what is the meal allowance for domestic travel?", k=4)
    assert hits, "the current policy must still be found"
    assert all(h.chunk.document_id != "expenses-2023" for h in hits)


def test_bm25_still_answers_from_a_retired_document_when_it_is_all_there_is() -> None:
    lonely = Document(
        "old-travel",
        "Travel Policy (retired)",
        "Status: superseded. Business class flights require director approval.",
        superseded=True,
    )
    retriever = BM25Retriever()
    retriever.index([lonely, PARKING])
    hits = retriever.search("who approves business class flights?", k=2)
    assert hits and hits[0].chunk.document_id == "old-travel"


def test_the_hybrid_fused_ranking_is_demoted_too() -> None:
    """Dense retrieval scores every chunk, so the archive can arrive through
    the dense half even when the lexical half dropped it."""

    class EveryoneEmbedder:
        model = "stub"
        base_url = "http://stub/v1"
        fingerprint = "stub@test"

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0] for _ in texts]

        def embed_query(self, text: str) -> list[float]:
            return [1.0, 0.0]

    retriever = HybridRetriever(lexical=BM25Retriever(), embedder=EveryoneEmbedder())  # type: ignore[arg-type]
    retriever.index([OLD, NEW, PARKING])
    hits = retriever.search("meal allowance for domestic travel", k=3)
    assert hits
    assert all(h.chunk.document_id != "expenses-2023" for h in hits)


# -- ingest: no money spent on, no alarms raised by, retired documents -------


class NoDraftProvider:
    """A provider that fails the test if the pipeline tries to spend money."""

    model_id = "must-not-be-called"

    async def chat(self, *args: object, **kwargs: object) -> object:
        raise AssertionError("drafting was attempted for a superseded document")

    async def chat_stream(self, *args: object, **kwargs: object) -> object:
        raise AssertionError("drafting was attempted for a superseded document")


async def test_superseded_documents_are_not_drafted_from() -> None:
    with KnowledgeStore() as store:
        report = await draft_for_documents(
            [OLD],
            store=store,
            provider=NoDraftProvider(),  # type: ignore[arg-type]
            corpus_version="c1",
            document_ids=frozenset({"expenses-2023"}),
        )
    assert report.documents_drafted == 0
    assert report.cost_usd == 0.0
    assert any("superseded" in note for note in report.notes)
    assert any("expenses-2023" in note for note in report.notes)


def test_an_arriving_archive_copy_does_not_contradict_stored_answers() -> None:
    """A stale copy disagreeing with the current answers is expected, not
    news - it must not page an admin the way a genuinely new document does."""
    with KnowledgeStore() as store:
        proposal = store.propose(
            canonical_query="what is the meal allowance",
            question="What is the meal allowance?",
            answer="The meal allowance is EUR 45 per day for domestic travel. [expenses]",
            citations=(Citation("expenses", "Expenses Policy", "meals", None),),
            origin_documents=("expenses",),
            corpus_version="c1",
            support_ratio=0.95,
        )
        assert proposal is not None
        store.approve(proposal.id, reviewer="finance")

        retriever = BM25Retriever()
        retriever.index([NEW, OLD])
        report = scan_documents([NEW, OLD], store=store, retriever=retriever)
        assert report.answers_contradicted == 0


def test_the_versions_conflict_stays_visible_to_admins() -> None:
    """Demotion decides retrieval, not governance: the duplicated pair still
    lands in /manage so a human can decide which copy stands."""
    with KnowledgeStore() as store:
        scan_documents([NEW, OLD], store=store)
        kinds = {c.kind for c in store.open_conflicts()}
        assert kinds, "the twin documents disagree and that must be recorded"


# -- the shipped corpus, as a spec -------------------------------------------

AVELINE = Path(__file__).resolve().parent.parent / "evals" / "corpus" / "aveline"


def test_the_aveline_archive_never_reaches_the_model() -> None:
    """The failure that motivated all of this, pinned to the shipped corpus:
    ask the questions whose golden cases forbid the archived figures, and the
    archive must not be in the retrieved context."""
    from openknowledge.connectors import LocalFilesConnector

    documents = LocalFilesConnector(AVELINE).fetch()
    archived = [d for d in documents if d.superseded]
    assert [d.document_id for d in archived] == ["archive-expenses-policy-2023"]

    retriever = BM25Retriever()
    retriever.index(documents)
    for question in (
        "How much can I claim for meals per day when travelling in Ireland?",
        "What is the approval limit for the Chief Financial Officer?",
    ):
        hits = retriever.search(question, k=6)
        assert hits, question
        assert all(h.chunk.document_id != "archive-expenses-policy-2023" for h in hits), question
