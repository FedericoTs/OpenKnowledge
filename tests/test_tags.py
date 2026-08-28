"""Tags derived at upload, and the retrieval radius they shrink.

The promise under test: indexing a document also derives a readable set of
tags from its name, title, headings and distinctive vocabulary - free, no
model - and a question that names its documents decisively is searched only
against those documents. The promise that matters more: any ambiguity means
the radius does not shrink at all, because the catastrophic failure here is
a question routed away from the document that held its answer.
"""

from __future__ import annotations

from pathlib import Path

from openknowledge.retrieval import BM25Retriever, Document
from openknowledge.retrieval.bm25 import BM25Retriever as _BM25
from openknowledge.retrieval.hybrid import HybridRetriever
from openknowledge.retrieval.tags import (
    corpus_document_frequency,
    derive_tags,
    fold_tags,
    route_by_tags,
)

EXPENSES = Document(
    "hr-expenses-policy",
    "Expenses Policy",
    "Meal allowances. The meal allowance is EUR 45 per day for domestic travel. "
    "Claims must be submitted within 60 days with receipts.",
)
SECURITY = Document(
    "security-information-security",
    "Information Security Policy",
    "Passwords must be rotated. USB storage devices are prohibited on all "
    "corporate laptops. Incidents go to the security team.",
)
PARKING = Document(
    "facilities-guide",
    "Facilities Guide",
    "Parking spaces are allocated by the office manager on request.",
)

CORPUS = [EXPENSES, SECURITY, PARKING]


def _tags(document: Document) -> tuple[str, ...]:
    return derive_tags(document, corpus_document_frequency(CORPUS), len(CORPUS))


# -- derivation --------------------------------------------------------------


def test_tags_come_from_the_id_the_title_and_the_body() -> None:
    tags = _tags(EXPENSES)
    assert "hr" in tags  # the operator's folder taxonomy survives
    assert "expenses" in tags
    assert "policy" in tags
    assert any(t.startswith("meal") for t in tags)  # distinctive body term


def test_tags_are_readable_words_not_stems() -> None:
    """They are shown in the document listing, so "expenses" must stay
    "expenses" - folding is a matching detail, done separately."""
    for tag in _tags(EXPENSES):
        assert tag.isalpha() or tag.isalnum()
    assert "expenses" in _tags(EXPENSES)
    assert "expen" not in _tags(EXPENSES)


def test_word_forms_share_one_tag() -> None:
    doc = Document(
        "hr-leave",
        "Leave",
        "Requesting leave. Leave requests and the requested leave days are "
        "approved by managers. " * 3,
    )
    tags = derive_tags(doc, corpus_document_frequency([doc]), 1)
    folded = fold_tags(tags)
    assert len(folded) == len(tags)  # dedupe happened at derivation


def test_derivation_is_deterministic() -> None:
    df = corpus_document_frequency(CORPUS)
    assert derive_tags(EXPENSES, df, 3) == derive_tags(EXPENSES, df, 3)


def test_noise_and_numbers_are_not_tags() -> None:
    tags = _tags(EXPENSES)
    assert "the" not in tags
    assert "45" not in tags
    assert "60" not in tags


# -- routing -----------------------------------------------------------------


def _folded(corpus: list[Document]) -> dict[str, frozenset[str]]:
    df = corpus_document_frequency(corpus)
    return {d.document_id: fold_tags(derive_tags(d, df, len(corpus))) for d in corpus}


def test_a_question_naming_its_document_routes_to_it() -> None:
    route = route_by_tags("what does the expenses policy say about meals?", _folded(CORPUS))
    assert route == {"hr-expenses-policy"}


def test_one_shared_word_is_coincidence_not_a_route() -> None:
    """ "policy" alone must not route: both policies carry it, and a single
    hit is exactly the false narrowing the thresholds exist to prevent."""
    assert route_by_tags("what is our policy?", _folded(CORPUS)) is None


def test_a_question_matching_nothing_fails_open() -> None:
    assert route_by_tags("how do I reset my printer?", _folded(CORPUS)) is None


def test_a_broad_question_fails_open() -> None:
    """When most of the corpus matches, narrowing adds risk, not precision."""
    twins = [
        Document(f"hr-policy-{i}", f"Policy {i}", "Expenses policy rules for claims.")
        for i in range(3)
    ]
    assert route_by_tags("expenses policy rules", _folded(twins)) is None


def test_word_forms_still_route() -> None:
    route = route_by_tags("expense policies for meals", _folded(CORPUS))
    assert route == {"hr-expenses-policy"}


# -- retrieval integration ---------------------------------------------------


def test_the_named_document_leads_the_radius() -> None:
    retriever = BM25Retriever()
    retriever.index(CORPUS)
    hits = retriever.search("what does the expenses policy say about meal claims?", k=6)
    assert hits
    assert hits[0].chunk.document_id == "hr-expenses-policy"


def test_a_route_never_thins_the_context() -> None:
    """The regression the repository golden set caught live: routed to a
    document that chunks to a single window, a hard filter handed the model
    a one-chunk context and it refused a question it answers happily with a
    fuller one. A route reorders; only the cut to k excludes."""
    retriever = BM25Retriever()
    retriever.index(CORPUS)
    question = "what does the expenses policy say about meal claims?"
    routed = retriever.search(question, k=6)
    unrouted_retriever = BM25Retriever(tag_routing=False)
    unrouted_retriever.index(CORPUS)
    unrouted = unrouted_retriever.search(question, k=6)
    assert len(routed) == len(unrouted)  # same radius, different order
    assert {h.chunk.chunk_id for h in routed} == {h.chunk.chunk_id for h in unrouted}


def test_exclusion_happens_when_the_named_documents_fill_the_radius() -> None:
    """At scale the named documents have chunks to spare, and the cut to k
    then excludes the strangers entirely - the radius decrease, earned only
    when it costs no context."""
    wordy = Document(
        "hr-expenses-policy",
        "Expenses Policy",
        "Expenses policy for meal claims. " * 12,
    )
    retriever = BM25Retriever(target_words=12, overlap_words=3)
    retriever.index([wordy, SECURITY, PARKING])
    hits = retriever.search("what does the expenses policy say about meal claims?", k=3)
    assert len(hits) == 3
    assert {h.chunk.document_id for h in hits} == {"hr-expenses-policy"}


def test_routing_off_restores_the_old_behaviour_exactly() -> None:
    routed = BM25Retriever()
    routed.index(CORPUS)
    unrouted = BM25Retriever(tag_routing=False)
    unrouted.index(CORPUS)

    # A question no route fires for must retrieve identically either way.
    question = "who allocates parking?"
    assert [h.chunk.chunk_id for h in routed.search(question, k=6)] == [
        h.chunk.chunk_id for h in unrouted.search(question, k=6)
    ]


def test_access_control_still_applies_inside_a_route() -> None:
    walled = Document(
        "hr-expenses-policy",
        "Expenses Policy",
        EXPENSES.text,
        allowed_principals=frozenset({"group:finance"}),
    )
    retriever = BM25Retriever()
    retriever.index([walled, SECURITY, PARKING])
    hits = retriever.search(
        "what does the expenses policy say about meal claims?",
        k=6,
        principals=frozenset({"user:someone", "authenticated"}),
    )
    # The route names a document the asker cannot see; they get whatever else
    # matched, exactly as before tags existed - never the walled document.
    assert all(h.chunk.document_id != "hr-expenses-policy" for h in hits)


def test_the_route_admits_the_archive_and_demotion_still_drops_it() -> None:
    """Tag routing and superseded demotion compose: the archive twin shares
    the route's tags, and then loses to the current copy as always."""
    archive = Document(
        "hr-expenses-policy-2023",
        "Expenses Policy (2023)",
        "Status: SUPERSEDED by v4.1. Retained for audit only. "
        "Meal allowances. The meal allowance is EUR 35 per day for domestic travel.",
        superseded=True,
    )
    retriever = BM25Retriever()
    retriever.index([EXPENSES, archive, SECURITY, PARKING])
    hits = retriever.search("what does the expenses policy say about meal allowances?", k=6)
    assert hits
    assert hits[0].chunk.document_id == "hr-expenses-policy"
    assert all(h.chunk.document_id != "hr-expenses-policy-2023" for h in hits)


def test_the_dense_half_respects_the_route() -> None:
    """Cosine scores every chunk, so without the shared route the dense half
    would smuggle excluded documents back into the fused ranking."""

    class LovesParking:
        model = "stub"
        base_url = "http://stub/v1"
        fingerprint = "stub@test"

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [[0.0, 1.0] if "Parking" in t else [1.0, 0.0] for t in texts]

        def embed_query(self, text: str) -> list[float]:
            return [0.0, 1.0]  # every question looks like parking to this stub

    retriever = HybridRetriever(lexical=_BM25(), embedder=LovesParking())  # type: ignore[arg-type]
    retriever.index(CORPUS)
    hits = retriever.search("what does the expenses policy say about meal claims?", k=3)
    assert hits
    # Without the fused-level route the parking chunk would lead - the stub
    # ranks it first for every question. The named document must lead.
    assert hits[0].chunk.document_id == "hr-expenses-policy"


# -- the shipped corpus, as a spec -------------------------------------------

AVELINE = Path(__file__).resolve().parent.parent / "evals" / "corpus" / "aveline"
GOLDEN_AVELINE = Path(__file__).resolve().parent.parent / "evals" / "golden-aveline"
DOCUMENTS = Path(__file__).resolve().parent.parent / "documents"
GOLDEN = Path(__file__).resolve().parent.parent / "evals" / "golden"


def test_the_repository_golden_set_survives_routing_too() -> None:
    """Added after the live golden run caught what the aveline preflight
    could not: both shipped corpus-and-set pairings are pinned here, so a
    routing change that strands either set's evidence fails the build."""
    from openknowledge.connectors import LocalFilesConnector
    from openknowledge.evaluation.dataset import load_cases
    from openknowledge.evaluation.preflight import preflight
    from openknowledge.retrieval.rerank import StructuralReranker

    retriever = BM25Retriever()
    retriever.index(LocalFilesConnector(DOCUMENTS).fetch())
    report = preflight(
        load_cases(GOLDEN),
        retriever=retriever,
        k=6,
        candidates=30,
        reranker=StructuralReranker(max_per_document=2),
    )
    assert report.passed, [f.describe() for f in report.failures]


def test_every_aveline_document_gets_tags_and_no_case_is_orphaned() -> None:
    """Two promises on the shipped corpus: every document has tags to be
    found by, and routing never strands a golden case's evidence - the
    preflight that gates eval runs, run here for free on every test run."""
    from openknowledge.connectors import LocalFilesConnector
    from openknowledge.evaluation.dataset import load_cases
    from openknowledge.evaluation.preflight import preflight
    from openknowledge.retrieval.rerank import StructuralReranker

    documents = LocalFilesConnector(AVELINE).fetch()
    retriever = BM25Retriever()
    retriever.index(documents)

    tags = retriever.document_tags()
    assert set(tags) == {d.document_id for d in documents}
    assert all(tags[d.document_id] for d in documents)
    assert "expenses" in tags["hr-expenses-policy"]

    report = preflight(
        load_cases(GOLDEN_AVELINE),
        retriever=retriever,
        k=6,
        candidates=30,
        reranker=StructuralReranker(max_per_document=2),
    )
    assert report.passed, [f.describe() for f in report.failures]


def test_tag_routing_is_part_of_the_cache_key() -> None:
    """Routing changes which chunks a question is answered from; flipping it
    must not serve answers produced under the other regime."""
    from openknowledge.cache import AnswerStore
    from openknowledge.cascade import Cascade
    from openknowledge.config import Settings

    def policy(**changes: object) -> str:
        settings = Settings(_env_file=None).model_copy(update=changes)  # type: ignore[call-arg]
        cascade = Cascade(
            store=AnswerStore(":memory:"), retriever=BM25Retriever(), settings=settings
        )
        return cascade._key_context().policy_version

    assert policy() != policy(tag_routing=False)
