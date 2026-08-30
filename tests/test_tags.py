"""Tags derived at upload, and the candidacy they guarantee.

The promise under test: indexing a document also derives a readable set of
tags from its name, title, headings and distinctive vocabulary - free, no
model - and a question that names its documents decisively is guaranteed to
find them among its candidates. The promise that matters more: the route
never removes and never reorders - a filter and a routed-first ordering
were both measured against the golden sets and both made the local model
worse. Ambiguity means no route at all.
"""

from __future__ import annotations

from pathlib import Path

from openknowledge.retrieval import BM25Retriever, Document
from openknowledge.retrieval.base import Chunk, ScoredChunk
from openknowledge.retrieval.bm25 import BM25Retriever as _BM25
from openknowledge.retrieval.hybrid import HybridRetriever
from openknowledge.retrieval.tags import (
    corpus_document_frequency,
    derive_tags,
    fold_tags,
    guarantee_routed,
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


def test_the_named_document_is_always_in_the_radius() -> None:
    retriever = BM25Retriever()
    retriever.index(CORPUS)
    hits = retriever.search("what does the expenses policy say about meal claims?", k=6)
    assert any(h.chunk.document_id == "hr-expenses-policy" for h in hits)


def test_a_route_rescues_a_buried_document_and_changes_nothing_else() -> None:
    """The mechanism itself, as a pure function: a named document below the
    cut is rescued at the tail, displacing the weakest stranger; a named
    document already present leaves the ranking untouched; no route, no
    change. Order is never rewritten - two stronger designs (filtering, and
    routed-first ordering) were measured and both made the model worse."""
    from openknowledge.retrieval.base import ScoredChunk, chunk_document
    from openknowledge.retrieval.tags import guarantee_routed

    ranked = [
        ScoredChunk(chunk=chunk_document(Document(doc_id, doc_id, "words " * 30))[0], score=score)
        for doc_id, score in (("a", 4.0), ("b", 3.0), ("c", 2.0), ("target", 1.0))
    ]

    rescued = guarantee_routed(ranked, frozenset({"target"}), k=2)
    assert [s.chunk.document_id for s in rescued] == ["a", "target"]

    untouched = guarantee_routed(ranked, frozenset({"a"}), k=2)
    assert [s.chunk.document_id for s in untouched] == ["a", "b"]

    unrouted = guarantee_routed(ranked, None, k=2)
    assert [s.chunk.document_id for s in unrouted] == ["a", "b"]


def test_a_route_never_thins_the_context() -> None:
    """The regression the repository golden set caught live: routed to a
    document that chunks to a single window, a hard filter handed the model
    a one-chunk context and it refused a question it answers happily with a
    fuller one. A route only rescues; it never removes or reorders."""
    retriever = BM25Retriever()
    retriever.index(CORPUS)
    question = "what does the expenses policy say about meal claims?"
    routed = retriever.search(question, k=6)
    unrouted_retriever = BM25Retriever(tag_routing=False)
    unrouted_retriever.index(CORPUS)
    unrouted = unrouted_retriever.search(question, k=6)
    # On a corpus where the named document already ranks in the head, the
    # context is byte-identical with routing on and off - measured on the
    # aveline corpus for the questions that regressed under stronger designs.
    assert [h.chunk.chunk_id for h in routed] == [h.chunk.chunk_id for h in unrouted]


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
    assert any(h.chunk.document_id == "hr-expenses-policy" for h in hits)
    assert all(h.chunk.document_id != "hr-expenses-policy-2023" for h in hits)


def test_the_route_applies_to_the_fused_ranking() -> None:
    """The guarantee runs on the combined view, after both halves vote -
    at k=1 the rescue is visible: the stub buries the named document under
    a parking chunk it loves, and the route pulls it back into the cut."""

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
    hits = retriever.search("what does the expenses policy say about meal claims?", k=1)
    assert [h.chunk.document_id for h in hits] == ["hr-expenses-policy"]


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


# -- what a thousand documents exposed (scale regression) ---------------------
#
# Both faults below were invisible on a corpus of three and severe on a
# thousand. Measured there: three of five realistic questions came back
# refused, one of them holding a passage ranked FIRST for that question -
# evicted from a k=6 search by its own routing.


def test_rescue_never_evicts_the_whole_scored_head() -> None:
    """The arithmetic that did: keep ``k - len(rescued)``, floored at zero,
    which means keep NOTHING once the route names k documents or more."""
    ranked = [
        ScoredChunk(Chunk(f"top#{i}", f"top-{i}", "Top", f"the best passage {i}"), 10.0 - i)
        for i in range(6)
    ] + [
        ScoredChunk(Chunk(f"far#{i}", f"routed-{i}", "Routed", f"a weaker passage {i}"), 1.0)
        for i in range(40)
    ]
    within = frozenset(f"routed-{i}" for i in range(40))

    got = guarantee_routed(ranked, within, 6)

    assert len(got) <= 6, "the cut is k, even when the route is wide"
    ids = [s.chunk.document_id for s in got]
    assert "top-0" in ids, "the best-scoring passage in the corpus must survive its own route"
    kept = sum(1 for i in ids if i.startswith("top-"))
    assert kept >= 3, f"score-earned chunks must keep a majority of k, kept {kept}"


def test_rescue_spends_its_budget_on_the_best_ranked_missing_document() -> None:
    """When the budget cannot cover every named document, it must buy the
    strongest candidates - not whichever document sorts first by id."""
    ranked = [
        ScoredChunk(Chunk(f"top#{i}", f"top-{i}", "Top", f"passage {i}"), 10.0 - i)
        for i in range(6)
    ] + [
        ScoredChunk(Chunk("z#0", "zzz-strong", "Strong", "a strong routed passage"), 5.0),
        ScoredChunk(Chunk("a#0", "aaa-weak", "Weak", "a weak routed passage"), 0.1),
    ]
    got = guarantee_routed(ranked, frozenset({"zzz-strong", "aaa-weak"}), 6)
    ids = [s.chunk.document_id for s in got]
    assert "zzz-strong" in ids, "the better-ranked routed document must be rescued first"


def test_a_route_naming_many_documents_is_not_a_route() -> None:
    """A third of a thousand documents is three hundred and thirty. The share
    rule alone called that decisive; it is a topic, and topics are what the
    score is for."""
    tags = {f"doc-{i}": fold_tags(("notice", "period")) for i in range(100)}
    tags.update({f"other-{i}": fold_tags(("unrelated", "words")) for i in range(900)})
    assert route_by_tags("what is the notice period for senior engineers?", tags) is None


def test_a_route_naming_a_couple_of_documents_still_routes() -> None:
    """The cap must not cost the feature its purpose."""
    tags = {
        "handbook": fold_tags(("parental", "leave")),
        "archive": fold_tags(("parental", "leave")),
    }
    tags.update({f"other-{i}": fold_tags(("unrelated", "words")) for i in range(50)})
    got = route_by_tags("how long is parental leave?", tags)
    assert got == frozenset({"handbook", "archive"})
