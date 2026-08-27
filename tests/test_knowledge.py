"""The knowledge lifecycle: draft at ingest, review once, flag contradictions.

The feature's promise is that maintaining the bot becomes a one-off at upload.
That only holds if drafting is bounded, review is ranked, and a contradiction
reaches a human instead of reaching an employee.
"""

from __future__ import annotations

import json

import pytest

from fakes import FakeProvider
from openknowledge.knowledge import (
    KnowledgeStore,
    ProposalStatus,
    draft_for_documents,
    draft_from_document,
    figure_changes,
    find_conflicts,
    ingest_documents,
    proposal_id,
    rank_by_demand,
    scan_documents,
)
from openknowledge.knowledge.reverify import reverify_changed_documents
from openknowledge.retrieval import BM25Retriever, Document
from openknowledge.types import Citation

EXPENSES_V1 = Document(
    "expenses",
    "Expenses Policy",
    "Travel expenses require prior written approval for any amount above EUR 500. "
    "Expense claims must be submitted within 60 days of the date incurred.",
)
EXPENSES_V2 = Document(
    "expenses",
    "Expenses Policy",
    "Travel expenses require prior written approval for any amount above EUR 1,000. "
    "Expense claims must be submitted within 60 days of the date incurred.",
)
LEAVE = Document(
    "leave",
    "Parental Leave",
    "Employees with at least 12 months of continuous service are entitled to 20 weeks "
    "of fully paid parental leave.",
)


@pytest.fixture
def knowledge() -> KnowledgeStore:
    with KnowledgeStore() as store:
        yield store


def faq_json(*pairs: tuple[str, str]) -> str:
    return json.dumps([{"question": q, "answer": a} for q, a in pairs])


# -- document versions -----------------------------------------------------


def test_first_sync_reports_everything_as_new(knowledge: KnowledgeStore) -> None:
    added, changed, removed = knowledge.sync_documents([EXPENSES_V1, LEAVE])
    assert added == {"expenses", "leave"}
    assert not changed and not removed


def test_an_unchanged_corpus_reports_nothing(knowledge: KnowledgeStore) -> None:
    """This is what keeps ingest-time work one-off rather than per-reindex."""
    knowledge.sync_documents([EXPENSES_V1, LEAVE])
    added, changed, removed = knowledge.sync_documents([EXPENSES_V1, LEAVE])
    assert not added and not changed and not removed


def test_an_edit_is_detected_by_content_not_name(knowledge: KnowledgeStore) -> None:
    knowledge.sync_documents([EXPENSES_V1])
    _, changed, _ = knowledge.sync_documents([EXPENSES_V2])
    assert changed == {"expenses"}


def test_deletion_is_detected(knowledge: KnowledgeStore) -> None:
    knowledge.sync_documents([EXPENSES_V1, LEAVE])
    _, _, removed = knowledge.sync_documents([LEAVE])
    assert removed == {"expenses"}


# -- proposals -------------------------------------------------------------


def propose(store: KnowledgeStore, question: str = "how much leave", **kw):
    defaults = dict(
        canonical_query=question,
        question=question,
        answer="20 weeks. [leave]",
        citations=(Citation("leave", "Parental Leave", "20 weeks", None),),
        origin_documents=("leave",),
        corpus_version="c1",
        support_ratio=0.9,
    )
    return store.propose(**{**defaults, **kw})


def test_a_proposal_starts_as_a_draft(knowledge: KnowledgeStore) -> None:
    proposal = propose(knowledge)
    assert proposal is not None and proposal.status is ProposalStatus.DRAFT


def test_re_proposing_the_same_question_updates_rather_than_duplicates(
    knowledge: KnowledgeStore,
) -> None:
    propose(knowledge)
    propose(knowledge, answer="Twenty weeks. [leave]")
    assert knowledge.counts() == {"draft": 1}


def test_a_rejected_proposal_never_comes_back(knowledge: KnowledgeStore) -> None:
    """A queue that re-offers what you already declined stops being read."""
    proposal = propose(knowledge)
    assert proposal is not None
    knowledge.reject(proposal.id, reviewer="hr")
    assert propose(knowledge) is None
    assert knowledge.pending() == []


def test_an_approved_proposal_is_not_re_proposed(knowledge: KnowledgeStore) -> None:
    proposal = propose(knowledge)
    assert proposal is not None
    knowledge.approve(proposal.id, reviewer="hr")
    assert propose(knowledge) is None


def test_only_drafts_can_be_reviewed(knowledge: KnowledgeStore) -> None:
    proposal = propose(knowledge)
    assert proposal is not None
    assert knowledge.approve(proposal.id) is not None
    assert knowledge.approve(proposal.id) is None, "approving twice must not succeed"


def test_a_draft_is_servable_and_an_approved_one_is_not(knowledge: KnowledgeStore) -> None:
    """Approval promotes to a pin, so the draft tier must stop offering it."""
    proposal = propose(knowledge)
    assert proposal is not None
    assert knowledge.draft_for("how much leave") is not None
    knowledge.approve(proposal.id)
    assert knowledge.draft_for("how much leave") is None


def test_drafts_are_retired_when_their_source_document_changes(
    knowledge: KnowledgeStore,
) -> None:
    """A draft verified against text that no longer exists is not trustworthy."""
    propose(knowledge)
    stale = knowledge.supersede_for_documents(frozenset({"leave"}))
    assert len(stale) == 1
    assert knowledge.draft_for("how much leave") is None


def test_approved_answers_survive_a_document_change(knowledge: KnowledgeStore) -> None:
    """Silently retiring an approval would erase a human decision."""
    proposal = propose(knowledge)
    assert proposal is not None
    knowledge.approve(proposal.id, reviewer="hr")
    assert knowledge.supersede_for_documents(frozenset({"leave"})) == []
    assert knowledge.counts()["approved"] == 1


def test_finding_approved_answers_that_cite_a_document(knowledge: KnowledgeStore) -> None:
    """The lookup that makes contradiction detection affordable."""
    proposal = propose(knowledge)
    assert proposal is not None
    knowledge.approve(proposal.id)
    assert [p.id for p in knowledge.approved_citing("leave")] == [proposal.id]
    assert knowledge.approved_citing("expenses") == []


def test_proposal_ids_are_stable_and_variant_aware() -> None:
    base = proposal_id("q", ("a", "b"))
    assert base == proposal_id("q", ("b", "a")), "document order must not matter"
    assert base != proposal_id("q", ("a", "b"), variant="corpus-2")


def test_review_queue_is_ranked_by_what_approving_is_worth(
    knowledge: KnowledgeStore,
) -> None:
    """Nobody reviews three thousand items; the top of the list has to earn it."""
    rare = propose(knowledge, question="obscure question")
    common = propose(knowledge, question="popular question")
    assert rare is not None and common is not None

    ranked = rank_by_demand(
        knowledge.pending(),
        demand={"popular question": 400, "obscure question": 1},
        cost_per_answer_usd=0.01,
    )
    assert ranked[0][0].id == common.id
    assert ranked[0][1] == pytest.approx(4.0)


def test_a_draft_nobody_asks_for_scores_zero(knowledge: KnowledgeStore) -> None:
    propose(knowledge, question="never asked")
    ranked = rank_by_demand(knowledge.pending(), demand={}, cost_per_answer_usd=0.01)
    assert ranked[0][1] == 0.0


# -- conflicts -------------------------------------------------------------


def test_conflicts_are_recorded_once(knowledge: KnowledgeStore) -> None:
    conflicts = find_conflicts([EXPENSES_V1, Document("expenses-new", "New", EXPENSES_V2.text)])
    assert conflicts
    for conflict in conflicts:
        knowledge.record_conflict(conflict)
        knowledge.record_conflict(conflict)
    assert len(knowledge.open_conflicts()) == len(conflicts)


def test_resolving_a_conflict_closes_it(knowledge: KnowledgeStore) -> None:
    (conflict, *_) = find_conflicts(
        [EXPENSES_V1, Document("expenses-new", "New", EXPENSES_V2.text)]
    )
    knowledge.record_conflict(conflict)
    assert knowledge.resolve_conflict(conflict.key, resolution="keep new", resolver="fin")
    assert knowledge.open_conflicts() == []
    assert knowledge.resolve_conflict(conflict.key, resolution="again") is None


def test_re_detecting_a_resolved_conflict_does_not_reopen_it(
    knowledge: KnowledgeStore,
) -> None:
    """Otherwise every re-index would undo the admin's decisions."""
    (conflict, *_) = find_conflicts(
        [EXPENSES_V1, Document("expenses-new", "New", EXPENSES_V2.text)]
    )
    knowledge.record_conflict(conflict)
    knowledge.resolve_conflict(conflict.key, resolution="keep new")
    knowledge.record_conflict(conflict)
    assert knowledge.open_conflicts() == []


def test_deleting_one_side_clears_the_conflict(knowledge: KnowledgeStore) -> None:
    """Removing a document resolves the disagreement; the flag must not persist."""
    (conflict, *_) = find_conflicts(
        [EXPENSES_V1, Document("expenses-new", "New", EXPENSES_V2.text)]
    )
    knowledge.record_conflict(conflict)
    assert knowledge.drop_conflicts_for_documents(frozenset({"expenses"})) == 1
    assert knowledge.open_conflicts() == []


# -- drafting --------------------------------------------------------------


async def test_drafts_are_parsed_and_gate_checked() -> None:
    provider = FakeProvider(
        replies=[
            faq_json(
                (
                    "What is the approval threshold?",
                    "Prior written approval is required above EUR 500. [expenses]",
                ),
                (
                    "How long do I have to submit a claim?",
                    "Within 60 days of the date incurred. [expenses]",
                ),
            )
        ]
    )
    result = await draft_from_document(provider, EXPENSES_V1)
    assert len(result.drafted) == 2
    assert result.rejected == ()
    assert all(d.citations for d in result.drafted)


async def test_an_invented_figure_never_reaches_the_review_queue() -> None:
    """The gate runs on drafts exactly as it runs on live answers."""
    provider = FakeProvider(
        replies=[
            faq_json(
                (
                    "What is the approval threshold?",
                    "Approval is required above EUR 5,000. [expenses]",
                ),
            )
        ]
    )
    result = await draft_from_document(provider, EXPENSES_V1)
    assert result.drafted == ()
    assert "5,000" in result.rejected[0][1]


async def test_an_invented_source_is_rejected() -> None:
    provider = FakeProvider(
        replies=[faq_json(("What is the threshold?", "EUR 500. [finance-handbook]"))]
    )
    result = await draft_from_document(provider, EXPENSES_V1)
    assert result.drafted == ()


async def test_a_markdown_fence_does_not_lose_the_document() -> None:
    """Tolerant parsing: rejecting a whole document over a backtick is a bad trade."""
    payload = faq_json(
        ("What is the approval threshold?", "Approval is required above EUR 500. [expenses]")
    )
    provider = FakeProvider(replies=[f"Here you go:\n```json\n{payload}\n```"])
    result = await draft_from_document(provider, EXPENSES_V1)
    assert len(result.drafted) == 1


async def test_unparseable_output_yields_nothing_rather_than_raising() -> None:
    provider = FakeProvider(replies=["I'd be happy to help with that!"])
    result = await draft_from_document(provider, EXPENSES_V1)
    assert result.drafted == () and result.rejected == ()


async def test_a_provider_outage_is_survivable() -> None:
    result = await draft_from_document(FakeProvider(fail=True), EXPENSES_V1)
    assert result.drafted == () and result.cost_usd == 0.0


async def test_duplicate_questions_within_one_document_are_dropped() -> None:
    answer = "Approval is required above EUR 500. [expenses]"
    provider = FakeProvider(
        replies=[
            faq_json(
                ("What is the approval threshold?", answer),
                ("what is the APPROVAL threshold??", answer),
            )
        ]
    )
    result = await draft_from_document(provider, EXPENSES_V1)
    assert len(result.drafted) == 1
    assert "duplicate" in result.rejected[0][1]


# -- pipeline --------------------------------------------------------------


def test_scanning_is_free_and_still_finds_conflicts(knowledge: KnowledgeStore) -> None:
    """An operator who re-indexes gets conflict detection without spending."""
    report = scan_documents(
        [EXPENSES_V1, Document("expenses-new", "New", EXPENSES_V2.text)], store=knowledge
    )
    assert report.conflicts_open > 0
    assert report.cost_usd == 0.0
    assert report.documents_drafted == 0


async def test_ingest_without_a_model_says_so(knowledge: KnowledgeStore) -> None:
    report = await ingest_documents(
        [EXPENSES_V1], store=knowledge, corpus_version="c1", provider=None
    )
    assert report.drafts_created == 0
    assert any("no model configured" in n for n in report.notes)


async def test_a_second_ingest_drafts_nothing(knowledge: KnowledgeStore) -> None:
    """The one-off promise: unchanged documents cost nothing to re-ingest."""
    reply = faq_json(
        ("What is the approval threshold?", "Approval is required above EUR 500. [expenses]")
    )
    provider = FakeProvider(replies=[reply, reply])

    first = await ingest_documents(
        [EXPENSES_V1], store=knowledge, corpus_version="c1", provider=provider
    )
    assert first.drafts_created == 1

    second = await ingest_documents(
        [EXPENSES_V1], store=knowledge, corpus_version="c1", provider=provider
    )
    assert second.documents_drafted == 0
    assert second.cost_usd == 0.0
    assert len(provider.calls) == 1


async def test_the_per_run_cap_is_reported_not_silent(knowledge: KnowledgeStore) -> None:
    """A silent truncation reads as 'covered everything' when it did not."""
    docs = [Document(f"d{i}", f"D{i}", f"The limit is EUR {i}00.") for i in range(5)]
    provider = FakeProvider(replies=["[]"] * 5)
    knowledge.sync_documents(docs)

    report = await draft_for_documents(
        docs,
        store=knowledge,
        provider=provider,
        corpus_version="c1",
        document_ids=frozenset(d.document_id for d in docs),
        max_documents=2,
    )
    assert report.documents_drafted == 2
    assert any("3 skipped" in n for n in report.notes)


# -- re-verification -------------------------------------------------------


def test_figure_changes_ignore_rewording() -> None:
    old = "Approval is needed above EUR 500. [expenses]"
    assert figure_changes(old, "You need approval above EUR 500. [expenses]") == ()


def test_figure_changes_catch_a_moved_threshold() -> None:
    old = "Approval is needed above EUR 500. [expenses]"
    assert figure_changes(old, "Approval is needed above EUR 1,000. [expenses]") == (
        ("EUR 500", "EUR 1,000"),
    )


async def test_a_moved_figure_raises_a_review_item(knowledge: KnowledgeStore) -> None:
    """The contradiction flag, anchored to an answer a human already approved."""
    approved = knowledge.propose(
        canonical_query="what is the approval threshold",
        question="What is the approval threshold?",
        answer="Prior written approval is required above EUR 500. [expenses]",
        citations=(Citation("expenses", "Expenses Policy", "EUR 500", None),),
        origin_documents=("expenses",),
        corpus_version="c1",
    )
    assert approved is not None
    knowledge.approve(approved.id, reviewer="finance")

    retriever = BM25Retriever()
    retriever.index([EXPENSES_V2])
    provider = FakeProvider(
        replies=["Prior written approval is required above EUR 1,000. [expenses]"]
    )

    revisions = await reverify_changed_documents(
        frozenset({"expenses"}),
        store=knowledge,
        retriever=retriever,
        provider=provider,
        corpus_version="c2",
    )
    assert len(revisions) == 1
    assert revisions[0].is_material
    assert revisions[0].figure_changes == (("EUR 500", "EUR 1,000"),)
    assert revisions[0].proposal is not None
    assert revisions[0].proposal.supersedes == approved.id


async def test_only_answers_citing_the_changed_document_are_re_asked(
    knowledge: KnowledgeStore,
) -> None:
    """This is what makes contradiction detection affordable rather than O(N^2)."""
    unrelated = knowledge.propose(
        canonical_query="how much leave",
        question="How much parental leave?",
        answer="20 weeks. [leave]",
        citations=(Citation("leave", "Parental Leave", "20 weeks", None),),
        origin_documents=("leave",),
        corpus_version="c1",
    )
    assert unrelated is not None
    knowledge.approve(unrelated.id)

    retriever = BM25Retriever()
    retriever.index([EXPENSES_V2, LEAVE])
    provider = FakeProvider(replies=["should not be called"])

    revisions = await reverify_changed_documents(
        frozenset({"expenses"}),
        store=knowledge,
        retriever=retriever,
        provider=provider,
        corpus_version="c2",
    )
    assert revisions == []
    assert provider.calls == [], "an unrelated approved answer must not be re-asked"


async def test_reverification_is_a_no_op_when_nothing_changed(
    knowledge: KnowledgeStore,
) -> None:
    provider = FakeProvider(replies=["x"])
    retriever = BM25Retriever()
    retriever.index([EXPENSES_V1])
    assert (
        await reverify_changed_documents(
            frozenset(),
            store=knowledge,
            retriever=retriever,
            provider=provider,
            corpus_version="c1",
        )
        == []
    )
    assert provider.calls == []
