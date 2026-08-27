"""Checking stored answers against newly arrived documents.

The case re-verification cannot see: a *new* document contradicting an existing
answer. No approved answer cites it, because it did not exist. This closes that
gap for free - one lexical search and some regex, no model call.
"""

from __future__ import annotations

import pytest

from openknowledge.knowledge import KnowledgeStore, crosscheck_answers, scan_documents
from openknowledge.retrieval import BM25Retriever, Document
from openknowledge.types import Citation

EXPENSES_2025 = Document(
    "expenses-2025",
    "Expenses Policy",
    "Alcohol is not reimbursable under any circumstances, including client entertainment. "
    "Travel above EUR 500 requires prior written approval.",
)
EXPENSES_2026 = Document(
    "expenses-2026",
    "Expenses Policy 2026",
    "Alcohol may be reimbursed for client entertainment up to EUR 60 per head. "
    "Travel above EUR 500 requires prior written approval.",
)
VPN = Document(
    "vpn",
    "Remote Access",
    "VPN access requests must be approved by IT Operations within 2 business days.",
)


@pytest.fixture
def knowledge() -> KnowledgeStore:
    with KnowledgeStore() as store:
        yield store


def approve_alcohol_answer(store: KnowledgeStore, *, approved: bool = True):
    proposal = store.propose(
        canonical_query="can i expense alcohol for client entertainment",
        question="Can I expense alcohol for client entertainment?",
        answer=(
            "No. Alcohol is not reimbursable under any circumstances, including "
            "client entertainment. [expenses-2025]"
        ),
        citations=(Citation("expenses-2025", "Expenses Policy", "alcohol", None),),
        origin_documents=("expenses-2025",),
        corpus_version="c1",
        support_ratio=0.95,
    )
    assert proposal is not None
    if approved:
        store.approve(proposal.id, reviewer="finance")
    return proposal


def indexed(*documents: Document) -> BM25Retriever:
    retriever = BM25Retriever()
    retriever.index(list(documents))
    return retriever


def test_a_new_document_contradicting_an_approved_answer_is_caught(
    knowledge: KnowledgeStore,
) -> None:
    approve_alcohol_answer(knowledge)
    findings = crosscheck_answers(
        store=knowledge,
        retriever=indexed(EXPENSES_2025, EXPENSES_2026),
        document_ids=frozenset({"expenses-2026"}),
    )
    assert len(findings) == 1
    assert findings[0].document_id == "expenses-2026"
    assert findings[0].is_approved
    assert "not allowed" in findings[0].describe()


def test_drafts_are_checked_too(knowledge: KnowledgeStore) -> None:
    approve_alcohol_answer(knowledge, approved=False)
    findings = crosscheck_answers(
        store=knowledge,
        retriever=indexed(EXPENSES_2025, EXPENSES_2026),
        document_ids=frozenset({"expenses-2026"}),
    )
    assert len(findings) == 1
    assert not findings[0].is_approved


def test_approved_answers_are_reported_before_drafts(knowledge: KnowledgeStore) -> None:
    """An approval is a decision somebody made, so a document that disagrees
    with one is more urgent than one that disagrees with an unreviewed draft."""
    approve_alcohol_answer(knowledge)
    draft = knowledge.propose(
        canonical_query="what happens to alcohol claims",
        question="What happens to alcohol claims?",
        answer="Alcohol is not reimbursable at all. [expenses-2025]",
        citations=(Citation("expenses-2025", "Expenses Policy", "alcohol", None),),
        origin_documents=("expenses-2025",),
        corpus_version="c1",
    )
    assert draft is not None

    findings = crosscheck_answers(
        store=knowledge,
        retriever=indexed(EXPENSES_2025, EXPENSES_2026),
        document_ids=frozenset({"expenses-2026"}),
    )
    assert findings and findings[0].is_approved


def test_an_agreeing_document_is_not_flagged(knowledge: KnowledgeStore) -> None:
    approve_alcohol_answer(knowledge)
    restated = Document(
        "expenses-summary",
        "Expenses Summary",
        "Alcohol is not reimbursable under any circumstances, including client entertainment.",
    )
    findings = crosscheck_answers(
        store=knowledge,
        retriever=indexed(EXPENSES_2025, restated),
        document_ids=frozenset({"expenses-summary"}),
    )
    assert findings == []


def test_an_unrelated_document_is_not_compared(knowledge: KnowledgeStore) -> None:
    """The retrieval pre-filter is what keeps this from being O(N^2)."""
    approve_alcohol_answer(knowledge)
    findings = crosscheck_answers(
        store=knowledge,
        retriever=indexed(EXPENSES_2025, VPN),
        document_ids=frozenset({"vpn"}),
    )
    assert findings == []


def test_an_answer_is_never_checked_against_its_own_source(
    knowledge: KnowledgeStore,
) -> None:
    """An answer derived from a document cannot disagree with it."""
    approve_alcohol_answer(knowledge)
    findings = crosscheck_answers(
        store=knowledge,
        retriever=indexed(EXPENSES_2025),
        document_ids=frozenset({"expenses-2025"}),
    )
    assert findings == []


def test_nothing_changed_means_nothing_checked(knowledge: KnowledgeStore) -> None:
    approve_alcohol_answer(knowledge)
    assert (
        crosscheck_answers(
            store=knowledge, retriever=indexed(EXPENSES_2025), document_ids=frozenset()
        )
        == []
    )


def test_a_numeric_disagreement_is_caught_too(knowledge: KnowledgeStore) -> None:
    proposal = knowledge.propose(
        canonical_query="what is the approval threshold",
        question="What is the approval threshold for travel?",
        answer="Travel above EUR 500 requires prior written approval. [expenses-2025]",
        citations=(Citation("expenses-2025", "Expenses Policy", "threshold", None),),
        origin_documents=("expenses-2025",),
        corpus_version="c1",
    )
    assert proposal is not None
    knowledge.approve(proposal.id)

    raised = Document(
        "expenses-2027",
        "Expenses Policy 2027",
        "Travel above EUR 2,000 requires prior written approval.",
    )
    findings = crosscheck_answers(
        store=knowledge,
        retriever=indexed(EXPENSES_2025, raised),
        document_ids=frozenset({"expenses-2027"}),
    )
    assert findings
    assert any(c.kind == "numeric" for c in findings[0].conflicts)


def test_the_ingest_scan_records_what_it_finds(knowledge: KnowledgeStore) -> None:
    """Wired into the free half of ingest, so it runs on every re-index."""
    knowledge.sync_documents([EXPENSES_2025])
    approve_alcohol_answer(knowledge)

    report = scan_documents(
        [EXPENSES_2025, EXPENSES_2026],
        store=knowledge,
        retriever=indexed(EXPENSES_2025, EXPENSES_2026),
    )
    assert report.answers_contradicted == 1
    assert report.cost_usd == 0.0, "the whole cross-check must be free"
    assert any("answer contradicted" in note for note in report.notes)
    assert knowledge.open_conflicts()
