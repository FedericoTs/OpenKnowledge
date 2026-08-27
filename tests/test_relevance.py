"""Deciding whether a conflict applies to the question being asked.

Precision matters in both directions. Too loose and every question touching a
document with one bad figure is refused, so the admin switches the feature off.
Too strict and someone is told EUR 500 while the current policy says EUR 1,000.
"""

from __future__ import annotations

import pytest

from openknowledge.knowledge import KnowledgeStore, find_conflicts
from openknowledge.knowledge.relevance import (
    _fold,
    describe_for_user,
    overlap_coefficient,
    relevant_conflicts,
)
from openknowledge.retrieval import Document

OLD = Document(
    "expenses-2025",
    "Expenses 2025",
    "Travel expenses require prior written approval for any amount above EUR 500. "
    "Expense claims must be submitted within 60 days of the date incurred. "
    "Meals are reimbursed up to EUR 45 per day.",
)
NEW = Document(
    "expenses-2026",
    "Expenses 2026",
    "Travel expenses require prior written approval for any amount above EUR 1,000. "
    "Expense claims must be submitted within 30 days of the date incurred. "
    "Meals are reimbursed up to EUR 45 per day.",
)
LEAVE = Document(
    "parental-leave",
    "Parental Leave",
    "Employees with 12 months of continuous service get 20 weeks of fully paid leave. "
    "Requests need 30 days notice.",
)


@pytest.fixture
def conflicts():
    store = KnowledgeStore()
    for conflict in find_conflicts([OLD, NEW, LEAVE]):
        store.record_conflict(conflict)
    opened = store.open_conflicts()
    store.close()
    return opened


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("expense", "expenses"),
        ("claim", "claims"),
        ("submit", "submitted"),
        ("day", "days"),
        ("approval", "approved"),
        ("week", "weeks"),
    ],
)
def test_word_forms_fold_together(a: str, b: str) -> None:
    """A suffix stripper turns 'expenses' into 'expens' and leaves 'expense'
    alone, so the two stop matching. Prefix folding does not have that hole."""
    assert _fold(a) == _fold(b)


@pytest.mark.parametrize("word", ["not", "no", "less", "unless", "is", "was"])
def test_meaning_bearing_short_words_are_left_alone(word: str) -> None:
    assert _fold(word) == word


def test_overlap_coefficient_handles_length_mismatch() -> None:
    """Questions are short and claim contexts are not; Jaccard would miss."""
    question = frozenset({"approval", "threshold", "expense"})
    context = frozenset({"travel", "expense", "require", "prior", "written", "approval"})
    assert overlap_coefficient(question, context) == pytest.approx(2 / 3)
    assert overlap_coefficient(frozenset(), context) == 0.0


@pytest.mark.parametrize(
    "question",
    [
        "What is the approval threshold for travel expenses?",
        "Above what amount do I need approval for travel expenses?",
        "How long do I have to submit an expense claim?",
        "When must expense claims be submitted?",
    ],
)
def test_contested_questions_are_flagged(question: str, conflicts) -> None:
    assert relevant_conflicts(question, conflicts), question


@pytest.mark.parametrize(
    "question",
    [
        "How much can I claim for meals per day?",
        "How do I connect to the VPN?",
        "How long does VPN access approval take?",
        "How much parental leave do I get?",
        "How much notice must I give for parental leave?",
        "Who approves a new starter's laptop?",
        "",
    ],
)
def test_uncontested_questions_are_left_alone(question: str, conflicts) -> None:
    assert relevant_conflicts(question, conflicts) == [], question


def test_a_single_shared_word_is_not_enough(conflicts) -> None:
    """One coincidental word must not trigger a refusal."""
    assert relevant_conflicts("What counts as travel?", conflicts) == []


def test_most_relevant_conflict_comes_first(conflicts) -> None:
    hits = relevant_conflicts("What is the approval threshold for travel expenses?", conflicts)
    assert hits[0].unit == "eur"


def test_the_user_message_names_both_figures_and_both_documents(conflicts) -> None:
    """'Ask your admin' without saying what is disputed wastes everyone's time."""
    hits = relevant_conflicts("What is the approval threshold for travel expenses?", conflicts)
    message = describe_for_user(hits)
    assert "EUR 500" in message and "EUR 1,000" in message
    assert "expenses-2025" in message and "expenses-2026" in message
    assert "administrator" in message


def test_thresholds_are_tunable(conflicts) -> None:
    strict = relevant_conflicts(
        "What is the approval threshold for travel expenses?",
        conflicts,
        min_overlap=0.99,
        min_shared=10,
    )
    assert strict == []
