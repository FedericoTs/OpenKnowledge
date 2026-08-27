"""Checking a golden set before paying to run it.

About half the failures a fresh golden set produces are the set's own - a typo
in `must_cite`, a phrase written differently from the corpus, a fact in a
document retrieval never surfaces. All three read as model failures in the
report and all three are free to find, which is the whole argument for running
this first.
"""

from __future__ import annotations

from openknowledge.evaluation import format_preflight, preflight
from openknowledge.evaluation.dataset import parse_cases
from openknowledge.retrieval import BM25Retriever, Document
from openknowledge.retrieval.rerank import StructuralReranker

EXPENSES = Document(
    "expenses-policy",
    "Expenses Policy",
    "Travel above EUR 500 requires prior approval. The daily allowance for meals "
    "is EUR 45 per day.",
)
LEAVE = Document(
    "parental-leave",
    "Parental Leave",
    "Employees with 12 months of service get 20 weeks of fully paid parental leave.",
)


#: Enough documents that k=6 is actually a choice. With two documents retrieval
#: returns both whatever you ask, and a pre-flight that cannot miss proves
#: nothing about a pre-flight that has to.
DISTRACTORS = [
    Document(f"policy-{n}", f"Policy {n}", body)
    for n, body in enumerate(
        [
            "Desk booking opens 14 days ahead and closes at 18:00 the previous day.",
            "The cycle to work scheme is administered by payroll each January.",
            "Visitor passes are issued at reception and must be returned on exit.",
            "Company cars are allocated by grade and reviewed every 36 months.",
            "Charity matching is capped at EUR 250 per employee per year.",
            "The staff canteen operates from 07:30 to 15:00 on working days.",
            "Fire drills are held twice a year and attendance is recorded.",
        ]
    )
]


def retriever() -> BM25Retriever:
    r = BM25Retriever()
    r.index([EXPENSES, LEAVE, *DISTRACTORS])
    return r


def check(raw: list[dict[str, object]]):
    return preflight(parse_cases(raw), retriever=retriever(), k=6)


def test_a_reachable_case_passes() -> None:
    report = check(
        [
            {
                "id": "meal",
                "question": "How much can I claim for meals?",
                "must_cite": ["expenses-policy"],
                "must_say": ["EUR 45"],
            }
        ]
    )
    assert report.passed
    assert "PASSED" in format_preflight(report)


def test_a_typo_in_must_cite_is_named_as_a_typo() -> None:
    """A missing document and an unranked one need opposite fixes."""
    report = check(
        [
            {
                "id": "typo",
                "question": "How much can I claim for meals?",
                "must_cite": ["expense-policy"],  # singular: does not exist
                "must_say": ["EUR 45"],
            }
        ]
    )
    assert not report.passed
    assert report.failures[0].unknown_documents == ("expense-policy",)
    assert report.failures[0].missing_citations == ()
    assert "not in the corpus" in format_preflight(report)


def test_a_real_document_that_was_not_retrieved_is_a_ranking_problem() -> None:
    report = check(
        [
            {
                "id": "wrong-doc",
                "question": "How much parental leave do I get?",
                "must_cite": ["expenses-policy"],
                "must_say": ["20 weeks"],
            }
        ]
    )
    failure = report.failures[0]
    assert failure.missing_citations == ("expenses-policy",)
    assert failure.unknown_documents == ()
    assert "was not retrieved" in format_preflight(report)


def test_a_phrase_the_corpus_never_says_is_caught() -> None:
    report = check(
        [
            {
                "id": "wrong-figure",
                "question": "How much can I claim for meals?",
                "must_cite": ["expenses-policy"],
                "must_say": ["EUR 55"],
            }
        ]
    )
    assert not report.passed
    assert "none of" in format_preflight(report)


def test_one_of_several_spellings_is_enough() -> None:
    """Cases list alternative wordings of one fact; requiring all would fail
    every well-written case."""
    report = check(
        [
            {
                "id": "either",
                "question": "How much parental leave do I get?",
                "must_cite": ["parental-leave"],
                "must_say": ["20 weeks", "twenty weeks"],
            }
        ]
    )
    assert report.passed


def test_refusal_cases_are_not_checked_here() -> None:
    """Refusals usually retrieve plenty and ground none of it, so demanding
    that nothing be retrieved would be wrong."""
    report = check(
        [
            {"id": "refuse", "question": "How many sick days do I get?", "kind": "refusal"},
        ]
    )
    assert report.checks == []
    assert report.skipped_refusals == 1
    assert report.passed


def test_it_costs_nothing_and_calls_nothing() -> None:
    """The property that makes it worth running first: no cascade, no provider."""
    report = preflight(
        parse_cases(
            [
                {
                    "id": "meal",
                    "question": "meals per day",
                    "must_cite": ["expenses-policy"],
                    "must_say": ["EUR 45"],
                }
            ]
        ),
        retriever=retriever(),
        reranker=StructuralReranker(),
    )
    assert report.passed
