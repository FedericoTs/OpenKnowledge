"""Telling a contradiction apart from a duplicate.

Two documents disagreeing on one figure is a decision somebody has to make.
Two documents disagreeing on ninety-eight is one stale copy, and saying it
ninety-eight times buries the findings that matter underneath it.
"""

from __future__ import annotations

from openknowledge.knowledge.claims import Claim, Conflict, compare_documents
from openknowledge.knowledge.variants import group_by_document_pair
from openknowledge.retrieval import Document


def claim(doc_id: str, raw: str, value: float, subject: str) -> Claim:
    return Claim(
        document_id=doc_id,
        document_title=doc_id.title(),
        raw=raw,
        value=value,
        unit="eur",
        context=frozenset(subject.split()),
        sentence=f"{subject} is {raw}.",
    )


def conflict(doc_a: str, doc_b: str, subject: str, left: float, right: float) -> Conflict:
    return Conflict(
        left=claim(doc_a, f"EUR {left:.0f}", left, subject),
        right=claim(doc_b, f"EUR {right:.0f}", right, subject),
        overlap=0.8,
    )


def test_a_pair_that_disagrees_a_little_is_listed_claim_by_claim() -> None:
    pairs = group_by_document_pair(
        [conflict("expenses", "travel", f"limit-{n} approval trips", 500, 1000) for n in range(2)],
        {("expenses", "travel"): 4},
    )
    assert len(pairs) == 1
    assert not pairs[0].is_variant
    assert len(pairs[0].conflicts) == 2


def test_a_pair_that_shares_a_document_worth_of_figures_reads_as_duplication() -> None:
    conflicts = [conflict("policy-v1", "policy-v2", f"limit-{n} grade", 500, 900) for n in range(9)]
    pairs = group_by_document_pair(conflicts, {("policy-v1", "policy-v2"): 40})

    assert pairs[0].is_variant
    assert pairs[0].compared == 49
    assert "two versions of the same document" in pairs[0].describe()


def test_two_long_documents_contradicting_on_a_few_points_are_not_duplicates() -> None:
    """The case the shared-subject test would otherwise get wrong."""
    conflicts = [conflict("handbook", "addendum", f"limit-{n} grade", 500, 900) for n in range(3)]
    pairs = group_by_document_pair(conflicts, {("handbook", "addendum"): 90})
    assert not pairs[0].is_variant


def test_agreement_counts_are_found_whichever_way_round_the_pair_is_keyed() -> None:
    conflicts = [conflict("b-doc", "a-doc", f"limit-{n} grade", 500, 900) for n in range(9)]
    pairs = group_by_document_pair(conflicts, {("b-doc", "a-doc"): 40})
    assert pairs[0].agreements == 40
    assert pairs[0].is_variant


def test_a_pair_found_only_through_prose_is_never_collapsed() -> None:
    """Deontic findings have no agreement tally, so they must stay listed."""
    conflicts = [
        Conflict(
            left=claim("a", "required", 1, f"rule-{n} approval"),
            right=claim("b", "allowed", 2, f"rule-{n} approval"),
            overlap=0.9,
            kind="deontic",
        )
        for n in range(12)
    ]
    pairs = group_by_document_pair(conflicts)
    assert not pairs[0].is_variant


def test_duplicated_pairs_sort_below_real_contradictions() -> None:
    conflicts = [conflict("v1", "v2", f"limit-{n} grade", 500, 900) for n in range(9)]
    conflicts += [conflict("expenses", "travel", "meal allowance limit", 45, 60)]
    pairs = group_by_document_pair(conflicts, {("v1", "v2"): 40})
    assert [p.is_variant for p in pairs] == [False, True]


def test_agreements_are_counted_from_real_documents() -> None:
    """The count has to come from claims that matched on subject, not all claims."""
    left = Document(
        "policy-v1",
        "Policy V1",
        "The travel approval limit is EUR 500. The meal allowance limit is EUR 45.",
    )
    right = Document(
        "policy-v2",
        "Policy V2",
        "The travel approval limit is EUR 900. The meal allowance limit is EUR 45.",
    )
    conflicts, agreements = compare_documents([left, right])

    assert [c.left.raw for c in conflicts] == ["EUR 500"]
    assert agreements[("policy-v1", "policy-v2")] == 1


def test_stored_conflicts_group_the_same_way_the_audit_does() -> None:
    """`openknowledge conflicts` and `openknowledge audit` read the same data.

    They disagreed for a while: the audit collapsed a duplicated pair into one
    line and the CLI listed all twenty-four findings, which made the review
    queue unusable on exactly the corpus the audit handled well.
    """
    from openknowledge.knowledge.variants import group_stored

    class Row:
        def __init__(self, left: str, right: str) -> None:
            self.left_document = left
            self.right_document = right

    rows = [Row("policy-v1", "policy-v2") for _ in range(24)]
    rows += [Row("expenses", "travel"), Row("expenses", "travel")]

    pairs = group_stored(rows)

    assert [p.is_variant for p in pairs] == [False, True], "real disagreements come first"
    assert len(pairs[0].conflicts) == 2
    assert "versioning problem" in pairs[1].describe()


def test_a_pair_with_one_disagreement_is_never_called_duplication() -> None:
    from openknowledge.knowledge.variants import group_stored

    class Row:
        left_document = "expenses"
        right_document = "travel"

    assert not group_stored([Row()])[0].is_variant
