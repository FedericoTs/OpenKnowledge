"""Contradictions in policy prose.

Accuracy in both directions. A missed contradiction tells somebody the wrong
policy with a citation attached; a false one blocks an answerable question, and
an admin who sees three bogus flags stops reading the fourth.
"""

from __future__ import annotations

import pytest

from openknowledge.knowledge.claims import find_conflicts
from openknowledge.knowledge.deontic import (
    Force,
    conflicts_between,
    extract_deontic_claims,
    predicate_families,
)
from openknowledge.retrieval import Document


def doc(text: str, doc_id: str = "d") -> Document:
    return Document(doc_id, doc_id.title(), text)


def forces(text: str) -> list[Force]:
    return [c.force for c in extract_deontic_claims(doc(text))]


def flags(text_a: str, text_b: str, **kw) -> bool:
    return bool(
        conflicts_between(
            extract_deontic_claims(doc(text_a, "a")),
            extract_deontic_claims(doc(text_b, "b")),
            **kw,
        )
    )


# -- force extraction ------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Approval must be obtained before travel.", Force.MANDATORY),
        ("Approval shall be obtained before travel.", Force.MANDATORY),
        ("Prior approval is required for all bookings.", Force.MANDATORY),
        ("Travel above EUR 500 requires approval.", Force.MANDATORY),
        ("Employees may work remotely.", Force.PERMITTED),
        ("Contractors are eligible for the scheme.", Force.PERMITTED),
        ("Meals are reimbursable up to EUR 45.", Force.PERMITTED),
        ("Alcohol must not be claimed.", Force.FORBIDDEN),
        ("Personal laptops are not permitted on the network.", Force.FORBIDDEN),
        ("Alcohol is not reimbursable.", Force.FORBIDDEN),
        ("Interns are excluded from the bonus scheme.", Force.FORBIDDEN),
        ("Client data may not be shared.", Force.FORBIDDEN),
    ],
)
def test_forces_are_read_correctly(text: str, expected: Force) -> None:
    assert expected in forces(text), forces(text)


def test_negated_modal_beats_the_modal_inside_it() -> None:
    """'must not' must never read as 'must'."""
    assert forces("Alcohol must not be claimed.") == [Force.FORBIDDEN]
    assert forces("Personal devices may not be connected.") == [Force.FORBIDDEN]


def test_not_required_means_optional_not_prohibited() -> None:
    """Reading it as a prohibition would invent disagreements between
    documents that agree."""
    assert Force.PERMITTED in forces("A receipt is not required for small expenses.")
    assert Force.FORBIDDEN not in forces("A receipt is not required for small expenses.")


def test_no_approval_required_is_permission() -> None:
    assert Force.PERMITTED in forces("No approval is required for expenses below EUR 50.")


def test_prose_without_a_rule_yields_nothing() -> None:
    assert extract_deontic_claims(doc("This handbook was last revised in spring.")) == []


# -- predicate families ----------------------------------------------------


def test_families_group_word_forms() -> None:
    assert "reimbursement" in predicate_families("Alcohol is not reimbursable.")
    assert "reimbursement" in predicate_families("Alcohol may be reimbursed.")
    assert "eligibility" in predicate_families("Contractors are excluded.")
    assert "eligibility" in predicate_families("Contractors are eligible.")


def test_a_sentence_can_belong_to_several_families() -> None:
    families = predicate_families("Employees may submit expense claims online.")
    assert {"submission", "reimbursement"} <= families


def test_unrelated_prose_has_no_family() -> None:
    assert predicate_families("The office opens at nine.") == frozenset()


# -- contradictions --------------------------------------------------------


@pytest.mark.parametrize(
    ("a", "b"),
    [
        (
            "Contractors are eligible for parental leave after 12 months.",
            "Contractors are excluded from parental leave regardless of tenure.",
        ),
        (
            "Alcohol is not reimbursable under any circumstances.",
            "Alcohol may be reimbursed for client entertainment.",
        ),
        (
            "Prior written approval must be obtained for all travel bookings.",
            "Prior written approval is not required for travel bookings.",
        ),
        (
            "Personal laptops are not permitted on the corporate network.",
            "Personal laptops may be used on the corporate network.",
        ),
        (
            "Client data must not be shared with third parties.",
            "Client data may be shared with approved third parties.",
        ),
    ],
)
def test_real_contradictions_are_flagged(a: str, b: str) -> None:
    assert flags(a, b), f"missed: {a!r} vs {b!r}"


@pytest.mark.parametrize(
    ("a", "b", "why"),
    [
        (
            "Alcohol is not reimbursable under any circumstances.",
            "Alcohol is not reimbursable, including client entertainment.",
            "a restatement with more detail is agreement",
        ),
        (
            "Employees must submit expense claims within 60 days.",
            "Employees may submit expense claims online through the portal.",
            "a deadline and a channel are both true at once",
        ),
        (
            "Contractors must have a sponsor in engineering leadership.",
            "Contractors may claim travel expenses with prior approval.",
            "same subject, unrelated kinds of rule",
        ),
        (
            "Alcohol is not reimbursable under any circumstances.",
            "VPN access requests must be approved by IT Operations.",
            "nothing in common",
        ),
        (
            "A receipt is not required for expenses below EUR 25.",
            "A receipt may be submitted for expenses below EUR 25.",
            "optional and permitted agree",
        ),
    ],
)
def test_near_misses_stay_quiet(a: str, b: str, why: str) -> None:
    assert not flags(a, b), f"false flag ({why}): {a!r} vs {b!r}"


def test_must_versus_may_needs_near_identical_context() -> None:
    """The soft pair. 'Must' and 'may' coexist constantly, so they only
    contradict when they describe the identical action."""
    assert flags(
        "Prior written approval must be obtained for all travel bookings.",
        "Prior written approval is not required for travel bookings.",
    )
    assert not flags(
        "Leave requests must be submitted 30 days in advance.",
        "Leave requests may be submitted through the HR portal.",
    )


def test_same_force_is_never_a_contradiction() -> None:
    assert not flags(
        "Travel above EUR 500 requires prior written approval.",
        "Travel above EUR 1,000 requires prior written approval.",
    )


def test_strictness_trades_recall_for_precision() -> None:
    pair = (
        "Contractors are eligible for parental leave after 12 months.",
        "Contractors are excluded from parental leave regardless of tenure.",
    )
    assert flags(*pair)
    assert not flags(*pair, strictness=2.0)


# -- integration with the numeric pass -------------------------------------


def test_find_conflicts_covers_both_kinds() -> None:
    old = doc(
        "Travel above EUR 500 requires approval. Alcohol is not reimbursable.",
        "policy-2025",
    )
    new = doc(
        "Travel above EUR 1,000 requires approval. Alcohol may be reimbursed for clients.",
        "policy-2026",
    )
    kinds = {c.kind for c in find_conflicts([old, new])}
    assert kinds == {"numeric", "deontic"}


def test_a_numeric_claim_never_pairs_with_a_deontic_one() -> None:
    """Different units: a figure cannot contradict a permission."""
    a = doc("The limit is EUR 500.", "a")
    b = doc("Claims may be submitted online.", "b")
    assert find_conflicts([a, b]) == []


def test_one_document_alone_produces_no_conflicts() -> None:
    handbook = doc(
        "Travel above EUR 500 requires approval. Meals are reimbursed up to EUR 45 "
        "per day. Alcohol is not reimbursable. Claims may be submitted online.",
        "handbook",
    )
    assert find_conflicts([handbook]) == []
