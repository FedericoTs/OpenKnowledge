"""Numeric claim extraction and conflict detection.

The free first pass at "do these documents contradict each other". It has to
catch the disagreements that matter and stay quiet about everything else - an
admin who sees three false flags stops reading the fourth.
"""

from __future__ import annotations

import pytest

from openknowledge.knowledge.claims import Claim, extract_claims, find_conflicts
from openknowledge.retrieval import Document


def doc(doc_id: str, text: str) -> Document:
    return Document(doc_id, doc_id.replace("-", " ").title(), text)


def raws(claims: list[Claim]) -> list[str]:
    return [c.raw for c in claims]


# -- extraction ------------------------------------------------------------


def test_extracts_currency_before_and_after_the_number() -> None:
    claims = extract_claims(doc("d", "Approval above EUR 500 is required. Meals cap at 45 EUR."))
    assert [(c.value, c.unit) for c in claims] == [(500.0, "eur"), (45.0, "eur")]


def test_extracts_durations_and_percentages() -> None:
    claims = extract_claims(
        doc("d", "Submit within 60 days. Notice is 4 weeks. The uplift is 15%.")
    )
    assert [(c.value, c.unit) for c in claims] == [
        (60.0, "days"),
        (4.0, "weeks"),
        (15.0, "percent"),
    ]


def test_business_days_normalise_to_days() -> None:
    (claim,) = extract_claims(doc("d", "Approval takes 2 business days."))
    assert claim.unit == "days" and claim.value == 2.0


def test_sentence_punctuation_is_not_part_of_the_figure() -> None:
    (claim,) = extract_claims(doc("d", "The limit is EUR 500."))
    assert claim.raw == "EUR 500"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("The cap is EUR 1,200 per quarter.", 1200.0),  # thousands, comma
        ("The cap is EUR 1.200 per quarter.", 1200.0),  # thousands, european dot
        ("The rate is 1.5%.", 1.5),  # genuine decimal
        ("The rate is 1,5%.", 1.5),  # european decimal
        ("The cap is EUR 1,234,567.89 total.", 1234567.89),
    ],
)
def test_number_formats_parse_to_the_same_value(text: str, expected: float) -> None:
    """Misreading a separator would invent a conflict between a document and itself."""
    (claim,) = extract_claims(doc("d", text))
    assert claim.value == pytest.approx(expected)


def test_one_figure_produces_one_claim() -> None:
    """'EUR 45 per day' matches both a currency and a unit pattern."""
    claims = extract_claims(doc("d", "Meals are reimbursed up to EUR 45 per day."))
    assert len(claims) == 1
    assert claims[0].unit == "eur"


def test_context_excludes_filler_and_keeps_topic_words() -> None:
    (claim,) = extract_claims(doc("d", "Travel expenses require approval above EUR 500."))
    assert {"travel", "expenses", "approval"} <= claim.context
    assert "the" not in claim.context and "above" not in claim.context


def test_a_document_with_no_numbers_yields_nothing() -> None:
    assert extract_claims(doc("d", "Be excellent to each other.")) == []


# -- conflicts -------------------------------------------------------------

OLD = doc(
    "expenses-2025",
    "Travel expenses require prior written approval for any amount above EUR 500. "
    "Meals are reimbursed up to EUR 45 per day. "
    "Expense claims must be submitted within 60 days of the date incurred.",
)
NEW = doc(
    "expenses-2026",
    "Travel expenses require prior written approval for any amount above EUR 1,000. "
    "Meals are reimbursed up to EUR 45 per day. "
    "Expense claims must be submitted within 30 days of the date incurred.",
)
UNRELATED = doc("vpn", "VPN access requests take 2 business days to approve.")


def test_finds_the_real_disagreements() -> None:
    conflicts = find_conflicts([OLD, NEW])
    pairs = {(c.left.value, c.right.value) for c in conflicts}
    assert (500.0, 1000.0) in pairs
    assert (60.0, 30.0) in pairs


def test_agreement_is_not_a_conflict() -> None:
    """Both documents say EUR 45 for meals. That is the system working."""
    conflicts = find_conflicts([OLD, NEW])
    assert all(45.0 not in (c.left.value, c.right.value) for c in conflicts)


def test_unrelated_documents_do_not_collide() -> None:
    """'2 business days' must not conflict with '60 days' for expense claims."""
    conflicts = find_conflicts([OLD, UNRELATED])
    assert conflicts == []


def test_different_rules_in_one_document_are_not_flagged() -> None:
    """A rule with conditions is not a contradiction, and flagging it teaches
    the admin to ignore the feature."""
    assert find_conflicts([OLD]) == []


def test_same_number_different_unit_is_not_a_conflict() -> None:
    a = doc("a", "The notice period is 30 days for all staff.")
    b = doc("b", "The notice period is 30 weeks for all staff.")
    # Different units are not comparable, so this is not reported as a numeric
    # conflict - it is a prose disagreement for the re-verification path.
    assert find_conflicts([a, b]) == []


def test_a_conflict_key_is_order_independent() -> None:
    forward = find_conflicts([OLD, NEW])
    backward = find_conflicts([NEW, OLD])
    assert {c.key for c in forward} == {c.key for c in backward}


def test_conflicts_are_ranked_by_confidence() -> None:
    conflicts = find_conflicts([OLD, NEW])
    assert conflicts == sorted(conflicts, key=lambda c: (-c.overlap, c.key))


def test_a_conflict_describes_itself_for_a_human() -> None:
    (conflict, *_) = find_conflicts([OLD, NEW])
    text = conflict.describe()
    assert "expenses-2025" in text and "expenses-2026" in text


def test_raising_the_threshold_reduces_flags() -> None:
    loose = find_conflicts([OLD, NEW], min_overlap=0.1, min_shared_words=1)
    strict = find_conflicts([OLD, NEW], min_overlap=0.9, min_shared_words=5)
    assert len(strict) <= len(loose)
