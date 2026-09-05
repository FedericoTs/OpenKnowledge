"""The apply-the-rule set is fair, and its forbidden phrases are safe.

A golden set that fails correct answers is worse than none, because it teaches
its author to loosen the checks that catch real errors - golden-ftr's notes
record three such failures. This set is scored by substring, and its answers
are derived rather than quoted, which is the combination most likely to produce
exactly that. So every requirement and every prohibition is checked here
against a written-out correct answer.
"""

from __future__ import annotations

import pathlib

import pytest

from openknowledge.evaluation.dataset import load_cases
from openknowledge.evaluation.runner import _normalise, _states

RULES = pathlib.Path(__file__).resolve().parents[1] / "evals" / "golden-rules" / "rules.yaml"

#: A correct answer to each answerable case, written out. Deliberately verbose,
#: quoting the rule, and phrased with negations where a real answer would use
#: them - "is not above", "does not exceed", "no additional requirement". That
#: last part is load-bearing: an earlier version of rule-12's answer avoided
#: the word "not" entirely, and the fairness test below then passed happily
#: with `must_not_say: [no]` restored, which is precisely the trap it exists
#: to catch. A correct answer that dodges the trap tests nothing.
CORRECT = {
    "rule-01-40k-needs-quotes": (
        "Yes - three competitive quotes are required for any contract with an annual value "
        "above EUR 25,000 [finance-procurement-policy]."
    ),
    "rule-02-40k-approver": (
        "A EUR 40,000 annual contract is approved by the Chief Financial Officer, who also "
        "requires three competitive quotes [finance-procurement-policy]."
    ),
    "rule-03-3k-approver": (
        "EUR 3,000 a year falls in the up to EUR 5,000 band, so a line manager approves it and "
        "there is no additional requirement - it does not need a business case "
        "[finance-procurement-policy]."
    ),
    "rule-04-60k-board": (
        "EUR 60,000 is above EUR 50,000, so the Board approves it and three quotes and legal "
        "review are required [finance-procurement-policy]."
    ),
    "rule-05-750-expense-approval": (
        "Yes. Any single item of travel expenditure above EUR 500 requires written approval "
        "from a line manager before the expense is incurred [hr-expenses-policy]."
    ),
    "rule-06-exactly-500": (
        "EUR 500 is not above the threshold - the rule applies above EUR 500 - so it is at or "
        "below it and may be claimed without prior approval [hr-expenses-policy]."
    ),
    "rule-07-exactly-25000-quotes": (
        "No, quotes are not required at exactly EUR 25,000: three competitive quotes apply to "
        "contracts above EUR 25,000, and 25,000 is not above 25,000 [finance-procurement-policy]."
    ),
    "rule-08-exactly-25000-approver": (
        "Exactly EUR 25,000 sits in the EUR 5,001 to EUR 25,000 band, so the Head of "
        "Department approves it, with a written business case [finance-procurement-policy]."
    ),
    "rule-09-exactly-5000": (
        "A EUR 5,000 annual contract is not above EUR 5,000, so it sits in the first band: a "
        "line manager approves it and no business case is required "
        "[finance-procurement-policy]."
    ),
    "rule-10-exactly-20-days-abroad": (
        "Requests to work from another country for more than 20 working days in a rolling 12 "
        "months will not be approved; exactly 20 working days does not exceed that, though "
        "prior written approval from People Operations is still required [hr-remote-work]."
    ),
    "rule-11-exactly-12-months-service": (
        "With 12 months of continuous service at the expected date of birth you are entitled "
        "to 20 weeks of fully paid parental leave [hr-parental-leave]."
    ),
    "rule-12-exactly-14-character-password": (
        "A 14 characters password is long enough: passwords must be at least 14 characters, "
        "so 14 is not too short [security-information-security]."
    ),
    "rule-13-analyst-cannot-approve-1200": (
        "No - an Analyst may authorise up to EUR 500, so a EUR 1,200 claim cannot be approved "
        "by an Analyst and escalates to a Manager, whose limit is EUR 1,500 "
        "[hr-expenses-policy]."
    ),
    "rule-14-split-purchase": (
        "Splitting a purchase to stay below a threshold is a disciplinary matter, so the "
        "EUR 60,000 purchase needs Board approval as one contract [finance-procurement-policy]."
    ),
}


@pytest.fixture(scope="module")
def cases():
    return load_cases(RULES)


def test_the_set_is_the_shape_it_claims(cases) -> None:
    assert len(cases) == 16
    assert sum(1 for c in cases if c.kind == "refusal") == 2
    assert sum(1 for c in cases if "boundary" in c.tags) == 7


def test_every_answerable_case_has_a_written_out_correct_answer(cases) -> None:
    """Adding a case without one would leave it unchecked by the two tests below."""
    answerable = {c.id for c in cases if c.kind == "answerable"}
    assert answerable == set(CORRECT)


@pytest.mark.parametrize("case_id", sorted(CORRECT))
def test_a_correct_answer_is_not_rejected(case_id: str, cases) -> None:
    """No forbidden phrase may appear in a right answer.

    `must_not_say: [no]` fails this, because "not" contains "no" - and that is
    how the first draft of rule-12 was written.
    """
    case = next(c for c in cases if c.id == case_id)
    text = _normalise(CORRECT[case_id])
    for forbidden in case.must_not_say:
        assert not _states(text, forbidden), f"{case_id}: {forbidden!r} is in a correct answer"


@pytest.mark.parametrize("case_id", sorted(CORRECT))
def test_a_correct_answer_is_accepted(case_id: str, cases) -> None:
    """Every required fact group must be satisfiable by a real answer."""
    case = next(c for c in cases if c.id == case_id)
    text = _normalise(CORRECT[case_id])
    for group in case.must_say:
        assert any(_states(text, form) for form in group), f"{case_id}: {group} not satisfied"


def test_the_boundary_cases_sit_on_a_threshold(cases) -> None:
    """A boundary case whose figure is not a threshold is an interior case
    mislabelled, and the label is what says which failures matter most."""
    thresholds = {"500", "5,000", "25,000", "20", "12", "14"}
    for case in (c for c in cases if "boundary" in c.tags):
        assert any(t in case.question for t in thresholds), case.id
