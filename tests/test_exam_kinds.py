"""Two defects the apply-the-rule runs exposed, and the system defect under one.

Both cases in `evals/golden-rules` were failing for reasons that were not the
system's. Establishing which is which took reading the corpus rather than
trusting the note I had written about it, and one of the two turned out to be a
real bug:

`rule-03` demanded a citation to `finance-procurement-policy` while the system
cited `finance-approval-limits`. Both state the approval bands, and the second
states them for the category the question actually names - "Software
subscription | 0 | 5000 | Line manager". The exam failed a correct answer for
citing the *more* specific source.

`rule-06` was written as an answerable boundary case for a corpus that cannot
answer it: two live documents put the travel approval threshold at EUR 500 and
EUR 1,000. Reporting that is the designed behaviour. But a THIRD document was
being reported alongside them - the 2023 archive, which declares itself
superseded and which retrieval already excludes - so the system was withholding
an answer partly on the authority of text the reader would never be shown.
"""

from __future__ import annotations

import pytest

from openknowledge.evaluation.dataset import Case, DatasetError, parse_cases


def test_must_cite_accepts_alternatives() -> None:
    """Two documents can each independently answer a question."""
    case = parse_cases(
        [
            {
                "id": "c",
                "question": "who approves it?",
                "must_cite": [["finance-procurement-policy", "finance-approval-limits"]],
            }
        ]
    )[0]
    assert case.must_cite == (("finance-procurement-policy", "finance-approval-limits"),)


def test_a_bare_string_must_cite_is_still_one_required_document() -> None:
    """Every set in this repository writes `must_cite: [hr-handbook]`, and that
    must keep meaning "cite this one"."""
    case = parse_cases(
        [{"id": "c", "question": "q?", "must_cite": ["hr-handbook", "hr-expenses-policy"]}]
    )[0]
    assert case.must_cite == (("hr-handbook",), ("hr-expenses-policy",))


def test_a_case_built_in_code_is_coerced_the_same_way() -> None:
    """`Case(must_cite=("hr-handbook",))` is what callers write. Left alone,
    each LETTER would become its own group of alternatives - and a group is
    satisfied by any citation containing it, so the requirement would vanish.
    The same trap `must_say` documents, one field along."""
    case = Case(id="c", question="q?", must_cite=("hr-handbook",))
    assert case.must_cite == (("hr-handbook",),)


def test_contested_is_a_kind_a_case_may_declare() -> None:
    case = parse_cases([{"id": "c", "question": "q?", "kind": "contested"}])[0]
    assert case.kind == "contested"


def test_an_unknown_kind_is_still_refused() -> None:
    with pytest.raises(DatasetError, match="kind must be"):
        parse_cases([{"id": "c", "question": "q?", "kind": "probably"}])


# --- the system defect: a retired document was still gating answers ---------


def _conflict(kind: str) -> object:
    from openknowledge.knowledge.store import StoredConflict

    return StoredConflict(
        key=f"k-{kind}",
        left_document="archive-expenses-policy-2023",
        left_raw="EUR 300",
        left_sentence="Any single item of travel expenditure above EUR 300 requires approval.",
        right_document="hr-travel-guidelines",
        right_raw="EUR 1,000",
        right_sentence="Travel expenditure above EUR 1,000 requires written approval.",
        unit="EUR",
        kind=kind,
        overlap=0.9,
        context=frozenset({"travel", "expenditure", "approval", "requires", "written"}),
        status="open",
    )


def test_a_superseded_document_does_not_withhold_an_answer() -> None:
    """Retrieval already excludes a document that declares itself superseded,
    so a conflict it raises withholds an answer on the authority of text the
    reader was never going to be shown.

    The variant grouping that predates this spared a retired copy only against
    the document that REPLACED it. Against a third document it still gated -
    which is how "do I need approval for a travel expense of exactly EUR 500?"
    came back refused, citing a 2023 policy the corpus had retired.
    """
    from openknowledge.knowledge.relevance import relevant_conflicts

    question = "Do I need written approval for travel expenditure?"
    assert relevant_conflicts(question, [_conflict("numeric")]), "the fixture must be relevant"
    assert not relevant_conflicts(question, [_conflict("superseded")])


def test_a_live_disagreement_still_withholds_one() -> None:
    """The point is not to stop gating. Two live documents disagreeing is
    exactly what the contested tier is for."""
    from openknowledge.knowledge.relevance import relevant_conflicts

    live = _conflict("numeric")
    assert relevant_conflicts("Do I need written approval for travel expenditure?", [live])
