"""The evaluation harness.

These tests script the model's answers, so they check the *scorer* rather than
any model's ability. A harness that cannot detect a wrong number or a fabricated
answer is worse than none, because it certifies whatever it is pointed at.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fakes import FakeProvider
from openknowledge.cascade import Cascade
from openknowledge.evaluation import (
    Case,
    DatasetError,
    compare,
    filter_cases,
    format_report,
    load_cases,
    parse_cases,
    run_eval,
)
from openknowledge.evaluation.runner import _normalise, _score, _states
from openknowledge.types import Answer, Citation, Tier

GOOD_LEAVE = (
    "Employees with 12 months of continuous service get 20 weeks of fully paid "
    "parental leave. Requests need 30 days notice. [hr-handbook]"
)
WRONG_NUMBER = (
    "Employees with 12 months of continuous service get 26 weeks of fully paid "
    "parental leave. Requests need 30 days notice. [hr-handbook]"
)

LEAVE_CASE = Case(
    id="leave",
    question="How much parental leave do I get?",
    must_cite=("hr-handbook",),
    must_say=("20 weeks", "12 months"),
    must_not_say=("26 weeks",),
)


def build(store, retriever, settings, replies: list[str]) -> Cascade:
    return Cascade(
        store=store,
        retriever=retriever,
        settings=settings,
        local=FakeProvider(replies=list(replies)),
    )


# -- dataset ---------------------------------------------------------------


def test_parses_a_minimal_case() -> None:
    (case,) = parse_cases([{"id": "a", "question": "Q?"}])
    assert case.kind == "answerable" and case.must_say == ()


def test_scalars_are_accepted_where_lists_are_expected() -> None:
    (case,) = parse_cases([{"id": "a", "question": "Q?", "must_say": "20 weeks"}])
    assert case.must_say == (("20 weeks",),)


def test_a_nested_list_means_any_one_of_these() -> None:
    (case,) = parse_cases(
        [{"id": "a", "question": "Q?", "must_say": [["two", "2"], "client-facing"]}]
    )
    assert case.must_say == (("two", "2"), ("client-facing",))


def test_a_bare_string_written_in_code_is_not_matched_character_by_character() -> None:
    """`Case(must_say=("20 weeks",))` is what anyone writes, and left alone it
    would iterate the string - "2" is in almost any answer, so the case would
    silently pass."""
    case = Case(id="a", question="Q?", must_say=("20 weeks",))  # type: ignore[arg-type]
    assert case.must_say == (("20 weeks",),)


@pytest.mark.parametrize(
    ("entry", "match"),
    [
        ({"id": "a"}, "'question' is required"),
        ({"id": "a", "question": "Q?", "kind": "maybe"}, "kind must be"),
        ({"id": "a", "question": "Q?", "must_say": 42}, "must be a string or list"),
    ],
)
def test_malformed_cases_fail_loudly(entry: dict, match: str) -> None:
    """A silently-skipped case is a test that stops testing without telling you."""
    with pytest.raises(DatasetError, match=match):
        parse_cases([entry])


def test_duplicate_ids_are_rejected() -> None:
    with pytest.raises(DatasetError, match="duplicate case id"):
        parse_cases([{"id": "a", "question": "Q?"}, {"id": "a", "question": "R?"}])


def test_the_shipped_golden_set_loads() -> None:
    cases = load_cases(Path(__file__).resolve().parents[1] / "evals" / "golden")
    assert len(cases) >= 10
    assert any(c.kind == "refusal" for c in cases), "a golden set with no safety cases is naive"


def test_every_asserted_fact_has_a_counterpart_where_it_matters() -> None:
    """Numeric facts should carry the plausible wrong answer in must_not_say."""
    cases = load_cases(Path(__file__).resolve().parents[1] / "evals" / "golden")
    numeric = [
        c
        for c in cases
        if c.kind == "answerable" and any(any(ch.isdigit() for ch in f) for f in c.must_say)
    ]
    assert numeric, "expected numeric cases in the sample set"
    without_guard = [c.id for c in numeric if not c.must_not_say]
    assert len(without_guard) <= 2, f"numeric cases lacking a wrong-answer guard: {without_guard}"


def test_filtering_by_kind_and_tag() -> None:
    cases = [
        Case(id="a", question="Q?", tags=("hr",)),
        Case(id="b", question="R?", kind="refusal", tags=("safety",)),
    ]
    assert [c.id for c in filter_cases(cases, kind="refusal")] == ["b"]
    assert [c.id for c in filter_cases(cases, tags=("hr",))] == ["a"]


# -- scoring ---------------------------------------------------------------


async def test_a_correct_answer_passes(store, retriever, settings) -> None:
    cascade = build(store, retriever, settings, [GOOD_LEAVE])
    report = await run_eval(cascade, [LEAVE_CASE])

    assert report.passed
    assert report.accuracy == 1.0
    assert report.false_answers == 0


async def test_a_wrong_number_fails_even_though_the_prose_is_right(
    store, retriever, settings
) -> None:
    """The grounding gate rejects it, so the cascade refuses - and that is a miss."""
    cascade = build(store, retriever, settings, [WRONG_NUMBER])
    report = await run_eval(cascade, [LEAVE_CASE])

    assert not report.passed
    assert report.accuracy == 0.0
    (result,) = report.results
    assert result.tier is Tier.REFUSED
    assert any("refused" in r for r in result.failures)


async def test_a_missing_fact_is_caught(store, retriever, settings) -> None:
    partial = "You get 20 weeks of fully paid parental leave. [hr-handbook]"
    cascade = build(store, retriever, settings, [partial, partial])
    report = await run_eval(cascade, [LEAVE_CASE])

    assert not report.passed
    assert any("12 months" in r for r in report.results[0].failures)


async def test_answering_a_question_the_corpus_cannot_answer_is_flagged_unsafe(
    store, retriever, settings
) -> None:
    """The metric that matters most: confident invention on an uncovered topic."""
    invented = "Employees receive 10 paid sick days per year. [hr-handbook]"
    case = Case(id="sick", question="How many sick days do I get?", kind="refusal")

    # Force it through: pretend the gate passed by pinning the invented answer.
    store.pin("how many sick days do i get", invented)
    cascade = build(store, retriever, settings, [])
    report = await run_eval(cascade, [case])

    assert report.false_answers == 1
    assert report.false_answer_rate == 1.0
    assert not report.passed
    assert report.results[0].false_answer


async def test_a_proper_refusal_passes_the_safety_set(store, retriever, settings) -> None:
    case = Case(id="sick", question="How many sick days do I get?", kind="refusal")
    cascade = build(store, retriever, settings, ["I don't know."])
    report = await run_eval(cascade, [case])

    assert report.passed
    assert report.false_answers == 0


async def test_non_determinism_is_caught(store, retriever, settings) -> None:
    """Two different answers to the same question must fail, however good both are."""
    variant = GOOD_LEAVE.replace("Requests need", "Requests require")
    cascade = Cascade(
        store=store,
        retriever=retriever,
        settings=settings,
        local=FakeProvider(replies=[GOOD_LEAVE, variant]),
    )
    # Defeat the cache so the second ask really re-runs the model.
    cascade.store.put = lambda *a, **k: None  # type: ignore[method-assign]

    report = await run_eval(cascade, [LEAVE_CASE])
    assert report.determinism == 0.0
    assert not report.passed
    assert any("two different answers" in r for r in report.results[0].failures)


async def test_paraphrase_drift_is_caught(store, retriever, settings) -> None:
    case = Case(
        id="leave",
        question="How much parental leave do I get?",
        must_cite=("hr-handbook",),
        must_say=("20 weeks",),
        paraphrases=("what is the parental leave entitlement",),
    )
    drifted = "Parental leave is available to eligible employees. [hr-handbook]"
    cascade = Cascade(
        store=store,
        retriever=retriever,
        settings=settings,
        local=FakeProvider(replies=[GOOD_LEAVE, drifted, drifted]),
    )
    cascade.store.put = lambda *a, **k: None  # type: ignore[method-assign]

    report = await run_eval(cascade, [case])
    assert report.paraphrase_consistency == 0.0
    assert not report.passed


async def test_cost_is_reported_alongside_accuracy(store, retriever, settings) -> None:
    """Either number alone is trivially gamed; the harness reports the pair."""
    settings = settings.model_copy(update={"local_enabled": False, "escalation_enabled": True})
    cascade = Cascade(
        store=store,
        retriever=retriever,
        settings=settings,
        frontier=FakeProvider(model_id="claude-opus-5", tier="frontier", replies=[GOOD_LEAVE]),
    )
    report = await run_eval(cascade, [LEAVE_CASE])

    assert report.accuracy == 1.0
    assert report.total_cost_usd == pytest.approx(0.02)
    # Asked twice for the determinism check; the second was a cache hit.
    assert report.free_share == 0.0
    assert "frontier" in report.tier_counts


# -- baseline comparison ---------------------------------------------------


async def test_regressions_are_detected(store, retriever, settings) -> None:
    cascade = build(store, retriever, settings, [WRONG_NUMBER])
    report = await run_eval(cascade, [LEAVE_CASE])

    result = compare(report, {"accuracy": 1.0, "false_answers": 0, "determinism": 1.0})
    assert not result.ok
    assert any("accuracy" in r for r in result.regressions)


async def test_improvements_are_reported(store, retriever, settings) -> None:
    cascade = build(store, retriever, settings, [GOOD_LEAVE])
    report = await run_eval(cascade, [LEAVE_CASE])

    result = compare(report, {"accuracy": 0.5, "false_answers": 1, "determinism": 1.0})
    assert result.ok
    assert len(result.improvements) == 2


async def test_a_new_false_answer_is_always_a_regression(store, retriever, settings) -> None:
    invented = "Employees receive 10 paid sick days per year. [hr-handbook]"
    store.pin("how many sick days do i get", invented)
    cascade = build(store, retriever, settings, [])
    report = await run_eval(
        cascade, [Case(id="sick", question="How many sick days do I get?", kind="refusal")]
    )

    result = compare(report, {"accuracy": 1.0, "false_answers": 0, "determinism": 1.0})
    assert not result.ok
    assert any("false answers" in r for r in result.regressions)


async def test_a_cost_jump_is_a_regression(store, retriever, settings) -> None:
    settings = settings.model_copy(update={"local_enabled": False, "escalation_enabled": True})
    cascade = Cascade(
        store=store,
        retriever=retriever,
        settings=settings,
        frontier=FakeProvider(model_id="claude-opus-5", tier="frontier", replies=[GOOD_LEAVE]),
    )
    report = await run_eval(cascade, [LEAVE_CASE])

    baseline = {
        "accuracy": 1.0,
        "false_answers": 0,
        "determinism": 1.0,
        "cost_per_question_usd": 0.001,
    }
    assert any("cost per question" in r for r in compare(report, baseline).regressions)


# -- report rendering ------------------------------------------------------


async def test_report_names_unsafe_failures_distinctly(store, retriever, settings) -> None:
    invented = "Employees receive 10 paid sick days per year. [hr-handbook]"
    store.pin("how many sick days do i get", invented)
    cascade = build(store, retriever, settings, [])
    report = await run_eval(
        cascade, [Case(id="sick", question="How many sick days do I get?", kind="refusal")]
    )

    text = format_report(report)
    assert "[UNSAFE]" in text
    assert "FAILED" in text


async def test_a_clean_run_says_so(store, retriever, settings) -> None:
    cascade = build(store, retriever, settings, [GOOD_LEAVE])
    text = format_report(await run_eval(cascade, [LEAVE_CASE]))
    assert "PASSED" in text


# -- contradiction detection -----------------------------------------------

from openknowledge.evaluation import (  # noqa: E402
    ConflictSetError,
    format_conflict_report,
    load_conflict_cases,
    parse_conflict_cases,
    run_conflict_eval,
)

REAL = {
    "id": "flip",
    "expect": "conflict",
    "documents": [
        {"id": "a", "text": "Contractors are eligible for parental leave."},
        {"id": "b", "text": "Contractors are excluded from parental leave."},
    ],
}
CLEAN = {
    "id": "quiet",
    "expect": "clean",
    "documents": [
        {"id": "a", "text": "Alcohol is not reimbursable under any circumstances."},
        {"id": "b", "text": "VPN access requests must be approved by IT Operations."},
    ],
}


def test_a_set_without_clean_cases_is_rejected() -> None:
    """A set of only real contradictions measures recall and cannot see a
    false positive - which is the failure that gets a detector switched off."""
    with pytest.raises(ConflictSetError, match="clean"):
        parse_conflict_cases([REAL])


@pytest.mark.parametrize(
    ("entry", "match"),
    [
        ({"id": "x", "expect": "maybe", "documents": [{"id": "a", "text": "t"}]}, "expect must"),
        ({"id": "x", "expect": "clean"}, "at least one document"),
        ({"id": "x", "expect": "clean", "documents": [{"id": "a"}]}, "id and text"),
    ],
)
def test_malformed_conflict_cases_fail_loudly(entry: dict, match: str) -> None:
    with pytest.raises(ConflictSetError, match=match):
        parse_conflict_cases([entry, CLEAN])


def test_both_directions_are_scored() -> None:
    report = run_conflict_eval(parse_conflict_cases([REAL, CLEAN]))
    assert report.precision == 1.0
    assert report.recall == 1.0
    assert report.passed


def test_a_miss_shows_up_as_lost_recall() -> None:
    """Strictness high enough to suppress a real contradiction."""
    report = run_conflict_eval(parse_conflict_cases([REAL, CLEAN]), deontic_strictness=2.5)
    assert report.recall == 0.0
    assert report.precision == 1.0, "suppressing everything cannot hurt precision"
    assert not report.passed
    assert "MISSED" in format_conflict_report(report)


def test_the_shipped_conflict_set_is_clean() -> None:
    """The standing measurement. Any regression here is an accuracy regression."""
    cases = load_conflict_cases(Path(__file__).resolve().parents[1] / "evals" / "conflicts")
    report = run_conflict_eval(cases)

    assert len(cases) >= 15
    assert sum(not c.expect_conflict for c in cases) >= len(cases) // 3, (
        "at least a third of the set should be near-misses that must stay quiet"
    )
    assert report.precision == 1.0, report.to_dict()["failures"]
    assert report.recall == 1.0, report.to_dict()["failures"]


def test_both_detectors_are_represented_in_the_set() -> None:
    cases = load_conflict_cases(Path(__file__).resolve().parents[1] / "evals" / "conflicts")
    kinds = {c.kind for c in cases if c.kind}
    assert {"numeric", "deontic"} <= kinds


# -- a contested response is a refusal --------------------------------------


def test_a_contested_response_passes_a_refusal_case() -> None:
    """Found by a live run, which scored the safest thing the cascade does as a
    fabrication.

    A contested response declines to answer and names the two documents that
    disagree. It is a *better* refusal than a plain one, and counting it as a
    false answer is the pressure that gets a contradiction feature switched off
    for hurting the numbers.
    """
    case = parse_cases(
        [{"id": "contested", "question": "What is the travel limit?", "kind": "refusal"}]
    )[0]
    answer = Answer(
        text="Your documents disagree on this, so I won't guess:\n"
        "  - [expenses-policy] says EUR 500, [travel-guidelines] says EUR 1,000",
        tier=Tier.CONTESTED,
        model_id="none",
        cache_key="k",
        grounded=True,
    )

    passed, failures, false_answer = _score(case, answer)

    assert passed
    assert not false_answer, "declining to answer is never a fabrication"
    assert failures == ()


def test_a_contested_answerable_case_says_so_instead_of_blaming_retrieval() -> None:
    """The two need different fixes, so they need different messages.

    Reported through the citation and content checks it read as "did not cite"
    and "contains incorrect content", which sends the reader looking for a
    retrieval bug that is not there. The documents disagree; somebody has to
    decide.
    """
    case = parse_cases(
        [
            {
                "id": "meals",
                "question": "How much for meals?",
                "must_cite": ["expenses-policy"],
                "must_say": ["45"],
            }
        ]
    )[0]
    answer = Answer(
        text="Your documents disagree on this, so I won't guess.",
        tier=Tier.CONTESTED,
        model_id="none",
        cache_key="k",
        grounded=True,
        notes=("[expenses-policy] says EUR 45 but [archive-2023] says EUR 35",),
    )

    passed, failures, false_answer = _score(case, answer)

    assert not passed and not false_answer
    assert len(failures) == 1
    assert "refused as contested" in failures[0]
    assert "archive-2023" in failures[0]
    assert not any("did not cite" in f for f in failures)


def test_answering_a_refusal_case_is_still_a_false_answer() -> None:
    """The point of widening the refusal check is not to make failures vanish."""
    case = parse_cases([{"id": "sick", "question": "How many sick days?", "kind": "refusal"}])[0]
    answer = Answer(
        text="You get 25 sick days a year. [hr-handbook]",
        tier=Tier.LOCAL,
        model_id="m",
        cache_key="k",
        grounded=True,
    )

    passed, _failures, false_answer = _score(case, answer)
    assert not passed and false_answer


def test_the_tiers_that_decline_are_named_in_one_place() -> None:
    assert Tier.REFUSED.declined and Tier.CONTESTED.declined
    for tier in (Tier.PINNED, Tier.EXACT_CACHE, Tier.DRAFT, Tier.LOCAL, Tier.FRONTIER):
        assert not tier.declined


# -- matching facts and forbidden content -----------------------------------


@pytest.mark.parametrize(
    ("text", "phrase", "expected"),
    [
        # A figure must not match inside a larger one. Found by a live run, which
        # scored a correct "EUR 25,000" as containing the forbidden "5,000".
        ("the cfo limit is eur 25,000", "5,000", False),
        ("the cfo limit is eur 25,000", "25,000", True),
        ("the allowance is eur 145 per night", "45", False),
        ("the allowance is eur 45 per day", "45", True),
        ("the rate is 0.42 per kilometre", "42", False),
        ("claims must be submitted within 60 days", "30 days", False),
        ("claims must be submitted within 60 days", "60 days", True),
        # Prose keeps ordinary substring behaviour.
        ("alcohol is not reimbursable", "not reimbursable", True),
        ("employees get 20 weeks of leave", "20 weeks", True),
    ],
)
def test_a_figure_never_matches_inside_a_larger_figure(
    text: str, phrase: str, expected: bool
) -> None:
    assert _states(_normalise(text), phrase) is expected


async def test_a_correct_answer_is_not_failed_by_a_substring_of_its_own_figure(
    store, retriever, settings
) -> None:
    """The end-to-end shape of the same bug.

    A golden set that fails correct answers is worse than none, because it
    teaches its author to loosen the checks that catch real errors.
    """
    case = Case(
        id="limit",
        question="How much parental leave do I get?",
        must_cite=("hr-handbook",),
        must_say=(("20 weeks",),),
        must_not_say=("0 weeks",),
    )
    cascade = build(store, retriever, settings, [GOOD_LEAVE, GOOD_LEAVE])
    report = await run_eval(cascade, [case])

    assert report.results[0].passed, report.results[0].failures


def test_a_fact_with_alternatives_reads_as_one_requirement_when_it_fails() -> None:
    case = parse_cases(
        [{"id": "a", "question": "Q?", "must_say": [["two", "2"]], "must_cite": []}]
    )[0]
    answer = Answer(
        text="Employees may work remotely for up to three days. [hr-remote-work]",
        tier=Tier.LOCAL,
        model_id="m",
        cache_key="k",
        citations=(Citation(document_id="hr-remote-work", document_title="Remote", snippet="x"),),
        grounded=True,
    )
    _passed, failures, _false = _score(case, answer)
    assert len(failures) == 1
    assert "'two' or '2'" in failures[0]


def test_every_answerable_case_in_the_shipped_set_guards_a_wrong_answer() -> None:
    """The rule the golden set's own header states, enforced.

    "20 weeks" appearing is half the check; "26 weeks" not appearing is the
    other half. A case with no `must_not_say` passes on any answer that happens
    to contain its phrase, which on a corpus full of "30 days" is most of them -
    and a set of those produces a 100% score that means nothing.
    """
    root = Path(__file__).resolve().parent.parent / "evals"
    for directory in ("golden", "golden-aveline"):
        for case in load_cases(root / directory):
            if case.kind != "answerable":
                continue
            assert case.must_say, f"{case.id}: asserts nothing"
            assert case.must_not_say, f"{case.id}: no wrong answer to rule out"
