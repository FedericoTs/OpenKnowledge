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
from openknowledge.types import Tier

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
    assert case.must_say == ("20 weeks",)


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
    cases = load_cases(Path(__file__).resolve().parents[1] / "evals")
    assert len(cases) >= 10
    assert any(c.kind == "refusal" for c in cases), "a golden set with no safety cases is naive"


def test_every_asserted_fact_has_a_counterpart_where_it_matters() -> None:
    """Numeric facts should carry the plausible wrong answer in must_not_say."""
    cases = load_cases(Path(__file__).resolve().parents[1] / "evals")
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
