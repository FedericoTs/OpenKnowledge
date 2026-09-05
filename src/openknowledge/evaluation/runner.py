"""Runs the golden set and scores it.

Reports accuracy *and* cost together, on purpose. Either number alone is easy to
win and meaningless: accuracy is trivially maximised by sending everything to a
frontier model, and cost is trivially minimised by answering everything from a
stale cache. The pair is the actual objective.

Three properties are measured that a generic RAG benchmark would not bother with,
because they are the ones this design stakes its claims on:

* **False answers** - questions the corpus does not cover that got an answer
  anyway. This is the safety metric, and it is reported separately from accuracy
  because a single false answer matters more than several missed ones.
* **Determinism** - every case is asked twice and the two answers must be
  byte-identical.
* **Paraphrase consistency** - the same question in different words must produce
  the same facts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..cascade.router import Cascade
from ..types import Answer, Tier
from .dataset import Case

_WHITESPACE = re.compile(r"\s+")


def _normalise(text: str) -> str:
    return _WHITESPACE.sub(" ", text.casefold()).strip()


def _states(text: str, phrase: str) -> bool:
    """Whether ``text`` states ``phrase``, with figures matched on boundaries.

    Plain substring matching is right for prose and wrong where a number touches
    the edge of the phrase, because every figure is a substring of a larger one.
    A live run scored a correct "EUR 25,000" as containing the forbidden
    "5,000"; the same rule would read "EUR 145" as containing "45" and "20
    weeks" as containing "0 weeks". A golden set that fails correct answers is
    worse than none, because it teaches its author to loosen the checks that
    catch real errors.

    The guard follows the digits rather than the whole phrase: a needle
    *starting* with a digit may not be preceded by one, and a needle *ending*
    with a digit may not be followed by one. Everything else keeps ordinary
    substring behaviour, which is what "not reimbursable" wants.
    """
    needle = _normalise(phrase)
    if not needle:
        return False
    left = r"(?<![\d.,])" if needle[0].isdigit() else ""
    right = r"(?![\d]|[.,]\d)" if needle[-1].isdigit() else ""
    if not left and not right:
        return needle in text
    return re.search(rf"{left}{re.escape(needle)}{right}", text) is not None


@dataclass(frozen=True, slots=True)
class CaseResult:
    case: Case
    answer: Answer
    passed: bool
    failures: tuple[str, ...] = ()
    #: Answered when it should have refused. Tracked separately - this is the
    #: failure mode that makes a bot untrustworthy rather than merely unhelpful.
    false_answer: bool = False
    #: Whether asking twice gave the same answer, or None when it was not
    #: asked twice. None and True are different facts and must not be
    #: reported as the same number - see `EvalReport.determinism`.
    deterministic: bool | None = None
    paraphrase_consistent: bool = True
    cost_usd: float = 0.0
    support: float | None = None

    @property
    def tier(self) -> Tier:
        return self.answer.tier


@dataclass
class EvalReport:
    results: list[CaseResult] = field(default_factory=list)

    # -- accuracy ---------------------------------------------------------
    @property
    def answerable(self) -> list[CaseResult]:
        return [r for r in self.results if r.case.kind == "answerable"]

    @property
    def refusal_cases(self) -> list[CaseResult]:
        return [r for r in self.results if r.case.kind == "refusal"]

    @property
    def accuracy(self) -> float:
        cases = self.answerable
        return sum(r.passed for r in cases) / len(cases) if cases else 0.0

    @property
    def false_answers(self) -> int:
        return sum(r.false_answer for r in self.results)

    @property
    def false_answer_rate(self) -> float:
        cases = self.refusal_cases
        return self.false_answers / len(cases) if cases else 0.0

    @property
    def determinism(self) -> float | None:
        """How often asking twice gave the same answer, or None if nobody asked.

        None rather than 1.0, because `--no-determinism` asks each question
        once and a metric that then reports 100% is claiming a measurement it
        did not take. That number went into saved baselines too, so a later
        run that genuinely checked and scored 90% was compared against a
        figure nobody had ever measured.
        """
        checked = [r.deterministic for r in self.results if r.deterministic is not None]
        return sum(checked) / len(checked) if checked else None

    @property
    def paraphrase_consistency(self) -> float:
        checked = [r for r in self.results if r.case.paraphrases]
        return sum(r.paraphrase_consistent for r in checked) / len(checked) if checked else 1.0

    # -- cost -------------------------------------------------------------
    @property
    def total_cost_usd(self) -> float:
        return sum(r.cost_usd for r in self.results)

    @property
    def cost_per_question_usd(self) -> float:
        return self.total_cost_usd / len(self.results) if self.results else 0.0

    @property
    def tier_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in self.results:
            counts[r.tier.value] = counts.get(r.tier.value, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    @property
    def support_separation(self) -> tuple[float, float] | None:
        """Mean grounding support on the cases that passed, and on those that failed.

        The only claim worth making for any answer-level score: that it is lower
        where the answer was wrong. Reported rather than asserted, because on a
        given corpus it may simply not be true - and a score that does not
        separate is worse than none, since it makes wrong answers look checked.

        A scored confidence built on top of this was measured and withdrawn; see
        `retrieval/confidence.py`. Anything proposed to replace it should be
        checked here **and** against deliberately degraded retrieval, which is
        the test the first attempt failed.
        """
        scored = [(r.support, r.passed) for r in self.answerable if r.support is not None]
        passed = [c for c, ok in scored if ok]
        failed = [c for c, ok in scored if not ok]
        if not passed or not failed:
            return None
        return sum(passed) / len(passed), sum(failed) / len(failed)

    @property
    def free_share(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.tier.is_cache_hit for r in self.results) / len(self.results)

    @property
    def passed(self) -> bool:
        """Overall verdict: perfect safety, plus full accuracy and determinism.

        An empty answerable set (``--only refusal``) is not an accuracy failure -
        there was nothing to get right.
        """
        accuracy_ok = not self.answerable or self.accuracy == 1.0
        # An unchecked determinism does not fail the run - the caller asked
        # for it to be skipped - but it does not pass it either: there is
        # nothing to pass.
        determinism_ok = self.determinism is None or self.determinism == 1.0
        return self.false_answers == 0 and accuracy_ok and determinism_ok

    def to_dict(self) -> dict[str, object]:
        return {
            "cases": len(self.results),
            "accuracy": round(self.accuracy, 4),
            "false_answers": self.false_answers,
            "false_answer_rate": round(self.false_answer_rate, 4),
            "determinism": None if self.determinism is None else round(self.determinism, 4),
            "paraphrase_consistency": round(self.paraphrase_consistency, 4),
            "cost_per_question_usd": round(self.cost_per_question_usd, 6),
            "total_cost_usd": round(self.total_cost_usd, 6),
            "free_share": round(self.free_share, 4),
            "support_separation": (
                [round(v, 4) for v in self.support_separation] if self.support_separation else None
            ),
            "tiers": self.tier_counts,
            "failures": [
                {"id": r.case.id, "tier": r.tier.value, "reasons": list(r.failures)}
                for r in self.results
                if not r.passed
            ],
        }


def _score(case: Case, answer: Answer) -> tuple[bool, tuple[str, ...], bool]:
    """Grade one answer. Returns (passed, failures, false_answer)."""
    text = _normalise(answer.text)
    failures: list[str] = []

    if case.kind == "refusal":
        if answer.tier.declined:
            return True, (), False
        return (
            False,
            (f"answered a question the corpus does not cover: {answer.text[:120]!r}",),
            True,
        )

    if answer.tier is Tier.CONTESTED:
        # Distinguished from a plain refusal because the fix is different: the
        # documents disagree and somebody has to decide, whereas a refusal
        # usually means the fact was never retrieved. Reporting it as "did not
        # cite" and "contains incorrect content" - which is what the checks below
        # produce - sends the reader looking for a retrieval bug that is not
        # there.
        contested = ", ".join(answer.notes[:2]) or "sources disagree"
        return False, (f"refused as contested: {contested}",), False

    if answer.tier is Tier.REFUSED:
        return False, ("refused a question the corpus does cover",), False

    cited = {c.document_id for c in answer.citations}
    missing = [doc for doc in case.must_cite if doc not in cited]
    if missing:
        failures.append(f"did not cite {', '.join(missing)} (cited: {', '.join(sorted(cited))})")

    for alternatives in case.must_say:
        # Any one spelling satisfies the fact. Requiring all of them would fail
        # every correct answer, because no answer says a number both ways.
        if not any(_states(text, form) for form in alternatives):
            failures.append(f"missing required fact {_describe(alternatives)}")

    for wrong in case.must_not_say:
        if _states(text, wrong):
            failures.append(f"contains incorrect content {wrong!r}")

    if case.must_list:
        # A share, not a verdict per item: the question this exists to answer
        # is "how much of the list did it manage", and a line per missing term
        # would bury that number under eighty lines nobody reads.
        listed = _items_listed(case, text)
        share = len(listed) / len(case.must_list)
        if share < case.min_share:
            failures.append(
                f"listed {len(listed)} of {len(case.must_list)} required items "
                f"({share:.0%}), needed {case.min_share:.0%}"
            )

    if not answer.grounded:
        failures.append("answer was not grounded")

    return not failures, tuple(failures), False


def _items_listed(case: Case, normalised_text: str) -> tuple[str, ...]:
    """Which of ``must_list`` the (already normalised) answer states."""
    return tuple(item for item in case.must_list if _states(normalised_text, item))


def _describe(alternatives: tuple[str, ...]) -> str:
    """How a fact reads in a failure line."""
    if len(alternatives) == 1:
        return repr(alternatives[0])
    return " or ".join(repr(a) for a in alternatives)


def _facts_present(case: Case, answer: Answer) -> frozenset[str]:
    """Which facts the answer states, identified by the fact rather than by
    which of its spellings was used.

    Paraphrase consistency compares two answers to the same question, and they
    may legitimately spell one figure differently. Keying on the fact rather
    than the wording is what keeps that from reading as an inconsistency.
    """
    text = _normalise(answer.text)
    facts = frozenset(
        alternatives[0]
        for alternatives in case.must_say
        if any(_states(text, form) for form in alternatives)
    )
    # Listed items count as facts too, so a paraphrase that lists a different
    # subset of the same glossary is reported as the inconsistency it is.
    return facts | frozenset(_items_listed(case, text))


async def run_case(cascade: Cascade, case: Case, *, check_determinism: bool = True) -> CaseResult:
    """Ask one case, score it, and check determinism and paraphrase consistency."""
    principals = frozenset(case.principals) if case.principals is not None else None

    answer = await cascade.answer(case.question, principals=principals, channel="eval")
    passed, failures, false_answer = _score(case, answer)
    cost = answer.cost_usd
    support = answer.support

    deterministic: bool | None = None
    if check_determinism:
        again = await cascade.answer(case.question, principals=principals, channel="eval")
        cost += again.cost_usd
        deterministic = again.text == answer.text
        if not deterministic:
            failures = (*failures, "asked twice, got two different answers")
            passed = False

    consistent = True
    if case.paraphrases:
        expected = _facts_present(case, answer)
        for phrasing in case.paraphrases:
            other = await cascade.answer(phrasing, principals=principals, channel="eval")
            cost += other.cost_usd
            if _facts_present(case, other) != expected:
                consistent = False
                failures = (*failures, f"paraphrase {phrasing!r} produced different facts")
                passed = False
                break

    return CaseResult(
        case=case,
        answer=answer,
        passed=passed,
        failures=failures,
        false_answer=false_answer,
        deterministic=deterministic,
        paraphrase_consistent=consistent,
        cost_usd=cost,
        support=support,
    )


async def run_eval(
    cascade: Cascade, cases: list[Case], *, check_determinism: bool = True
) -> EvalReport:
    report = EvalReport()
    for case in cases:
        report.results.append(await run_case(cascade, case, check_determinism=check_determinism))
    return report
