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


@dataclass(frozen=True, slots=True)
class CaseResult:
    case: Case
    answer: Answer
    passed: bool
    failures: tuple[str, ...] = ()
    #: Answered when it should have refused. Tracked separately - this is the
    #: failure mode that makes a bot untrustworthy rather than merely unhelpful.
    false_answer: bool = False
    deterministic: bool = True
    paraphrase_consistent: bool = True
    cost_usd: float = 0.0

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
    def determinism(self) -> float:
        return (
            sum(r.deterministic for r in self.results) / len(self.results) if self.results else 0.0
        )

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
        return self.false_answers == 0 and accuracy_ok and self.determinism == 1.0

    def to_dict(self) -> dict[str, object]:
        return {
            "cases": len(self.results),
            "accuracy": round(self.accuracy, 4),
            "false_answers": self.false_answers,
            "false_answer_rate": round(self.false_answer_rate, 4),
            "determinism": round(self.determinism, 4),
            "paraphrase_consistency": round(self.paraphrase_consistency, 4),
            "cost_per_question_usd": round(self.cost_per_question_usd, 6),
            "total_cost_usd": round(self.total_cost_usd, 6),
            "free_share": round(self.free_share, 4),
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
        if answer.tier is Tier.REFUSED:
            return True, (), False
        return (
            False,
            (f"answered a question the corpus does not cover: {answer.text[:120]!r}",),
            True,
        )

    if answer.tier is Tier.REFUSED:
        return False, ("refused a question the corpus does cover",), False

    cited = {c.document_id for c in answer.citations}
    missing = [doc for doc in case.must_cite if doc not in cited]
    if missing:
        failures.append(f"did not cite {', '.join(missing)} (cited: {', '.join(sorted(cited))})")

    for fact in case.must_say:
        if _normalise(fact) not in text:
            failures.append(f"missing required fact {fact!r}")

    for wrong in case.must_not_say:
        if _normalise(wrong) in text:
            failures.append(f"contains incorrect content {wrong!r}")

    if not answer.grounded:
        failures.append("answer was not grounded")

    return not failures, tuple(failures), False


def _facts_present(case: Case, answer: Answer) -> frozenset[str]:
    text = _normalise(answer.text)
    return frozenset(f for f in case.must_say if _normalise(f) in text)


async def run_case(cascade: Cascade, case: Case, *, check_determinism: bool = True) -> CaseResult:
    """Ask one case, score it, and check determinism and paraphrase consistency."""
    principals = frozenset(case.principals) if case.principals is not None else None

    answer = await cascade.answer(case.question, principals=principals, channel="eval")
    passed, failures, false_answer = _score(case, answer)
    cost = answer.cost_usd

    deterministic = True
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
    )


async def run_eval(
    cascade: Cascade, cases: list[Case], *, check_determinism: bool = True
) -> EvalReport:
    report = EvalReport()
    for case in cases:
        report.results.append(await run_case(cascade, case, check_determinism=check_determinism))
    return report
