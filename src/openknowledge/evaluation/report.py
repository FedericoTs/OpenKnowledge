"""Rendering an eval report, and comparing it against a baseline.

The comparison is what makes this useful in CI. An absolute score answers "is it
good"; a diff against the last known-good run answers "did I just break it",
which is the question a pull request actually poses.
"""

from __future__ import annotations

from dataclasses import dataclass

from .runner import EvalReport

#: Tolerance on cost regressions. Cost moves for legitimate reasons (a cold cache
#: on a fresh run), so only flag a real jump.
_COST_REGRESSION_FACTOR = 1.25


@dataclass(frozen=True, slots=True)
class Comparison:
    regressions: tuple[str, ...]
    improvements: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.regressions


def compare(current: EvalReport, baseline: dict[str, object]) -> Comparison:
    """Diff a run against a stored baseline."""
    regressions: list[str] = []
    improvements: list[str] = []

    def num(key: str, default: float = 0.0) -> float:
        value = baseline.get(key, default)
        return float(value) if isinstance(value, (int, float)) else default

    base_accuracy = num("accuracy")
    if current.accuracy < base_accuracy:
        regressions.append(f"accuracy {base_accuracy:.1%} -> {current.accuracy:.1%}")
    elif current.accuracy > base_accuracy:
        improvements.append(f"accuracy {base_accuracy:.1%} -> {current.accuracy:.1%}")

    base_false = int(num("false_answers"))
    if current.false_answers > base_false:
        regressions.append(
            f"false answers {base_false} -> {current.false_answers} "
            "(answered questions the corpus does not cover)"
        )
    elif current.false_answers < base_false:
        improvements.append(f"false answers {base_false} -> {current.false_answers}")

    base_determinism = num("determinism", 1.0)
    if current.determinism < base_determinism:
        regressions.append(f"determinism {base_determinism:.1%} -> {current.determinism:.1%}")

    base_cost = num("cost_per_question_usd")
    if base_cost > 0 and current.cost_per_question_usd > base_cost * _COST_REGRESSION_FACTOR:
        regressions.append(
            f"cost per question ${base_cost:.5f} -> ${current.cost_per_question_usd:.5f}"
        )
    elif base_cost > 0 and current.cost_per_question_usd < base_cost:
        improvements.append(
            f"cost per question ${base_cost:.5f} -> ${current.cost_per_question_usd:.5f}"
        )

    return Comparison(tuple(regressions), tuple(improvements))


def format_report(report: EvalReport, *, verbose: bool = False) -> str:
    """Render a report for a terminal."""
    lines: list[str] = []
    n = len(report.results)
    answerable = len(report.answerable)
    refusals = len(report.refusal_cases)

    lines.append(f"{n} cases  ({answerable} answerable, {refusals} must-refuse)")
    lines.append("")

    lines.append("Correctness")
    if answerable:
        lines.append(f"  accuracy                 {report.accuracy:>8.1%}  ({answerable} cases)")
    else:
        lines.append(f"  accuracy                 {'n/a':>8}  (no answerable cases selected)")
    lines.append(
        f"  false answers            {report.false_answers:>8}  "
        f"({report.false_answer_rate:.1%} of must-refuse cases)"
    )
    lines.append(f"  determinism              {report.determinism:>8.1%}  (same question twice)")
    lines.append(
        f"  paraphrase consistency   {report.paraphrase_consistency:>8.1%}"
        "  (same facts, other words)"
    )

    lines.append("")
    lines.append("Cost")
    lines.append(f"  per question             ${report.cost_per_question_usd:>7.5f}")
    lines.append(f"  total for this run       ${report.total_cost_usd:>7.5f}")
    lines.append(f"  answered without a model {report.free_share:>8.1%}")
    if report.tier_counts:
        tiers = "  ".join(f"{tier}={count}" for tier, count in report.tier_counts.items())
        lines.append(f"  tiers                    {tiers}")

    failed = [r for r in report.results if not r.passed]
    if failed:
        lines.append("")
        lines.append(f"Failures ({len(failed)})")
        for r in failed:
            marker = "UNSAFE" if r.false_answer else "fail"
            lines.append(f"  [{marker}] {r.case.id}  ({r.tier.value})")
            for reason in r.failures:
                lines.append(f"      - {reason}")
            if verbose:
                lines.append(f"      answer: {r.answer.text[:200]!r}")

    lines.append("")
    if report.false_answers:
        lines.append(
            f"FAILED - {report.false_answers} question(s) the corpus does not cover were "
            "answered anyway."
        )
    elif failed:
        lines.append(f"FAILED - {len(failed)} case(s) did not meet expectations.")
    else:
        lines.append("PASSED - all cases met expectations.")
    return "\n".join(lines)
