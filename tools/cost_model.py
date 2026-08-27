#!/usr/bin/env python3
"""Reproduce the cost numbers in the README and docs/COST-MODEL.md.

Run it rather than trusting the tables:

    uv run python tools/cost_model.py [questions_per_day]

Every figure comes from ``openknowledge.costs`` and the rates in
``pricing.yaml``, so if a vendor changes its prices, updating that one file
updates the whole argument.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

from openknowledge.costs import Usage, cost_usd, get_price, self_hosted_cost_usd

WORKING_DAYS = 250
BUSY_HOURS_PER_DAY = 8

#: A single mid-range GPU instance, the sort of box a local 8B model needs.
#: Change this to your own hardware rate - amortised capex plus power for
#: on-prem, or the hourly instance rate in the cloud.
GPU_HOURLY_USD = 1.20


@dataclass(frozen=True)
class Step:
    """One optimisation, priced per paid call and per question asked."""

    name: str
    usage: Usage
    model: str
    #: Share of all questions that reach a paid model at all.
    paid_share: float = 1.0

    def per_paid_call(self) -> float:
        return cost_usd(self.usage, get_price(self.model))

    def per_question(self) -> float:
        return self.per_paid_call() * self.paid_share


# Each row adds one lever to the row above it.
STEPS = [
    Step(
        "Naive RAG",
        Usage(input_tokens=15_000, output_tokens=1_000),
        "claude-opus-5",
    ),
    Step(
        "+ prompt caching",
        Usage(input_tokens=13_000, cache_read_tokens=2_000, output_tokens=1_000),
        "claude-opus-5",
    ),
    Step(
        "+ tighter retrieval",
        Usage(input_tokens=2_500, cache_read_tokens=2_000, output_tokens=400),
        "claude-opus-5",
    ),
    Step(
        "+ smaller model",
        Usage(input_tokens=2_500, cache_read_tokens=2_000, output_tokens=400),
        "claude-sonnet-5",
    ),
    Step(
        "+ pins and cache (45% free)",
        Usage(input_tokens=2_500, cache_read_tokens=2_000, output_tokens=400),
        "claude-sonnet-5",
        paid_share=0.55,
    ),
]

#: Two end states worth comparing head to head.
API_ONLY = STEPS[-1]  # 45% free, 55% to a mid-tier API model. No hardware.
CASCADE_ESCALATIONS = Step(  # 45% free, 45% local, 10% escalated to frontier
    "full cascade",
    Usage(input_tokens=2_500, cache_read_tokens=2_000, output_tokens=400),
    "claude-opus-5",
    paid_share=0.10,
)


def hardware_per_question(questions_per_day: float) -> float:
    if questions_per_day <= 0:
        return 0.0
    return self_hosted_cost_usd(
        hourly_rate_usd=GPU_HOURLY_USD,
        questions_per_hour=questions_per_day / BUSY_HOURS_PER_DAY,
    )


def cascade_per_question(questions_per_day: float) -> float:
    """Total cost per question with a local tier, hardware included."""
    return CASCADE_ESCALATIONS.per_question() + hardware_per_question(questions_per_day)


def crossover_questions_per_day() -> float | None:
    """Volume at which running a local model beats paying per token.

    Below this, the GPU sits idle enough that its fixed cost per question
    exceeds what the API tier would have charged. Above it, the fixed cost is
    spread thin and the local tier wins.
    """
    api = API_ONLY.per_question()
    escalations = CASCADE_ESCALATIONS.per_question()
    headroom = api - escalations
    if headroom <= 0:
        return None
    daily_hardware = GPU_HOURLY_USD * BUSY_HOURS_PER_DAY
    return daily_hardware / headroom


def main(argv: list[str]) -> int:
    questions_per_day = float(argv[1]) if len(argv) > 1 else 2_000

    def annual(per_question: float) -> float:
        return per_question * questions_per_day * WORKING_DAYS

    print(f"Assuming {questions_per_day:,.0f} questions/day over {WORKING_DAYS} working days.\n")

    print("API cost, one lever at a time")
    print(f"  {'':<30}{'per paid call':>15}{'per question':>15}{'per year':>14}")
    for step in STEPS:
        print(
            f"  {step.name:<30}"
            f"{step.per_paid_call():>15.5f}"
            f"{step.per_question():>15.5f}"
            f"{annual(step.per_question()):>14,.0f}"
        )

    baseline = STEPS[0].per_question()
    api_only = API_ONLY.per_question()
    print(
        f"\n  Those levers alone, with no local model and no new hardware: "
        f"{baseline / api_only:.0f}x cheaper,\n"
        f"  {annual(baseline):,.0f} -> {annual(api_only):,.0f} per year."
    )

    # -- the local tier ------------------------------------------------------
    print("\n\nAdding a local model: the fixed cost has to be carried too")
    print(
        f"  {'questions/day':>16}{'hardware/question':>20}{'cascade total':>16}{'vs API-only':>17}"
    )
    for per_day in (250, 1_000, 2_000, 5_000, 10_000, 25_000):
        hardware = hardware_per_question(per_day)
        total = cascade_per_question(per_day)
        verdict = "cheaper" if total < api_only else "MORE expensive"
        print(f"  {per_day:>16,}{hardware:>20.5f}{total:>16.5f}{verdict:>17}")
    print(f"\n  (A ${GPU_HOURLY_USD:.2f}/hour GPU running {BUSY_HOURS_PER_DAY}h/day.)")

    crossover = crossover_questions_per_day()
    if crossover is not None:
        print(
            f"\n  Break-even is around {crossover:,.0f} questions/day. Below that, a local\n"
            f"  model costs more per question than the API tier it replaces - it is a\n"
            f"  privacy decision, not a savings one. Above it, the fixed cost is spread\n"
            f"  thin enough to win, and keeps winning as volume grows."
        )

    # -- bottom line ---------------------------------------------------------
    total = cascade_per_question(questions_per_day)
    print("\n\nAt your volume")
    rows = (
        ("naive RAG today", baseline),
        ("API-only, tuned", api_only),
        ("full cascade + hardware", total),
    )
    for label, per_q in rows:
        print(f"  {label:<26}{per_q:>10.5f} / question   {annual(per_q):>12,.0f} / year")
    best = min(api_only, total)
    print(
        f"\n  Best available: {baseline / best:.0f}x cheaper than today, "
        f"saving {annual(baseline) - annual(best):,.0f} per year."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
