#!/usr/bin/env python3
"""Reproduce the cost numbers in the README and docs/COST-MODEL.md.

Run it rather than trusting the tables:

    uv run python tools/cost_model.py [questions_per_day]

Every rate comes from ``pricing.yaml``, so if a vendor changes its prices,
updating that one file updates the whole argument.

**Token counts come from a measurement, not from this file.** They are read from
``evals/measured/real-contracts.json``, produced by ``tools/measure_prompts.py``
against a real corpus. This matters, because the assumptions this tool used to
carry were wrong in three compounding ways:

* it assumed a **2,000-token cacheable system prompt**; the real one measures
  476 and is *under the 512-token floor*, so it caches nothing at all;
* it assumed a **4,500-token prompt at six chunks**; the real one is 2,313;
* it quietly cut the answer from 1,000 tokens to 400 in the same row as the
  retrieval change, giving retrieval discipline credit for a 60% output
  reduction that retrieval does not cause.

Corrected, the headline drops from 19x to 11x. Every row below says whether its
numbers are measured or assumed, and the run says which source it used.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

from openknowledge.costs import Usage, cost_usd, get_price, self_hosted_cost_usd
from openknowledge.providers.anthropic_provider import CACHE_MIN_TOKENS

WORKING_DAYS = 250
BUSY_HOURS_PER_DAY = 8

MEASUREMENTS = Path(__file__).resolve().parent.parent / "evals" / "measured" / "real-contracts.json"

#: Used only when no measurement file is present. Deliberately the *measured*
#: values rather than the old guesses, so a missing file degrades to the same
#: answer rather than silently to a more flattering one.
FALLBACK = {
    "naive_input_tokens": 13_097,
    "tight_input_tokens": 2_313,
    "system_prompt_tokens": 476,
    "source": "fallback constants (no measurement file found)",
}


def measurements() -> dict[str, object]:
    """Token counts from a real corpus, or the documented fallback."""
    if not MEASUREMENTS.exists():
        return FALLBACK
    data = json.loads(MEASUREMENTS.read_text(encoding="utf-8"))
    rows = {int(row["chunks"]): int(row["input_tokens"]) for row in data["rows"]}
    return {
        # The "generous retrieval" build: real retrieval, but keep everything
        # that scored. This is what $0.10 per question actually looks like.
        "naive_input_tokens": rows[40],
        "tight_input_tokens": rows[6],
        "system_prompt_tokens": int(data["system_prompt_tokens"]),
        "source": (
            f"{MEASUREMENTS.parent.name}/{MEASUREMENTS.name}: "
            f"{data['documents']} documents, {data['chunks']} chunks, "
            f"{data['questions']} questions"
        ),
    }


#: Held constant across every row. Nothing in this architecture shortens an
#: answer, so letting it vary between rows would credit a lever for an effect it
#: does not have. That shorter answers are themselves a real lever is true, and
#: unmeasured, and therefore not claimed here.
ANSWER_TOKENS = 1_000

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


# Each row adds one lever to the row above it. Token counts are measured; the
# free share is the one assumption, and `openknowledge costs` replaces it with
# your own figure from the ledger.
_M = measurements()
_NAIVE = int(_M["naive_input_tokens"])  # type: ignore[call-overload]
_TIGHT = int(_M["tight_input_tokens"])  # type: ignore[call-overload]
_SYSTEM = int(_M["system_prompt_tokens"])  # type: ignore[call-overload]

#: The API declines to cache a prefix under its floor, so a system prompt below
#: it earns nothing however the cache_control marker is placed. Priced at what
#: it actually returns rather than at what it would return if the prompt were
#: longer.
_CACHES = _SYSTEM >= CACHE_MIN_TOKENS
_CACHED = _SYSTEM if _CACHES else 0

STEPS = [
    Step(
        "Generous retrieval",
        Usage(input_tokens=_NAIVE, output_tokens=ANSWER_TOKENS),
        "claude-opus-5",
    ),
    Step(
        "+ prompt caching"
        if _CACHES
        else f"+ prompt caching (inert, {_SYSTEM} < {CACHE_MIN_TOKENS})",
        Usage(
            input_tokens=_NAIVE - _CACHED,
            cache_read_tokens=_CACHED,
            output_tokens=ANSWER_TOKENS,
        ),
        "claude-opus-5",
    ),
    Step(
        "+ tighter retrieval (6 chunks)",
        Usage(
            input_tokens=_TIGHT - _CACHED,
            cache_read_tokens=_CACHED,
            output_tokens=ANSWER_TOKENS,
        ),
        "claude-opus-5",
    ),
    Step(
        "+ smaller model",
        Usage(
            input_tokens=_TIGHT - _CACHED,
            cache_read_tokens=_CACHED,
            output_tokens=ANSWER_TOKENS,
        ),
        "claude-sonnet-5",
    ),
    Step(
        "+ pins and cache (45% free)",
        Usage(
            input_tokens=_TIGHT - _CACHED,
            cache_read_tokens=_CACHED,
            output_tokens=ANSWER_TOKENS,
        ),
        "claude-sonnet-5",
        paid_share=0.55,
    ),
]

#: Two end states worth comparing head to head.
API_ONLY = STEPS[-1]  # 45% free, 55% to a mid-tier API model. No hardware.
CASCADE_ESCALATIONS = Step(  # 45% free, 45% local, 10% escalated to frontier
    "full cascade",
    Usage(input_tokens=_TIGHT - _CACHED, cache_read_tokens=_CACHED, output_tokens=ANSWER_TOKENS),
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


# --- moving work from query time to ingest time ---------------------------
#
# The cascade makes each question cheaper. Drafting answers at upload makes some
# questions cost nothing at all, forever, for a price paid once.

CORPUS_DOCUMENTS = 500
#: Reading one document and drafting ~8 Q&A pairs from it. The system prompt is
#: identical every time, so it is a cache read after the first document.
DRAFT_USAGE = Usage(input_tokens=2_000, cache_read_tokens=600, output_tokens=800)
DRAFTING_MODEL = "claude-sonnet-5"


def drafting_cost(documents: int = CORPUS_DOCUMENTS) -> float:
    """One-off cost of drafting an FAQ across the corpus."""
    return cost_usd(DRAFT_USAGE, get_price(DRAFTING_MODEL)) * documents


def naive_contradiction_cost(documents: int = CORPUS_DOCUMENTS) -> float:
    """Comparing one uploaded document against every other document."""
    return cost_usd(Usage(input_tokens=4_000, output_tokens=200), get_price(DRAFTING_MODEL)) * (
        documents
    )


def anchored_contradiction_cost(affected_answers: int = 8) -> float:
    """Re-asking only the approved answers that cite the changed document."""
    return affected_answers * API_ONLY.per_paid_call()


def main(argv: list[str]) -> int:
    questions_per_day = float(argv[1]) if len(argv) > 1 else 2_000

    def annual(per_question: float) -> float:
        return per_question * questions_per_day * WORKING_DAYS

    print(f"Assuming {questions_per_day:,.0f} questions/day over {WORKING_DAYS} working days.")
    print(f"Token counts from {_M['source']}.")
    print(f"Answer length held at {ANSWER_TOKENS:,} tokens on every row - the one assumption.\n")

    print("API cost, one lever at a time")
    name_width = max(len(step.name) for step in STEPS) + 2
    print(f"  {'':<{name_width}}{'per paid call':>15}{'per question':>15}{'per year':>14}")
    for step in STEPS:
        print(
            f"  {step.name:<{name_width}}"
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
    if not _CACHES:
        print(
            f"\n  Note: prompt caching contributes nothing here. The system prompt is "
            f"{_SYSTEM} tokens,\n"
            f"  under the {CACHE_MIN_TOKENS}-token minimum cacheable prefix, so the "
            f"cache_control marker\n"
            f"  on it is inert. Retrieval discipline and the free tier do all of the work."
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

    # -- moving the work to ingest time --------------------------------------
    #
    # Drafting does not "free N questions" on top of the existing saving - the
    # cache was already answering some of them. What it does is raise the share
    # of traffic that never reaches a model, so the saving is the *increase* in
    # that share priced at what one paid answer costs.
    one_off = drafting_cost()
    per_paid = API_ONLY.per_paid_call()
    base_free = 1 - API_ONLY.paid_share

    print("\n\nDrafting answers at upload instead of at question time")
    print(
        f"  An FAQ drafted across {CORPUS_DOCUMENTS} documents costs ${one_off:.2f}, once.\n"
        f"  It raises the share of questions answered without a model, which is the\n"
        f"  single number the whole cost model turns on."
    )
    print(
        f"\n  {'free share':>12}{'$/question':>13}{'$/year':>11}{'saved/year':>13}{'payback':>11}"
    )
    for free_share in (base_free, 0.60, 0.75, 0.85):
        per_question = (1 - free_share) * per_paid
        saved = annual((1 - base_free) * per_paid) - annual(per_question)
        label = f"{free_share:.0%}" + (" (today)" if free_share == base_free else "")
        payback = f"{one_off / (saved / WORKING_DAYS):.1f} d" if saved > 0 else "-"
        print(
            f"  {label:>12}{per_question:>13.5f}{annual(per_question):>11,.0f}"
            f"{saved:>13,.0f}{payback:>11}"
        )

    print(
        "\n  The bigger win is cold start. A fresh deployment has an empty cache and\n"
        "  climbs to its free share over months of traffic. Drafting at upload means\n"
        "  day one starts near the top of that table instead of at the bottom."
    )

    naive = naive_contradiction_cost()
    anchored = anchored_contradiction_cost()
    print("\n\nDetecting contradictions when a document is uploaded")
    print(f"  compare it against all {CORPUS_DOCUMENTS} documents   ${naive:>8.2f} per upload")
    print(f"  re-ask only the answers that cite it   ${anchored:>8.4f} per upload")
    print(f"  -> {naive / anchored:,.0f}x cheaper, and it names the claim that moved")
    print("\n  Numeric conflicts between documents are found without a model at all,")
    print("  so that pass costs nothing and runs on every re-index.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
