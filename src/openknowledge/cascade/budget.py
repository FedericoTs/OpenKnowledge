"""Spending on pace: how far up the ladder this question may climb.

A ladder without a governor spends whatever the traffic asks for. That is fine
until a bad week of unanswerable questions sends every one of them to the top
rung, and the first anybody hears about it is the invoice.

The governor turns a declared budget into a **ceiling on what one question may
cost**, recomputed from the ledger on every question:

    ceiling = budget remaining today / questions still expected today

Spend ahead of pace and the ceiling drops, so the expensive rungs stop being
tried. Spend behind it and the ceiling rises again. Nothing is scheduled, nothing
is reset by hand, and the arithmetic is one division an operator can check.

Three properties this deliberately has:

**The first rung is always allowed.** A budget is a limit on escalation, not an
off switch. A deployment that stops answering entirely because it is 3% over
pace has turned a cost control into an outage.

**Refusal, never a guess.** When the ceiling blocks the rungs that could have
grounded an answer, the question is refused and says why. The alternative -
serving the cheap rung's ungrounded attempt because the good one was too
expensive - is the one thing this project will not do.

**Refusals are not cached.** A question refused on budget gets a fresh attempt
once the ceiling recovers, because nothing about the corpus made it unanswerable.

Determinism is unaffected for answers that were served: those are cached under a
key that does not include spend, so an identical question keeps returning the
identical answer. What the budget changes is whether a *new* question may reach a
paid rung today. See docs/DETERMINISM.md.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ..cache import AnswerStore
from ..costs import PricingError, Usage, cost_usd, get_price
from .ladder import Ladder, Rung

log = logging.getLogger(__name__)

#: Characters per token, for forecasting a call *before* making it. Only ever
#: used to decide whether a rung is affordable - never recorded as a cost. The
#: ledger takes token counts from the provider's own usage response, because a
#: guess in the ledger is how a cost report quietly becomes fiction.
_CHARS_PER_TOKEN = 4


@dataclass(frozen=True, slots=True)
class Budget:
    """What a deployment has declared it is willing to spend."""

    #: Cap for a rolling 24 hours. None means no limit and no governor.
    daily_usd: float | None = None
    #: Used to spread the cap across the day. It does not have to be right - it
    #: only sets the pace, and the ceiling self-corrects as real volume arrives.
    expected_questions_per_day: int = 2_000

    @property
    def enabled(self) -> bool:
        return self.daily_usd is not None and self.daily_usd > 0


@dataclass(frozen=True, slots=True)
class BudgetState:
    """What the governor decided, in terms an operator can check."""

    enabled: bool
    spent_usd: float = 0.0
    limit_usd: float | None = None
    answered: int = 0
    ceiling_usd: float | None = None

    @property
    def exhausted(self) -> bool:
        return self.limit_usd is not None and self.spent_usd >= self.limit_usd

    def describe(self) -> str:
        if not self.enabled or self.limit_usd is None:
            return "no budget configured"
        pace = f"${self.ceiling_usd:.5f}/question" if self.ceiling_usd is not None else "-"
        return (
            f"${self.spent_usd:.4f} of ${self.limit_usd:.2f} spent over "
            f"{self.answered} question(s); ceiling {pace}"
        )


class BudgetGovernor:
    """Decides which rungs this question may use, from the ledger."""

    #: A rolling 24 hours rather than a calendar day: no timezone to agree on,
    #: and no midnight cliff where a throttled deployment suddenly opens up.
    WINDOW_SECONDS = 24 * 60 * 60

    def __init__(self, *, store: AnswerStore, budget: Budget) -> None:
        self.store = store
        self.budget = budget

    def state(self) -> BudgetState:
        if not self.budget.enabled:
            return BudgetState(enabled=False)

        assert self.budget.daily_usd is not None
        spent, answered = self.store.spend_since(self.store.now() - self.WINDOW_SECONDS)
        remaining = max(self.budget.daily_usd - spent, 0.0)
        expected_left = max(self.budget.expected_questions_per_day - answered, 1)

        return BudgetState(
            enabled=True,
            spent_usd=spent,
            limit_usd=self.budget.daily_usd,
            answered=answered,
            ceiling_usd=remaining / expected_left,
        )

    def allowed(
        self, ladder: Ladder, *, prompt_chars: int, max_tokens: int
    ) -> tuple[tuple[Rung, ...], BudgetState, tuple[str, ...]]:
        """The rungs this question may climb, plus why any were withheld."""
        state = self.state()
        if not state.enabled or state.ceiling_usd is None:
            return tuple(ladder), state, ()

        kept: list[Rung] = []
        withheld: list[str] = []
        for index, rung in enumerate(ladder):
            forecast = forecast_usd(
                rung.model_id, prompt_chars=prompt_chars, max_tokens=rung.max_tokens or max_tokens
            )
            # Rung 0 is never withheld: a budget limits escalation, not service.
            # An unpriced rung is not withheld either - refusing to try a model
            # because we could not forecast it would let a missing price table
            # entry silently degrade answers.
            if index == 0 or forecast is None or forecast <= state.ceiling_usd:
                kept.append(rung)
                continue
            # Named by model, not just by rung: the rung name is whatever the
            # operator called it, and the actionable fact is which model this
            # deployment stopped being able to afford.
            withheld.append(
                f"{rung.name} ({rung.model_id}) not tried: forecast ${forecast:.5f} "
                f"exceeds the ${state.ceiling_usd:.5f} budget ceiling ({state.describe()})"
            )

        if withheld:
            log.info("budget ceiling withheld %d rung(s): %s", len(withheld), state.describe())
        return tuple(kept), state, tuple(withheld)


def forecast_usd(model_id: str, *, prompt_chars: int, max_tokens: int) -> float | None:
    """Worst-case cost of one call, before making it.

    Deliberately pessimistic: it assumes the model writes right up to its output
    cap. Under-forecasting would let a rung through that the budget cannot
    actually afford, and the whole point is to decide *before* spending.

    Returns None when the model has no verified price, which the caller must
    treat as "cannot judge" rather than as "free".
    """
    try:
        price = get_price(model_id)
    except (KeyError, PricingError):
        return None
    try:
        return cost_usd(
            Usage(
                input_tokens=max(prompt_chars // _CHARS_PER_TOKEN, 0),
                output_tokens=max_tokens,
            ),
            price,
        )
    except PricingError:
        return None
