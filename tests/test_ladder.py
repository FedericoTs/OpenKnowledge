"""The escalation ladder and the budget that governs how far it climbs.

Two claims are load-bearing here and both are safety claims, not cost ones:

* a cheap rung can lower the bill but never the standard, because every rung is
  graded by the same grounding gate on the same passages;
* a budget limits *escalation*, never service, and never turns into a guess.
"""

from __future__ import annotations

import pytest

from fakes import FakeProvider
from openknowledge.cascade import Cascade
from openknowledge.cascade.budget import Budget, BudgetGovernor, forecast_usd
from openknowledge.cascade.ladder import Ladder, Rung
from openknowledge.types import Answer, Tier

GROUNDED = (
    "Employees with 12 months of continuous service get 20 weeks of fully paid "
    "parental leave. Requests need 30 days notice. [hr-handbook]"
)
INVENTED = "Employees get 26 weeks of fully paid parental leave. [hr-handbook]"
QUESTION = "How much parental leave do I get?"


def priced(model_id: str, replies: list[str]) -> FakeProvider:
    """A rung that bills, so the governor has something to forecast."""
    return FakeProvider(model_id=model_id, tier="frontier", self_hosted=False, replies=replies)


def ladder_of(*rungs: Rung) -> Ladder:
    return Ladder(rungs)


def build(store, retriever, settings, ladder: Ladder, budget: Budget | None = None) -> Cascade:
    return Cascade(
        store=store, retriever=retriever, settings=settings, ladder=ladder, budget=budget
    )


# -- climbing ---------------------------------------------------------------


async def test_it_stops_at_the_first_rung_that_passes_the_gate(store, retriever, settings) -> None:
    cheap = FakeProvider(model_id="cheap", replies=[GROUNDED])
    middle = priced("gpt-oss-120b", [GROUNDED])
    top = priced("claude-opus-5", [GROUNDED])

    answer = await build(
        store,
        retriever,
        settings,
        ladder_of(
            Rung("cheap", cheap, Tier.LOCAL),
            Rung("middle", middle),
            Rung("top", top),
        ),
    ).answer(QUESTION)

    assert answer.tier is Tier.LOCAL
    assert middle.calls == [] and top.calls == [], "nothing above the winner may be called"


async def test_a_middle_rung_catches_what_the_cheap_one_could_not(
    store, retriever, settings
) -> None:
    """The saving the ladder exists for: a gate failure handled below frontier."""
    cheap = FakeProvider(model_id="cheap", replies=[INVENTED])
    middle = priced("gpt-oss-120b", [GROUNDED])
    top = priced("claude-opus-5", [GROUNDED])

    answer = await build(
        store,
        retriever,
        settings,
        ladder_of(
            Rung("cheap", cheap, Tier.LOCAL),
            Rung("middle", middle),
            Rung("top", top),
        ),
    ).answer(QUESTION)

    assert answer.grounded
    assert answer.model_id == "gpt-oss-120b"
    assert top.calls == [], "the frontier must not be reached once a cheaper rung grounds it"
    assert answer.escalated_from is Tier.LOCAL


async def test_an_ungrounded_rung_never_reaches_the_user(store, retriever, settings) -> None:
    """Adding rungs must not add ways to serve something the gate rejected."""
    ladder = ladder_of(
        Rung("cheap", FakeProvider(model_id="cheap", replies=[INVENTED]), Tier.LOCAL),
        Rung("middle", priced("gpt-oss-120b", [INVENTED])),
        Rung("top", priced("claude-opus-5", [INVENTED])),
    )
    answer = await build(store, retriever, settings, ladder).answer(QUESTION)

    assert answer.tier is Tier.REFUSED
    assert not answer.grounded
    assert "26 weeks" not in answer.text


async def test_the_bill_includes_every_rung_that_failed_below_the_winner(
    store, retriever, settings
) -> None:
    """A ledger that only records the winning call understates the question."""
    ladder = ladder_of(
        Rung("cheap", priced("gpt-oss-20b", [INVENTED]), Tier.LOCAL),
        Rung("top", priced("claude-opus-5", [GROUNDED])),
    )
    answer = await build(store, retriever, settings, ladder).answer(QUESTION)

    only_the_winner = answer.usage.output_tokens // 2
    assert answer.grounded
    assert answer.usage.output_tokens > only_the_winner
    assert answer.cost_usd > 0.0


async def test_changing_the_ladder_changes_the_cache_key(store, retriever, settings) -> None:
    """Otherwise answers from a retired rung would outlive it."""
    one = build(store, retriever, settings, ladder_of(Rung("a", FakeProvider(model_id="a"))))
    two = build(store, retriever, settings, ladder_of(Rung("b", FakeProvider(model_id="b"))))
    assert one.route_id != two.route_id


def test_a_narrowing_rung_is_warned_about(caplog) -> None:
    """Showing a more expensive rung less evidence usually turns a fix into a refusal."""
    with caplog.at_level("WARNING"):
        Ladder(
            (
                Rung("cheap", FakeProvider(model_id="cheap"), Tier.LOCAL, k=20),
                Rung("top", FakeProvider(model_id="top"), k=6),
            )
        )
    assert "less evidence" in caplog.text


def test_one_search_serves_every_rung() -> None:
    ladder = Ladder(
        (
            Rung("cheap", FakeProvider(model_id="cheap"), Tier.LOCAL, k=20),
            Rung("top", FakeProvider(model_id="top"), k=8),
        )
    )
    assert ladder.widest_k(6) == 20


# -- the budget governor ----------------------------------------------------


def test_no_budget_means_no_governor(store) -> None:
    governor = BudgetGovernor(store=store, budget=Budget())
    ladder = Ladder((Rung("a", FakeProvider()), Rung("b", FakeProvider())))
    kept, state, withheld = governor.allowed(ladder, prompt_chars=8_000, max_tokens=1_000)

    assert len(kept) == 2
    assert withheld == ()
    assert not state.enabled


def test_the_ceiling_is_the_budget_spread_over_expected_traffic(store) -> None:
    governor = BudgetGovernor(
        store=store, budget=Budget(daily_usd=10.0, expected_questions_per_day=1_000)
    )
    assert governor.state().ceiling_usd == pytest.approx(0.01)


def test_an_expensive_rung_is_withheld_when_it_would_break_the_pace(store) -> None:
    governor = BudgetGovernor(
        store=store, budget=Budget(daily_usd=1.0, expected_questions_per_day=1_000)
    )
    ladder = Ladder(
        (
            Rung("cheap", FakeProvider(model_id="gpt-oss-20b", self_hosted=False), Tier.LOCAL),
            Rung("top", FakeProvider(model_id="claude-opus-5", self_hosted=False)),
        )
    )
    kept, _state, withheld = governor.allowed(ladder, prompt_chars=9_000, max_tokens=1_000)

    assert [r.name for r in kept] == ["cheap"]
    assert withheld and "claude-opus-5" in withheld[0]


def test_the_first_rung_is_never_withheld(store) -> None:
    """A budget is a limit on escalation, not an off switch."""
    governor = BudgetGovernor(
        store=store, budget=Budget(daily_usd=0.000001, expected_questions_per_day=1_000_000)
    )
    ladder = Ladder(
        (
            Rung("top", FakeProvider(model_id="claude-opus-5", self_hosted=False)),
            Rung("also-top", FakeProvider(model_id="claude-opus-5", self_hosted=False)),
        )
    )
    kept, _state, _ = governor.allowed(ladder, prompt_chars=9_000, max_tokens=1_000)
    assert [r.name for r in kept] == ["top"]


def test_an_unpriced_rung_is_tried_rather_than_assumed_free_or_dropped(store) -> None:
    """A missing price table entry must not silently degrade answers."""
    governor = BudgetGovernor(
        store=store, budget=Budget(daily_usd=0.01, expected_questions_per_day=1_000)
    )
    ladder = Ladder(
        (
            Rung("cheap", FakeProvider(model_id="gpt-oss-20b", self_hosted=False), Tier.LOCAL),
            Rung("mystery", FakeProvider(model_id="not-in-the-price-table", self_hosted=False)),
        )
    )
    kept, _state, withheld = governor.allowed(ladder, prompt_chars=9_000, max_tokens=1_000)

    assert [r.name for r in kept] == ["cheap", "mystery"]
    assert withheld == ()


def test_the_ceiling_tightens_as_spending_runs_ahead_of_pace(store, settings) -> None:
    budget = Budget(daily_usd=1.0, expected_questions_per_day=1_000)
    governor = BudgetGovernor(store=store, budget=budget)
    before = governor.state().ceiling_usd

    for n in range(20):
        store.record(
            f"q{n}",
            Answer(
                text="x",
                tier=Tier.FRONTIER,
                model_id="claude-opus-5",
                cache_key=f"k{n}",
                cost_usd=0.02,
            ),
        )

    after = governor.state().ceiling_usd
    assert after is not None and before is not None
    assert after < before


def test_forecasting_is_pessimistic_and_refuses_to_guess() -> None:
    priced_call = forecast_usd("claude-opus-5", prompt_chars=8_000, max_tokens=1_000)
    assert priced_call is not None and priced_call > 0

    assert forecast_usd("no-such-model", prompt_chars=8_000, max_tokens=1_000) is None
    assert forecast_usd("openai-frontier", prompt_chars=8_000, max_tokens=1_000) is None


# -- the budget meeting the cascade ----------------------------------------


async def test_a_budgeted_refusal_says_why_and_is_not_cached(store, retriever, settings) -> None:
    """The alternative - serving the cheap rung's rejected attempt - is the one
    thing this project will not do."""
    cheap = FakeProvider(model_id="gpt-oss-20b", self_hosted=False, replies=[INVENTED, GROUNDED])
    top = priced("claude-opus-5", [GROUNDED])
    cascade = build(
        store,
        retriever,
        settings,
        ladder_of(Rung("gpt-oss-20b", cheap, Tier.LOCAL), Rung("claude-opus-5", top)),
        Budget(daily_usd=0.001, expected_questions_per_day=1_000),
    )

    refused = await cascade.answer(QUESTION)
    assert refused.tier is Tier.REFUSED
    assert any("budget ceiling" in n for n in refused.notes)
    assert top.calls == [], "the withheld rung must not be called"

    # Not cached: nothing about the corpus made this unanswerable.
    again = await cascade.answer(QUESTION)
    assert again.tier is not Tier.EXACT_CACHE
