"""End-to-end routing: the behaviour the cost and safety claims rest on."""

from __future__ import annotations

import pytest

from fakes import FakeProvider
from openknowledge.cache import AnswerStore
from openknowledge.cascade import Cascade
from openknowledge.config import Settings
from openknowledge.retrieval import BM25Retriever, Document
from openknowledge.types import Tier


def replace_settings(settings: Settings, **changes: object) -> Settings:
    """Settings is a pydantic model, so copying goes through model_copy."""
    return settings.model_copy(update=changes)


GROUNDED = (
    "Employees with 12 months of continuous service get 20 weeks of fully paid "
    "parental leave. Requests need 30 days notice. [hr-handbook]"
)
INVENTED = "Employees get 26 weeks of fully paid parental leave. [hr-handbook]"
QUESTION = "How much parental leave do I get?"


def build(store, retriever, settings, *, local=None, frontier=None) -> Cascade:
    return Cascade(
        store=store, retriever=retriever, settings=settings, local=local, frontier=frontier
    )


async def test_pinned_answer_wins_and_calls_nothing(store, retriever, settings) -> None:
    store.pin("how much parental leave do i get", "20 weeks, fully paid.", author="hr")
    local = FakeProvider(replies=[GROUNDED])
    answer = await build(store, retriever, settings, local=local).answer(QUESTION)

    assert answer.tier is Tier.PINNED
    assert answer.text == "20 weeks, fully paid."
    assert answer.cost_usd == 0.0
    assert local.calls == [], "a pinned answer must not reach a model"


async def test_local_model_answers_and_is_cached(store, retriever, settings) -> None:
    local = FakeProvider(replies=[GROUNDED])
    cascade = build(store, retriever, settings, local=local)

    first = await cascade.answer(QUESTION)
    assert first.tier is Tier.LOCAL
    assert first.cost_usd == 0.0, "a self-hosted model has no per-token invoice"

    second = await cascade.answer("hi, how much PARENTAL LEAVE do i get??")
    assert second.tier is Tier.EXACT_CACHE
    assert second.text == first.text
    assert len(local.calls) == 1, "the second phrasing must not reach the model at all"


async def test_ungrounded_local_answer_escalates(store, retriever, settings) -> None:
    """The invented figure is exactly what the paid tier is for."""
    settings = replace_settings(settings, escalation_enabled=True)
    local = FakeProvider(replies=[INVENTED])
    frontier = FakeProvider(model_id="claude-opus-5", tier="frontier", replies=[GROUNDED])

    answer = await build(store, retriever, settings, local=local, frontier=frontier).answer(
        QUESTION
    )

    assert answer.tier is Tier.FRONTIER
    assert answer.escalated_from is Tier.LOCAL
    assert answer.cost_usd > 0
    assert local.calls and frontier.calls


async def test_provider_outage_escalates_rather_than_failing(store, retriever, settings) -> None:
    settings = replace_settings(settings, escalation_enabled=True)
    local = FakeProvider(fail=True)
    frontier = FakeProvider(model_id="claude-opus-5", tier="frontier", replies=[GROUNDED])

    answer = await build(store, retriever, settings, local=local, frontier=frontier).answer(
        QUESTION
    )
    assert answer.tier is Tier.FRONTIER


async def test_refuses_rather_than_guessing_when_nothing_is_grounded(
    store, retriever, settings
) -> None:
    local = FakeProvider(replies=[INVENTED])
    answer = await build(store, retriever, settings, local=local).answer(QUESTION)

    assert answer.tier is Tier.REFUSED
    assert not answer.grounded
    assert any("escalation is disabled" in n for n in answer.notes)


async def test_refusals_are_not_cached(store, retriever, settings) -> None:
    local = FakeProvider(replies=[INVENTED, GROUNDED])
    cascade = build(store, retriever, settings, local=local)

    assert (await cascade.answer(QUESTION)).tier is Tier.REFUSED
    # A later, better answer must still be reachable - the refusal must not stick.
    assert (await cascade.answer(QUESTION)).tier is Tier.LOCAL


async def test_no_matching_documents_refuses_without_calling_a_model(store, settings) -> None:
    empty = BM25Retriever()
    empty.index([Document("d", "D", "unrelated content about printers")])
    local = FakeProvider(replies=[GROUNDED])

    answer = await build(store, empty, settings, local=local).answer("zzzz qqqq")
    assert answer.tier is Tier.REFUSED
    assert local.calls == []


async def test_editing_a_document_invalidates_the_cached_answer(
    store, retriever, documents, settings
) -> None:
    # The second reply reflects the edited notice period. If the cascade served
    # the cached answer instead, the user would still be told 30 days.
    after_edit = GROUNDED.replace("30 days", "45 days")
    local = FakeProvider(replies=[GROUNDED, after_edit])
    cascade = build(store, retriever, settings, local=local)

    await cascade.answer(QUESTION)
    assert len(local.calls) == 1

    updated = [
        Document(
            "hr-handbook",
            "HR Handbook",
            "Parental leave. Employees with at least 12 months of continuous service are "
            "entitled to 20 weeks of fully paid parental leave. Requests must be submitted "
            "at least 45 days in advance through the HR portal.",
        ),
        *documents[1:],
    ]
    retriever.index(updated)

    answer = await cascade.answer(QUESTION)
    assert answer.tier is Tier.LOCAL
    assert len(local.calls) == 2, "a corpus edit must force a fresh answer"
    assert "45 days" in answer.text and "30 days" not in answer.text


async def test_a_cached_answer_is_withheld_from_someone_who_cannot_see_its_sources(
    store, retriever, settings
) -> None:
    """The cache is shared across users; access is re-checked on every read."""
    board_answer = "Bands run from EUR 180000 to EUR 240000 pending approval. [board-comp]"
    local = FakeProvider(replies=[board_answer, board_answer])
    cascade = build(store, retriever, settings, local=local)

    q = "What are the executive salary bands?"
    privileged = await cascade.answer(q, principals=frozenset({"board"}))
    assert privileged.tier is Tier.LOCAL
    assert "board-comp" in {c.document_id for c in privileged.citations}

    staff = await cascade.answer(q, principals=frozenset({"staff"}))
    assert staff.tier is not Tier.EXACT_CACHE, "restricted content leaked out of the cache"
    assert "board-comp" not in {c.document_id for c in staff.citations}


async def test_a_pin_is_also_access_checked(store, retriever, settings) -> None:
    from openknowledge.types import Citation

    store.pin(
        "what are the executive salary bands",
        "EUR 180000 to EUR 240000.",
        citations=(Citation("board-comp", "Board Compensation", "bands", None),),
    )
    cascade = build(store, retriever, settings, local=FakeProvider(replies=["I don't know."]))

    assert (
        await cascade.answer(
            "What are the executive salary bands?", principals=frozenset({"board"})
        )
    ).tier is Tier.PINNED
    assert (
        await cascade.answer(
            "What are the executive salary bands?", principals=frozenset({"staff"})
        )
    ).tier is not Tier.PINNED


async def test_the_ledger_shows_the_blended_cost_falling(store, retriever, settings) -> None:
    settings = replace_settings(settings, local_enabled=False, escalation_enabled=True)
    frontier = FakeProvider(model_id="claude-opus-5", tier="frontier", replies=[GROUNDED])
    cascade = build(store, retriever, settings, frontier=frontier)

    await cascade.answer(QUESTION)
    paid_only = store.cost_report()["cost_per_question_usd"]

    for _ in range(19):
        await cascade.answer(QUESTION)

    report = store.cost_report()
    assert report["questions"] == 20
    assert len(frontier.calls) == 1, "19 of 20 questions were served free"
    assert report["cost_per_question_usd"] == pytest.approx(paid_only / 20)
    assert report["by_tier"]["exact"]["spend_usd"] == 0.0


async def test_admin_prompt_changes_invalidate_cached_answers(store, retriever, settings) -> None:
    local = FakeProvider(replies=[GROUNDED, GROUNDED])
    assert (
        await build(store, retriever, settings, local=local).answer(QUESTION)
    ).tier is Tier.LOCAL

    tuned = replace_settings(settings, system_prompt_suffix="Always answer in British English.")
    answer = await build(store, retriever, tuned, local=local).answer(QUESTION)
    assert answer.tier is Tier.LOCAL
    assert len(local.calls) == 2, "an edited system prompt must not serve old answers"


async def test_switching_models_invalidates_cached_answers(store, retriever, settings) -> None:
    a = FakeProvider(model_id="qwen3:8b", replies=[GROUNDED])
    assert (await build(store, retriever, settings, local=a).answer(QUESTION)).tier is Tier.LOCAL

    b = FakeProvider(model_id="llama3.1:8b", replies=[GROUNDED])
    assert (await build(store, retriever, settings, local=b).answer(QUESTION)).tier is Tier.LOCAL
    assert b.calls, "a different model must produce its own answer"


async def test_channel_is_recorded_for_per_surface_cost_reporting(
    store: AnswerStore, retriever, settings
) -> None:
    local = FakeProvider(replies=[GROUNDED])
    await build(store, retriever, settings, local=local).answer(QUESTION, channel="teams")
    assert [e.channel for e in store.recent_questions()] == ["teams"]


# -- cost accounting across tiers -----------------------------------------


async def test_a_refusal_after_a_paid_attempt_is_not_reported_as_free(
    store, retriever, settings
) -> None:
    """Rejected answers still burned tokens. The ledger has to say so."""
    settings = replace_settings(settings, escalation_enabled=True)
    local = FakeProvider(replies=[INVENTED])
    frontier = FakeProvider(model_id="claude-opus-5", tier="frontier", replies=[INVENTED])

    answer = await build(store, retriever, settings, local=local, frontier=frontier).answer(
        QUESTION
    )

    assert answer.tier is Tier.REFUSED
    assert answer.cost_usd == pytest.approx(0.02), "the escalation call was billed"
    assert store.cost_report()["spend_usd"] == pytest.approx(0.02)


async def test_an_escalated_answer_carries_the_cost_of_the_failed_attempt(
    store, retriever, settings
) -> None:
    settings = replace_settings(settings, escalation_enabled=True)
    local = FakeProvider(replies=[INVENTED])
    frontier = FakeProvider(model_id="claude-opus-5", tier="frontier", replies=[GROUNDED])

    answer = await build(store, retriever, settings, local=local, frontier=frontier).answer(
        QUESTION
    )

    assert answer.tier is Tier.FRONTIER
    # The local attempt costs nothing per token, so the total is the frontier
    # call; the usage totals cover both so token accounting stays honest.
    assert answer.cost_usd == pytest.approx(0.02)
    assert answer.usage.input_tokens == 6000, "both attempts' tokens are counted"


async def test_a_provider_outage_costs_nothing(store, retriever, settings) -> None:
    """Nothing was generated, so nothing should be billed."""
    settings = replace_settings(settings, escalation_enabled=True)
    frontier = FakeProvider(model_id="claude-opus-5", tier="frontier", fail=True)

    answer = await build(
        store, retriever, settings, local=FakeProvider(fail=True), frontier=frontier
    ).answer(QUESTION)
    assert answer.tier is Tier.REFUSED
    assert answer.cost_usd == 0.0


async def test_unpriced_models_are_flagged_rather_than_counted_as_zero(
    store, retriever, settings
) -> None:
    settings = replace_settings(settings, escalation_enabled=True)
    frontier = FakeProvider(model_id="some-new-model", tier="frontier", replies=[GROUNDED])

    answer = await build(store, retriever, settings, frontier=frontier).answer(QUESTION)
    assert answer.tier is Tier.FRONTIER
    assert any("no verified price" in n for n in answer.notes)


def test_refused_is_not_counted_as_a_cache_hit() -> None:
    """It can be expensive, so it must not be lumped in with the free tiers."""
    assert not Tier.REFUSED.is_cache_hit
    assert Tier.PINNED.is_cache_hit and Tier.EXACT_CACHE.is_cache_hit
    assert not Tier.LOCAL.is_cache_hit and not Tier.FRONTIER.is_cache_hit
