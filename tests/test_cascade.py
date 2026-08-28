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


async def test_an_unreachable_model_does_not_blame_the_documents(
    store, retriever, settings
) -> None:
    """The two refusals are different statements and must not be swapped.

    Reported from a real deployment: Ollama was not running, and the widget
    answered "that isn't covered by the documents I have" over a corpus it had
    never read. That sends the operator to look at their documents when the
    problem is their model server - a wrong answer about why there is no answer,
    which is the one kind of wrong answer this design cannot afford.
    """
    local = FakeProvider(fail=True)
    answer = await build(store, retriever, settings, local=local).answer(QUESTION)

    assert answer.tier is Tier.REFUSED
    assert "never read" in answer.text
    assert "isn't covered by the documents" not in answer.text

    # And no sources: listing them under this claim implies something read them.
    assert answer.citations == ()
    assert any("tier unavailable" in n for n in answer.notes)
    assert any("model status" in n for n in answer.notes)
    assert any("nothing read them" in n for n in answer.notes)


async def test_a_model_that_read_them_and_failed_does_blame_the_documents(
    store, retriever, settings
) -> None:
    """The other half. Here the sentence is true, and the sources are worth showing."""
    local = FakeProvider(replies=[INVENTED])
    answer = await build(store, retriever, settings, local=local).answer(QUESTION)

    assert answer.tier is Tier.REFUSED
    assert "isn't covered by the documents" in answer.text
    assert "never read" not in answer.text
    assert answer.citations, "it read these; say which"


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


# -- drafted answers and contested claims ----------------------------------

from openknowledge.knowledge import KnowledgeStore, find_conflicts  # noqa: E402
from openknowledge.types import Citation  # noqa: E402

DRAFTED = "Employees with 12 months of service get 20 weeks of fully paid leave. [hr-handbook]"


def with_knowledge(store, retriever, settings, knowledge, *, local=None) -> Cascade:
    return Cascade(
        store=store,
        retriever=retriever,
        settings=settings,
        local=local,
        knowledge=knowledge,
    )


async def test_a_drafted_answer_is_served_and_marked_unreviewed(
    store, retriever, settings, knowledge
) -> None:
    """It passed the same gate a live answer does, so it is a precomputed cache
    entry - served, but never presented as though a person wrote it."""
    knowledge.propose(
        canonical_query="how much parental leave do i get",
        question=QUESTION,
        answer=DRAFTED,
        citations=(Citation("hr-handbook", "HR Handbook", "20 weeks", None),),
        origin_documents=("hr-handbook",),
        corpus_version="c1",
        support_ratio=0.9,
    )
    local = FakeProvider(replies=[GROUNDED])
    answer = await with_knowledge(store, retriever, settings, knowledge, local=local).answer(
        QUESTION
    )

    assert answer.tier is Tier.DRAFT
    assert answer.needs_review
    assert answer.cost_usd == 0.0
    assert any("not yet reviewed" in n for n in answer.notes)
    assert local.calls == [], "a drafted answer must not reach a model"


async def test_serving_drafts_can_be_switched_off(store, retriever, settings, knowledge) -> None:
    """For a posture where nothing machine-written reaches staff before sign-off."""
    knowledge.propose(
        canonical_query="how much parental leave do i get",
        question=QUESTION,
        answer=DRAFTED,
        citations=(Citation("hr-handbook", "HR Handbook", "20 weeks", None),),
        origin_documents=("hr-handbook",),
        corpus_version="c1",
    )
    off = replace_settings(settings, serve_drafts=False)
    local = FakeProvider(replies=[GROUNDED])
    answer = await with_knowledge(store, retriever, off, knowledge, local=local).answer(QUESTION)

    assert answer.tier is Tier.LOCAL
    assert local.calls


async def test_a_draft_is_access_checked_like_any_other_answer(
    store, retriever, settings, knowledge
) -> None:
    knowledge.propose(
        canonical_query="what are the executive salary bands",
        question="What are the executive salary bands?",
        answer="EUR 180000 to EUR 240000. [board-comp]",
        citations=(Citation("board-comp", "Board Compensation", "bands", None),),
        origin_documents=("board-comp",),
        corpus_version="c1",
    )
    cascade = with_knowledge(
        store, retriever, settings, knowledge, local=FakeProvider(replies=["I don't know."])
    )
    staff = await cascade.answer(
        "What are the executive salary bands?", principals=frozenset({"staff"})
    )
    assert staff.tier is not Tier.DRAFT


def seed_conflict(knowledge: KnowledgeStore) -> None:
    old = Document(
        "expenses-2025",
        "Expenses 2025",
        "Travel expenses require prior written approval for any amount above EUR 500.",
    )
    new = Document(
        "expenses-2026",
        "Expenses 2026",
        "Travel expenses require prior written approval for any amount above EUR 1,000.",
    )
    for conflict in find_conflicts([old, new]):
        knowledge.record_conflict(conflict)


async def test_a_contested_claim_is_refused_rather_than_guessed(
    store, retriever, settings, knowledge
) -> None:
    seed_conflict(knowledge)
    local = FakeProvider(replies=[GROUNDED])
    answer = await with_knowledge(store, retriever, settings, knowledge, local=local).answer(
        "What is the approval threshold for travel expenses?"
    )

    assert answer.tier is Tier.CONTESTED
    assert not answer.is_answerable
    assert "EUR 500" in answer.text and "EUR 1,000" in answer.text
    assert local.calls == [], "a contested question must not reach a model"


async def test_questions_the_documents_agree_on_are_unaffected(
    store, retriever, settings, knowledge
) -> None:
    """Blocking every question that touches a document with one bad figure
    would make the feature intolerable."""
    seed_conflict(knowledge)
    local = FakeProvider(replies=[GROUNDED])
    answer = await with_knowledge(store, retriever, settings, knowledge, local=local).answer(
        QUESTION
    )
    assert answer.tier is Tier.LOCAL


async def test_a_pin_made_before_the_conflict_is_withheld(
    store, retriever, settings, knowledge
) -> None:
    """The stale-pin trap: a person pinned an answer, then a document arrived
    that disagrees. The pin looks authoritative and is out of date."""
    store.pin(
        "what is the approval threshold for travel expenses",
        "Approval is required above EUR 500.",
        author="finance",
    )
    seed_conflict(knowledge)  # detected after the pin was written

    answer = await with_knowledge(store, retriever, settings, knowledge).answer(
        "What is the approval threshold for travel expenses?"
    )
    assert answer.tier is Tier.CONTESTED


async def test_a_timestamp_tie_withholds_the_pin(
    store, retriever, settings, knowledge, monkeypatch
) -> None:
    """Windows CI caught this: time.time() there ticks every ~15.6 ms, so a
    pin and a conflict written back to back can share one timestamp, and a
    strict after-comparison then reads the conflict as already accounted for.
    In-process writes now never tie (clock.ordered_now), but a tie remains
    possible across processes - and equality cannot prove the pinner saw the
    disagreement, so it must withhold."""
    import openknowledge.cache.store as cache_store
    import openknowledge.knowledge.store as knowledge_store

    frozen = 1_700_000_000.0
    monkeypatch.setattr(cache_store, "ordered_now", lambda: frozen)
    monkeypatch.setattr(knowledge_store, "ordered_now", lambda: frozen)

    store.pin(
        "what is the approval threshold for travel expenses",
        "Approval is required above EUR 500.",
        author="finance",
    )
    seed_conflict(knowledge)  # same instant, by construction

    answer = await with_knowledge(store, retriever, settings, knowledge).answer(
        "What is the approval threshold for travel expenses?"
    )
    assert answer.tier is Tier.CONTESTED


async def test_a_pin_made_after_the_conflict_wins(store, retriever, settings, knowledge) -> None:
    """Pinning with the disagreement visible *is* resolving it."""
    seed_conflict(knowledge)
    store.pin(
        "what is the approval threshold for travel expenses",
        "Approval is required above EUR 1,000 per the 2026 policy.",
        author="finance",
    )
    answer = await with_knowledge(store, retriever, settings, knowledge).answer(
        "What is the approval threshold for travel expenses?"
    )
    assert answer.tier is Tier.PINNED
    assert "1,000" in answer.text


async def test_resolving_a_conflict_unblocks_the_question(
    store, retriever, settings, knowledge
) -> None:
    seed_conflict(knowledge)
    for conflict in knowledge.open_conflicts():
        knowledge.resolve_conflict(conflict.key, resolution="2026 is authoritative")

    local = FakeProvider(replies=["Approval is required above EUR 500. [expenses]"])
    answer = await with_knowledge(store, retriever, settings, knowledge, local=local).answer(
        "What is the approval threshold for travel expenses?"
    )
    assert answer.tier is not Tier.CONTESTED


async def test_conflict_blocking_can_be_switched_off(store, retriever, settings, knowledge) -> None:
    seed_conflict(knowledge)
    off = replace_settings(settings, block_on_conflict=False)
    local = FakeProvider(replies=[GROUNDED])
    answer = await with_knowledge(store, retriever, off, knowledge, local=local).answer(
        "What is the approval threshold for travel expenses?"
    )
    assert answer.tier is not Tier.CONTESTED


async def test_a_contested_refusal_is_not_counted_as_a_cache_hit() -> None:
    """It answers nothing, so it must not inflate the free share."""
    assert not Tier.CONTESTED.is_cache_hit
    assert Tier.DRAFT.is_cache_hit
