"""Follow-ups: "what about contractors?" has to mean something.

The design constraint these tests hold: history is material for INTERPRETING
a question, never part of the cache key. The rewritten standalone question is
what gets keyed, so the same follow-up in the same conversation hits the exact
cache the second time, and the raw fragment never becomes a key at all.
"""

from __future__ import annotations

import pytest
from tests.fakes import FakeProvider
from tests.test_cascade import GROUNDED, build

from openknowledge.cascade.followup import looks_dependent, resolve
from openknowledge.providers.base import Message
from openknowledge.types import Tier

HISTORY = (
    Message(role="user", content="How many weeks of parental leave do employees get?"),
    Message(role="assistant", content="Employees get 20 weeks of fully paid parental leave."),
)

STANDALONE = "Are contractors eligible for parental leave?"


DEPENDENT = [
    "what about contractors?",
    "and during probation?",
    "is it paid?",
    "does that include weekends",
    "how about part-timers?",
    "why not?",
    "can they carry it over",
]

SELF_CONTAINED = [
    "What is the daily meal allowance?",
    "How many days of parental leave do employees get?",
    "what documents do you have",
    "What is the approval threshold for travel expenditure under the policy?",
]


@pytest.mark.parametrize("question", DEPENDENT)
def test_questions_that_lean_backwards_are_recognised(question: str) -> None:
    assert looks_dependent(question)


@pytest.mark.parametrize("question", SELF_CONTAINED)
def test_complete_questions_pay_no_interpretation_call(question: str) -> None:
    assert not looks_dependent(question)


async def test_a_follow_up_is_rewritten_using_the_history() -> None:
    provider = FakeProvider(replies=[STANDALONE])
    resolution = await resolve("what about contractors?", HISTORY, provider)

    assert resolution.rewritten
    assert resolution.question == STANDALONE
    assert "interpreted as" in resolution.note
    # The transcript reached the model; without it there is nothing to resolve from.
    assert "parental leave" in provider.contexts[0]


async def test_a_standalone_question_is_not_sent_for_rewriting() -> None:
    provider = FakeProvider(replies=["should never be used"])
    resolution = await resolve(SELF_CONTAINED[0], HISTORY, provider)
    assert not resolution.rewritten
    assert provider.calls == []


async def test_no_provider_degrades_to_answering_as_asked() -> None:
    resolution = await resolve("what about contractors?", HISTORY, None)
    assert not resolution.rewritten
    assert "no local model was reachable" in resolution.note


async def test_a_dead_provider_degrades_instead_of_erroring() -> None:
    resolution = await resolve("what about contractors?", HISTORY, FakeProvider(fail=True))
    assert not resolution.rewritten
    assert "answered as asked" in resolution.note


async def test_a_degenerate_rewrite_is_discarded() -> None:
    essay = "Well, considering the conversation, " * 30
    resolution = await resolve("what about contractors?", HISTORY, FakeProvider(replies=[essay]))
    assert not resolution.rewritten
    assert resolution.question == "what about contractors?"


# --- through the cascade ----------------------------------------------------


async def test_the_cache_is_keyed_on_the_resolved_question(store, retriever, settings) -> None:
    """The same follow-up, twice, in the same conversation: the second one must
    hit the exact cache - proof that the fragment never became a key."""
    local = FakeProvider(replies=[STANDALONE, GROUNDED, STANDALONE])
    cascade = build(store, retriever, settings, local=local)

    first = await cascade.answer("what about contractors?", history=HISTORY)
    assert first.tier is Tier.LOCAL
    assert any("interpreted as" in n for n in first.notes)

    again = await cascade.answer("what about contractors?", history=HISTORY)
    assert again.tier is Tier.EXACT_CACHE
    assert again.text == first.text


async def test_asking_the_standalone_question_directly_hits_the_same_entry(
    store, retriever, settings
) -> None:
    """The rewritten question is a real question: someone asking it cold, with
    no conversation at all, gets the identical cached answer."""
    local = FakeProvider(replies=[STANDALONE, GROUNDED])
    cascade = build(store, retriever, settings, local=local)
    followed_up = await cascade.answer("what about contractors?", history=HISTORY)

    cold = await cascade.answer(STANDALONE)
    assert cold.tier is Tier.EXACT_CACHE
    assert cold.text == followed_up.text


async def test_the_interpretation_call_is_billed_onto_the_answer(
    store, retriever, settings
) -> None:
    local = FakeProvider(replies=[STANDALONE, GROUNDED])
    answer = await build(store, retriever, settings, local=local).answer(
        "what about contractors?", history=HISTORY
    )
    # Two calls' worth of tokens: the rewrite and the answer.
    assert answer.usage.input_tokens == 2 * local.usage.input_tokens
    assert answer.usage.output_tokens == 2 * local.usage.output_tokens


async def test_without_history_nothing_changes(store, retriever, settings) -> None:
    """No conversation, no interpretation, no extra call - the single-question
    path is exactly what it was, even for a question phrased like a follow-up."""
    local = FakeProvider(replies=[GROUNDED])
    answer = await build(store, retriever, settings, local=local).answer(
        "what about the parental leave weeks?"  # dependent-shaped, but retrievable
    )
    assert len(local.calls) == 1, "only the answering call may happen"
    assert local.calls == ["what about the parental leave weeks?"]
    assert not any("interpreted" in n for n in answer.notes)


async def test_the_stream_names_the_interpretation(store, retriever, settings) -> None:
    local = FakeProvider(replies=[STANDALONE, GROUNDED])
    cascade = build(store, retriever, settings, local=local)
    events = [e async for e in cascade.answer_stream("what about contractors?", history=HISTORY)]

    resolved = [e for e in events if e["type"] == "resolved"]
    assert resolved and resolved[0]["question"] == STANDALONE
    assert events[-1]["answer"].grounded
