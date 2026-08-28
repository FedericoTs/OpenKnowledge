"""Streaming: the same answer, narrated - and withdrawn when the gate says so.

The design constraint everything here pins: there is ONE resolution path.
answer() drains the same event stream answer_stream() exposes, so the two can
never disagree about tier, caching, notes or cost. A separate streaming
resolver would eventually fork, and the determinism argument rests on there
being exactly one way a question gets answered.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from tests.fakes import FakeProvider
from tests.test_cascade import GROUNDED, INVENTED, QUESTION, build

from openknowledge.providers.base import Completion, ProviderError
from openknowledge.types import Tier


@dataclass
class StreamingFakeProvider(FakeProvider):
    """A FakeProvider whose answers also arrive as word-by-word deltas."""

    stream_calls: list[str] = field(default_factory=list)

    async def stream(
        self,
        *,
        system: str,
        context: str,
        question: str,
        history: tuple = (),
        max_tokens: int = 1500,
    ) -> AsyncIterator[str | Completion]:
        self.stream_calls.append(question)
        if self.fail:
            raise ProviderError(f"{self.model_id} is unavailable")
        reply = self.replies.pop(0) if self.replies else "I don't know."
        words = reply.split(" ")
        for index, word in enumerate(words):
            yield word if index == 0 else f" {word}"
        yield Completion(text=reply, usage=self.usage, model_id=self.model_id)


async def collect(cascade, question: str) -> list[dict]:
    return [event async for event in cascade.answer_stream(question)]


async def test_a_grounded_answer_streams_then_finalises(store, retriever, settings) -> None:
    local = StreamingFakeProvider(replies=[GROUNDED])
    events = await collect(build(store, retriever, settings, local=local), QUESTION)

    kinds = [e["type"] for e in events]
    assert kinds[0] == "status"
    assert "provisional" in kinds
    assert kinds[-1] == "final"

    deltas = "".join(e["text"] for e in events if e["type"] == "delta")
    final = events[-1]["answer"]
    assert deltas == final.text, "the streamed text and the final answer diverged"
    assert final.tier is Tier.LOCAL
    assert final.grounded
    assert local.stream_calls == [QUESTION], "streamed rung must not also call complete()"
    assert local.calls == []


async def test_streaming_and_plain_produce_the_same_answer(store, retriever, settings) -> None:
    """Byte-identical, because it is literally the same code path."""
    streamed_provider = StreamingFakeProvider(replies=[GROUNDED])
    events = await collect(build(store, retriever, settings, local=streamed_provider), QUESTION)
    streamed = events[-1]["answer"]

    # Compare against a fresh non-streaming cascade over a fresh store, so the
    # second run cannot simply hit the cache the first one wrote.
    from openknowledge.cache import AnswerStore

    with AnswerStore() as fresh_store:
        plain = await build(
            fresh_store, retriever, settings, local=FakeProvider(replies=[GROUNDED])
        ).answer(QUESTION)

    assert streamed.text == plain.text
    assert streamed.tier == plain.tier
    assert streamed.citations == plain.citations


async def test_ungated_text_is_retracted_in_front_of_the_reader(store, retriever, settings) -> None:
    """The honesty half of the streaming bargain.

    The reader watched INVENTED text appear. When the gate rejects it, the
    reader must watch it withdrawn - a stream that silently swaps a bad answer
    for a refusal is indistinguishable from one that never showed it.
    """
    local = StreamingFakeProvider(replies=[INVENTED])
    events = await collect(build(store, retriever, settings, local=local), QUESTION)

    kinds = [e["type"] for e in events]
    assert "provisional" in kinds
    assert "retract" in kinds, "rejected streamed text was never retracted"
    assert kinds.index("retract") > kinds.index("provisional")

    final = events[-1]["answer"]
    assert final.tier is Tier.REFUSED
    retract = next(e for e in events if e["type"] == "retract")
    assert retract["reason"], "a retraction must say why"


async def test_a_stream_that_dies_midway_is_not_the_documents_fault(
    store, retriever, settings
) -> None:
    local = StreamingFakeProvider(fail=True)
    events = await collect(build(store, retriever, settings, local=local), QUESTION)

    final = events[-1]["answer"]
    assert final.tier is Tier.REFUSED
    assert "never read" in final.text  # the unavailable refusal, not the corpus one
    assert any(e["type"] == "retract" for e in events)


async def test_instant_tiers_stream_nothing_but_the_final(store, retriever, settings) -> None:
    """A cache hit narrated as 'generating...' would be theatre."""
    local = StreamingFakeProvider(replies=[GROUNDED, GROUNDED])
    cascade = build(store, retriever, settings, local=local)
    await cascade.answer(QUESTION)  # populate the exact cache

    events = await collect(cascade, QUESTION)
    assert [e["type"] for e in events] == ["final"]
    assert events[-1]["answer"].tier is Tier.EXACT_CACHE


async def test_a_billed_rung_is_never_streamed(store, retriever, settings) -> None:
    """Some runtimes report no usage on a stream; a zero in the ledger for a
    billed call would understate the one number this project is judged on."""
    billed = StreamingFakeProvider(replies=[GROUNDED], self_hosted=False)
    events = await collect(build(store, retriever, settings, local=billed), QUESTION)

    assert not any(e["type"] in ("provisional", "delta") for e in events)
    assert billed.stream_calls == []
    assert billed.calls == [QUESTION]
    assert events[-1]["answer"].tier is Tier.LOCAL


def test_the_http_stream_carries_the_same_final_payload() -> None:
    """The SSE endpoint's final frame is exactly what /chat would return."""
    import json

    from fastapi.testclient import TestClient

    from openknowledge.api.app import create_app
    from openknowledge.config import Settings

    settings = Settings(
        data_dir="./data-stream-test",
        local_enabled=False,
        embedding_enabled=False,
        _env_file=None,  # type: ignore[call-arg]
    )
    with TestClient(create_app(settings)) as client:
        with client.stream(
            "POST", "/chat/stream", json={"question": "what documents do you have?"}
        ) as response:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
            frames = [
                json.loads(line[len("data:") :])
                for line in response.iter_lines()
                if line.startswith("data:")
            ]
        plain = client.post("/chat", json={"question": "what documents do you have?"}).json()

    assert frames[-1]["type"] == "final"
    assert frames[-1]["response"]["tier"] == plain["tier"] == "corpus"
    assert frames[-1]["response"]["answer"] == plain["answer"]

    import shutil

    shutil.rmtree("./data-stream-test", ignore_errors=True)
