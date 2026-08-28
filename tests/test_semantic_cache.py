"""The semantic cache: similarity nominates, the gate decides.

The measurement that shaped this design, against the real embedding model:
genuine paraphrases scored 0.727-0.849 cosine, while "parental leave weeks"
vs "annual leave days" - two questions with different correct answers -
scored 0.810, inside the paraphrase band. No threshold separates them, so
cosine is only allowed to pick a candidate; check_grounding, run against the
NEW question's own retrieval, decides whether the cached answer may be
served. The trap test below is that 0.810 pair, reconstructed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from tests.fakes import FakeProvider
from tests.test_cascade import build

from openknowledge.cache import AnswerStore
from openknowledge.cache.semantic import SemanticIndex
from openknowledge.config import Settings
from openknowledge.retrieval import BM25Retriever
from openknowledge.retrieval.base import Document
from openknowledge.types import Tier

LEAVE_DOC = Document(
    "hr-parental-leave",
    "Parental Leave Policy",
    "Parental leave. Employees with 12 months of continuous service are entitled "
    "to 20 weeks of fully paid parental leave.",
)
HOLIDAY_DOC = Document(
    "hr-annual-leave",
    "Annual Leave Policy",
    "Annual leave. Employees receive 25 days of annual leave per year, plus public holidays.",
)

ANSWER = (
    "Employees with 12 months of continuous service are entitled to 20 weeks of "
    "fully paid parental leave [hr-parental-leave]."
)

PHRASING_A = "How many weeks of parental leave do employees get?"
PHRASING_B = "how much parental leave do I get"
TRAP = "How many days of annual leave do employees get?"


@dataclass
class TopicEmbedder:
    """Vectors by crude topic, with the trap built in.

    'leave' questions - parental AND annual - land near each other, exactly as
    the real model measured (0.810 for the trap pair). If similarity alone
    decided, the trap question would be served the parental answer.
    """

    model: str = "topic-stub"
    base_url: str = "http://stub/v1"
    document_prefix: str = ""
    query_prefix: str = ""
    calls: list[str] = field(default_factory=list)

    @property
    def fingerprint(self) -> str:
        return "topic-stub@stub"

    def _vector(self, text: str) -> list[float]:
        low = text.lower()
        leave = 1.0 if "leave" in low else 0.0
        meals = 1.0 if "meal" in low or "dinner" in low else 0.0
        return [leave, meals, 0.3]

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.extend(texts)
        return [self._vector(t) for t in texts]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]


@pytest.fixture
def semantic_setup(settings: Settings):
    retriever = BM25Retriever()
    retriever.index([LEAVE_DOC, HOLIDAY_DOC])
    store = AnswerStore()
    settings.semantic_cache_enabled = True
    index = SemanticIndex(store, TopicEmbedder())
    yield store, retriever, index
    store.close()


async def test_a_paraphrase_is_served_from_the_semantic_cache(semantic_setup, settings) -> None:
    store, retriever, index = semantic_setup
    local = FakeProvider(replies=[ANSWER])
    cascade = build(store, retriever, settings, local=local)
    cascade.semantic = index

    first = await cascade.answer(PHRASING_A)
    assert first.tier is Tier.LOCAL

    second = await cascade.answer(PHRASING_B)
    assert second.tier is Tier.SEMANTIC_CACHE
    assert second.text == first.text
    assert second.grounded
    assert any("matched an earlier phrasing" in n for n in second.notes)
    assert len(local.calls) == 1, "the paraphrase must not cost a model call"


async def test_the_near_neighbour_trap_falls_through_to_the_model(semantic_setup, settings) -> None:
    """The 0.810 pair. Similarity nominates the parental answer for the
    annual-leave question; the gate must reject it, because its figures find
    no support in what the annual-leave question retrieves."""
    store, retriever, index = semantic_setup
    holiday_answer = "Employees receive 25 days of annual leave per year [hr-annual-leave]."
    local = FakeProvider(replies=[ANSWER, holiday_answer])
    cascade = build(store, retriever, settings, local=local)
    cascade.semantic = index

    await cascade.answer(PHRASING_A)
    trapped = await cascade.answer(TRAP)

    assert trapped.tier is Tier.LOCAL, "served a cached answer to a different question"
    assert "25 days" in trapped.text
    assert "20 weeks" not in trapped.text
    assert len(local.calls) == 2


async def test_a_nominee_the_asker_cannot_see_is_never_served(settings) -> None:
    secret = Document(
        "restricted-leave",
        "Executive Leave Terms",
        "Parental leave. Executives are entitled to 30 weeks of fully paid parental leave.",
        allowed_principals=frozenset({"executives"}),
    )
    retriever = BM25Retriever()
    retriever.index([secret])
    store = AnswerStore()
    settings.semantic_cache_enabled = True
    local = FakeProvider(replies=["Executives get 30 weeks of parental leave [restricted-leave]."])
    cascade = build(store, retriever, settings, local=local)
    cascade.semantic = SemanticIndex(store, TopicEmbedder())

    first = await cascade.answer(
        "How many weeks of parental leave do executives get?",
        principals=frozenset({"executives"}),
    )
    assert first.tier is Tier.LOCAL

    outsider = await cascade.answer(
        "how much parental leave for executives", principals=frozenset({"staff"})
    )
    assert outsider.tier is not Tier.SEMANTIC_CACHE
    assert "30 weeks" not in outsider.text
    store.close()


async def test_a_corpus_change_evicts_the_vectors(semantic_setup, settings) -> None:
    store, retriever, index = semantic_setup
    local = FakeProvider(replies=[ANSWER, ANSWER])
    cascade = build(store, retriever, settings, local=local)
    cascade.semantic = index

    await cascade.answer(PHRASING_A)
    assert index.evict_other_corpus_versions("a-new-corpus") == 1
    vector = index.embed(PHRASING_B)
    assert index.nominate(vector, retriever.corpus_version, threshold=0.7) is None


async def test_refusals_never_enter_the_semantic_cache(semantic_setup, settings) -> None:
    store, retriever, index = semantic_setup
    local = FakeProvider(replies=["This is invented nonsense with no citation."])
    cascade = build(store, retriever, settings, local=local)
    cascade.semantic = index

    refused = await cascade.answer(PHRASING_A)
    assert refused.tier is Tier.REFUSED
    vector = index.embed(PHRASING_B)
    assert index.nominate(vector, retriever.corpus_version, threshold=0.5) is None


async def test_without_an_index_nothing_changes(semantic_setup, settings) -> None:
    store, retriever, _ = semantic_setup
    local = FakeProvider(replies=[ANSWER, ANSWER])
    cascade = build(store, retriever, settings, local=local)
    assert cascade.semantic is None

    await cascade.answer(PHRASING_A)
    second = await cascade.answer(PHRASING_B)
    assert second.tier is Tier.LOCAL
    assert len(local.calls) == 2


async def test_the_gate_alone_was_not_enough_and_the_first_version_proves_it(
    semantic_setup, settings
) -> None:
    """Regression pin for the hole the trap test found in the first design.

    With near-topic documents, the trap question retrieves the parental chunk
    somewhere in its top-k, so the cached parental answer grounds at full
    support for the wrong question - the gate judges grounding, not aboutness.
    What dismisses the nominee is retrieval's own first choice: the top-ranked
    document for the new question is not one the cached answer cites.
    """
    store, retriever, index = semantic_setup
    local = FakeProvider(replies=[ANSWER, "unused"])
    cascade = build(store, retriever, settings, local=local)
    cascade.semantic = index
    await cascade.answer(PHRASING_A)

    hits = retriever.search(TRAP, k=settings.retrieval_k)
    assert any(h.chunk.document_id == "hr-parental-leave" for h in hits), (
        "the trap has lost its teeth: the parental chunk no longer reaches "
        "the trap question's top-k, so this test would pass vacuously"
    )
    assert hits[0].chunk.document_id == "hr-annual-leave"


async def test_the_live_trap_a_corpus_with_no_annual_leave_document(
    settings,
) -> None:
    """The live run's failure, reconstructed exactly.

    On the real corpus there IS no annual-leave document, so for "how many
    days of annual leave" retrieval's top-ranked document was the parental
    one, the gate grounded the cached parental answer at full support, and
    the wrong answer was served at similarity 0.81. Both earlier arbiters
    judge "closest thing we have"; neither asks whether the closest thing
    answers the question. The coverage arbiter does: "annual" and "days"
    appear nowhere in the cached question or answer, so the nominee dies and
    the question falls through to the model - which is the rung entitled to
    refuse it.
    """
    retriever = BM25Retriever()
    retriever.index([LEAVE_DOC])  # deliberately no annual-leave document
    store = AnswerStore()
    settings.semantic_cache_enabled = True
    local = FakeProvider(replies=[ANSWER, "I don't know."])
    cascade = build(store, retriever, settings, local=local)
    cascade.semantic = SemanticIndex(store, TopicEmbedder())

    await cascade.answer(PHRASING_A)
    trapped = await cascade.answer(TRAP)

    assert trapped.tier is not Tier.SEMANTIC_CACHE, "the live failure is back"
    assert "20 weeks" not in trapped.text
    assert len(local.calls) == 2, "the trap must reach the model, which may refuse it"
    store.close()


async def test_an_answers_ramblings_do_not_widen_what_it_can_be_served_for(
    semantic_setup, settings
) -> None:
    """The golden set's catch, reproduced.

    The live entitlement answer volunteered a sentence about contractors, so
    "do contractors get parental leave?" matched the cached entry through its
    ANSWER text and was served "20 weeks" - forbidden content for that
    question. A cache entry can vouch for its question and nothing else.
    """
    rambling = (
        "Employees are entitled to 20 weeks of fully paid parental leave "
        "[hr-parental-leave]. Contractors are not eligible for company-paid "
        "parental leave [hr-parental-leave]."
    )
    store, retriever, index = semantic_setup
    local = FakeProvider(replies=[rambling, "Contractors are not eligible [hr-parental-leave]."])
    cascade = build(store, retriever, settings, local=local)
    cascade.semantic = index

    await cascade.answer(PHRASING_A)
    contractor = await cascade.answer("do contractors get parental leave?")

    assert contractor.tier is not Tier.SEMANTIC_CACHE, "vouched beyond its question"
    assert "20 weeks" not in contractor.text
    assert len(local.calls) == 2
