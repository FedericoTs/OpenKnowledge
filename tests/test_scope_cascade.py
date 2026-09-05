"""Whole-document questions through the cascade, over the scope corpus.

The unit tests prove the recogniser and the structure reader; this proves the
router does what evals/golden-scope will measure with a model: an enumeration
the structure holds is answered without calling one, an ordinal past the end
refuses without calling one, a summary reads the named section in order, and a
fact question is untouched.
"""

from __future__ import annotations

import pathlib

import pytest

from fakes import FakeProvider
from openknowledge.cascade import Cascade
from openknowledge.documents import parse_file
from openknowledge.retrieval.base import Document
from openknowledge.retrieval.bm25 import BM25Retriever
from openknowledge.types import Tier

pytestmark = pytest.mark.asyncio

DOCS = pathlib.Path(__file__).resolve().parents[1] / "evals" / "golden-scope" / "documents"


@pytest.fixture(scope="module")
def scope_retriever() -> BM25Retriever:
    """The corpus the way the connector would hand it over: parsed, with blocks."""
    documents = []
    for path in sorted(DOCS.glob("*.md")):
        parsed = parse_file(path)
        documents.append(
            Document(
                document_id=path.stem,
                title=parsed.title or path.stem,
                text=parsed.text,
                blocks=parsed.blocks,
            )
        )
    r = BM25Retriever()
    r.index(documents)
    return r


def _cascade(store, scope_retriever, settings, provider: FakeProvider) -> Cascade:
    return Cascade(store=store, retriever=scope_retriever, settings=settings, local=provider)


async def test_the_cast_list_is_answered_from_structure_with_no_model(
    store, scope_retriever, settings
) -> None:
    provider = FakeProvider()
    answer = await _cascade(store, scope_retriever, settings, provider).answer(
        "Who are the persons in The Importance of Being Earnest?"
    )
    assert answer.tier is Tier.OUTLINE
    assert provider.calls == [], "no model should have been asked"
    assert answer.cost_usd == 0
    for name in ("John Worthing", "Algernon Moncrieff", "Lady Bracknell", "Miss Prism"):
        assert name in answer.text
    assert [c.document_id for c in answer.citations] == ["earnest"]


async def test_the_glossary_is_listed_in_full_without_a_model(
    store, scope_retriever, settings
) -> None:
    """Retrieval at k=6 shows a model 26 of these; the structure shows all 82."""
    provider = FakeProvider()
    answer = await _cascade(store, scope_retriever, settings, provider).answer(
        "What terms does the glossary of terms define?"
    )
    assert answer.tier is Tier.OUTLINE and provider.calls == []
    assert "has 82 terms" in answer.text
    assert "Accompanied baggage" in answer.text and "Usually traveled route" in answer.text


async def test_an_ordinal_names_the_item(store, scope_retriever, settings) -> None:
    provider = FakeProvider()
    answer = await _cascade(store, scope_retriever, settings, provider).answer(
        "What is the title of the second chapter of Alice in Wonderland?"
    )
    assert answer.tier is Tier.OUTLINE and provider.calls == []
    assert "Pool of Tears" in answer.text


@pytest.mark.parametrize(
    ("question", "count"),
    [
        ("What happens in the fourth act of The Importance of Being Earnest?", "3 acts"),
        ("What is the title of chapter thirteen of Alice in Wonderland?", "12 chapters"),
    ],
)
async def test_an_ordinal_past_the_end_refuses_with_the_count_and_no_model(
    store, scope_retriever, settings, question: str, count: str
) -> None:
    """The alternative is a model inventing a fourth act."""
    provider = FakeProvider()
    answer = await _cascade(store, scope_retriever, settings, provider).answer(question)
    assert answer.tier is Tier.REFUSED
    assert count in answer.text
    assert provider.calls == []


async def test_a_summary_reads_the_named_section_in_order(store, scope_retriever, settings) -> None:
    provider = FakeProvider()
    await _cascade(store, scope_retriever, settings, provider).answer(
        "Summarise the first act of The Importance of Being Earnest."
    )
    assert len(provider.calls) == 1
    context = provider.contexts[0]
    passages = context.count("\n[earnest]")
    # Window unknown in tests, so the cap is three times retrieval_k.
    assert 6 < passages <= 3 * settings.retrieval_k, passages
    assert "FIRST ACT" in context
    # The other acts are not read for a question about the first.
    assert "Drawing-Room at the Manor House" not in context or "THIRD ACT" not in context
    # In order: the first passage of the act precedes the last one shown.
    assert context.index("Lane is arranging afternoon tea") < context.rindex("\n[earnest]")


async def test_a_fact_question_keeps_the_ranked_path(store, scope_retriever, settings) -> None:
    """Six ranked passages, exactly as before - the control stays a control."""
    provider = FakeProvider()
    await _cascade(store, scope_retriever, settings, provider).answer(
        "What transportation methods may an agency authorize?"
    )
    assert len(provider.calls) == 1
    passages = provider.contexts[0].count("\n[")
    assert passages <= settings.retrieval_k


async def test_the_vote_reads_the_candidates_before_the_reranker(
    store, scope_retriever, settings
) -> None:
    """No title named. After reranking the six hits spread two per document
    and there is no target; before it, sixteen of thirty come from the play.
    Measured, and the reason the router keeps the pre-rerank list."""
    provider = FakeProvider()
    answer = await _cascade(store, scope_retriever, settings, provider).answer(
        "who are the characters?"
    )
    assert answer.tier is Tier.OUTLINE and provider.calls == []
    assert "Lady Bracknell" in answer.text


async def test_chapters_of_a_corpus_with_a_play_are_not_the_plays_acts(
    store, scope_retriever, settings
) -> None:
    """The ranking concentrates on the play for "what are the chapters?"; the
    play has no chapters; the structure must not answer with its acts."""
    provider = FakeProvider()
    answer = await _cascade(store, scope_retriever, settings, provider).answer(
        "what are the chapters?"
    )
    assert answer.tier is not Tier.OUTLINE or "FIRST ACT" not in answer.text
