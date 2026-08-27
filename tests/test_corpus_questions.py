"""Questions about the collection, not from it.

"What documents do you have?" is the first thing people type into a document
assistant, and it was answered with "that isn't covered by the documents I
have" - a system that knows exactly what it holds, saying it does not. No
retriever can fix that: the question has no subject to match on.

The recogniser carries all the risk. Hijacking a real document question would
be much worse than missing a meta-question, so the bar is not "does this look
like a meta-question" but "is there anything else in it".
"""

from __future__ import annotations

import pytest

from openknowledge.cascade.corpus import describe, recognise
from openknowledge.retrieval import BM25Retriever
from openknowledge.retrieval.base import Document
from openknowledge.types import Tier

ABOUT_THE_CORPUS = [
    "what document do we have?",
    "what documents do you have",
    "How many documents are indexed?",
    "list your documents",
    "what files can you read",
    "what is in your index",
    "which files do you have access to",
    "how many docs are loaded",
    "what policies do you know about",
]

#: Every one of these mentions the collection and is still a question about
#: content. Answering any of them from the index would be a regression, and a
#: worse failure than the one this module fixes.
ABOUT_THE_CONTENT = [
    "which documents mention parental leave",
    "how many days of parental leave",
    "what is the daily meal allowance",
    "who is the document owner",
    "what does the expenses policy say about alcohol",
    "which policy covers contractors",
    "how many documents mention GDPR",
    "what is the document retention period",
    "list the approval limits",
]


@pytest.mark.parametrize("question", ABOUT_THE_CORPUS)
def test_questions_about_the_collection_are_recognised(question: str) -> None:
    assert recognise(question) is not None


@pytest.mark.parametrize("question", ABOUT_THE_CONTENT)
def test_questions_about_content_are_left_alone(question: str) -> None:
    assert recognise(question) is None, "would hijack a real document question"


def test_the_answer_lists_what_is_there() -> None:
    text = describe(["Expenses Policy", "Travel Guidelines"], chunks=6)
    assert "2 documents indexed, in 6 passages" in text
    assert "- Expenses Policy" in text
    assert "- Travel Guidelines" in text


def test_an_empty_corpus_says_what_to_do_about_it() -> None:
    text = describe([], chunks=0)
    assert "no documents indexed" in text
    assert "openknowledge index" in text


def test_titles_are_access_controlled_like_search_is() -> None:
    """A title is not nothing.

    "Project Northstar - Redundancy Plan" tells you what it is without opening
    it, so listing titles without filtering would route around the ACL that
    retrieval respects.
    """
    retriever = BM25Retriever()
    retriever.index(
        [
            Document("public", "Expenses Policy", "Meals are reimbursed up to EUR 45."),
            Document(
                "secret",
                "Redundancy Plan",
                "Roles at risk in the coming quarter.",
                allowed_principals=frozenset({"hr-leadership"}),
            ),
        ]
    )

    everyone, hidden = retriever.documents_visible_to(frozenset({"all-staff"}))
    assert everyone == ["Expenses Policy"]
    assert hidden == 1

    leadership, hidden_from_them = retriever.documents_visible_to(
        frozenset({"hr-leadership", "all-staff"})
    )
    assert leadership == ["Expenses Policy", "Redundancy Plan"]
    assert hidden_from_them == 0

    assert describe(everyone, chunks=2, hidden=hidden).endswith("with the source.")
    assert "1 further document you do not have access to" in describe(
        everyone, chunks=2, hidden=hidden
    )


async def test_the_cascade_answers_it_without_calling_a_model(store, retriever, settings) -> None:
    """Free and instant, and it must not reach the ladder at all."""
    from tests.fakes import FakeProvider
    from tests.test_cascade import build

    local = FakeProvider(replies=["should never be called"])
    answer = await build(store, retriever, settings, local=local).answer(
        "what documents do you have?"
    )

    assert answer.tier is Tier.CORPUS
    assert answer.cost_usd == 0.0
    assert local.calls == [], "called a model for a question the index answers"
    assert "documents indexed" in answer.text
