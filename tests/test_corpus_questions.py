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


def test_the_first_question_every_new_user_asks_is_free() -> None:
    """ "What can you help me with?" mentions no document, and in the field it
    cost a frontier call and came back "that isn't covered by the documents I
    have" - the worst possible first impression. It is a question about the
    assistant, answered from the index like every other meta-question."""
    for question in (
        "what can you help me with?",
        "how can you help me?",
        "what are you able to do?",
    ):
        got = recognise(question)
        assert got is not None, question
        assert got.wants == "help", question


def test_a_capability_question_with_a_real_subject_is_not_hijacked() -> None:
    assert recognise("how can you help me claim travel expenses?") is None
    assert recognise("can you help me with the parental leave policy?") is None


def test_topic_questions_are_about_the_collection() -> None:
    """Field case, verbatim typo included: it went to a frontier model and was
    refused for a derived count the gate could never verify."""
    for question in (
        "what are the macro-categories of info they covers?",
        "what topics do you cover?",
        "what categories of information do you have?",
    ):
        assert recognise(question) is not None, question


def test_the_capability_answer_leads_with_what_the_assistant_does() -> None:
    text = describe(
        ["Expenses Policy"],
        chunks=6,
        tags={"Expenses Policy": ("expenses", "policy", "meals")},
        wants="help",
    )
    assert text.startswith("I answer questions from the documents indexed here")
    assert "covering: expenses, policy, meals" in text


def test_tags_answer_what_topics_are_covered() -> None:
    text = describe(
        ["Expenses Policy", "Security Policy"],
        chunks=12,
        tags={"Expenses Policy": ("expenses", "meals"), "Security Policy": ("security", "usb")},
    )
    assert "Expenses Policy - covering: expenses, meals" in text
    assert "Security Policy - covering: security, usb" in text


def test_what_can_you_do_is_about_the_assistant() -> None:
    """Field case: this reached the frontier model, which answered as if it
    were the organisation in the documents - grounded, cited, and the wrong
    speaker entirely. No subject plus "you" is a question about the assistant."""
    for question in ("what can you do for me?", "what do you do?", "who are you?"):
        got = recognise(question)
        assert got is not None and got.wants == "help", question


def test_do_you_summarize_documents_is_a_capability_question() -> None:
    """Field case, both sentences verbatim: an honest refusal that still cost
    a frontier call, for a question about the assistant's own features."""
    got = recognise("do you summarize documents? any document I provide you?")
    assert got is not None and got.wants == "help"
    assert recognise("can you compare documents?") is not None


def test_an_imperative_is_work_not_a_capability_question() -> None:
    """ "Summarize the documents" is an instruction. Sending it to the listing
    would swallow real work - the you-frame never opens for an imperative."""
    assert recognise("summarize the documents") is None
    assert recognise("summarise the expenses policy") is None


def test_a_you_question_with_a_named_subject_still_goes_to_retrieval() -> None:
    assert recognise("do you cover parental leave in the documents?") is None
    assert recognise("can you summarize the caterina debrief?") is None


def test_the_help_answer_mentions_summarising() -> None:
    text = describe(["Expenses Policy"], chunks=6, wants="help")
    assert "summarise" in text


# -- "what files can you manage?" (field regression) --------------------------
#
# Asked by every new user, and the worst possible outcome: it fell out of the
# free tier on the verb alone - "files" was already meta vocabulary, "manage"
# was not - escalated to a paid frontier call, and was refused.


def test_asking_what_files_can_be_managed_is_free() -> None:
    got = recognise("what files can you manage?")
    assert got is not None and got.wants == "help"


def test_asking_about_supported_formats_is_free() -> None:
    for question in (
        "what file types do you support?",
        "what documents can you read?",
        "what formats can you handle?",
    ):
        got = recognise(question)
        assert got is not None and got.wants == "help", question


def test_the_help_answer_names_the_formats_it_can_parse() -> None:
    """The question deserves the actual list, read from the registry that
    does the parsing so the promise cannot drift from the code."""
    from openknowledge.documents import SUPPORTED_SUFFIXES

    answer = describe(["Handbook"], chunks=3, wants="help")
    assert "PDF" in answer and "Word" in answer and "Excel" in answer
    assert ".pdf" not in answer, "say PDF, not .pdf - this is prose, not a file dialog"
    assert ".xlsx" in SUPPORTED_SUFFIXES  # the registry still backs the claim


def test_managing_verbs_do_not_hijack_real_work() -> None:
    """The verbs are spent only in the you-frame. An imperative, or a
    question with a real subject, still goes to retrieval."""
    for question in (
        "manage the vendor list",
        "read the contract and tell me the notice period",
        "what files did Giuseppe send about the website?",
        "can you read the expenses policy?",
    ):
        assert recognise(question) is None, question


def test_one_named_document_is_not_the_whole_shelf() -> None:
    """"What does this document cover?" is about a document, not the index.

    This is the worst failure the corpus tier can produce, and the only one
    it produces silently. Every other question in the gap report came back a
    refusal; this one came back a confident inventory of the whole server,
    marked grounded, from the first branch of the cascade - before the cache,
    before retrieval, before the contradiction check, and before anything that
    could have disagreed with it. A person asks what they just uploaded and is
    told what else is on the shelf.
    """
    for question in (
        "what does the document covers",
        "what does the document cover",
        "what does this document cover",
        "what is in the document",
        "what does the file contain",
        "what does that report say",
        "what does the handbook cover",
    ):
        assert recognise(question) is None, question


def test_the_collection_still_answers_for_itself() -> None:
    """The other side of the same line, and the reason the fix is narrow.

    "What are the macro-categories of info they covers?" is the field
    question that put "cover"/"covers" into the no-subject vocabulary in the
    first place. Fixing the singular case by dropping those words again would
    trade one field failure for the other.
    """
    for question in (
        "what do the documents cover",
        "what documents do you have",
        "what topics do the documents cover",
        "what are the macro-categories of info they covers",
        "how many documents do you have",
    ):
        assert recognise(question) is not None, question
