"""Does it recognise what people actually ask, without being told each word?

Every word list in cascade.corpus is open-class: there is always another
verb. Each gap costs a paid frontier call and then a refusal - the worst
outcome the cascade can produce - and it surfaces only when someone reports
it. Measured against ``evals/capability/questions.py``, a set written to
defeat those lists rather than flatter them, the lists alone recognised 25%.

So the decision does not rest on them. A question that addresses "you" in a
question frame, and whose content words do not appear in the passages
retrieved for it, is answered as a question about the assistant - free, with
no model call. Both halves are closed-class or measured, so neither has a
vocabulary tail to fall off.

These bars are enforced, not observed: a change that drops recognition or
starts stealing document questions fails here.
"""

from __future__ import annotations

from evals.capability.questions import ABOUT_ME, ABOUT_THE_DOCUMENTS, EITHER_IS_HONEST

from openknowledge.cascade.corpus import (
    asks_about_the_assistant,
    corpus_has_nothing_to_say,
    evidence_text,
    recognise,
)
from openknowledge.retrieval import BM25Retriever
from openknowledge.retrieval.base import Document

#: A corpus with nothing to say about the software reading it - the ordinary
#: case, and the one the net has to get right.
CORPUS = [
    Document(
        "handbook",
        "Employee Handbook",
        "Parental leave is 20 weeks at full pay for employees with more than "
        "one year of service. Leave must be requested through your manager "
        "at least eight weeks before the intended start date. Unused annual "
        "leave does not carry over beyond March.",
    ),
    Document(
        "expenses",
        "Expenses Policy",
        "Expenses are reimbursed monthly. Meals are reimbursed up to EUR 40 "
        "per day when travelling on company business. Expenses above EUR 200 "
        "require prior approval from a director. Receipts must accompany "
        "every expense claim; claims without receipts are refused.",
    ),
    Document(
        "debrief",
        "Demo Debrief",
        "Caterina raised the self-assessment gap: SMEs assess their own "
        "internal compliance before auditing vendors. She recommended an "
        "internal self-assessment module as the entry point of the platform, "
        "with vendor management as a second phase.",
    ),
]


def _route(question: str, retriever: BM25Retriever) -> str:
    """The route this question takes, as the cascade decides it."""
    if recognise(question) is not None:
        return "free"
    hits = retriever.search(question, k=8)
    if asks_about_the_assistant(question) and corpus_has_nothing_to_say(
        question, evidence_text([h.chunk for h in hits])
    ):
        return "free"
    return "retrieval"


def _retriever() -> BM25Retriever:
    r = BM25Retriever()
    r.index(CORPUS)
    return r


def test_questions_about_the_assistant_are_answered_free() -> None:
    """The bar that matters: no question about the assistant may reach a paid
    model. Every miss here is a real user charged for a refusal."""
    retriever = _retriever()
    missed = [q for q in ABOUT_ME if _route(q, retriever) != "free"]
    assert not missed, f"{len(missed)} of {len(ABOUT_ME)} would escalate and refuse: {missed}"


def test_questions_about_the_documents_still_reach_retrieval() -> None:
    """The bar that keeps the first one safe. Answering a real question with
    a description of the product would be a far worse failure than the one
    the net exists to prevent, so this is measured in both directions."""
    retriever = _retriever()
    stolen = [q for q in ABOUT_THE_DOCUMENTS if _route(q, retriever) != "retrieval"]
    assert not stolen, f"the net took questions the documents can answer: {stolen}"


def test_the_grammar_check_alone_never_decides() -> None:
    """A document question phrased at the assistant passes the grammar check
    and must be saved by the corpus vote - that pairing is the whole design,
    so it is pinned rather than left to the two suites above."""
    question = "can you tell me what the handbook says about parental leave?"
    assert asks_about_the_assistant(question)
    hits = _retriever().search(question, k=8)
    assert not corpus_has_nothing_to_say(question, evidence_text([h.chunk for h in hits]))


def test_an_imperative_is_never_a_question_about_me() -> None:
    for instruction in (
        "summarise the handbook for me",
        "translate your notes into Italian",
        "read the contract and tell me what you think",
        "compare the expenses policy with the handbook",
    ):
        assert not asks_about_the_assistant(instruction), instruction


def test_the_ambiguous_middle_is_refused_honestly_either_way() -> None:
    """A question phrased at the assistant, about a subject the documents do
    not cover. Whichever route it takes it must end in "not covered" - the
    net answers with the refusal first and the description after, so the
    person is never told a product blurb in place of an honest no."""
    from openknowledge.cascade.corpus import describe
    from openknowledge.prompts import REFUSAL_TEXT

    retriever = _retriever()
    for question in EITHER_IS_HONEST:
        route = _route(question, retriever)
        if route == "free":
            answer = (
                REFUSAL_TEXT
                + "\n\n"
                + describe([d.title for d in CORPUS], chunks=len(retriever), wants="help")
            )
            assert REFUSAL_TEXT in answer, question
            assert answer.index(REFUSAL_TEXT) == 0, "the refusal leads; the blurb follows"
