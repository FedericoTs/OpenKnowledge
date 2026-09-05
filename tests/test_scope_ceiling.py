"""What retrieval can show the model when the answer is a whole document.

The `/manage` gap report on a real install listed sixteen unanswered questions
and eleven were one shape: enumerate a list, name the characters, summarise a
document. They route to retrieval correctly. They fail after it, because the
answer is spread over more chunks than `retrieval_k` returns.

`evals/golden-scope/README.md` has the measurement. These pin the two ends of
it so a change to chunking, ranking or tag routing cannot quietly move them:
the control must stay complete, and the glossary must stay a long way from it.
No model is involved - this is the ceiling, not the answer.
"""

from __future__ import annotations

import pytest
from tools.measure_scope import (  # noqa: PLC2701 - the harness is the subject
    glossary_terms,
    load_corpus,
    transportation_methods,
)

from openknowledge.retrieval.bm25 import BM25Retriever


@pytest.fixture(scope="module")
def retriever() -> BM25Retriever:
    r = BM25Retriever()
    r.index(load_corpus())
    return r


def _visible(retriever: BM25Retriever, question: str, expected: list[str], k: int) -> int:
    seen = " ".join(s.chunk.text for s in retriever.search(question, k=k)).lower()
    return sum(1 for item in expected if item.lower() in seen)


def test_the_reference_list_is_the_regulations_own(retriever: BM25Retriever) -> None:
    """82 terms, read out of the corpus rather than typed here.

    A hand-copied list drifts from the corpus when the corpus is refreshed,
    and drifts in the direction that flatters the score.
    """
    terms = glossary_terms()
    assert len(terms) == 82
    assert terms[0] == "Accompanied baggage"
    assert terms[-1] == "Usually traveled route"


def test_shown_everything_the_measure_finds_everything(retriever: BM25Retriever) -> None:
    """The sabotage that makes the shortfall believable.

    If coverage cannot reach 100% with every chunk in hand, the extractor or
    the substring check is wrong and every other number here is noise.
    """
    terms = glossary_terms()
    assert _visible(retriever, "what terms does the glossary define", terms, k=92) == 82


def test_a_whole_document_enumeration_is_mostly_invisible(retriever: BM25Retriever) -> None:
    """At the shipped budget the model sees about a third of the answer.

    Pinned as a band, not a point: the exact figure moves with chunking, and
    asserting 26 exactly would fail on an unrelated improvement. What must not
    change silently is that this is nowhere near complete.
    """
    terms = glossary_terms()
    found = _visible(retriever, "what terms does the glossary define", terms, k=6)
    assert found < 40, f"expected well under half of 82 terms at k=6, saw {found}"


def test_more_chunks_do_not_rescue_it(retriever: BM25Retriever) -> None:
    """k=50 is eight times the budget and over half the corpus, and still short.

    This is the load-bearing result: the obvious fix does not work, because
    BM25 ranks by term overlap and the query shares no vocabulary with the
    terms it is asking to enumerate.
    """
    terms = glossary_terms()
    assert _visible(retriever, "what terms does the glossary define", terms, k=50) < 82


def test_a_list_inside_one_chunk_is_complete(retriever: BM25Retriever) -> None:
    """The control. If this ever drops, retrieval broke - not scope."""
    methods = transportation_methods()
    for question in (
        "what transportation methods are authorized",
        "list the authorized methods of transportation",
    ):
        assert _visible(retriever, question, methods, k=6) == len(methods), question


def test_the_control_is_not_passing_on_ambient_vocabulary(retriever: BM25Retriever) -> None:
    """ "common carrier" is ordinary phrasing in a travel regulation.

    So the control has to be shown to depend on retrieval finding § 301-10.2,
    not on those words being everywhere. Its floor is chance, not zero.
    """
    methods = transportation_methods()
    unrelated = _visible(retriever, "what is the per diem rate for lodging", methods, k=6)
    assert unrelated < len(methods)
