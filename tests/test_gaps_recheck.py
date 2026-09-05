"""A gap the reader has already closed is not a gap.

The ledger records what happened when a question was asked, which is the right
source for "this was not answered" and the wrong one for "this is not
answerable". Measured on a real install: of sixteen questions listed, four were
answered free at the time of reading and had been for a week - the row clears
only when somebody asks the question again, and nobody re-asks a question that
failed them once.
"""

from __future__ import annotations

import pytest

from openknowledge.documents.blocks import Block, BlockKind
from openknowledge.gaps import free_tier_for, mark_answerable
from openknowledge.retrieval.base import Document
from openknowledge.retrieval.bm25 import BM25Retriever

HANDBOOK = """# Field Handbook

## Priorities
- Ship the installer
- Sign it
- Publish to PyPI

## Notice period
Three months for permanent staff.
"""


@pytest.fixture
def retriever() -> BM25Retriever:
    parsed = (
        Block(kind=BlockKind.HEADING, text="Field Handbook", level=1),
        Block(kind=BlockKind.HEADING, text="Priorities", level=2, heading_path=("Field Handbook",)),
        Block(kind=BlockKind.LIST_ITEM, text="Ship the installer"),
        Block(kind=BlockKind.LIST_ITEM, text="Sign it"),
        Block(kind=BlockKind.LIST_ITEM, text="Publish to PyPI"),
        Block(kind=BlockKind.HEADING, text="Notice period", level=2),
        Block(kind=BlockKind.PARAGRAPH, text="Three months for permanent staff."),
    )
    r = BM25Retriever()
    r.index(
        [Document(document_id="handbook", title="Field Handbook", text=HANDBOOK, blocks=parsed)]
    )
    return r


def test_a_question_about_the_collection_is_answered_now(retriever: BM25Retriever) -> None:
    """The measured case: these four were fixed on 2026-08-29 and still listed."""
    for question in (
        "what documents do you have",
        "what files can you manage",
        "what can you help me with",
    ):
        assert free_tier_for(question, retriever) == "corpus", question


def test_a_question_the_structure_now_answers_is_closed(retriever: BM25Retriever) -> None:
    """ "What are the priorities?" was a gap until the outline tier existed."""
    assert free_tier_for("what are the priorities?", retriever) == "outline"


def test_a_genuine_gap_stays_open(retriever: BM25Retriever) -> None:
    """The handbook says nothing about parental leave, and no free tier
    pretends otherwise. This is the direction that matters: a report which
    quietly closes real gaps is worse than one that keeps stale rows."""
    assert free_tier_for("how many weeks of parental leave do I get?", retriever) is None


def test_a_fact_question_the_documents_do_cover_is_not_closed(retriever: BM25Retriever) -> None:
    """It is in the handbook, so it was never a corpus or outline question -
    if it was recorded as a gap, a model refused it and only a model can say
    it would not now."""
    assert free_tier_for("what is the notice period?", retriever) is None


def test_marking_keeps_every_row_and_its_counts(retriever: BM25Retriever) -> None:
    rows = [
        {"question": "what documents do you have", "asked": 4, "kind": "refused"},
        {"question": "how many weeks of parental leave do I get?", "asked": 9, "kind": "refused"},
    ]
    marked = mark_answerable(rows, retriever)
    assert [r["asked"] for r in marked] == [4, 9], "counts survive"
    assert [r["answered_now"] for r in marked] == ["corpus", None]
