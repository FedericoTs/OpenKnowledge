"""Whole-document questions: two votes to recognise one, structure to answer it.

The recogniser is tested on the sixteen field questions verbatim and on the
fact questions that must never match. The structure reader is tested on
hand-built blocks here and on the committed corpus in test_scope_corpus.py.
"""

from __future__ import annotations

import pytest

from openknowledge.cascade.scope import (
    Outline,
    assemble,
    outline,
    recognise_scope,
    render,
    section_chunks,
)
from openknowledge.documents.blocks import Block, BlockKind
from openknowledge.retrieval.base import Chunk, ScoredChunk


def _hit(doc: str, i: int = 0, title: str | None = None) -> ScoredChunk:
    return ScoredChunk(
        Chunk(chunk_id=f"{doc}#{i}", document_id=doc, document_title=title or doc, text="x"),
        score=1.0,
    )


def _concentrated(doc: str = "handbook", n: int = 6) -> list[ScoredChunk]:
    return [_hit(doc, i) for i in range(n)]


def _spread() -> list[ScoredChunk]:
    return [_hit(f"doc{i}", 0) for i in range(6)]


# --- the shape vote ------------------------------------------------------------


@pytest.mark.parametrize(
    ("question", "kind", "wants"),
    [
        ("what are the main characters from mythos", "enumerate", "persons"),
        ("what are the characters in the books", "enumerate", "persons"),
        ("what are the step by step activities we should cover for arvexlab", "enumerate", "items"),
        ("what are the priorities for arvexlab", "enumerate", "items"),
        ("what is the list of priorities we have in the file", "enumerate", "items"),
        ("what does the document covers", "enumerate", "any"),
        ("summarize the feedback from caterina", "summarise", "any"),
        ("what terms does the glossary of terms define?", "enumerate", "terms"),
        ("what are the chapters of the book?", "enumerate", "headings"),
        ("what does part 301-10 cover?", "enumerate", "headings"),
    ],
)
def test_field_questions_have_the_shape(question: str, kind: str, wants: str) -> None:
    scope = recognise_scope(question, _concentrated())
    assert scope is not None, question
    assert (scope.kind, scope.wants) == (kind, wants)


@pytest.mark.parametrize(
    "question",
    [
        "how much is the meal allowance",
        "what is the notice period",
        "how many days of annual leave do I get",
        "can I claim a taxi",
        "what transportation methods may an agency authorize",
        "list the authorized methods of transportation",
        "which documents mention parental leave",
        "who is the murderer in the book",
    ],
)
def test_fact_questions_never_match_even_when_hits_concentrate(question: str) -> None:
    """The control set stays on the ordinary path whatever retrieval did."""
    assert recognise_scope(question, _concentrated()) is None, question


# --- the target vote -----------------------------------------------------------


def test_shape_alone_is_not_enough() -> None:
    """Hits spread over six documents: no target, so nothing changes."""
    assert recognise_scope("what are the chapters?", _spread()) is None


def test_half_the_hits_from_one_document_is_a_target() -> None:
    hits = [_hit("a", 0), _hit("a", 1), _hit("a", 2), _hit("b", 0), _hit("c", 0), _hit("d", 0)]
    scope = recognise_scope("what are the chapters?", hits)
    assert scope is not None and scope.document_id == "a"


def test_two_of_six_is_not_a_target() -> None:
    hits = [_hit("a", 0), _hit("a", 1), _hit("b", 0), _hit("c", 0), _hit("d", 0), _hit("e", 0)]
    assert recognise_scope("what are the chapters?", hits) is None


def test_a_named_title_beats_the_ranking() -> None:
    hits = _spread()
    titles = {"alice": "Alice's Adventures in Wonderland", "earnest": "Earnest"}
    scope = recognise_scope("what are the chapters of alice in wonderland?", hits, titles=titles)
    assert scope is not None and scope.document_id == "alice"


def test_one_word_of_a_long_title_does_not_name_it() -> None:
    """A question about a person called Alice must not hijack the book."""
    titles = {"alice": "Alice's Adventures in Wonderland"}
    assert recognise_scope("what are alice's priorities?", _spread(), titles=titles) is None


def test_two_titles_named_is_no_target() -> None:
    titles = {"a": "Expenses Policy", "b": "Travel Policy"}
    q = "list the rules in the expenses policy and the travel policy"
    assert recognise_scope(q, _spread(), titles=titles) is None


# --- ordinals -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("question", "ordinal", "noun"),
    [
        ("what is the title of the second chapter?", 2, "chapter"),
        ("what is the title of chapter thirteen?", 13, "chapter"),
        ("what happens in the fourth act?", 4, "act"),
        ("what is priority 2?", 2, "priority"),
        ("what is the last chapter about?", -1, "chapter"),
        ("what is step 3 of the process?", 3, "step"),
    ],
)
def test_ordinals_carry_their_noun(question: str, ordinal: int, noun: str) -> None:
    scope = recognise_scope(question, _concentrated())
    assert scope is not None
    assert (scope.ordinal, scope.ordinal_noun) == (ordinal, noun)


def test_a_document_identifier_is_not_an_ordinal() -> None:
    """ "part 301-10" is a name, not the three-hundred-and-first part."""
    scope = recognise_scope("what does part 301-10 cover?", _concentrated())
    assert scope is not None and scope.ordinal is None


def test_a_summary_cue_wins_over_an_ordinal() -> None:
    """ "Summarise the first act" wants a summary; the ordinal says of what."""
    scope = recognise_scope("summarise the first act of the play", _concentrated())
    assert scope is not None
    assert scope.kind == "summarise"
    assert (scope.ordinal, scope.ordinal_noun) == (1, "act")


# --- structure ---------------------------------------------------------------------------


def _play() -> tuple[Block, ...]:
    h = BlockKind.HEADING
    li = BlockKind.LIST_ITEM
    p = BlockKind.PARAGRAPH
    return (
        Block(kind=h, text="THE PERSONS IN THE PLAY", level=2),
        Block(kind=li, text="John Worthing, J.P."),
        Block(kind=li, text="Algernon Moncrieff"),
        Block(kind=li, text="Lady Bracknell"),
        Block(kind=h, text="THE SCENES OF THE PLAY", level=2),
        Block(kind=p, text="ACT I. Algernon's flat."),
        Block(kind=h, text="FIRST ACT", level=2),
        Block(kind=p, text="Lane is arranging afternoon tea."),
        Block(kind=h, text="SECOND ACT", level=2),
        Block(kind=p, text="Garden at the Manor House."),
        Block(kind=h, text="THIRD ACT", level=2),
        Block(kind=p, text="Drawing-room."),
    )


def test_headings_are_the_sections() -> None:
    found = outline(_play(), "headings", "what are the sections?")
    assert found is not None
    assert found.items == (
        "THE PERSONS IN THE PLAY",
        "THE SCENES OF THE PLAY",
        "FIRST ACT",
        "SECOND ACT",
        "THIRD ACT",
    )


def test_an_ordinal_noun_narrows_the_headings_to_the_ones_that_say_it() -> None:
    """Asked about acts, the cast list is not one. This is what makes "the
    fourth act" a refusal instead of an answer of "SECOND ACT"."""
    found = outline(_play(), "headings", "the fourth act", noun="act")
    assert found is not None
    assert found.items == ("FIRST ACT", "SECOND ACT", "THIRD ACT")
    assert found.noun == "acts"


def test_a_noun_no_heading_says_means_no_answer() -> None:
    assert outline(_play(), "headings", "the second chapter", noun="chapter") is None


def test_a_lone_title_heading_is_not_a_section() -> None:
    blocks = (
        Block(kind=BlockKind.HEADING, text="Alice's Adventures", level=1),
        Block(kind=BlockKind.HEADING, text="CHAPTER I. Down the Rabbit-Hole", level=2),
        Block(kind=BlockKind.HEADING, text="CHAPTER II. The Pool of Tears", level=2),
    )
    found = outline(blocks, "headings", "what are the chapters?")
    assert found is not None and len(found.items) == 2


def test_all_headings_at_one_level_are_all_sections() -> None:
    """Twelve chapters and no title: the first chapter must not be dropped."""
    blocks = tuple(
        Block(kind=BlockKind.HEADING, text=f"CHAPTER {i}", level=2) for i in range(1, 13)
    )
    found = outline(blocks, "headings", "what are the chapters?")
    assert found is not None and len(found.items) == 12


def test_persons_come_from_the_list_under_the_heading_that_says_so() -> None:
    found = outline(_play(), "persons", "who are the persons in the play?")
    assert found is not None
    assert found.items == ("John Worthing, J.P.", "Algernon Moncrieff", "Lady Bracknell")
    assert found.under == "THE PERSONS IN THE PLAY"


def test_a_question_about_people_is_never_answered_from_an_unlabelled_list() -> None:
    blocks = (
        Block(kind=BlockKind.HEADING, text="Shopping", level=2),
        Block(kind=BlockKind.LIST_ITEM, text="milk"),
        Block(kind=BlockKind.LIST_ITEM, text="eggs"),
    )
    assert outline(blocks, "persons", "who are the characters?") is None


def test_items_come_from_the_matching_heading_or_the_only_list() -> None:
    blocks = (
        Block(kind=BlockKind.HEADING, text="Priorities for 2026", level=2),
        Block(kind=BlockKind.LIST_ITEM, text="Ship the installer"),
        Block(kind=BlockKind.LIST_ITEM, text="Sign it"),
        Block(kind=BlockKind.HEADING, text="Risks", level=2),
        Block(kind=BlockKind.LIST_ITEM, text="Quadratic index"),
    )
    found = outline(blocks, "items", "what are the priorities?")
    assert found is not None and found.items == ("Ship the installer", "Sign it")
    # Two lists and a question naming neither heading: no answer, not a guess.
    assert outline(blocks, "items", "what are the things?") is None


def test_terms_are_the_paragraphs_that_open_with_one() -> None:
    p = BlockKind.PARAGRAPH
    blocks = (
        Block(kind=p, text="Accompanied baggage. Government property of the traveler."),
        Block(kind=p, text="Actual expense. Payment of authorized actual expenses incurred."),
        Block(kind=p, text="Agency. For purposes of temporary duty allowances, agency means:"),
        Block(kind=p, text="This is an ordinary sentence and not a defined term at all."),
    )
    found = outline(blocks, "terms", "what terms are defined?")
    assert found is not None
    assert found.items == ("Accompanied baggage", "Actual expense", "Agency")


def test_no_structure_means_none_not_a_guess() -> None:
    blocks = (Block(kind=BlockKind.PARAGRAPH, text="Just prose."),)
    assert outline(blocks, "any", "what does it cover?") is None


# --- rendering -------------------------------------------------------------------------------


def test_the_whole_list_is_rendered_with_a_citation() -> None:
    text, refused = render(
        Outline(items=("A", "B"), noun="chapters"), title="Book", document_id="book", ordinal=None
    )
    assert not refused
    assert text == "Book has 2 chapters:\n  1. A\n  2. B\n[book]"


def test_an_ordinal_in_range_names_the_item() -> None:
    text, refused = render(
        Outline(items=("A", "B", "C"), noun="acts"), title="Play", document_id="p", ordinal=2
    )
    assert not refused and "second of the 3 acts" in text and "“B”" in text


def test_an_ordinal_past_the_end_refuses_and_says_how_many() -> None:
    text, refused = render(
        Outline(items=("A", "B", "C"), noun="acts"), title="Play", document_id="p", ordinal=4
    )
    assert refused
    assert text == "Play has 3 acts; there is no fourth one. [p]"


def test_last_means_the_end_of_the_list() -> None:
    text, refused = render(
        Outline(items=("A", "B", "C"), noun="acts"), title="Play", document_id="p", ordinal=-1
    )
    assert not refused and "“C”" in text


# --- assembly ---------------------------------------------------------------------------------


def _chunks(n: int, words: int = 350) -> list[Chunk]:
    body = " ".join(["word"] * words)
    return [
        Chunk(chunk_id=f"d#{i}", document_id="d", document_title="D", text=body, section="S")
        for i in range(n)
    ]


def test_assembly_fits_as_many_passages_as_the_window_allows() -> None:
    # 8192 * 0.9 = 7372 tokens; minus 2600 reserved = 4772 tokens = 19,088 chars.
    # Each passage is ~1,750 chars + 80 header, so ten fit and twelve do not.
    kept, left = assemble(_chunks(22), context_tokens=8192, reserved_tokens=2600, fallback_max=18)
    assert 9 <= len(kept) <= 11
    assert left == 22 - len(kept)
    assert [c.chunk_id for c in kept] == [f"d#{i}" for i in range(len(kept))], "in order"


def test_an_unknown_window_caps_at_the_fallback_not_the_document() -> None:
    kept, left = assemble(_chunks(22), context_tokens=0, reserved_tokens=0, fallback_max=18)
    assert (len(kept), left) == (18, 4)


def test_at_least_one_passage_is_always_kept() -> None:
    kept, _ = assemble(_chunks(3), context_tokens=1000, reserved_tokens=990, fallback_max=18)
    assert len(kept) == 1


def test_section_chunks_follow_the_heading_trail() -> None:
    chunks = [
        Chunk(
            chunk_id="d#0",
            document_id="d",
            document_title="D",
            text="a",
            section="Play > FIRST ACT",
        ),
        Chunk(
            chunk_id="d#1",
            document_id="d",
            document_title="D",
            text="b",
            section="Play > SECOND ACT",
        ),
        Chunk(
            chunk_id="d#2",
            document_id="d",
            document_title="D",
            text="c",
            section="Play > FIRST ACT",
        ),
    ]
    assert [c.chunk_id for c in section_chunks(chunks, "FIRST ACT")] == ["d#0", "d#2"]


# --- the noun in the question, without an ordinal --------------------------------------


def test_asking_for_chapters_of_a_play_lists_nothing() -> None:
    """A play has acts. Its headings are not the answer to "what are the chapters?"."""
    scope = recognise_scope("what are the chapters?", _concentrated())
    assert scope is not None and scope.heading_noun == "chapter"
    assert outline(_play(), scope.wants, "what are the chapters?", noun=scope.noun) is None


def test_asking_for_acts_of_a_play_lists_the_acts_and_calls_them_acts() -> None:
    scope = recognise_scope("what are the acts of the play?", _concentrated())
    assert scope is not None
    found = outline(_play(), scope.wants, "what are the acts?", noun=scope.noun)
    assert found is not None and found.items == ("FIRST ACT", "SECOND ACT", "THIRD ACT")
    assert found.noun == "acts"


def test_generic_heading_words_do_not_filter() -> None:
    """ "Sections" of a regulation are headed with a section sign, not the word."""
    scope = recognise_scope("what are the sections?", _concentrated())
    assert scope is not None and scope.heading_noun is None


def test_defined_is_an_enumeration_cue() -> None:
    scope = recognise_scope("what terms are defined?", _concentrated())
    assert scope is not None and scope.wants == "terms"
