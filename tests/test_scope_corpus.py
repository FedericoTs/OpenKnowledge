"""The structure reader against the committed corpus, through the real parser.

test_scope_questions.py proves the reader on hand-built blocks. This proves it on the
documents evals/golden-scope actually asks about, parsed the way the connector
parses them - so a change to the Markdown parser that stopped producing list
items or heading levels would fail here, not in a model run an hour later.
"""

from __future__ import annotations

import pathlib

from tools.measure_scope import glossary_terms

from openknowledge.cascade.scope import outline, render
from openknowledge.documents import parse_file
from openknowledge.evaluation.dataset import load_cases

SCOPE = pathlib.Path(__file__).resolve().parents[1] / "evals" / "golden-scope"
DOCS = SCOPE / "documents"


def _blocks(name: str):
    parsed = parse_file(DOCS / name)
    assert parsed.blocks, f"{name} parsed to no blocks"
    return parsed.blocks


def _case(case_id: str):
    return next(c for c in load_cases(SCOPE / "scope.yaml") if c.id == case_id)


def test_alice_has_twelve_chapters_and_they_are_the_reference_list() -> None:
    found = outline(_blocks("alice.md"), "headings", "what are the chapters?")
    assert found is not None and len(found.items) == 12
    for heading, title in zip(found.items, _case("scope-03-alice-chapters").must_list, strict=True):
        assert title in heading, (title, heading)


def test_the_second_chapter_is_the_pool_of_tears() -> None:
    found = outline(_blocks("alice.md"), "headings", "the second chapter", noun="chapter")
    assert found is not None
    text, refused = render(found, title="Alice", document_id="alice", ordinal=2)
    assert not refused and "Pool of Tears" in text


def test_chapter_thirteen_is_refused_with_the_count() -> None:
    found = outline(_blocks("alice.md"), "headings", "chapter thirteen", noun="chapter")
    assert found is not None
    text, refused = render(found, title="Alice", document_id="alice", ordinal=13)
    assert refused and "has 12 chapters" in text


def test_the_persons_in_the_play_are_wildes_nine() -> None:
    found = outline(_blocks("earnest.md"), "persons", "who are the persons in the play?")
    assert found is not None
    assert found.under == "THE PERSONS IN THE PLAY"
    assert len(found.items) == 9
    for key, printed in zip(
        _case("scope-02-persons-in-the-play").must_list, found.items, strict=True
    ):
        assert key in printed, (key, printed)


def test_the_fourth_act_is_refused_because_there_are_three() -> None:
    """The case that would have answered "SECOND ACT" without the noun filter."""
    found = outline(_blocks("earnest.md"), "headings", "the fourth act", noun="act")
    assert found is not None and found.items == ("FIRST ACT", "SECOND ACT", "THIRD ACT")
    text, refused = render(found, title="Earnest", document_id="earnest", ordinal=4)
    assert refused and text.startswith("Earnest has 3 acts; there is no fourth one.")


def test_the_glossary_terms_are_the_regulations_82() -> None:
    found = outline(_blocks("ftr-300-1.md"), "terms", "what terms does the glossary define?")
    assert found is not None
    assert list(found.items) == glossary_terms()


def test_part_301_10_covers_its_forty_sections() -> None:
    found = outline(_blocks("ftr-301-10.md"), "headings", "what does part 301-10 cover?")
    assert found is not None and len(found.items) >= 40
    for title in _case("scope-05-part-301-10-sections").must_list:
        assert any(title in heading for heading in found.items), title


def test_wants_any_reads_headings_first() -> None:
    """ "What does this document cover?" names no noun; the sections answer it."""
    found = outline(_blocks("alice.md"), "any", "what does the document cover?")
    assert found is not None and found.noun == "sections" and len(found.items) == 12
