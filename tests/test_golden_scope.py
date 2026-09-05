"""The scope set's reference answers come from its documents, and stay there.

evals/golden-scope asks whether the system can find *all* of something. Every
list it scores against was read out of the corpus by the script that wrote the
YAML: Wilde's cast list, Carroll's chapter titles, the terms the regulation
defines. These tests regenerate each one and fail if the YAML has drifted from
the documents - the direction such drift takes is always the one that flatters
the score.
"""

from __future__ import annotations

import pathlib
import re

from tools.gutenberg_to_md import convert
from tools.measure_scope import glossary_terms

from openknowledge.evaluation.dataset import load_cases

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCOPE = ROOT / "evals" / "golden-scope"
DOCS = SCOPE / "documents"


def _case(case_id: str):
    return next(c for c in load_cases(SCOPE / "scope.yaml") if c.id == case_id)


def test_the_ftr_copies_are_byte_identical_to_golden_ftr() -> None:
    """Two parts are copied so the set runs off one documents dir. Copied, not
    forked: a divergence would make two sets disagree about one regulation."""
    for name in ("ftr-300-1.md", "ftr-301-10.md"):
        assert (DOCS / name).read_bytes() == (
            ROOT / "evals" / "golden-ftr" / "documents" / name
        ).read_bytes(), name


def test_the_converter_is_idempotent_and_changes_no_words() -> None:
    """Structure markers become headings; the text is otherwise the book."""
    src = (SCOPE / "sources" / "pg11-alice.txt").read_text(encoding="utf-8")
    once = convert(src)
    assert convert(once) == once, "a second pass must be a no-op"
    assert once == (DOCS / "alice.md").read_text(encoding="utf-8"), "committed copy has drifted"

    def words(text: str) -> list[str]:
        return re.sub(r"^## |^- |^\*\*\*.*$", "", text, flags=re.M).split()

    body = src.replace("\r\n", "\n").split("*** START")[1].split("\n", 1)[1].split("*** END")[0]
    assert words(once) == words(body)


def test_the_glossary_list_is_the_regulations_own() -> None:
    assert list(_case("scope-04-glossary-terms").must_list) == glossary_terms()


def test_the_chapter_list_is_carrolls_own() -> None:
    alice = (DOCS / "alice.md").read_text(encoding="utf-8")
    chapters = re.findall(r"^## CHAPTER [IVX]+\. (.+)$", alice, re.M)
    assert len(chapters) == 12
    assert list(_case("scope-03-alice-chapters").must_list) == chapters


def test_the_cast_list_is_wildes_own() -> None:
    earnest = (DOCS / "earnest.md").read_text(encoding="utf-8")
    block = earnest.split("## THE PERSONS IN THE PLAY")[1].split("## ")[0]
    printed = [line[2:].strip() for line in block.strip().splitlines() if line.startswith("- ")]
    assert len(printed) == 9
    # The YAML keys each person by the name a summary would use; every key must
    # be a substring of the line Wilde printed, in order.
    for key, line in zip(_case("scope-02-persons-in-the-play").must_list, printed, strict=True):
        assert key in line, (key, line)


def test_the_documents_are_larger_than_the_budget() -> None:
    """The whole point of this corpus. Six chunks of ~350 words is ~2,100
    words; the two books are ten times that."""
    for name in ("alice.md", "earnest.md"):
        assert len((DOCS / name).read_text(encoding="utf-8").split()) > 20_000, name


def test_refusals_share_vocabulary_with_answerable_cases() -> None:
    """ "The fourth act" of a three-act play must be refused with the same
    machinery that answers "the second chapter", or the ordinal path invents."""
    cases = load_cases(SCOPE / "scope.yaml")
    refusals = [c for c in cases if c.kind == "refusal"]
    assert len(refusals) == 3
    assert any("fourth act" in c.question.lower() for c in refusals)
    assert any("thirteen" in c.question.lower() for c in refusals)
