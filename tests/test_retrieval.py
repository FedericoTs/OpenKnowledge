"""Retrieval, chunking, access control, and the grounding gate."""

from __future__ import annotations

from openknowledge.retrieval import BM25Retriever, Document, check_grounding, chunk_document
from openknowledge.retrieval.base import Chunk


def test_finds_the_right_document(retriever: BM25Retriever) -> None:
    hits = retriever.search("how much parental leave do I get", k=3)
    assert hits and hits[0].chunk.document_id == "hr-handbook"


def test_ranking_is_stable_across_runs(retriever: BM25Retriever, documents) -> None:
    """Identical questions must retrieve identical context, or the cache is a lie."""
    first = [h.chunk.chunk_id for h in retriever.search("expenses approval", k=5)]
    other = BM25Retriever()
    other.index(list(reversed(documents)))
    assert [h.chunk.chunk_id for h in other.search("expenses approval", k=5)] == first


def test_corpus_version_is_content_addressed(documents) -> None:
    a, b = BM25Retriever(), BM25Retriever()
    a.index(documents)
    b.index(list(reversed(documents)))
    assert a.corpus_version == b.corpus_version, "re-sync order must not invalidate the cache"

    edited = [*documents[:-1], Document("board-comp", "Board Compensation", "changed text")]
    c = BM25Retriever()
    c.index(edited)
    assert c.corpus_version != a.corpus_version


def test_restricted_documents_are_filtered_during_scoring(retriever: BM25Retriever) -> None:
    staff = retriever.search("executive salary bands", k=5, principals=frozenset({"staff"}))
    assert all(h.chunk.document_id != "board-comp" for h in staff)

    board = retriever.search("executive salary bands", k=5, principals=frozenset({"board"}))
    assert any(h.chunk.document_id == "board-comp" for h in board)


def test_removing_a_document_removes_it_from_results(documents) -> None:
    r = BM25Retriever()
    r.index(documents)
    assert r.search("alcohol reimbursable", k=3)
    r.index([d for d in documents if d.document_id != "expenses"])
    assert all(h.chunk.document_id != "expenses" for h in r.search("alcohol reimbursable", k=3))


def test_chunks_overlap_so_split_rules_stay_retrievable() -> None:
    doc = Document("d", "D", " ".join(f"w{i}" for i in range(1000)))
    chunks = chunk_document(doc, target_words=100, overlap_words=20)
    assert len(chunks) > 1
    first_words = chunks[0].text.split()
    second_words = chunks[1].text.split()
    assert first_words[-20:] == second_words[:20]


def test_empty_document_yields_no_chunks() -> None:
    assert chunk_document(Document("d", "D", "   ")) == []


def test_empty_index_returns_nothing_rather_than_erroring() -> None:
    r = BM25Retriever()
    assert r.search("anything") == []
    assert r.corpus_version == "empty"


# -- grounding ------------------------------------------------------------


def _chunks(retriever: BM25Retriever):
    return [h.chunk for h in retriever.search("parental leave", k=3)]


def test_grounded_answer_passes(retriever: BM25Retriever) -> None:
    report = check_grounding(
        "Employees with 12 months of continuous service get 20 weeks of fully paid "
        "parental leave. [hr-handbook]",
        _chunks(retriever),
    )
    assert report.passed, report.reasons
    assert report.cited_ids == ("hr-handbook",)


def test_invented_number_is_caught(retriever: BM25Retriever) -> None:
    """The most damaging error in a policy bot, and the cheapest to detect."""
    report = check_grounding(
        "Employees get 26 weeks of paid leave. [hr-handbook]", _chunks(retriever)
    )
    assert not report.passed
    assert report.unsupported_numbers == ("26",)


def test_invented_source_is_caught(retriever: BM25Retriever) -> None:
    report = check_grounding("You get 20 weeks. [employee-benefits-2024]", _chunks(retriever))
    assert not report.passed
    assert report.unknown_ids == ("employee-benefits-2024",)


def test_uncited_answer_is_rejected(retriever: BM25Retriever) -> None:
    report = check_grounding("You get 20 weeks of fully paid parental leave.", _chunks(retriever))
    assert not report.passed
    assert any("cites no sources" in r for r in report.reasons)


def test_abstention_is_flagged_but_not_treated_as_a_lie(retriever: BM25Retriever) -> None:
    report = check_grounding("I don't know - that isn't covered.", _chunks(retriever))
    assert not report.passed
    assert report.abstained


def test_fluent_invention_fails_the_overlap_check(retriever: BM25Retriever) -> None:
    report = check_grounding(
        "Our organisation deeply values work-life harmony and encourages colleagues to "
        "discuss flexible sabbatical arrangements with their designated wellbeing "
        "partner during quarterly check-ins. [hr-handbook]",
        _chunks(retriever),
    )
    assert not report.passed


def test_number_formatting_differences_are_not_false_positives(retriever: BM25Retriever) -> None:
    r = BM25Retriever()
    r.index([Document("d", "D", "The cap is EUR 1200 per quarter.")])
    chunks = [h.chunk for h in r.search("cap", k=1)]
    assert (
        check_grounding("The cap is EUR 1,200 per quarter. [d]", chunks).unsupported_numbers == ()
    )


def test_empty_answer_is_rejected(retriever: BM25Retriever) -> None:
    assert not check_grounding("   ", _chunks(retriever)).passed


# -- structure-aware chunking ----------------------------------------------

from openknowledge.documents import parse_text  # noqa: E402
from openknowledge.retrieval.base import chunk_blocks  # noqa: E402

POLICY_MD = """# Expenses Policy

## Approval thresholds

Any single expense above EUR 500 requires prior written approval.

| Grade | Meal allowance | Notice |
|---|---|---|
| Junior | EUR 35 | 5 days |
| Senior | EUR 45 | 2 days |

## Meals

Meals are reimbursed up to EUR 45 per day when travelling.
"""


def parsed_document(text: str = POLICY_MD, doc_id: str = "expenses") -> Document:
    parsed = parse_text(text)
    return Document(doc_id, parsed.title or doc_id, parsed.text, blocks=parsed.blocks)


def test_a_heading_starts_a_new_chunk() -> None:
    """Content under different headings is about different things; merging them
    lets retrieval return a chunk whose heading contradicts half its body."""
    chunks = chunk_document(parsed_document())
    meals = [c for c in chunks if "Meals" in c.text]
    assert meals, "expected a chunk for the Meals section"
    assert all("Approval thresholds" not in c.text for c in meals)


def test_table_rows_are_never_split(retriever) -> None:
    """A row cut in half is a number with no label attached."""
    chunks = chunk_document(parsed_document(), target_words=8, overlap_words=0)
    for chunk in chunks:
        for line in chunk.text.split("\n"):
            if "Meal allowance:" in line:
                assert "Grade:" in line and "Notice:" in line


def test_every_chunk_carries_its_heading_trail() -> None:
    chunks = chunk_document(parsed_document())
    body = [c for c in chunks if "EUR 500" in c.text]
    assert body and "Approval thresholds" in body[0].text


def test_chunks_keep_a_real_locator() -> None:
    """The citation has to point somewhere a person can open."""
    parsed = parse_text("# T\n\nbody text here")
    doc = Document("d", "T", parsed.text, blocks=parsed.blocks)
    assert chunk_document(doc)[0].locator


def test_an_over_long_paragraph_still_gets_windowed() -> None:
    long_text = "# T\n\n" + " ".join(f"w{i}" for i in range(900))
    chunks = chunk_document(parsed_document(long_text, "long"), target_words=100)
    assert len(chunks) > 1
    assert all("T:" in c.text for c in chunks), "windows keep the heading trail"


def test_a_document_without_blocks_falls_back_to_word_windows() -> None:
    doc = Document("d", "D", " ".join(f"w{i}" for i in range(1000)))
    chunks = chunk_document(doc, target_words=100, overlap_words=20)
    assert len(chunks) > 1
    assert chunks[0].text.split()[-20:] == chunks[1].text.split()[:20]


def test_chunk_ids_are_stable_and_unique() -> None:
    doc = parsed_document()
    first = [c.chunk_id for c in chunk_document(doc)]
    assert first == [c.chunk_id for c in chunk_document(doc)]
    assert len(set(first)) == len(first)


def test_structure_beats_flattening_on_a_table_question() -> None:
    """The accuracy claim, stated as a test: a labelled row retrieves for the
    question its labels answer, and a flattened corpus does not carry them."""
    structured = parsed_document()
    flat = Document("expenses", "Expenses Policy", structured.text)  # no blocks

    structured_index = BM25Retriever()
    structured_index.index([structured])
    hit = structured_index.search("what is the meal allowance for senior grade", k=1)[0]
    assert "Grade: Senior" in hit.chunk.text
    assert "Meal allowance: EUR 45" in hit.chunk.text

    flat_index = BM25Retriever()
    flat_index.index([flat])
    flat_hit = flat_index.search("what is the meal allowance for senior grade", k=1)[0]
    # The flattened chunk still contains the words, but as one undifferentiated
    # blob covering every section - so it cannot be cited to a location.
    assert flat_hit.chunk.locator.startswith("chunk ")


def test_blocks_survive_into_the_document_for_claim_extraction() -> None:
    """The claim extractors read Document.text, which must include the labels."""
    doc = parsed_document()
    assert "Grade: Senior | Meal allowance: EUR 45" in doc.text


def test_chunk_blocks_on_an_empty_document() -> None:
    assert chunk_blocks(Document("d", "D", "")) == []


def test_headings_alone_do_not_become_chunks() -> None:
    """A heading with nothing under it is a fine retrieval signal and a useless
    answer, so it attaches to the content beneath instead of standing alone."""
    parsed = parse_text("# Only A Heading")
    doc = Document("d", "D", parsed.text, blocks=parsed.blocks)
    assert all(c.text.strip() for c in chunk_document(doc))


# -- the cited floor: summaries earn relaxation per answer -------------------

_MAIL = Chunk(
    chunk_id="site-mail#1",
    document_id="site-mail",
    document_title="Website Changes",
    text=(
        "Changes for the website before the next outreach wave. "
        "Priority 1: change the platform description to an agent-run TPRM platform. "
        "Priority 2: remove references to regulators and internal compliance evidence, "
        "emphasise external TPRM rules. "
        "Priority 3: update the call to action to book a demo of the agent. "
        "Priority 4: replace the NIS2 wording with many regulations require. "
        "Update the screenshots to show TPRM workflows instead of compliance."
    ),
)

#: The field shape: every substantive claim cited, wording compressed the way
#: real summaries compress - support lands between the two floors.
_CITED_SUMMARY = (
    "The document sets out the site edits requested ahead of the next prospecting "
    "wave. [site-mail]\n"
    "- Rebrand the product blurb around an agent-operated TPRM product [site-mail]\n"
    "- Strip the regulator references and internal audit evidence, spotlighting "
    "external TPRM rules [site-mail]\n"
    "- Repoint the call to action toward booking a demo of the agent [site-mail]\n"
    "- Trade the NIS2 wording for a broader many regulations formulation [site-mail]\n"
)


def test_a_fully_cited_summary_is_graded_against_the_lower_floor() -> None:
    """The field case, reproduced in shape: a faithful summary compresses and
    rephrases, which is exactly what the single ratio penalised. Full
    citation discipline - every claim cited, ids real, figures verified -
    earns the lower floor for this answer alone."""
    report = check_grounding(_CITED_SUMMARY, [_MAIL])
    assert 0.30 <= report.support_ratio < 0.45, (
        f"the fixture must land between the floors to prove anything; got {report.support_ratio}"
    )
    assert report.cited_coverage == 1.0
    assert report.passed, report.reasons


def test_an_uncited_claim_forfeits_the_relaxation() -> None:
    uncited_intro = _CITED_SUMMARY.replace(
        "ahead of the next prospecting wave. [site-mail]",
        "ahead of the next prospecting wave.",
    )
    report = check_grounding(uncited_intro, [_MAIL])
    assert report.cited_coverage < 1.0
    assert not report.passed
    assert any("45%" in r for r in report.reasons)


def test_fully_cited_nonsense_still_fails() -> None:
    """The lower floor is a floor, not a pass: invention in fresh vocabulary
    scores far below it even when every sentence carries a real citation."""
    nonsense = (
        "The memo mandates quarterly submarine maintenance for every branch office. [site-mail]\n"
        "- All employees must certify their juggling proficiency annually [site-mail]\n"
    )
    report = check_grounding(nonsense, [_MAIL])
    assert not report.passed
    assert any("fully cited" in r for r in report.reasons)


def test_a_wrong_figure_forfeits_the_relaxation_and_fails() -> None:
    wrong_number = _CITED_SUMMARY + "- Priority 9 raises the budget to EUR 500 [site-mail]\n"
    report = check_grounding(wrong_number, [_MAIL])
    assert not report.passed
    assert "500" in " ".join(report.unsupported_numbers)


def test_connective_lines_are_not_claims() -> None:
    """'Key changes include:' asserts nothing and needs no citation."""
    with_glue = _CITED_SUMMARY + "\nKey changes include:\n"
    report = check_grounding(with_glue, [_MAIL])
    assert report.cited_coverage == 1.0
    assert report.passed, report.reasons


def test_short_answers_keep_their_original_grading() -> None:
    """An answer with no claim long enough to need a citation earns nothing:
    the relaxation cannot leak into the short-extraction path."""
    report = check_grounding("Priority 3: book a demo. [site-mail]", [_MAIL])
    assert report.cited_coverage == 0.0
    assert report.passed, report.reasons
