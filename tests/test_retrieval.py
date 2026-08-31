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


def test_a_partial_gap_does_not_withdraw_the_half_that_was_answered(
    retriever: BM25Retriever,
) -> None:
    """The field case. Asked whether taxis and meals are covered, the model
    answered meals from the sources and said plainly that taxis were not
    there - and the whole answer was withdrawn for containing "no
    information", so the reader lost the half the documents did cover.

    A question with two parts usually has two answers, and one of them being
    "not here" is this product working, not failing.
    """
    report = check_grounding(
        "Employees with 12 months of continuous service get 20 weeks of fully paid "
        "parental leave. [hr-handbook]\n\nHowever, there is no information in the "
        "provided documents about adoption leave.",
        _chunks(retriever),
    )
    assert not report.abstained, "a gap in one part is not a refusal of the whole"
    assert report.passed, report.reasons
    assert report.cited_ids == ("hr-handbook",)


def test_a_refusal_that_happens_to_cite_is_still_a_refusal(
    retriever: BM25Retriever,
) -> None:
    """The citation has to be doing work outside the decline.

    "There is no information about X [hr-handbook]" is a statement about the
    sources, not a claim drawn from them - so the citation inside it earns
    nothing, and the answer is still a refusal.
    """
    report = check_grounding(
        "There is no information about sabbatical leave in the documents. [hr-handbook]",
        _chunks(retriever),
    )
    assert report.abstained


def test_several_ways_of_saying_only_no_are_still_only_no(
    retriever: BM25Retriever,
) -> None:
    """Nothing outside the declining sentences cites anything, so there is
    nothing for the answer to stand on - however many sentences it takes."""
    report = check_grounding(
        "I don't know. The documents do not discuss this. You may want to ask HR.",
        _chunks(retriever),
    )
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


# -- the context's own labels are not inventions (field regression) ----------
#
# Three Azure answers in a row were refused on a real corpus for "inventing"
# figures like 17 and 4,5,6 - every one a chunk label the SOURCES block itself
# taught the model. Each rejection from that transcript is pinned here.


def _labelled(doc_id: str, title: str, n: int, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"{doc_id}#{n - 1}",
        document_id=doc_id,
        document_title=title,
        text=text,
        locator=f"chunk {n}",
    )


_DEBRIEF = [
    _labelled(
        "demo-debrief",
        "TL;DR - Executive Summary",
        2,
        "Caterina validated the market opportunity for compliance automation. "
        "The vendor-focused approach serves mid-to-large companies; SMEs first "
        "assess their own internal compliance before auditing vendors.",
    ),
    _labelled(
        "demo-debrief",
        "TL;DR - Executive Summary",
        4,
        "Caterina recommends an internal self-assessment module as the entry "
        "point of the platform, before vendor management begins.",
    ),
]


def test_echoing_a_retrieved_chunk_label_is_not_an_invented_figure() -> None:
    report = check_grounding(
        "Caterina validated the market opportunity for compliance automation "
        "[demo-debrief] (chunk 2). Her recommendation of an internal "
        "self-assessment module as the entry point is reinforced in chunk 4 "
        "[demo-debrief].",
        _DEBRIEF,
    )
    assert report.passed, report.reasons
    assert not report.unsupported_numbers


def test_a_comma_run_of_chunk_labels_resolves_instead_of_becoming_one_figure() -> None:
    """The number regex reads "chunks 2,4" as the single figure 2,4 - which no
    source will ever contain. Resolved references leave the text entirely."""
    report = check_grounding(
        "The platform must open with self-assessment before vendor management, "
        "as covered in chunks 2, 4 [demo-debrief].",
        _DEBRIEF,
    )
    assert report.passed, report.reasons


def test_an_unretrieved_chunk_number_is_still_an_invention() -> None:
    report = check_grounding(
        "The self-assessment module is described in chunk 17 [demo-debrief].",
        _DEBRIEF,
    )
    assert not report.passed
    assert "17" in " ".join(report.reasons)


def test_label_only_references_count_as_citing_sources() -> None:
    """A tidy frontier model that references (chunk 2) without the [id]
    brackets has cited what it saw; "answer cites no sources" was wrong."""
    report = check_grounding(
        "Caterina validated the market opportunity for compliance automation, "
        "and SMEs first assess their own internal compliance (chunk 2).",
        _DEBRIEF,
    )
    assert report.passed, report.reasons
    assert "demo-debrief" in report.cited_ids


def test_a_year_that_lives_only_in_the_title_is_evidence() -> None:
    chunk = Chunk(
        chunk_id="pol#0",
        document_id="expenses-2023",
        document_title="Expenses Policy (2023)",
        text="The meal allowance is EUR 45 per day for domestic travel.",
        locator="chunk 1",
    )
    report = check_grounding(
        "The 2023 policy sets the meal allowance at EUR 45 per day [expenses-2023].",
        [chunk],
    )
    assert report.passed, report.reasons


def test_genuinely_invented_figures_are_still_rejected() -> None:
    report = check_grounding(
        "The meal allowance is EUR 80 per day [demo-debrief] (chunk 2).",
        _DEBRIEF,
    )
    assert not report.passed
    assert "80" in " ".join(report.reasons)


# -- an answer's own numbering is not a figure (field regression) -------------
#
# "What are the step by step activities we should cover?" - answered as a
# correct, fully cited list and refused for the figure 3, which was the third
# item's own marker. Prompt rule 6 asks for exactly this enumeration, so the
# gate was punishing the shape it requests.


def test_list_numbering_is_not_an_invented_figure() -> None:
    report = check_grounding(
        "The debrief lists these activities [demo-debrief]:\n"
        "1. SMEs first assess their own internal compliance [demo-debrief]\n"
        "2. Vendor management begins after self-assessment [demo-debrief]\n"
        "3. The entry point of the platform is the self-assessment module "
        "[demo-debrief]\n",
        _DEBRIEF,
    )
    assert report.passed, report.reasons
    assert not report.unsupported_numbers


def test_paren_style_numbering_is_also_structure() -> None:
    report = check_grounding(
        "The order is [demo-debrief]:\n"
        "1) SMEs first assess their own internal compliance [demo-debrief]\n"
        "2) Vendor management begins after that self-assessment "
        "[demo-debrief]\n"
        "3) The self-assessment module is the entry point [demo-debrief]\n",
        _DEBRIEF,
    )
    assert report.passed, report.reasons


def test_a_figure_inside_a_numbered_item_is_still_audited() -> None:
    """Only the marker is structure. What the item claims is checked exactly
    as before - otherwise the fix would be a hole, not a correction."""
    report = check_grounding(
        "The steps are [demo-debrief]:\n"
        "1. SMEs first assess their own internal compliance [demo-debrief]\n"
        "2. Vendor management covers 47 supplier categories [demo-debrief]\n",
        _DEBRIEF,
    )
    assert not report.passed
    assert "47" in " ".join(report.reasons)


def test_a_year_opening_a_line_is_not_treated_as_numbering() -> None:
    """Markers are capped at two digits, so a line that opens with a year is
    still the factual claim it looks like."""
    report = check_grounding(
        "2019. The self-assessment module was introduced then [demo-debrief]",
        _DEBRIEF,
    )
    assert not report.passed
    assert "2019" in " ".join(report.reasons)


def test_a_chunk_states_its_heading_trail_once() -> None:
    """The heading trail as metadata, so a citation can name where to look.

    The trail is also repeated through the chunk's text - once as the heading
    and again in front of every block - and that repetition is left exactly
    where it is. Removing it is a real improvement to the index and a
    measurable change to what the model reads: see ROADMAP.
    """
    doc = parsed_document(
        "# Remote Access and VPN\n\n"
        "To connect from outside the office, install the client.\n\n"
        "Access requests are approved by IT Operations.\n",
        "vpn",
    )
    (chunk,) = chunk_document(doc)

    assert chunk.section == "Remote Access and VPN"
    assert chunk.locator == "chunk 1", "the gate resolves 'chunk 4' against this"
    # Once, not three times: as the trail at the top and nowhere else.
    assert chunk.text.count("Remote Access and VPN") == 1
    assert chunk.text.startswith("Remote Access and VPN:")
    # What the trail is for survives: the passage still says what it is about.
    assert "install the client" in chunk.text
    assert "approved by IT Operations" in chunk.text


def test_a_nested_section_keeps_the_whole_trail() -> None:
    """A subsection must not lose the section it belongs to."""
    doc = parsed_document(
        "# Expenses Policy\n\n## Meals and subsistence\n\n"
        "Meals are reimbursed up to EUR 45 per day.\n",
        "expenses",
    )
    chunk = next(c for c in chunk_document(doc) if "EUR 45" in c.text)
    assert chunk.section == "Expenses Policy > Meals and subsistence"
    assert chunk.text.count("Meals and subsistence") == 1
    assert chunk.text.startswith("Expenses Policy > Meals and subsistence:")


def test_a_document_with_no_headings_has_no_section() -> None:
    """Nothing is invented: no trail means no section to name."""
    doc = Document("d", "D", " ".join(f"w{i}" for i in range(50)))
    (chunk,) = chunk_document(doc)
    assert chunk.section is None
    assert chunk.locator == "chunk 1"
