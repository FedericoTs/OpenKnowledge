"""Retrieval, chunking, access control, and the grounding gate."""

from __future__ import annotations

from openknowledge.retrieval import BM25Retriever, Document, check_grounding, chunk_document


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
