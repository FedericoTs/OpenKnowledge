"""Remembering claims must not change what is found.

Conflict detection re-ran over the whole corpus on every rebuild, and the
comment justifying it was right about the reason and wrong about the cost:
conflicts are free of *money*, which is why re-running was affordable, and at
400 markdown documents they were 57% of a rebuild's clock. A rebuild happens on
every upload, every delete and every access rule an admin changes, inside the
request that is waiting for it.

So the extracting is cached and the comparing is not. These tests hold that
line: the same conflicts are found with the cache as without it, a document
whose text changes has its claims pulled again, and the cache does not grow
without end.
"""

from __future__ import annotations

from pathlib import Path

from openknowledge.api.engine import build_engine
from openknowledge.config import Settings
from openknowledge.knowledge.claims import ClaimCache, compare_documents
from openknowledge.retrieval.base import Document

POLICY = """# Expenses

Travel above EUR 500 requires prior approval from a line manager.
The meal allowance limit is EUR 45 per day.
Contractors must submit receipts within 30 days.
"""

MOVED = """# Travel Guidelines

Travel above EUR 1,000 requires prior approval from a line manager.
Contractors may submit receipts within 30 days.
"""

QUIET = """# Office

The kitchen is on the third floor and the coffee is free.
"""


def _doc(doc_id: str, text: str) -> Document:
    return Document(document_id=doc_id, title=doc_id, text=text)


def test_the_same_conflicts_are_found_with_the_cache_and_without() -> None:
    """The property the whole change rests on. If these ever differ, the cache
    is not an optimisation, it is a behaviour change wearing one."""
    documents = [_doc("expenses.md", POLICY), _doc("travel.md", MOVED), _doc("office.md", QUIET)]

    plain, plain_agreed = compare_documents(documents)
    cache = ClaimCache()
    cached, cached_agreed = compare_documents(documents, cache=cache)
    # And again on the same cache, which is what a second rebuild does.
    again, again_agreed = compare_documents(documents, cache=cache)

    assert plain, "the corpus does contradict itself, so there is something to compare"
    assert [c.key for c in cached] == [c.key for c in plain]
    assert [c.key for c in again] == [c.key for c in plain]
    assert cached_agreed == plain_agreed == again_agreed


def test_a_document_whose_text_changes_is_read_again() -> None:
    """The failure a content-keyed cache exists to avoid: serving yesterday's
    claims for today's text, so an edit that introduces a contradiction is
    never seen."""
    cache = ClaimCache()
    before = [_doc("expenses.md", POLICY), _doc("travel.md", QUIET)]
    assert not compare_documents(before, cache=cache)[0], "nothing disagrees yet"

    after = [_doc("expenses.md", POLICY), _doc("travel.md", MOVED)]
    conflicts, _ = compare_documents(after, cache=cache)
    assert conflicts, "the edited document is compared on its new text"


def test_two_documents_with_the_same_words_keep_their_own_claims() -> None:
    """Keyed by (document_id, content_hash), not by the hash alone: identical
    text is ordinary in a corpus with an archive copy, and a claim cites where
    it came from."""
    cache = ClaimCache()
    twins = [_doc("a.md", POLICY), _doc("b.md", POLICY)]
    compare_documents(twins, cache=cache)

    sources = {claim.document_id for claim in cache.numeric(twins[0])}
    sources |= {claim.document_id for claim in cache.numeric(twins[1])}
    assert sources == {"a.md", "b.md"}


def test_documents_that_leave_the_corpus_are_forgotten() -> None:
    """Otherwise it is a slow leak keyed by everything ever indexed."""
    cache = ClaimCache()
    compare_documents([_doc("a.md", POLICY), _doc("b.md", MOVED)], cache=cache)
    assert len(cache._numeric) == 2  # noqa: SLF001 - the thing being tested

    cache.keep_only([_doc("a.md", POLICY)])
    assert len(cache._numeric) == 1  # noqa: SLF001
    assert len(cache._deontic) == 1  # noqa: SLF001


def test_a_rebuild_finds_the_same_conflicts_the_first_index_did(tmp_path: Path) -> None:
    """End to end, through the engine that holds the cache across rebuilds -
    which is where a stale claim would actually reach somebody."""
    documents = tmp_path / "documents"
    documents.mkdir()
    (documents / "expenses.md").write_text(POLICY)
    (documents / "travel.md").write_text(MOVED)
    engine = build_engine(
        Settings(
            data_dir=str(tmp_path / "data"),
            documents_dir=str(documents),
            local_enabled=False,
            embedding_enabled=False,
            _env_file=None,  # type: ignore[call-arg]
        )
    )
    try:
        engine.reindex()
        first = {c.key for c in engine.knowledge.open_conflicts()}
        engine.reindex()
        assert {c.key for c in engine.knowledge.open_conflicts()} == first
        assert first, "the fixture corpus contradicts itself"

        # The moved figure is corrected. Its conflict must go - from a cache
        # that still holds the old claims for the old text - and the deontic
        # disagreement this fixture also carries ("must submit" against "may
        # submit") must stay, because that one was not corrected. Asserting
        # "no conflicts at all" would pass for the wrong reason the day the
        # reconciliation cleared too much.
        (documents / "travel.md").write_text(MOVED.replace("EUR 1,000", "EUR 500"))
        engine.reindex()
        after = {c.key for c in engine.knowledge.open_conflicts()}
        assert after == {"expenses:required|travel:allowed"}
        assert after < first, "the corrected figure stopped being a conflict"
    finally:
        engine.store.close()
        engine.knowledge.close()


# -- the bug this work found on its way past ---------------------------------


def test_correcting_a_document_clears_the_conflict_it_caused(tmp_path: Path) -> None:
    """A contradiction is resolved two ways, and only one was handled.

    Deleting one of the documents cleared the flag. *Correcting the figure*
    did not: nothing ever compared the open conflicts against what a scan
    currently finds, so the row stayed open for a disagreement the corpus no
    longer contained. With block_on_conflict on - the default - every question
    it gated stayed refused after the documents were already right, and the
    only way out was an admin resolving something that did not exist.

    Found by a test written to prove a claims cache was faithful: the cached
    and uncached runs agreed with each other, and both disagreed with a fresh
    engine that had never seen the old text. The cache was innocent; what it
    exposed was not.
    """
    documents = tmp_path / "documents"
    documents.mkdir()
    (documents / "expenses.md").write_text(
        "# Expenses\n\nTravel above EUR 500 requires prior approval.\n"
    )
    (documents / "travel.md").write_text(
        "# Travel\n\nTravel above EUR 1,000 requires prior approval.\n"
    )
    engine = build_engine(
        Settings(
            data_dir=str(tmp_path / "data"),
            documents_dir=str(documents),
            local_enabled=False,
            embedding_enabled=False,
            _env_file=None,  # type: ignore[call-arg]
        )
    )
    try:
        engine.reindex()
        assert engine.knowledge.open_conflicts(), "the two documents disagree"

        (documents / "travel.md").write_text(
            "# Travel\n\nTravel above EUR 500 requires prior approval.\n"
        )
        report = engine.reindex()
        assert not engine.knowledge.open_conflicts(), (
            "the documents now agree, so nothing should still be gating answers"
        )
        assert report  # the reindex reported something
    finally:
        engine.store.close()
        engine.knowledge.close()


def test_a_resolution_somebody_made_is_not_undone_by_a_rescan(tmp_path: Path) -> None:
    """Reconciling open flags must not reach into decisions. A resolved row is
    the record of a judgement, it gates nothing, and a rebuild that quietly
    deleted it would erase what the admin log points at."""
    documents = tmp_path / "documents"
    documents.mkdir()
    (documents / "a.md").write_text("# A\n\nTravel above EUR 500 requires prior approval.\n")
    (documents / "b.md").write_text("# B\n\nTravel above EUR 1,000 requires prior approval.\n")
    engine = build_engine(
        Settings(
            data_dir=str(tmp_path / "data"),
            documents_dir=str(documents),
            local_enabled=False,
            embedding_enabled=False,
            _env_file=None,  # type: ignore[call-arg]
        )
    )
    try:
        engine.reindex()
        key = engine.knowledge.open_conflicts()[0].key
        engine.knowledge.resolve_conflict(key, resolution="a.md stands", resolver="alice")

        engine.reindex()
        assert not engine.knowledge.open_conflicts(), "resolved stays resolved"

        # And still resolved after the documents are corrected, when the
        # conflict stops being detected at all.
        (documents / "b.md").write_text("# B\n\nTravel above EUR 500 requires prior approval.\n")
        engine.reindex()
        rows = engine.knowledge._conn.execute(  # noqa: SLF001 - reading the record itself
            "SELECT status, resolution FROM conflicts WHERE key = ?", (key,)
        ).fetchall()
        assert [(r["status"], r["resolution"]) for r in rows] == [("resolved", "a.md stands")]
    finally:
        engine.store.close()
        engine.knowledge.close()
