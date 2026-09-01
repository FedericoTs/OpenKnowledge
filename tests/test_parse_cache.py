"""The same bytes are never parsed twice.

Measured on this project's own parsers, per document: markdown 5.9ms, docx
56.7ms, PDF 780ms. That last one is not a slow parser - profiling put 99.2%
of a PDF rebuild in a Java process being started once per file, almost all of
it JVM startup rather than reading the document. Sixty small PDFs rebuilt in
46 seconds, and a thousand policy PDFs - an ordinary corpus for the company
this is built for - is minutes, paid again on every upload and every delete.

The tests that matter here are not the fast ones. A cache holding the text of
every document is a way to serve last week's policy for ever if its key is
wrong, so the key is the content and these hold it to that: the same corpus
comes out of a warm cache as out of no cache at all, an edit is seen even when
the clock says nothing changed, and two parsers never share an entry.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from openknowledge.api.engine import build_engine
from openknowledge.config import Settings
from openknowledge.connectors.local_files import LocalFilesConnector
from openknowledge.documents.blocks import Block, BlockKind, ParsedDocument
from openknowledge.documents.cache import ParseCache, cache_key

POLICY = "# Expenses\n\nTravel above EUR 500 requires prior approval.\n"
CHANGED = "# Expenses\n\nTravel above EUR 1,000 requires prior approval.\n"


def _settings(root: Path, documents: Path) -> Settings:
    return Settings(
        data_dir=str(root / "data"),
        documents_dir=str(documents),
        local_enabled=False,
        embedding_enabled=False,
        _env_file=None,  # type: ignore[call-arg]
    )


def _passages(engine) -> list[tuple[str, str, str]]:
    return [(c.document_id, c.section or "", c.text) for c in engine.retriever.chunks]


# -- the key -----------------------------------------------------------------


def test_the_key_is_the_content_not_the_name() -> None:
    assert cache_key(b"same", parser=".md") == cache_key(b"same", parser=".md")
    assert cache_key(b"same", parser=".md") != cache_key(b"other", parser=".md")


def test_two_pdf_backends_never_share_an_entry(tmp_path: Path) -> None:
    """Two PDF backends extract slightly different text - the parser says so
    itself - so a shared entry would hand one backend's words to a corpus
    fingerprinted under the other's.

    Asserted against the connector rather than against ``cache_key``. The
    first version of this checked that the key function distinguishes two
    names it was handed, which it does and always did - and passed happily
    while the connector handed it the same name for both backends. A test of
    the half that was never wrong is not a test.
    """
    pdf = tmp_path / "policy.pdf"
    other = tmp_path / "policy.md"
    one = LocalFilesConnector(tmp_path, pdf_backend="opendataloader")
    two = LocalFilesConnector(tmp_path, pdf_backend="pdfplumber")

    assert one._parser_name(pdf) != two._parser_name(pdf)  # noqa: SLF001 - the seam
    assert cache_key(b"same", parser=one._parser_name(pdf)) != cache_key(  # noqa: SLF001
        b"same", parser=two._parser_name(pdf)
    )
    # A format with one parser is not split by a setting that cannot affect it.
    assert one._parser_name(other) == two._parser_name(other)  # noqa: SLF001


def test_a_touched_file_is_still_a_hit_and_an_edit_is_still_a_miss(tmp_path: Path) -> None:
    """Why the key is the content and not the clock.

    ``rsync -t``, ``git checkout`` and every restore-from-backup put old
    timestamps on new bytes. A cache keyed on mtime and size would believe
    them and serve the previous version of a policy for ever - and the size
    is unchanged here, which is exactly the case that slips through.
    """
    documents = tmp_path / "documents"
    documents.mkdir()
    path = documents / "expenses.md"
    path.write_text(POLICY)
    engine = build_engine(_settings(tmp_path, documents))
    try:
        engine.reindex()
        misses = engine.connector.parses.misses

        # Same bytes, older clock: a hit.
        os.utime(path, (1_000_000, 1_000_000))
        engine.reindex()
        assert engine.connector.parses.misses == misses, "the clock moved, the bytes did not"

        # Different bytes, same length, clock rolled backwards: a miss.
        rewritten = POLICY.replace("EUR 500", "EUR 900")
        assert len(rewritten) == len(POLICY)
        path.write_text(rewritten)
        os.utime(path, (1_000_000, 1_000_000))
        engine.reindex()
        assert engine.connector.parses.misses > misses, "the bytes moved and it noticed"
        assert any("EUR 900" in text for _, _, text in _passages(engine))
    finally:
        engine.store.close()
        engine.knowledge.close()


# -- the property ------------------------------------------------------------


def test_a_warm_cache_produces_the_same_corpus_as_no_cache(tmp_path: Path) -> None:
    """If these ever differ, this is not an optimisation - it is a corpus
    quietly made of stale text."""
    documents = tmp_path / "documents"
    documents.mkdir()
    (documents / "expenses.md").write_text(POLICY)
    (documents / "travel.md").write_text(CHANGED)

    warm = build_engine(_settings(tmp_path / "a", documents))
    plain = build_engine(_settings(tmp_path / "b", documents))
    plain.connector.parses = None
    try:
        warm.reindex()
        warm.reindex()  # second scan: entirely from the cache
        plain.reindex()

        assert _passages(warm) == _passages(plain)
        assert warm.retriever.corpus_version == plain.retriever.corpus_version
        assert warm.connector.parses.hits > 0, "it really did serve from the cache"
    finally:
        for engine in (warm, plain):
            engine.store.close()
            engine.knowledge.close()


def test_it_survives_a_restart(tmp_path: Path) -> None:
    """The first build is where the whole cost lands. Paying it again every
    time the server comes up would leave most of the win on the table."""
    documents = tmp_path / "documents"
    documents.mkdir()
    (documents / "expenses.md").write_text(POLICY)

    first = build_engine(_settings(tmp_path, documents))
    first.reindex()
    stored = len(first.connector.parses)
    first.store.close()
    first.knowledge.close()

    again = build_engine(_settings(tmp_path, documents))
    try:
        again.reindex()
        assert len(again.connector.parses) == stored
        assert again.connector.parses.hits > 0, "a new process read the old cache"
    finally:
        again.store.close()
        again.knowledge.close()


# -- failing safe ------------------------------------------------------------


def test_an_unreadable_row_is_a_miss_and_not_a_crash(tmp_path: Path) -> None:
    """The file it came from is still on disk, so the worst case is paying
    for the parse again - much better than a crash, or half a document."""
    cache = ParseCache(tmp_path / "parses.db")
    key = cache_key(b"x", parser=".md")
    cache.put(key, ParsedDocument(blocks=(Block(kind=BlockKind.PARAGRAPH, text="hello"),)))
    assert cache.get(key) is not None

    raw = sqlite3.connect(tmp_path / "parses.db")
    raw.execute("UPDATE parses SET parsed = 'not json'")
    raw.commit()
    raw.close()
    assert cache.get(key) is None
    cache.close()


def test_a_parse_that_found_nothing_is_not_remembered(tmp_path: Path) -> None:
    """A failed parse usually failed for a reason outside the file - a
    missing backend, a JVM that did not start, a full disk - so the next scan
    should try again rather than inherit the failure for ever."""
    documents = tmp_path / "documents"
    documents.mkdir()
    (documents / "empty.md").write_text("")
    engine = build_engine(_settings(tmp_path, documents))
    try:
        engine.reindex()
        assert len(engine.connector.parses) == 0
    finally:
        engine.store.close()
        engine.knowledge.close()


def test_parses_of_files_that_are_gone_are_forgotten(tmp_path: Path) -> None:
    """Otherwise it grows by a row per edit for ever - every draft of a policy
    ever saved into the folder, kept because it existed once."""
    documents = tmp_path / "documents"
    documents.mkdir()
    path = documents / "expenses.md"
    path.write_text(POLICY)
    engine = build_engine(_settings(tmp_path, documents))
    try:
        engine.reindex()
        assert len(engine.connector.parses) == 1
        path.write_text(CHANGED)
        engine.reindex()
        assert len(engine.connector.parses) == 1, "the old text is not kept alongside the new"
        path.unlink()
        engine.reindex()
        assert len(engine.connector.parses) == 0
    finally:
        engine.store.close()
        engine.knowledge.close()


def test_deleting_the_cache_costs_a_rebuild_and_nothing_else(tmp_path: Path) -> None:
    """It is pure derived data, in its own file so that promise can be made."""
    documents = tmp_path / "documents"
    documents.mkdir()
    (documents / "expenses.md").write_text(POLICY)
    settings = _settings(tmp_path, documents)
    engine = build_engine(settings)
    engine.reindex()
    before = _passages(engine)
    engine.connector.parses.close()
    engine.store.close()
    engine.knowledge.close()

    Path(settings.parse_cache_path).unlink()
    after = build_engine(settings)
    try:
        after.reindex()
        assert _passages(after) == before
    finally:
        after.store.close()
        after.knowledge.close()


def test_audit_still_writes_nothing(tmp_path: Path) -> None:
    """`openknowledge audit` promises no database and nothing written - the
    one command for somebody who has not decided whether to trust this yet.
    A connector built without a cache parses every time, which is the right
    trade for a command that runs once."""
    documents = tmp_path / "documents"
    documents.mkdir()
    (documents / "expenses.md").write_text(POLICY)
    connector = LocalFilesConnector(documents)
    assert connector.parses is None
    connector.fetch()
    assert list(tmp_path.rglob("*.db")) == []


def test_the_backup_does_not_carry_it(tmp_path: Path) -> None:
    """A backup already carries the documents themselves. Adding a parse of
    every one of them would double the archive to save a rebuild."""
    from openknowledge.backup import _DATABASES

    assert "parses.db" not in _DATABASES
