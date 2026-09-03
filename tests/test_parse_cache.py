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

import hashlib
import os
import sqlite3
from pathlib import Path
from unittest import mock

from openknowledge.api.engine import build_engine
from openknowledge.config import Settings
from openknowledge.connectors.local_files import LocalFilesConnector
from openknowledge.documents import blocks as blocks_module
from openknowledge.documents import parse_bytes
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
    path.write_text(POLICY, encoding="utf-8")
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
        path.write_text(rewritten, encoding="utf-8")
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
    (documents / "expenses.md").write_text(POLICY, encoding="utf-8")
    (documents / "travel.md").write_text(CHANGED, encoding="utf-8")

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
    (documents / "expenses.md").write_text(POLICY, encoding="utf-8")

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
    (documents / "empty.md").write_text("", encoding="utf-8")
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
    path.write_text(POLICY, encoding="utf-8")
    engine = build_engine(_settings(tmp_path, documents))
    try:
        engine.reindex()
        assert len(engine.connector.parses) == 1
        path.write_text(CHANGED, encoding="utf-8")
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
    (documents / "expenses.md").write_text(POLICY, encoding="utf-8")
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
    (documents / "expenses.md").write_text(POLICY, encoding="utf-8")
    connector = LocalFilesConnector(documents)
    assert connector.parses is None
    connector.fetch()
    assert list(tmp_path.rglob("*.db")) == []


def test_the_backup_does_not_carry_it(tmp_path: Path) -> None:
    """A backup already carries the documents themselves. Adding a parse of
    every one of them would double the archive to save a rebuild."""
    from openknowledge.backup import _DATABASES

    assert "parses.db" not in _DATABASES


# -- the flattened text, stored beside the blocks --------------------------
#
# Flattening is not free: `normalise` makes six full passes over a document,
# and on 1,200 files that was 1.09 of the 1.54 seconds every upload spent
# re-reading a folder in which one file had changed. It is a pure function of
# the blocks, so the cache stores it. What these hold is that storing it
# cannot change what a document says.


def test_a_cached_parse_flattens_to_exactly_what_it_flattened_before() -> None:
    """The whole point, and the only thing that would matter if it were wrong.

    Every content hash, every chunk and every corpus_version is built from
    this text. A stored copy that differed from a recomputed one by so much as
    a newline would fingerprint the same document two ways.
    """
    parsed = parse_bytes(
        b"# Leave\n\nParental leave is 20 weeks.\n\n## Notice\n\nGive 8 weeks.\n", suffix=".md"
    )
    cache = ParseCache()
    key = cache_key(b"anything", parser=".md")
    cache.put(key, parsed)

    restored = cache.get(key)
    assert restored is not None
    assert restored.flattened is not None, "the text was not stored"
    assert restored.text == parsed.text


def test_a_stored_parse_still_equals_a_freshly_parsed_one() -> None:
    """One carries the memo and one does not, and they are the same document.

    Excluded from equality on purpose: several tests assert that a cached
    parse equals an uncached one, and they are asserting something true. A
    memo counted as a difference would fail them for a difference that is not
    one.
    """
    raw = b"# Expenses\n\nApproval above EUR 500.\n"
    parsed = parse_bytes(raw, suffix=".md")
    cache = ParseCache()
    key = cache_key(raw, parser=".md")
    cache.put(key, parsed)

    restored = cache.get(key)
    assert restored == parsed
    assert restored is not parsed


def test_a_document_that_was_never_cached_still_flattens() -> None:
    """The memo is an optimisation, not a requirement. Anything that builds a
    ParsedDocument directly - every parser does - gets the text computed."""
    parsed = parse_bytes(b"# Title\n\nBody text.\n", suffix=".md")
    assert parsed.flattened is None
    assert "Body text." in parsed.text


def test_reading_a_row_does_not_flatten_the_document_again() -> None:
    """Recomputing it on read would leave the cost exactly where it was."""
    raw = b"# Leave\n\nParental leave is 20 weeks.\n"
    parsed = parse_bytes(raw, suffix=".md")
    cache = ParseCache()
    key = cache_key(raw, parser=".md")
    cache.put(key, parsed)

    calls = 0
    real = blocks_module.normalise

    def counted(text: str) -> str:
        nonlocal calls
        calls += 1
        return real(text)

    with mock.patch.object(blocks_module, "normalise", counted):
        restored = cache.get(key)
        assert restored is not None
        _ = restored.text
        _ = restored.text
    assert calls == 0


def test_a_row_written_by_an_older_shape_is_a_miss(tmp_path: Path) -> None:
    """FORMAT is in the key, so a row from before the text was stored is not
    read as one that has it - it is simply not found, and re-parsed."""
    raw = b"# Leave\n\nParental leave is 20 weeks.\n"
    assert cache_key(raw, parser=".md") != f"1:.md:{hashlib.sha256(raw).hexdigest()}"

    cache = ParseCache(tmp_path / "parses.db")
    cache.put(f"1:.md:{hashlib.sha256(raw).hexdigest()}", parse_bytes(raw, suffix=".md"))
    assert cache.get(cache_key(raw, parser=".md")) is None
