"""One JVM for a folder of PDFs, not one per file.

The parse cache (``documents/cache.py``) made a re-index nearly free. It did
nothing for the *first* index, which is where a real corpus spends its time:
OpenDataLoader starts a Java process per document, and on this box that was
~640 ms of the ~656 ms a four-page PDF cost. A thousand policy PDFs - an
ordinary corpus for the company this is built for - is most of a quarter of an
hour spent starting and stopping JVMs.

The parser takes a batch. These tests hold the two things that makes true:
every document must parse to exactly what it parsed to alone, and no document
may be lost or confused with another on the way through.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from openknowledge.connectors.local_files import LocalFilesConnector
from openknowledge.documents import opendataloader as odl
from openknowledge.documents.cache import ParseCache
from openknowledge.documents.pdf import parse_pdf, parse_pdfs

needs_java = pytest.mark.skipif(
    not odl.is_available(), reason="OpenDataLoader needs a JVM and the wrapper package"
)

_WORDS = [
    "policy",
    "travel",
    "expense",
    "approval",
    "manager",
    "leave",
    "parental",
    "notice",
    "period",
    "contractor",
    "allowance",
    "reimbursement",
    "receipt",
    "threshold",
    "euro",
    "invoice",
    "deadline",
    "submission",
]


def write_pdf(path: Path, *, seed: int, pages: int = 2) -> bytes:
    """A PDF whose words depend on the seed, so two of them cannot be confused."""
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas as pdfcanvas

    rng = random.Random(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf = pdfcanvas.Canvas(str(path), pagesize=A4)
    for page in range(pages):
        pdf.setFont("Helvetica-Bold", 14)
        pdf.drawString(60, 800, f"Marker {seed} Section {page + 1}")
        pdf.setFont("Helvetica", 9)
        y = 770
        for line in range(8):
            # Not the first line on the page: the parser drops a page's topmost
            # line as a running header, which is right for real documents and
            # would quietly hide this marker.
            if line == 3:
                pdf.drawString(60, y, f"Reference number MARKER-{seed} applies here.")
            else:
                pdf.drawString(60, y, " ".join(rng.choice(_WORDS) for _ in range(9)))
            y -= 14
        pdf.showPage()
    pdf.save()
    return path.read_bytes()


@needs_java
def test_a_batched_parse_is_the_same_parse(tmp_path: Path) -> None:
    """The property every other claim rests on.

    The corpus fingerprint is built from this text. A batch that parsed even
    slightly differently would invalidate every cached answer the moment a
    corpus grew past the point where batching kicked in - and worse, would do
    it silently.
    """
    blobs = [write_pdf(tmp_path / f"d{i}.pdf", seed=i) for i in range(6)]

    alone = [parse_pdf(blob, backend="opendataloader") for blob in blobs]
    together = parse_pdfs(blobs, backend="opendataloader")

    assert together is not None
    assert len(together) == len(alone)
    for one, many in zip(alone, together, strict=True):
        assert many.blocks == one.blocks
        assert (many.title, many.pages, many.warnings) == (one.title, one.pages, one.warnings)


@needs_java
def test_two_documents_with_the_same_filename_stay_two_documents(tmp_path: Path) -> None:
    """`HR/policy.pdf` and `Finance/policy.pdf` are not the same document.

    The parser names each output after its input's basename, so a batch handed
    the real paths writes one `policy.json` and silently loses a document - or
    answers questions about one policy out of the other one's text. Measured,
    not feared: two different PDFs with the same basename produced exactly one
    output file. This goes through the connector, because that is where real
    paths exist and where the mistake would be reintroduced.
    """
    root = tmp_path / "corpus"
    write_pdf(root / "hr" / "policy.pdf", seed=11)
    write_pdf(root / "finance" / "policy.pdf", seed=22)

    documents = LocalFilesConnector(root, parses=ParseCache()).fetch()

    by_id = {doc.document_id: doc.text for doc in documents}
    assert set(by_id) == {"hr-policy", "finance-policy"}
    assert "MARKER-11" in by_id["hr-policy"] and "MARKER-22" not in by_id["hr-policy"]
    assert "MARKER-22" in by_id["finance-policy"] and "MARKER-11" not in by_id["finance-policy"]


@needs_java
def test_one_unreadable_file_does_not_take_the_batch_with_it(tmp_path: Path) -> None:
    """The CLI exits non-zero on a bad file, having written the good ones.

    A corpus with one corrupt PDF in it must still index every other document,
    and the bad one must still be *named* - a file that contributes nothing
    without saying so is how a corpus develops a hole.
    """
    root = tmp_path / "corpus"
    write_pdf(root / "good-one.pdf", seed=3)
    write_pdf(root / "good-two.pdf", seed=4)
    (root / "broken.pdf").write_bytes(b"%PDF-1.4\nnot actually a pdf\n")

    connector = LocalFilesConnector(root, parses=ParseCache())
    documents = connector.fetch()

    assert {doc.document_id for doc in documents} == {"good-one", "good-two"}
    assert [skipped.path for skipped in connector.skipped] == ["broken.pdf"]
    # The same sentence it would carry had it never been batched. `auto` is
    # what the connector uses, so that is what this has to compare against -
    # under `auto` a rejected PDF is handed to pdfplumber, whose reading of
    # the same file is what the operator is told about.
    alone = parse_pdf((root / "broken.pdf").read_bytes(), backend="auto")
    assert connector.skipped[0].reason == alone.warnings[0]


@needs_java
def test_a_folder_of_pdfs_costs_one_jvm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The whole point, counted rather than timed."""
    import opendataloader_pdf

    root = tmp_path / "corpus"
    for i in range(5):
        write_pdf(root / f"policy-{i}.pdf", seed=i)

    calls = 0
    real = opendataloader_pdf.convert

    def counted(**kwargs: object) -> None:
        nonlocal calls
        calls += 1
        real(**kwargs)

    monkeypatch.setattr(opendataloader_pdf, "convert", counted)
    documents = LocalFilesConnector(root, parses=ParseCache()).fetch()

    assert len(documents) == 5
    assert calls == 1


@needs_java
def test_a_batch_is_split_so_it_never_holds_a_whole_corpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every document in a batch is staged on disk and held by the parser.

    Unbounded, a corpus of large PDFs would trade a JVM problem for a memory
    one, so the batch is split. Driven with a small size rather than by writing
    sixty-five PDFs.
    """
    import opendataloader_pdf

    from openknowledge.connectors import local_files

    # Only the connector's, deliberately: the parser splits too, and patching
    # both would let either one alone satisfy this. Its own splitting is
    # pinned separately, against the parser directly.
    monkeypatch.setattr(local_files, "BATCH_SIZE", 2)

    root = tmp_path / "corpus"
    for i in range(5):
        write_pdf(root / f"policy-{i}.pdf", seed=i)

    sizes: list[int] = []
    real = opendataloader_pdf.convert

    def counted(**kwargs: object) -> None:
        paths = kwargs["input_path"]
        assert isinstance(paths, list)
        sizes.append(len(paths))
        real(**kwargs)

    monkeypatch.setattr(opendataloader_pdf, "convert", counted)
    documents = LocalFilesConnector(root, parses=ParseCache()).fetch()

    assert len(documents) == 5
    assert sizes == [2, 2, 1]


@needs_java
def test_a_cached_corpus_starts_no_jvm_at_all(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Batching and the parse cache compose: the second index parses nothing."""
    import opendataloader_pdf

    root = tmp_path / "corpus"
    for i in range(4):
        write_pdf(root / f"policy-{i}.pdf", seed=i)

    cache = ParseCache(tmp_path / "parses.db")
    first = LocalFilesConnector(root, parses=cache).fetch()

    calls = 0
    real = opendataloader_pdf.convert

    def counted(**kwargs: object) -> None:
        nonlocal calls
        calls += 1
        real(**kwargs)

    monkeypatch.setattr(opendataloader_pdf, "convert", counted)
    second = LocalFilesConnector(root, parses=cache).fetch()

    assert calls == 0
    assert [(doc.document_id, doc.text) for doc in first] == [
        (doc.document_id, doc.text) for doc in second
    ]


@needs_java
def test_a_corpus_indexes_identically_with_and_without_batching(tmp_path: Path) -> None:
    """End to end: the documents a scan produces do not depend on the batch."""
    root = tmp_path / "corpus"
    for i in range(4):
        write_pdf(root / f"policy-{i}.pdf", seed=i)
    (root / "notes.md").write_text(
        "# Notes\n\nA markdown file, parsed the ordinary way.\n", encoding="utf-8"
    )

    batched = LocalFilesConnector(root, parses=ParseCache()).fetch()
    one_at_a_time = LocalFilesConnector(root).fetch()  # no cache, no prefetch path

    assert [(d.document_id, d.title, d.text) for d in batched] == [
        (d.document_id, d.title, d.text) for d in one_at_a_time
    ]


def test_pdfplumber_gains_nothing_from_a_batch(tmp_path: Path) -> None:
    """No process to start, so no batch: one code path in the fallback parser."""
    blob = write_pdf(tmp_path / "one.pdf", seed=1)
    assert parse_pdfs([blob], backend="pdfplumber") is None


def test_a_batch_without_the_java_backend_declines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On a machine with no JVM the caller parses the ordinary way."""
    monkeypatch.setattr(odl, "is_available", lambda: False)
    blob = write_pdf(tmp_path / "one.pdf", seed=1)
    assert parse_pdfs([blob], backend="auto") is None


@needs_java
def test_the_parser_splits_its_own_batches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Not only the connector.

    ``parse_pdfs`` is reachable by anything that has a pile of PDFs, and a
    caller who hands it ten thousand of them must not get ten thousand files
    staged on disk and held by the parser at once. The connector splits too,
    for the same reason - which is exactly why this asserts against the parser
    directly rather than through it.
    """
    import opendataloader_pdf

    blobs = [write_pdf(tmp_path / f"d{i}.pdf", seed=i, pages=1) for i in range(5)]
    monkeypatch.setattr(odl, "BATCH_SIZE", 2)

    sizes: list[int] = []
    real = opendataloader_pdf.convert

    def counted(**kwargs: object) -> None:
        paths = kwargs["input_path"]
        assert isinstance(paths, list)
        sizes.append(len(paths))
        real(**kwargs)

    monkeypatch.setattr(opendataloader_pdf, "convert", counted)
    parsed = odl.parse_pdfs_opendataloader(blobs)

    assert sizes == [2, 2, 1]
    assert [len(document.blocks) > 0 for document in parsed] == [True] * 5


@needs_java
def test_a_rejected_file_is_re_parsed_alone_for_its_own_reason(tmp_path: Path) -> None:
    """A missing output means that document failed, and it must say why itself.

    Under ``auto`` a failed parse is rescued by pdfplumber, which produces its
    own sentence and would hide this. Pinned against the Java backend on its
    own, where the sentence has to come from the batch's fallback and must be
    word for word the one the file would have carried had it never been
    batched: an operator acts on that sentence.
    """
    good = write_pdf(tmp_path / "good.pdf", seed=7)
    bad = b"%PDF-1.4\nnot actually a pdf\n"

    batched = odl.parse_pdfs_opendataloader([good, bad])
    alone = odl.parse_pdf_opendataloader(bad)

    assert batched[0].blocks and "MARKER-7" in batched[0].text
    assert not batched[1].blocks
    assert batched[1].warnings == alone.warnings
    assert batched[1].warnings == (
        "OpenDataLoader: this file is not a valid PDF file (corrupted or truncated content).",
    )


def test_no_batch_is_collected_for_a_backend_that_gains_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """pdfplumber has no process to start, so nothing is read ahead for it.

    Collecting a group means reading and hashing every file in it. Doing that
    to build a batch the parser then declines would read the whole corpus an
    extra time for no gain at all.
    """
    from openknowledge.connectors import local_files

    root = tmp_path / "corpus"
    for i in range(3):
        write_pdf(root / f"policy-{i}.pdf", seed=i)

    def refuse(blobs: object, backend: str = "auto") -> None:
        raise AssertionError("a batch was collected for a backend that declines batches")

    monkeypatch.setattr(local_files, "parse_pdfs", refuse)
    documents = LocalFilesConnector(root, pdf_backend="pdfplumber", parses=ParseCache()).fetch()

    assert len(documents) == 3


@needs_java
def test_never_more_than_one_group_of_parses_is_held(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reason the read-ahead is driven by the walk rather than done up front.

    Parsing every PDF before indexing any of them would hold a whole corpus of
    parsed documents in memory on top of the corpus being built - a JVM problem
    traded for a memory one. Only the group about to be needed is read ahead,
    and this counts what is actually held rather than trusting the comment.
    """
    from openknowledge.connectors import local_files

    monkeypatch.setattr(local_files, "BATCH_SIZE", 2)

    root = tmp_path / "corpus"
    for i in range(5):
        write_pdf(root / f"policy-{i}.pdf", seed=i)

    connector = LocalFilesConnector(root, parses=ParseCache())
    held: list[int] = []
    parse = connector._parse

    def watched(path: Path) -> object:
        parsed = parse(path)
        held.append(len(connector._parsed_ahead))
        return parsed

    monkeypatch.setattr(connector, "_parse", watched)
    documents = connector.fetch()

    assert len(documents) == 5
    assert max(held) < 2  # a group of two, one of which has just been handed over
