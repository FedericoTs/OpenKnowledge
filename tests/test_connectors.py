"""Local files connector."""

from __future__ import annotations

from pathlib import Path

import pytest

from openknowledge.connectors import Connector, LocalFilesConnector


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    (tmp_path / "policies").mkdir()
    (tmp_path / "policies" / "expenses.md").write_text(
        "# Expenses Policy\nMeals to EUR 45.", encoding="utf-8"
    )
    (tmp_path / "notes.txt").write_text("plain text note", encoding="utf-8")
    (tmp_path / "empty.md").write_text("   ", encoding="utf-8")
    (tmp_path / "image.png").write_bytes(b"\x89PNG not text")
    return tmp_path


def test_satisfies_the_connector_protocol(corpus: Path) -> None:
    assert isinstance(LocalFilesConnector(corpus), Connector)


def test_reads_text_files_recursively(corpus: Path) -> None:
    ids = {d.document_id for d in LocalFilesConnector(corpus).fetch()}
    assert ids == {"policies-expenses", "notes"}


def test_unreadable_files_are_named_not_silently_dropped(corpus: Path) -> None:
    """A file contributing nothing without saying so is how a corpus grows a
    hole that only surfaces months later as a wrong answer."""
    connector = LocalFilesConnector(corpus)
    connector.fetch()
    skipped = {s.path: s.reason for s in connector.skipped}
    assert "image.png" in skipped
    assert "no parser for .png" in skipped["image.png"]
    assert "empty.md" in skipped


def test_a_legacy_office_file_says_how_to_fix_it(corpus: Path) -> None:
    (corpus / "handbook.doc").write_bytes(b"\xd0\xcf\x11\xe0 legacy")
    connector = LocalFilesConnector(corpus)
    connector.fetch()
    reason = next(s.reason for s in connector.skipped if s.path == "handbook.doc")
    assert "re-save it as .docx" in reason


def test_office_lock_files_are_ignored_entirely(corpus: Path) -> None:
    """Word leaves ~$ files behind; they are not documents and not worth
    reporting as failures either."""
    (corpus / "~$expenses.docx").write_bytes(b"lock")
    connector = LocalFilesConnector(corpus)
    connector.fetch()
    assert not any("~$" in s.path for s in connector.skipped)


def test_parsed_documents_carry_their_structure(corpus: Path) -> None:
    doc = next(
        d for d in LocalFilesConnector(corpus).fetch() if d.document_id == "policies-expenses"
    )
    assert doc.blocks, "the connector must pass parsed structure through"
    assert doc.title == "Expenses Policy"


def test_document_ids_are_stable_and_citable(corpus: Path) -> None:
    """Models must reproduce these exactly, so they stay simple."""
    doc = next(d for d in LocalFilesConnector(corpus).fetch() if "expenses" in d.document_id)
    assert doc.document_id == "policies-expenses"
    assert doc.title == "Expenses Policy"


def test_relative_root_still_produces_a_usable_url(corpus: Path, monkeypatch) -> None:
    monkeypatch.chdir(corpus)
    docs = LocalFilesConnector(".").fetch()
    assert docs and all(d.url and d.url.startswith("file://") for d in docs)


def test_missing_root_is_not_fatal(tmp_path: Path) -> None:
    assert LocalFilesConnector(tmp_path / "nope").fetch() == []


def test_content_hash_tracks_edits(corpus: Path) -> None:
    """The corpus version is built from these, so an edit has to move the hash."""

    def hash_of(doc_id: str) -> str:
        return next(
            d.content_hash for d in LocalFilesConnector(corpus).fetch() if d.document_id == doc_id
        )

    before = hash_of("notes")
    (corpus / "notes.txt").write_text("edited note", encoding="utf-8")
    assert hash_of("notes") != before


def test_every_connector_this_package_exports_is_built() -> None:
    """There used to be a `cloud_stubs` module holding connectors that were
    specified and not written, so the shape of the work was visible. Both are
    written now - SharePoint and Google Drive mirror into the documents folder
    and stamp each file with its readers - so the placeholder is gone, and
    this stands where its test did: nothing here raises NotImplementedError."""
    import openknowledge.connectors as package

    for name in package.__all__:
        exported = getattr(package, name)
        assert not getattr(exported, "setup_hint", None), f"{name} is still a placeholder"


def test_an_email_from_line_is_not_a_title(tmp_path: Path) -> None:
    """Reported from the first real-tenant corpus: a mail printed to PDF
    opens with its From line, and the corpus listing then introduced the
    document as 'Federico Sciuca <federico@...>'. A heading containing an
    email address names a correspondent; the filename names the document."""
    from openknowledge.connectors.local_files import _usable_title

    assert _usable_title("Federico Sciuca <federico@arvexlab.com>") is None
    assert _usable_title("mail from federico@arvexlab.com today") is None
    assert _usable_title(None) is None
    assert _usable_title("Expenses Policy 2026") == "Expenses Policy 2026"

    (tmp_path / "ArvexLab-Mail---Cambiamenti-WEBSITE.md").write_text(
        "# Federico Sciuca <federico@arvexlab.com>\n\nPriority 1: change the site.\n",
        encoding="utf-8",
    )
    from openknowledge.connectors.local_files import LocalFilesConnector

    (document,) = LocalFilesConnector(tmp_path).fetch()
    assert document.title == "Arvexlab Mail   Cambiamenti Website"
