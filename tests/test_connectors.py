"""Local files connector."""

from __future__ import annotations

from pathlib import Path

import pytest

from openknowledge.connectors import Connector, LocalFilesConnector, SharePointConnector


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    (tmp_path / "policies").mkdir()
    (tmp_path / "policies" / "expenses.md").write_text("# Expenses Policy\nMeals to EUR 45.")
    (tmp_path / "notes.txt").write_text("plain text note")
    (tmp_path / "empty.md").write_text("   ")
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
    (corpus / "notes.txt").write_text("edited note")
    assert hash_of("notes") != before


def test_unimplemented_connectors_say_what_they_need() -> None:
    with pytest.raises(NotImplementedError, match="Entra ID"):
        SharePointConnector().fetch()
