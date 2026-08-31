"""A file the system cannot read must say so when it arrives.

A scan is a picture of a document. Every check the upload path had was about
the container - the extension, the size, whether any bytes arrived - and a
scanned PDF passes all of them. It was stored, reported as stored, listed in
the sidebar, and then refused every question about itself with "not covered
by the documents I have". True, and useless: the document is right there.

The diagnosis was never missing. The PDF parser has always said "no text
layer on any of 3 pages - this looks like a scan", the connector has always
recorded it, and the scan report has always carried it. It went to a log
nobody reads. These tests are about the last three inches: getting it in
front of the person whose file it is.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from openknowledge.api.app import create_app
from openknowledge.config import Settings

pypdf = pytest.importorskip("PIL", reason="needs Pillow to build a scan")


def _scan(pages: int = 1) -> bytes:
    """A PDF that is pictures of words - what a photocopier produces."""
    from PIL import Image, ImageDraw

    sheets = []
    for n in range(pages):
        img = Image.new("RGB", (620, 877), "white")
        ImageDraw.Draw(img).text((50, 100), f"VENDOR AGREEMENT page {n + 1}", fill="black")
        sheets.append(img)
    buffer = io.BytesIO()
    sheets[0].save(buffer, "PDF", save_all=True, append_images=sheets[1:], resolution=100)
    return buffer.getvalue()


@pytest.fixture
def client(tmp_path: Path):
    (tmp_path / "documents").mkdir()
    settings = Settings(
        data_dir=str(tmp_path / "data"),
        documents_dir=str(tmp_path / "documents"),
        local_enabled=False,
        embedding_enabled=False,
        upload_enabled=True,
        _env_file=None,  # type: ignore[call-arg]
    )
    with TestClient(create_app(settings)) as c:
        yield c


def test_a_scan_is_reported_unreadable_when_it_is_uploaded(client) -> None:
    """The moment that matters. Afterwards the person is in the chat asking
    questions of a document that holds nothing, blaming the assistant."""
    response = client.post(
        "/documents",
        files=[("files", ("scanned-contract.pdf", _scan(), "application/pdf"))],
    )
    assert response.status_code == 201, response.text
    stored = response.json()["stored"]
    assert len(stored) == 1
    note = stored[0].get("unreadable")
    assert note, "a stored file that yielded nothing must say so"
    assert "scan" in note.lower()
    assert "ocr" in note.lower(), "say what would fix it, not just that it failed"


def test_a_readable_document_carries_no_such_note(client) -> None:
    response = client.post(
        "/documents",
        files=[
            ("files", ("handbook.md", b"# Notice\n\nNotice is three months.\n", "text/markdown"))
        ],
    )
    stored = response.json()["stored"]
    assert "unreadable" not in stored[0]


def test_the_listing_keeps_saying_why(client) -> None:
    """Not only at upload: someone who returns a week later, wondering why a
    document never answers, must find the reason beside its name."""
    client.post(
        "/documents", files=[("files", ("scanned-contract.pdf", _scan(), "application/pdf"))]
    )
    rows = {row["name"]: row["skipped"] for row in client.get("/documents").json()["files"]}
    assert rows["scanned-contract.pdf"], "the listing must carry the reason too"
    assert "scan" in rows["scanned-contract.pdf"].lower()


def test_the_file_is_kept_not_thrown_away(client) -> None:
    """Refusing to read it is not a reason to delete someone's document: they
    may well OCR it and want the original in place."""
    client.post(
        "/documents", files=[("files", ("scanned-contract.pdf", _scan(), "application/pdf"))]
    )
    names = [row["name"] for row in client.get("/documents").json()["files"]]
    assert "scanned-contract.pdf" in names
