"""Uploading documents: the door through which knowledge arrives.

The endpoints exist behind an explicit gate (a desktop install opens it at
first serve, a server keeps it shut until told), and every filename a client
sends is treated as hostile until flattened and whitelisted - multipart
filenames are attacker-controlled strings, not paths.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from openknowledge.api.app import _safe_document_name, create_app
from openknowledge.config import Settings


@pytest.fixture
def client(tmp_path: Path):
    settings = Settings(
        data_dir=str(tmp_path / "data"),
        documents_dir=str(tmp_path / "documents"),
        upload_enabled=True,
        local_enabled=False,
        embedding_enabled=False,
        _env_file=None,  # type: ignore[call-arg]
    )
    with TestClient(create_app(settings)) as c:
        yield c, tmp_path / "documents"


def _file(name: str, content: bytes = b"# Expenses Policy\n\nMeals are EUR 45 per day.\n"):
    return ("files", (name, io.BytesIO(content), "application/octet-stream"))


def test_uploads_are_off_by_default(tmp_path: Path) -> None:
    """A running answer engine has no business accepting writes unasked."""
    settings = Settings(
        data_dir=str(tmp_path / "data"),
        local_enabled=False,
        embedding_enabled=False,
        _env_file=None,  # type: ignore[call-arg]
    )
    with TestClient(create_app(settings)) as client:
        assert client.post("/documents", files=[_file("a.md")]).status_code == 404
        assert client.get("/documents").status_code == 404


def test_a_dropped_file_becomes_knowledge_in_one_request(client) -> None:
    """Store, index, and answer about it - the whole point, in one round trip."""
    c, root = client
    response = c.post("/documents", files=[_file("expenses-policy.md")])
    assert response.status_code == 201
    body = response.json()

    assert body["stored"] == [
        {"name": "expenses-policy.md", "bytes": body["stored"][0]["bytes"], "replaced": False}
    ]
    assert body["skipped"] == []
    assert body["corpus"]["documents"] == 1
    assert (root / "expenses-policy.md").is_file()

    # The very next question already knows about it, with no extra step.
    health = c.get("/healthz").json()
    assert health["documents_indexed"] == 1
    chat = c.post("/chat", json={"question": "what documents do you have?"}).json()
    assert "Expenses Policy" in chat["answer"] or "expenses" in chat["answer"].lower()


def test_hostile_filenames_are_flattened_or_refused(client) -> None:
    c, root = client
    response = c.post(
        "/documents",
        files=[
            _file("../../evil.md"),
            _file("..\\..\\windows-evil.md"),
            _file("/etc/absolute.md"),
            _file(".hidden.md"),
            _file("c:drive.md"),
        ],
    )
    body = response.json()
    stored_names = {s["name"] for s in body["stored"]}
    # Traversal prefixes are flattened to the final component...
    assert stored_names == {"evil.md", "windows-evil.md", "absolute.md"}
    # ...and nothing escaped the folder.
    outside = [p for p in root.parent.rglob("*.md") if root not in p.parents]
    assert outside == []
    # Dotfiles and drive-prefixed names are refused outright.
    refused = {s["name"] for s in body["skipped"]}
    assert ".hidden.md" in refused and "c:drive.md" in refused


def test_the_unreadable_are_refused_with_the_reason_not_stored(client) -> None:
    c, root = client
    response = c.post(
        "/documents",
        files=[_file("notes.md"), _file("old-report.doc"), _file("photo.xyz")],
    )
    body = response.json()
    assert {s["name"] for s in body["stored"]} == {"notes.md"}
    reasons = {s["name"]: s["reason"] for s in body["skipped"]}
    assert "re-save it as .docx" in reasons["old-report.doc"]
    assert "no parser" in reasons["photo.xyz"]
    assert not (root / "old-report.doc").exists(), "stored a file it cannot read"


def test_an_oversized_file_is_refused_and_named(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=str(tmp_path / "data"),
        documents_dir=str(tmp_path / "documents"),
        upload_enabled=True,
        upload_max_mb=1,
        local_enabled=False,
        embedding_enabled=False,
        _env_file=None,  # type: ignore[call-arg]
    )
    with TestClient(create_app(settings)) as c:
        big = b"x" * (1_000_001)
        body = c.post("/documents", files=[_file("huge.md", big)]).json()
        assert body["stored"] == []
        assert "1 MB" in body["skipped"][0]["reason"]


def test_replacing_a_document_says_so_and_reindexes(client) -> None:
    c, _ = client
    c.post("/documents", files=[_file("policy.md", b"# Old\n\nThe limit is EUR 100.\n")])
    body = c.post(
        "/documents", files=[_file("policy.md", b"# New\n\nThe limit is EUR 200.\n")]
    ).json()
    assert body["stored"][0]["replaced"] is True

    answer = c.post("/chat", json={"question": "what documents do you have?"}).json()
    assert answer["tier"] == "corpus"


def test_deleting_a_document_removes_its_knowledge(client) -> None:
    c, root = client
    c.post("/documents", files=[_file("doomed.md")])
    assert c.get("/healthz").json()["documents_indexed"] == 1

    response = c.delete("/documents/doomed.md")
    assert response.status_code == 200
    assert not (root / "doomed.md").exists()
    assert c.get("/healthz").json()["documents_indexed"] == 0

    assert c.delete("/documents/doomed.md").status_code == 404
    assert c.delete("/documents/..%2F..%2Fescape.md").status_code in (400, 404)


def test_listing_shows_what_indexed_and_what_could_not(client) -> None:
    c, root = client
    root.mkdir(parents=True, exist_ok=True)
    (root / "good.md").write_text("# Good\n\ncontent")
    (root / "legacy.doc").write_bytes(b"old")
    rows = {f["name"]: f for f in c.get("/documents").json()["files"]}
    assert rows["good.md"]["skipped"] is None
    assert "re-save" in rows["legacy.doc"]["skipped"]


def test_safe_name_is_a_whitelist_not_a_blacklist() -> None:
    assert _safe_document_name("Annual Report (2026).pdf") == "Annual Report (2026).pdf"
    assert _safe_document_name("a/b/c.md") == "c.md"
    assert _safe_document_name("..") is None
    assert _safe_document_name("") is None
    assert _safe_document_name("con:aux.md") is None
    assert _safe_document_name("x" * 200 + ".md") is None
