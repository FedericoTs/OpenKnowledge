"""The backup button: the same file the CLI writes, handed to the browser.

``backup.py`` has its own tests for what goes into the archive. This file is
the route and the page: the zip that comes back is a real one with the right
members, leaving the documents out leaves them out, the download is in the
admin log, the file the response was served from does not stay behind, and
only the admin routes into /manage draw the controls.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

from fastapi.testclient import TestClient
from tests.test_knowledge_gaps import _boot_paths

from openknowledge.api.app import create_app
from openknowledge.config import Settings

TOKEN = "t0ken"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


def _client(tmp_path: Path) -> TestClient:
    docs = tmp_path / "documents"
    (docs / "hr").mkdir(parents=True)
    (docs / "hr" / "leave.md").write_text("# Parental Leave\nEmployees get 20 weeks fully paid.")
    settings = Settings(
        data_dir=str(tmp_path / "data"),
        documents_dir=str(docs),
        admin_token=TOKEN,
        local_enabled=False,
        escalation_enabled=False,
        _env_file=None,  # type: ignore[call-arg]
    )
    return TestClient(create_app(settings))


def test_the_download_is_the_archive_the_cli_writes(tmp_path: Path) -> None:
    with _client(tmp_path) as c:
        response = c.get("/admin/backup", headers=AUTH)
        assert response.status_code == 200, response.text
        assert response.headers["content-type"] == "application/zip"
        disposition = response.headers["content-disposition"]
        assert "openknowledge-backup-" in disposition and disposition.endswith('.zip"')

        archive = zipfile.ZipFile(io.BytesIO(response.content))
        names = set(archive.namelist())
        assert "manifest.json" in names
        assert any(n.startswith("data/") and n.endswith(".db") for n in names), names
        assert "documents/hr/leave.md" in names

        # Served, then gone: nothing accumulates under the data directory.
        assert list((tmp_path / "data" / "backups").glob("*.zip")) == []

        log = c.get("/admin/log", headers=AUTH).json()
        (entry,) = [e for e in log["entries"] if e["action"] == "backup.download"]
        assert entry["target"].startswith("openknowledge-backup-")
        assert entry["detail"]["documents"] == 1
        assert entry["detail"]["bytes"] == len(response.content)


def test_the_documents_can_be_left_out(tmp_path: Path) -> None:
    with _client(tmp_path) as c:
        response = c.get("/admin/backup?documents=false", headers=AUTH)
        names = zipfile.ZipFile(io.BytesIO(response.content)).namelist()
        assert not any(n.startswith("documents/") for n in names), names
        assert "manifest.json" in names
        log = c.get("/admin/log", headers=AUTH).json()
        (entry,) = [e for e in log["entries"] if e["action"] == "backup.download"]
        assert entry["detail"]["documents"] == 0


def test_the_archive_carries_no_secret(tmp_path: Path) -> None:
    """The file goes to whoever keeps the backups; the token that fetched it
    must not travel with it."""
    with _client(tmp_path) as c:
        response = c.get("/admin/backup", headers=AUTH)
    archive = zipfile.ZipFile(io.BytesIO(response.content))
    manifest = archive.read("manifest.json").decode()
    assert TOKEN not in manifest
    assert "OK_ADMIN_TOKEN" in manifest, "the manifest names what has to be set again"


def test_only_the_admin_routes_into_the_page_draw_the_controls(tmp_path: Path) -> None:
    with _client(tmp_path) as c:
        page = c.get("/manage").text
    assert "async function refreshBackup()" in page
    paths = _boot_paths(page)
    assert "refreshBackup" in paths["token"]
    assert "refreshBackup" in paths["admin session"]
    assert "refreshBackup" not in paths["curator session"]


def test_the_page_says_what_is_in_it_and_what_is_not(tmp_path: Path) -> None:
    with _client(tmp_path) as c:
        page = c.get("/manage").text
    assert "<h2>Backup</h2>" in page
    assert 'id="backup"' in page
    # The hint's own sentence, not the one the button says after a download.
    assert "Secrets are not in it: the admin token and any API keys" in page
    assert "openknowledge restore" in page
