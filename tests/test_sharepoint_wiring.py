"""The mirror inside the running app: files in the corpus, readers enforced,
the mirror not editable through the app, and the refusal when sign-in is off.

The fake Graph plays the tenant. Nothing here proves the connector against
a real one - that run is still owed - but everything about how the mirror
meets the rest of the product is proved here: what the local connector
stamps, what a restricted viewer can see, what the page is told.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from tests.fake_graph import (
    CLIENT_ID,
    CLIENT_SECRET,
    TENANT,
    FakeGraph,
    group_grant,
    site_group_grant,
)

from openknowledge.api.app import create_app
from openknowledge.config import Settings

HR = "11111111-1111-1111-1111-111111111111"
FINANCE = "22222222-2222-2222-2222-222222222222"
TOKEN = "t0ken"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture
def graph() -> Iterator[FakeGraph]:
    g = FakeGraph()
    g.add_drive("drive-1", "Documents")
    g.add_file(
        "drive-1",
        "i-leave",
        "HR/parental-leave.md",
        b"# Parental Leave\nEmployees get 20 weeks fully paid.",
        [group_grant(HR)],
    )
    g.add_file(
        "drive-1",
        "i-expenses",
        "Finance/expenses.md",
        b"# Expenses Policy\nTravel above EUR 500 needs approval.",
        [group_grant(FINANCE)],
    )
    g.add_file(
        "drive-1",
        "i-minutes",
        "HR/board-minutes.md",
        b"# Board Minutes\nThe merger closes in May.",
        [site_group_grant("Owners")],
    )
    try:
        yield g
    finally:
        g.close()


def _settings(tmp_path: Path, graph: FakeGraph, **overrides: object) -> Settings:
    docs = tmp_path / "documents"
    docs.mkdir(exist_ok=True)
    (docs / "handbook.md").write_text("# Handbook\nThe office closes at 18:00.")
    values: dict[str, object] = {
        "data_dir": str(tmp_path / "data"),
        "documents_dir": str(docs),
        "admin_token": TOKEN,
        "local_enabled": False,
        "escalation_enabled": False,
        "upload_enabled": True,
        "sharepoint_enabled": True,
        "sharepoint_tenant_id": TENANT,
        "sharepoint_client_id": CLIENT_ID,
        "sharepoint_client_secret": CLIENT_SECRET,
        "sharepoint_site": "contoso.sharepoint.com:/sites/HR",
        "sharepoint_graph_url": f"{graph.base}/v1.0",
        "sharepoint_login_url": graph.base,
        "sharepoint_poll_seconds": 0,
        "sharepoint_require_signin": False,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)  # type: ignore[call-arg, arg-type]


def test_mirrored_files_join_the_corpus_with_their_readers(
    tmp_path: Path, graph: FakeGraph
) -> None:
    with TestClient(create_app(_settings(tmp_path, graph))) as c:
        engine = c.app.state.engine
        summary = engine.sync_sharepoint()
        assert summary is not None and summary.errors == []
        assert summary.added == 3

        listed = c.get("/documents").json()
        names = {f["name"] for f in listed["files"]}
        assert "sharepoint/Documents/HR/parental-leave.md" in names
        assert "handbook.md" in names, "the folder's own files are still there"
        assert listed["sharepoint"]["documents"] == 3
        assert listed["sharepoint"]["withheld"] == 1

        def titles(viewer: frozenset[str] | None) -> set[str]:
            visible, _ = engine.retriever.documents_visible_to(viewer)
            return set(visible)

        hr = titles(frozenset({f"group:{HR}"}))
        assert "Parental Leave" in hr and "Handbook" in hr
        assert "Expenses Policy" not in hr, "Finance's file is Finance's"
        assert "Board Minutes" not in hr, "readers that could not be mapped are nobody"
        outsider = titles(frozenset({"group:sales"}))
        assert outsider == {"Handbook"}, "the folder's unrestricted file only"
        assert "Board Minutes" in titles(None), "the unrestricted view is the admin's"


def test_the_mirror_cannot_be_edited_through_the_app(tmp_path: Path, graph: FakeGraph) -> None:
    with TestClient(create_app(_settings(tmp_path, graph))) as c:
        c.app.state.engine.sync_sharepoint()
        upload = c.post(
            "/documents",
            headers=AUTH,
            data={"folder": "sharepoint/Documents/HR"},
            files={"files": ("new.md", b"# New\nText.", "text/markdown")},
        )
        assert upload.status_code == 409, upload.text
        assert "SharePoint" in upload.json()["detail"]

        removal = c.delete("/documents/sharepoint/Documents/HR/parental-leave.md", headers=AUTH)
        assert removal.status_code == 409, removal.text
        assert (tmp_path / "documents/sharepoint/Documents/HR/parental-leave.md").is_file()

        # The folder's own files are as editable as ever.
        assert c.delete("/documents/handbook.md", headers=AUTH).status_code == 200


def test_sign_in_off_refuses_to_mirror_unless_told_otherwise(
    tmp_path: Path, graph: FakeGraph
) -> None:
    settings = _settings(tmp_path, graph, sharepoint_require_signin=True)
    with TestClient(create_app(settings)) as c:
        engine = c.app.state.engine
        summary = engine.sync_sharepoint()
        assert summary is not None and len(summary.errors) == 1
        assert "sign-in is off" in summary.errors[0]
        assert not (tmp_path / "documents" / "sharepoint").exists(), "nothing was mirrored"
        assert graph.requests == [], "Graph was never called"
        status = c.get("/documents").json()["sharepoint"]
        assert "sign-in is off" in status["refusal"]


def test_a_missing_setting_is_a_refusal_the_page_can_show(tmp_path: Path, graph: FakeGraph) -> None:
    settings = _settings(tmp_path, graph, sharepoint_client_secret=None)
    with TestClient(create_app(settings)) as c:
        status = c.get("/documents").json()["sharepoint"]
        assert "OK_SHAREPOINT_CLIENT_SECRET" in status["refusal"]


def test_an_admin_can_sync_now_and_it_is_logged(tmp_path: Path, graph: FakeGraph) -> None:
    with TestClient(create_app(_settings(tmp_path, graph))) as c:
        response = c.post("/admin/sharepoint/sync", headers=AUTH)
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["added"] == 3 and body["errors"] == []
        log = c.get("/admin/log", headers=AUTH).json()
        (entry,) = [e for e in log["entries"] if e["action"] == "sharepoint.sync"]
        assert entry["detail"]["documents"] == 3


def test_without_a_mirror_the_sync_route_says_so(tmp_path: Path, graph: FakeGraph) -> None:
    settings = _settings(tmp_path, graph, sharepoint_enabled=False)
    with TestClient(create_app(settings)) as c:
        assert c.post("/admin/sharepoint/sync", headers=AUTH).status_code == 404
        assert c.get("/documents").json()["sharepoint"] is None


def test_the_cli_syncs_and_reports(
    tmp_path: Path, graph: FakeGraph, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    from openknowledge.cli import main

    docs = tmp_path / "documents"
    docs.mkdir()
    for key, value in {
        "OK_STATE_DIR": str(tmp_path),
        "OK_DATA_DIR": str(tmp_path / "data"),
        "OK_DOCUMENTS_DIR": str(docs),
        "OK_LOCAL_ENABLED": "false",
        "OK_EMBEDDING_ENABLED": "false",
        "OK_SHAREPOINT_ENABLED": "true",
        "OK_SHAREPOINT_TENANT_ID": TENANT,
        "OK_SHAREPOINT_CLIENT_ID": CLIENT_ID,
        "OK_SHAREPOINT_CLIENT_SECRET": CLIENT_SECRET,
        "OK_SHAREPOINT_SITE": "contoso.sharepoint.com:/sites/HR",
        "OK_SHAREPOINT_GRAPH_URL": f"{graph.base}/v1.0",
        "OK_SHAREPOINT_LOGIN_URL": graph.base,
        "OK_SHAREPOINT_REQUIRE_SIGNIN": "false",
    }.items():
        monkeypatch.setenv(key, value)
    for key in list(os.environ):
        if (
            key.startswith("OK_")
            and key
            not in {
                "OK_STATE_DIR",
                "OK_DATA_DIR",
                "OK_DOCUMENTS_DIR",
                "OK_LOCAL_ENABLED",
                "OK_EMBEDDING_ENABLED",
            }
            and not key.startswith("OK_SHAREPOINT_")
        ):
            monkeypatch.delenv(key, raising=False)

    assert main(["sharepoint", "status"]) == 0
    assert "last sync: never" in capsys.readouterr().out
    assert main(["sharepoint", "sync"]) == 0
    out = capsys.readouterr().out
    assert "3 added" in out and "3 document(s) mirrored, 1 withheld" in out
    assert (docs / "sharepoint/Documents/Finance/expenses.md").is_file()
