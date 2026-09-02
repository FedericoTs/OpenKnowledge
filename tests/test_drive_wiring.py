"""The Drive mirror inside the running app, and the identity it depends on.

Drive names people by email; the directory names them by id. The test that
matters most here is that the two meet: a person whose verified email is the
one Drive granted gets the file, and somebody else does not.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from tests.fake_drive import (
    CLIENT_EMAIL,
    DOMAIN,
    SUBJECT,
    FakeDrive,
    domain_grant,
    group_grant,
    private_key_pem,
    user_grant,
)

from openknowledge.api.app import create_app
from openknowledge.auth.oidc import Identity
from openknowledge.auth.sessions import Session
from openknowledge.config import Settings
from openknowledge.connectors.mirror import WITHHELD

ALICE = "alice@contoso.com"
BOB = "bob@contoso.com"
HR_GROUP = "hr-team@contoso.com"
TOKEN = "t0ken"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture
def drive() -> Iterator[FakeDrive]:
    d = FakeDrive()
    d.add_drive("drive-1", "Company")
    d.add_file(
        "drive-1",
        "f-leave",
        "parental-leave.md",
        b"# Parental Leave\nEmployees get 20 weeks fully paid.",
        [user_grant(ALICE), group_grant(HR_GROUP)],
    )
    d.add_file(
        "drive-1",
        "f-all",
        "handbook.md",
        b"# Handbook\nThe office closes at 18:00.",
        [domain_grant(DOMAIN)],
    )
    d.add_file(
        "drive-1",
        "f-board",
        "board.md",
        b"# Board\nSecret.",
        [domain_grant("partner.example")],
    )
    try:
        yield d
    finally:
        d.close()


def _settings(tmp_path: Path, drive: FakeDrive, **overrides: object) -> Settings:
    docs = tmp_path / "documents"
    docs.mkdir(exist_ok=True)
    (docs / "notice.md").write_text("# Notice\nThe canteen is closed on Monday.")
    values: dict[str, object] = {
        "data_dir": str(tmp_path / "data"),
        "documents_dir": str(docs),
        "admin_token": TOKEN,
        "local_enabled": False,
        "embedding_enabled": False,
        "escalation_enabled": False,
        "upload_enabled": True,
        "drive_enabled": True,
        "drive_client_email": CLIENT_EMAIL,
        "drive_private_key": private_key_pem(),
        "drive_subject": SUBJECT,
        "drive_domain": DOMAIN,
        "drive_api_url": f"{drive.base}/v3",
        "drive_token_url": f"{drive.base}/token",
        "drive_poll_seconds": 0,
        "drive_require_signin": False,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)  # type: ignore[call-arg, arg-type]


# -- the two vocabularies meeting ---------------------------------------------------


def test_a_verified_email_becomes_a_principal_and_an_unverified_one_does_not() -> None:
    """The whole bridge between Drive's names and the directory's. An address
    the provider does not vouch for is something the person typed, and minting
    a principal from it would let anyone read another person's files."""
    signed_in = Session(
        subject="oid-1", name="Alice", groups=("group-guid",), expires_at=0.0, email=ALICE
    )
    assert signed_in.principals == {
        "authenticated",
        "user:oid-1",
        "group:group-guid",
        f"user:{ALICE}",
    }
    without = Session(subject="oid-1", name="Alice", groups=(), expires_at=0.0)
    assert without.principals == {"authenticated", "user:oid-1"}


def test_an_unverified_email_claim_is_dropped_at_sign_in() -> None:
    from openknowledge.auth.oidc import OidcClient

    client = OidcClient(issuer="https://example", client_id="app")
    verified = client._identity_from({"oid": "1", "email": "Alice@Contoso.com", "groups": []})
    assert verified.email == ALICE, "and normalised, so it matches however it was typed"

    unverified = client._identity_from(
        {"oid": "1", "email": ALICE, "email_verified": False, "groups": []}
    )
    assert unverified.email == ""
    assert isinstance(unverified, Identity)


# -- the mirror in the app ------------------------------------------------------------


def test_mirrored_files_join_the_corpus_with_their_readers(
    tmp_path: Path, drive: FakeDrive
) -> None:
    with TestClient(create_app(_settings(tmp_path, drive))) as c:
        engine = c.app.state.engine
        summary = engine.sync_drive()
        assert summary is not None and summary.errors == []
        assert summary.added == 3

        listed = c.get("/documents").json()
        assert "gdrive/Company/parental-leave.md" in {f["name"] for f in listed["files"]}
        assert "notice.md" in {f["name"] for f in listed["files"]}
        assert listed["drive"]["documents"] == 3
        assert listed["drive"]["withheld"] == 1
        assert listed["sharepoint"] is None

        def titles(viewer: frozenset[str] | None) -> set[str]:
            visible, _ = engine.retriever.documents_visible_to(viewer)
            return set(visible)

        alice = titles(frozenset({"authenticated", "user:oid-1", f"user:{ALICE}"}))
        assert "Parental Leave" in alice, "Drive granted her address; sign-in mints it"
        assert "Handbook" in alice, "and the whole-domain grant reaches everyone signed in"
        assert "Notice" in alice, "the folder's own files are unaffected"
        assert "Board" not in alice, "a partner domain's grant is nobody here"

        bob = titles(frozenset({"authenticated", "user:oid-2", f"user:{BOB}"}))
        assert "Parental Leave" not in bob, "the file was granted to Alice, not to Bob"
        assert "Handbook" in bob

        in_hr = titles(frozenset({"authenticated", f"user:{BOB}", f"group:{HR_GROUP}"}))
        assert "Parental Leave" in in_hr, "a group grant reaches the group's members"


def test_the_mirror_cannot_be_edited_through_the_app(tmp_path: Path, drive: FakeDrive) -> None:
    with TestClient(create_app(_settings(tmp_path, drive))) as c:
        c.app.state.engine.sync_drive()
        upload = c.post(
            "/documents",
            headers=AUTH,
            data={"folder": "gdrive/Company"},
            files={"files": ("new.md", b"# New\nText.", "text/markdown")},
        )
        assert upload.status_code == 409
        assert "Google Drive" in upload.json()["detail"], "the refusal names where to go instead"

        removal = c.delete("/documents/gdrive/Company/handbook.md", headers=AUTH)
        assert removal.status_code == 409 and "Google Drive" in removal.json()["detail"]
        assert (tmp_path / "documents/gdrive/Company/handbook.md").is_file()
        assert c.delete("/documents/notice.md", headers=AUTH).status_code == 200


def test_sign_in_off_refuses_to_mirror_unless_told_otherwise(
    tmp_path: Path, drive: FakeDrive
) -> None:
    settings = _settings(tmp_path, drive, drive_require_signin=True)
    with TestClient(create_app(settings)) as c:
        summary = c.app.state.engine.sync_drive()
        assert summary is not None and len(summary.errors) == 1
        assert "sign-in is off" in summary.errors[0]
        assert not (tmp_path / "documents" / "gdrive").exists()
        assert drive.requests == []
        assert "sign-in is off" in c.get("/documents").json()["drive"]["refusal"]


def test_a_missing_setting_is_a_refusal_the_page_can_show(tmp_path: Path, drive: FakeDrive) -> None:
    settings = _settings(tmp_path, drive, drive_domain="")
    with TestClient(create_app(settings)) as c:
        assert "OK_DRIVE_DOMAIN" in c.get("/documents").json()["drive"]["refusal"]


def test_an_admin_can_sync_now_and_it_is_logged(tmp_path: Path, drive: FakeDrive) -> None:
    with TestClient(create_app(_settings(tmp_path, drive))) as c:
        response = c.post("/admin/drive/sync", headers=AUTH)
        assert response.status_code == 200, response.text
        assert response.json()["added"] == 3
        log = c.get("/admin/log", headers=AUTH).json()
        (entry,) = [e for e in log["entries"] if e["action"] == "drive.sync"]
        assert entry["detail"]["documents"] == 3


def test_without_a_mirror_the_sync_route_says_so(tmp_path: Path, drive: FakeDrive) -> None:
    with TestClient(create_app(_settings(tmp_path, drive, drive_enabled=False))) as c:
        assert c.post("/admin/drive/sync", headers=AUTH).status_code == 404
        assert c.get("/documents").json()["drive"] is None


def test_two_mirrors_stamp_their_own_files_and_nobody_elses(
    tmp_path: Path, drive: FakeDrive
) -> None:
    """Both mirrors can run at once. Each owns a folder, so merging what they
    stamped is a union and never a disagreement about who may read what."""
    from tests.fake_graph import CLIENT_ID, CLIENT_SECRET, TENANT, FakeGraph
    from tests.fake_graph import group_grant as graph_group

    graph = FakeGraph()
    graph.add_drive("d-1", "Documents")
    graph.add_file("d-1", "i-1", "policy.md", b"# Policy\nText.", [graph_group("guid-hr")])
    try:
        settings = _settings(
            tmp_path,
            drive,
            sharepoint_enabled=True,
            sharepoint_tenant_id=TENANT,
            sharepoint_client_id=CLIENT_ID,
            sharepoint_client_secret=CLIENT_SECRET,
            sharepoint_site="contoso.sharepoint.com:/sites/HR",
            sharepoint_graph_url=f"{graph.base}/v1.0",
            sharepoint_login_url=graph.base,
            sharepoint_poll_seconds=0,
            sharepoint_require_signin=False,
        )
        with TestClient(create_app(settings)) as c:
            engine = c.app.state.engine
            engine.sync_drive()
            engine.sync_sharepoint()
            stamped = engine.mirror_principals()
            assert stamped["gdrive/Company/parental-leave.md"] == {
                f"user:{ALICE}",
                f"group:{HR_GROUP}",
            }
            assert stamped["sharepoint/Documents/policy.md"] == {"group:guid-hr"}
            assert stamped["gdrive/Company/board.md"] == {WITHHELD}
            assert engine.mirror_owns("gdrive/Company/board.md") == "Google Drive"
            assert engine.mirror_owns("sharepoint/Documents/policy.md") == "SharePoint"
            assert engine.mirror_owns("notice.md") is None

            body = c.get("/documents").json()
            assert body["drive"]["documents"] == 3 and body["sharepoint"]["documents"] == 1
    finally:
        graph.close()
