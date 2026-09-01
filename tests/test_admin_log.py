"""Who changed what, and who is allowed to.

Two questions an enterprise buyer asks that this project could not answer
until now. The first is attribution: *somebody* walled off the HR folder
last Tuesday - who? The second is least privilege: the person who writes
the FAQ should not also hold the keys to who may read it.

The tests that matter most here are the two refusals. A signed-in employee
in no groups could delete any document they could read - a 200, the file
gone, the corpus one document smaller - which is the kind of hole that is
obvious once seen and invisible until someone drives it. And the log must
never carry a secret: an editable setting can be an API key, so the entry
records which settings changed and never what they were set to.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi.testclient import TestClient

from fake_idp import FakeIdp
from openknowledge.api.app import create_app
from openknowledge.config import Settings
from openknowledge.knowledge.store import Actor, KnowledgeStore

CLIENT_ID = "ok-roles-test"
ADMIN_TOKEN = "admin-token-for-tests"
ADMIN_GROUP = "g-admins"
CURATOR_GROUP = "g-curators"


# -- the store ---------------------------------------------------------------


def test_an_entry_says_who_what_and_when(tmp_path: Path) -> None:
    with KnowledgeStore(tmp_path / "k.db") as store:
        store.record_action(
            Actor(id="oid-1", name="Alice Moreau", kind="person"),
            "access.set",
            "hr",
            {"principals": ["group:hr"]},
        )
        (entry,) = store.admin_actions()

    assert entry.actor.id == "oid-1"
    assert entry.actor.name == "Alice Moreau"
    assert entry.actor.kind == "person"
    assert entry.action == "access.set"
    assert entry.target == "hr"
    assert entry.detail == {"principals": ["group:hr"]}
    assert entry.at > 0


def test_the_newest_change_is_the_one_you_read_first(tmp_path: Path) -> None:
    with KnowledgeStore(tmp_path / "k.db") as store:
        for name in ("first", "second", "third"):
            store.record_action(Actor.token(), "reindex", name)
        assert [e.target for e in store.admin_actions()] == ["third", "second", "first"]
        assert [e.target for e in store.admin_actions(limit=2)] == ["third", "second"]


def test_a_shared_token_names_nobody_and_says_so(tmp_path: Path) -> None:
    """The one thing a log cannot do is recover an identity that was never
    captured. It records that, rather than inventing an admin."""
    with KnowledgeStore(tmp_path / "k.db") as store:
        store.record_action(Actor.token(), "reindex")
        (entry,) = store.admin_actions()
    assert entry.actor.kind == "token"
    assert entry.actor.name == "shared admin token"


def test_one_unreadable_row_does_not_hide_the_rest(tmp_path: Path) -> None:
    """A restored or hand-edited database can hold anything in the detail
    column; a viewer that raises on one row hides every row after it."""
    path = tmp_path / "k.db"
    with KnowledgeStore(path) as store:
        store.record_action(Actor.token(), "reindex", "before")
        store.record_action(Actor.token(), "reindex", "after")
        raw = sqlite3.connect(path)
        raw.execute("UPDATE admin_actions SET detail = 'not json at all' WHERE target = 'before'")
        raw.commit()
        raw.close()
        entries = store.admin_actions()

    assert [e.target for e in entries] == ["after", "before"]
    assert entries[1].detail == {}


def test_a_failed_write_never_fails_the_change_it_records(tmp_path: Path) -> None:
    """Losing the record of a change is bad; refusing the change because the
    record would not write turns an audit trail into an outage."""
    store = KnowledgeStore(tmp_path / "k.db")
    store.close()
    store.record_action(Actor.token(), "reindex")  # must not raise


def test_a_detail_that_is_not_json_is_still_recorded(tmp_path: Path) -> None:
    with KnowledgeStore(tmp_path / "k.db") as store:
        store.record_action(Actor.token(), "reindex", detail={"when": object()})
        (entry,) = store.admin_actions()
    assert "when" in entry.detail


def test_asking_a_question_writes_nothing_here(tmp_path: Path) -> None:
    """The deliberate asymmetry: admin actions are attributed, questions are
    not. The gaps report has its own test that it cannot name anyone; this
    is the other half of the same promise."""
    with KnowledgeStore(tmp_path / "k.db") as store:
        assert store.admin_action_count() == 0


# -- through HTTP ------------------------------------------------------------


@pytest.fixture
def idp():
    provider = FakeIdp()
    yield provider
    provider.close()


def _settings(tmp_path: Path, idp: FakeIdp, **overrides: object) -> Settings:
    values: dict = {
        "data_dir": str(tmp_path / "data"),
        "documents_dir": str(tmp_path / "documents"),
        "auth_mode": "oidc",
        "oidc_issuer": idp.issuer,
        "oidc_client_id": CLIENT_ID,
        "oidc_client_secret": "s3cret",
        "oidc_admin_group": ADMIN_GROUP,
        "oidc_curator_group": CURATOR_GROUP,
        "admin_token": ADMIN_TOKEN,
        "upload_enabled": True,
        "local_enabled": False,
        "embedding_enabled": False,
        "_env_file": None,
    }
    values.update(overrides)
    return Settings(**values)


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    documents = tmp_path / "documents"
    (documents / "hr").mkdir(parents=True)
    (documents / "handbook.md").write_text("# Handbook\n\nThe office closes at 18:00.\n")
    (documents / "hr" / "leave.md").write_text("# Leave\n\nParental leave is 20 weeks.\n")
    return documents


@pytest.fixture
def app(tmp_path: Path, idp: FakeIdp, corpus: Path):
    return create_app(_settings(tmp_path, idp))


def sign_in(app, idp: FakeIdp, *, subject: str, name: str, groups: tuple[str, ...]) -> TestClient:
    """A browser that has been through the whole flow, as this person."""
    client = TestClient(app)
    started = client.get("/auth/login", follow_redirects=False)
    sent = {k: v[0] for k, v in parse_qs(urlparse(started.headers["location"]).query).items()}
    code = idp.mint_code(
        audience=CLIENT_ID, nonce=sent["nonce"], subject=subject, name=name, groups=groups
    )
    landed = client.get(f"/auth/callback?code={code}&state={sent['state']}", follow_redirects=False)
    assert landed.status_code == 302
    return client


@pytest.fixture
def admin(app, idp: FakeIdp) -> TestClient:
    return sign_in(app, idp, subject="alice-oid", name="Alice Moreau", groups=(ADMIN_GROUP,))


@pytest.fixture
def curator(app, idp: FakeIdp) -> TestClient:
    return sign_in(app, idp, subject="cara-oid", name="Cara Nadeau", groups=(CURATOR_GROUP,))


@pytest.fixture
def employee(app, idp: FakeIdp) -> TestClient:
    return sign_in(app, idp, subject="bob-oid", name="Bob Ordinary", groups=())


def test_a_signed_in_admin_is_named_in_the_log(app, admin) -> None:
    with admin:
        assert admin.put("/admin/access/hr", json={"principals": ["group:hr"]}).status_code == 200
        log = admin.get("/admin/log").json()

    assert (log["attributed"], log["returned"], log["total"]) == (1, 1, 1)
    (entry,) = log["entries"]
    assert entry["actor"] == "Alice Moreau"
    assert entry["actor_id"] == "alice-oid"
    assert entry["actor_kind"] == "person"
    assert entry["action"] == "access.set"
    assert entry["target"] == "hr"


def test_the_shared_token_is_recorded_as_naming_nobody(app) -> None:
    with TestClient(app) as client:
        headers = {"authorization": f"Bearer {ADMIN_TOKEN}"}
        assert client.post("/admin/reindex", headers=headers).status_code == 200
        log = client.get("/admin/log", headers=headers).json()

    assert log["attributed"] == 0
    assert log["entries"][0]["actor_kind"] == "token"


def test_a_settings_change_records_the_names_and_never_the_values(app) -> None:
    """An editable setting's value can be a credential - a base URL with a
    password in it is the ordinary way that happens. A log that recorded
    values would put one in every backup taken afterwards, so it records
    which settings changed and never what they became."""
    secret = "https://svc:sup3r-secret@embeddings.internal"
    with TestClient(app) as client:
        headers = {"authorization": f"Bearer {ADMIN_TOKEN}"}
        applied = client.put(
            "/admin/settings",
            json={"embedding_base_url": secret, "retrieval_k": 9},
            headers=headers,
        )
        assert applied.status_code == 200, applied.text
        log = client.get("/admin/log", headers=headers).json()

    assert "sup3r-secret" not in json.dumps(log)
    (entry,) = [e for e in log["entries"] if e["action"] == "settings.update"]
    assert sorted(entry["detail"]["settings"]) == ["embedding_base_url", "retrieval_k"]


def test_the_log_names_who_deleted_a_document(app, curator) -> None:
    """The question this whole feature exists to answer."""
    with curator:
        assert curator.delete("/documents/handbook.md").status_code == 200
    with TestClient(app) as client:
        log = client.get("/admin/log", headers={"authorization": f"Bearer {ADMIN_TOKEN}"}).json()

    (entry,) = [e for e in log["entries"] if e["action"] == "document.delete"]
    assert entry["actor"] == "Cara Nadeau"
    assert entry["target"] == "handbook.md"


# -- roles -------------------------------------------------------------------

GOVERNANCE = [
    ("GET", "/admin/log", None),
    ("GET", "/admin/settings", None),
    ("GET", "/admin/config", None),
    ("GET", "/admin/access", None),
    ("PUT", "/admin/access/hr", {"principals": ["group:hr"]}),
    ("DELETE", "/admin/access/hr", None),
]
CURATION = [
    ("GET", "/admin/pins", None),
    ("GET", "/admin/proposals", None),
    ("GET", "/admin/conflicts", None),
    ("GET", "/admin/costs", None),
    ("GET", "/admin/gaps", None),
    ("POST", "/admin/reindex", None),
]


@pytest.mark.parametrize(("verb", "path", "body"), GOVERNANCE)
def test_a_curator_does_not_hold_governance(app, curator, verb, path, body) -> None:
    with curator:
        response = curator.request(verb, path, json=body) if body else curator.request(verb, path)
    assert response.status_code == 403
    assert "administrator" in response.json()["detail"]


@pytest.mark.parametrize(("verb", "path", "body"), CURATION)
def test_a_curator_curates(app, curator, verb, path, body) -> None:
    with curator:
        response = curator.request(verb, path, json=body) if body else curator.request(verb, path)
    assert response.status_code == 200, response.text


@pytest.mark.parametrize(("verb", "path", "body"), GOVERNANCE + CURATION)
def test_an_ordinary_employee_holds_neither(app, employee, verb, path, body) -> None:
    with employee:
        r = employee.request(verb, path, json=body) if body else employee.request(verb, path)
    assert r.status_code == 403


def test_a_curator_can_pin_and_the_pin_is_theirs(app, curator) -> None:
    with curator:
        created = curator.post(
            "/admin/pins",
            json={"question": "when does the office close?", "answer": "18:00.", "cite": []},
        )
        assert created.status_code == 201, created.text
    with TestClient(app) as client:
        log = client.get("/admin/log", headers={"authorization": f"Bearer {ADMIN_TOKEN}"}).json()
    (entry,) = [e for e in log["entries"] if e["action"] == "pin.create"]
    assert entry["actor"] == "Cara Nadeau"


def test_a_signed_in_person_is_told_they_are_not_an_admin(app, employee) -> None:
    """Not 'invalid admin token': they hold no token and never should. The
    old message sent curators hunting for a credential that was not theirs."""
    with employee:
        refused = employee.get("/admin/settings")
    assert refused.status_code == 403
    detail = refused.json()["detail"]
    assert "not an administrator" in detail
    assert "admin group" in detail, "and where the answer lives"
    assert "token" not in detail


def test_a_wrong_token_is_a_wrong_token_even_from_a_signed_in_person(app, employee) -> None:
    """Somebody who presented a token meant to use one. Telling them their
    account is not an administrator would hide a typo in an integration."""
    with employee:
        refused = employee.get("/admin/settings", headers={"authorization": "Bearer nope"})
    assert refused.status_code == 401
    assert refused.json()["detail"] == "invalid admin token"


def test_the_admin_token_still_works_for_anyone_holding_it(app, employee) -> None:
    """Automation carries the token, not a session. It must keep working."""
    with employee:
        allowed = employee.get(
            "/admin/settings", headers={"authorization": f"Bearer {ADMIN_TOKEN}"}
        )
    assert allowed.status_code == 200


def test_with_no_curator_group_the_curator_surface_is_the_admin_surface(
    tmp_path: Path, idp: FakeIdp, corpus: Path
) -> None:
    """An install that never configures one behaves exactly as it always did."""
    app = create_app(_settings(tmp_path, idp, oidc_curator_group=""))
    client = sign_in(app, idp, subject="cara-oid", name="Cara", groups=(CURATOR_GROUP,))
    with client:
        assert client.get("/admin/pins").status_code == 403
        assert client.post("/admin/reindex").status_code == 403


# -- the hole this closed ----------------------------------------------------


def test_an_employee_cannot_delete_a_document(app, employee, corpus: Path) -> None:
    """Before this, a signed-in employee in no groups could delete anything
    they could read: a 200, and the handbook gone."""
    with employee:
        refused = employee.delete("/documents/handbook.md")
    assert refused.status_code == 403
    assert (corpus / "handbook.md").is_file()


def test_an_employee_cannot_overwrite_a_document(app, employee, corpus: Path) -> None:
    """Replacing a document's contents is a delete wearing an upload's
    clothes, and was the same hole by another route."""
    with employee:
        response = employee.post(
            "/documents",
            files={"files": ("handbook.md", b"# Handbook\n\nThe office never closes.\n")},
        )
    assert response.status_code == 201
    assert response.json()["stored"] == []
    assert "administrator" in response.json()["skipped"][0]["reason"]
    assert "18:00" in (corpus / "handbook.md").read_text()


def test_an_employee_can_still_contribute_a_new_document(app, employee, corpus: Path) -> None:
    """Contribution is the point of the uploads switch; only replacing and
    removing needed a role."""
    with employee:
        response = employee.post(
            "/documents",
            files={"files": ("expenses.md", b"# Expenses\n\nThe meal allowance is EUR 45.\n")},
        )
    assert response.status_code == 201, response.text
    assert [s["name"] for s in response.json()["stored"]] == ["expenses.md"]
    assert (corpus / "expenses.md").is_file()


def test_a_curator_may_delete_and_replace(app, curator, corpus: Path) -> None:
    with curator:
        assert curator.delete("/documents/handbook.md").status_code == 200
    assert not (corpus / "handbook.md").exists()


def test_with_sign_in_off_nothing_changed(tmp_path: Path, corpus: Path) -> None:
    """The desktop app and a trusted LAN have no identity to check against.
    Reaching the port was always full control there; a role check against an
    identity that does not exist would be theatre, not security."""
    settings = Settings(
        data_dir=str(tmp_path / "data"),
        documents_dir=str(corpus),
        upload_enabled=True,
        local_enabled=False,
        embedding_enabled=False,
        _env_file=None,  # type: ignore[call-arg]
    )
    with TestClient(create_app(settings)) as client:
        assert client.delete("/documents/handbook.md").status_code == 200
    assert not (corpus / "handbook.md").exists()
