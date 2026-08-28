"""Sign-in through real HTTP: the gate, the flow, and the ACL truth.

The test that matters most here is the last kind: a document restricted to
an HR group, served to an HR member and refused to everyone else - through
the same endpoints, the same pinned tier, the same session machinery a
deployment runs. The enforcement inside retrieval and the cache has its own
tests; these prove the badge reader in front of it mints the principals
those checks were waiting for.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi.testclient import TestClient

from fake_idp import FakeIdp
from openknowledge.api.app import create_app
from openknowledge.config import Settings
from openknowledge.retrieval.base import Document

CLIENT_ID = "ok-test-client"
ADMIN_TOKEN = "admin-token-for-tests"


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
        "admin_token": ADMIN_TOKEN,
        "upload_enabled": True,
        "local_enabled": False,
        "embedding_enabled": False,
        "_env_file": None,
    }
    values.update(overrides)
    return Settings(**values)


@pytest.fixture
def client(tmp_path: Path, idp: FakeIdp):
    with TestClient(create_app(_settings(tmp_path, idp))) as c:
        yield c


def sign_in(client: TestClient, idp: FakeIdp, *, subject="alice", groups=()) -> None:
    """Drive the whole flow the way a browser would."""
    started = client.get("/auth/login", follow_redirects=False)
    assert started.status_code == 302
    sent = {k: v[0] for k, v in parse_qs(urlparse(started.headers["location"]).query).items()}
    code = idp.mint_code(audience=CLIENT_ID, nonce=sent["nonce"], subject=subject, groups=groups)
    landed = client.get(f"/auth/callback?code={code}&state={sent['state']}", follow_redirects=False)
    assert landed.status_code == 302 and landed.headers["location"] == "/"


# -- the gate ---------------------------------------------------------------


def test_signed_out_browsers_are_sent_to_sign_in(client) -> None:
    response = client.get("/", headers={"accept": "text/html"}, follow_redirects=False)
    assert response.status_code == 302
    assert response.headers["location"] == "/auth/login"


def test_signed_out_api_calls_get_401_not_a_redirect(client) -> None:
    assert client.post("/chat", json={"question": "hi"}).status_code == 401
    assert client.get("/documents").status_code == 401


def test_monitoring_needs_no_identity(client) -> None:
    assert client.get("/healthz").status_code == 200
    assert client.get("/favicon.ico").status_code == 200


def test_the_full_round_trip_signs_a_browser_in(client, idp) -> None:
    sign_in(client, idp)
    page = client.get("/", headers={"accept": "text/html"})
    assert page.status_code == 200 and "OpenKnowledge" in page.text


def test_sign_out_ends_the_session(client, idp) -> None:
    sign_in(client, idp)
    out = client.post("/auth/logout", follow_redirects=False)
    assert out.status_code == 302
    after = client.get("/", headers={"accept": "text/html"}, follow_redirects=False)
    assert after.status_code == 302, "the session survived sign-out"


def test_an_expired_session_stops_working(client, idp, monkeypatch) -> None:
    from openknowledge.auth import sessions as module

    sign_in(client, idp)
    monkeypatch.setattr(module, "_now", lambda: module.time.time() + 9 * 3600)
    assert client.post("/chat", json={"question": "hi"}).status_code == 401


def test_a_state_cannot_be_replayed(client, idp) -> None:
    started = client.get("/auth/login", follow_redirects=False)
    sent = {k: v[0] for k, v in parse_qs(urlparse(started.headers["location"]).query).items()}
    code = idp.mint_code(audience=CLIENT_ID, nonce=sent["nonce"])
    first = client.get(f"/auth/callback?code={code}&state={sent['state']}", follow_redirects=False)
    assert first.status_code == 302
    replay = client.get(f"/auth/callback?code={code}&state={sent['state']}", follow_redirects=False)
    assert replay.status_code == 400


def test_the_idp_saying_no_is_relayed_readably(client) -> None:
    response = client.get("/auth/callback?error=access_denied&error_description=Blocked+by+policy")
    assert response.status_code == 400
    assert "Blocked by policy" in response.text


def test_a_rejected_token_names_the_reason(client, idp) -> None:
    started = client.get("/auth/login", follow_redirects=False)
    sent = {k: v[0] for k, v in parse_qs(urlparse(started.headers["location"]).query).items()}
    code = idp.mint_code(audience=CLIENT_ID, nonce=sent["nonce"], overage=True)
    response = client.get(f"/auth/callback?code={code}&state={sent['state']}")
    assert response.status_code == 400
    assert "groups assigned to the application" in response.text


def test_the_session_cookie_is_hardened(client, idp) -> None:
    started = client.get("/auth/login", follow_redirects=False)
    sent = {k: v[0] for k, v in parse_qs(urlparse(started.headers["location"]).query).items()}
    code = idp.mint_code(audience=CLIENT_ID, nonce=sent["nonce"])
    landed = client.get(f"/auth/callback?code={code}&state={sent['state']}", follow_redirects=False)
    cookie = landed.headers["set-cookie"]
    assert "HttpOnly" in cookie and "SameSite=lax" in cookie


def test_an_https_public_url_marks_the_cookie_secure(tmp_path, idp) -> None:
    settings = _settings(tmp_path, idp, public_url="https://kb.example.com")
    with TestClient(create_app(settings)) as client:
        started = client.get("/auth/login", follow_redirects=False)
        sent = {k: v[0] for k, v in parse_qs(urlparse(started.headers["location"]).query).items()}
        assert sent["redirect_uri"] == "https://kb.example.com/auth/callback"
        code = idp.mint_code(audience=CLIENT_ID, nonce=sent["nonce"])
        landed = client.get(
            f"/auth/callback?code={code}&state={sent['state']}", follow_redirects=False
        )
        assert "Secure" in landed.headers["set-cookie"]


# -- principals: minted, never asserted -------------------------------------


def test_a_signed_in_request_cannot_assert_principals(client, idp) -> None:
    sign_in(client, idp)
    response = client.post("/chat", json={"question": "hi", "principals": ["group:g-hr"]})
    assert response.status_code == 400
    assert "sign-in" in response.json()["detail"]


def test_the_admin_token_keeps_the_trusted_caller_mode(client) -> None:
    """A bot backend relaying per-user principals authenticates with the
    admin token and keeps working - it already holds every admin write."""
    headers = {"Authorization": f"Bearer {ADMIN_TOKEN}"}
    assert client.get("/admin/costs", headers=headers).status_code == 200
    response = client.post(
        "/chat", json={"question": "hi", "principals": ["group:g-hr"]}, headers=headers
    )
    assert response.status_code == 200


def test_with_sign_in_off_nothing_changes(tmp_path) -> None:
    settings = Settings(
        data_dir=str(tmp_path / "data"),
        documents_dir=str(tmp_path / "documents"),
        local_enabled=False,
        embedding_enabled=False,
        _env_file=None,  # type: ignore[call-arg]
    )
    with TestClient(create_app(settings)) as client:
        assert client.get("/auth/login").status_code == 404
        assert client.get("/auth/me").status_code == 404
        response = client.post("/chat", json={"question": "hi", "principals": ["anything"]})
        assert response.status_code == 200, "trusted-caller mode must survive"


def test_misconfigured_sign_in_fails_at_startup_not_at_noon(tmp_path, idp) -> None:
    with pytest.raises(RuntimeError, match="OK_OIDC_CLIENT_ID"):
        create_app(_settings(tmp_path, idp, oidc_client_id=""))


# -- who am I, and admins as a group ----------------------------------------


def test_auth_me_renders_the_session_and_nothing_else(client, idp) -> None:
    assert client.get("/auth/me").json() == {"signed_in": False}
    sign_in(client, idp, subject="alice", groups=("g-hr",))
    body = client.get("/auth/me").json()
    assert body == {"signed_in": True, "name": "Test Person", "admin": False}


def test_the_admin_group_is_the_admin_credential(tmp_path, idp) -> None:
    """Membership in OK_OIDC_ADMIN_GROUP grants /manage's API without the
    shared token - granted and revoked in the directory, like everything
    else about a person."""
    settings = _settings(tmp_path, idp, oidc_admin_group="g-admins", admin_token=None)
    with TestClient(create_app(settings)) as client:
        sign_in(client, idp, subject="root", groups=("g-admins",))
        assert client.get("/admin/costs").status_code == 200, "no token, group is enough"
        assert client.get("/auth/me").json()["admin"] is True

        client.cookies.clear()
        sign_in(client, idp, subject="pleb", groups=("g-hr",))
        denied = client.get("/admin/costs")
        assert denied.status_code == 403
        assert "admin group" in denied.json()["detail"]
        assert client.get("/auth/me").json()["admin"] is False


def test_the_token_still_works_beside_the_admin_group(tmp_path, idp) -> None:
    settings = _settings(tmp_path, idp, oidc_admin_group="g-admins")
    with TestClient(create_app(settings)) as client:
        headers = {"Authorization": f"Bearer {ADMIN_TOKEN}"}
        assert client.get("/admin/costs", headers=headers).status_code == 200


# -- the ACL truth, end to end ----------------------------------------------


RESTRICTED = Document(
    document_id="salary-bands",
    title="Salary Bands",
    text="# Salary Bands\n\nBand C pays EUR 70,000 per year.",
    allowed_principals=frozenset({"group:g-hr"}),
)
PUBLIC = Document(
    document_id="handbook",
    title="Office Handbook",
    text="# Office Handbook\n\nThe office closes at 18:00.",
)


def _index_and_pin(client: TestClient) -> None:
    client.app.state.engine.retriever.index([RESTRICTED, PUBLIC])
    pinned = client.post(
        "/admin/pins",
        json={
            "question": "What does band C pay?",
            "answer": "Band C pays EUR 70,000 per year. [salary-bands]",
            "cite": ["salary-bands"],
        },
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    assert pinned.status_code == 201


def test_an_hr_member_gets_the_answer_everyone_else_does_not(client, idp) -> None:
    _index_and_pin(client)

    sign_in(client, idp, subject="alice", groups=("g-hr",))
    served = client.post("/chat", json={"question": "What does band C pay?"}).json()
    assert served["tier"] == "pinned"
    assert "70,000" in served["answer"]

    client.cookies.clear()
    sign_in(client, idp, subject="bob", groups=())
    denied = client.post("/chat", json={"question": "What does band C pay?"}).json()
    assert denied["tier"] != "pinned"
    assert "70,000" not in denied["answer"], "the ACL leaked through the pinned tier"


def test_the_corpus_listing_hides_what_the_asker_cannot_read(client, idp) -> None:
    _index_and_pin(client)

    sign_in(client, idp, subject="alice", groups=("g-hr",))
    hr_view = client.post("/chat", json={"question": "What documents do you have?"}).json()
    assert "Salary Bands" in hr_view["answer"]

    client.cookies.clear()
    sign_in(client, idp, subject="bob", groups=())
    outsider = client.post("/chat", json={"question": "What documents do you have?"}).json()
    assert "Salary Bands" not in outsider["answer"]
    assert "Office Handbook" in outsider["answer"]


def test_the_stream_is_gated_and_serves_the_same_answer(client, idp) -> None:
    _index_and_pin(client)
    assert client.post("/chat/stream", json={"question": "hi"}).status_code == 401

    sign_in(client, idp, subject="alice", groups=("g-hr",))
    with client.stream(
        "POST", "/chat/stream", json={"question": "What does band C pay?"}
    ) as stream:
        assert stream.status_code == 200
        body = "".join(stream.iter_text())
    assert '"final"' in body and "70,000" in body
