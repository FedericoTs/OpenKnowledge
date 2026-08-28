"""The OIDC client and the session store, against a provider we control.

Every negative test here is a lie the fake provider was told to tell -
wrong issuer, wrong audience, expired token, replayed nonce, the groups
overage - and the client must refuse each one with a reason a person can
read. The positive path is exercised end-to-end through HTTP in
test_auth_api.py; these tests hold the pieces to their contracts.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from fake_idp import FakeIdp
from openknowledge.auth.oidc import Identity, OidcClient, OidcError
from openknowledge.auth.sessions import SessionStore

CLIENT_ID = "ok-test-client"


@pytest.fixture
def idp():
    provider = FakeIdp()
    yield provider
    provider.close()


@pytest.fixture
async def oidc(idp: FakeIdp):
    client = OidcClient(issuer=idp.issuer, client_id=CLIENT_ID, client_secret="s3cret")
    yield client
    await client.aclose()


async def _login(oidc: OidcClient, idp: FakeIdp, **mint: object) -> Identity:
    """Drive the full client-side flow for an identity the test chose."""
    url, pending = await oidc.begin_login("http://app.example/auth/callback")
    code = idp.mint_code(audience=CLIENT_ID, nonce=pending.nonce, **mint)  # type: ignore[arg-type]
    return await oidc.complete_login(code, pending, "http://app.example/auth/callback")


# -- the flow itself --------------------------------------------------------


async def test_the_authorization_url_carries_the_whole_contract(oidc, idp) -> None:
    url, pending = await oidc.begin_login("http://app.example/auth/callback")
    parsed = urlparse(url)
    params = {k: v[0] for k, v in parse_qs(parsed.query).items()}
    assert url.startswith(f"{idp.issuer}/authorize?")
    assert params["response_type"] == "code"
    assert params["client_id"] == CLIENT_ID
    assert params["redirect_uri"] == "http://app.example/auth/callback"
    assert params["state"] == pending.state
    assert params["nonce"] == pending.nonce
    assert params["code_challenge_method"] == "S256"
    assert params["code_challenge"]  # derived from the verifier, never the verifier


async def test_a_signed_token_becomes_an_identity(oidc, idp) -> None:
    identity = await _login(oidc, idp, subject="alice", groups=("g-hr", "g-all"))
    assert identity.subject == "alice"
    assert identity.name == "Test Person"
    assert identity.groups == ("g-hr", "g-all")


async def test_the_exchange_sends_the_verifier_and_the_secret(oidc, idp) -> None:
    await _login(oidc, idp)
    sent = idp.last_token_request
    assert sent["grant_type"] == "authorization_code"
    assert sent["client_id"] == CLIENT_ID
    assert sent["client_secret"] == "s3cret"
    assert len(sent["code_verifier"]) >= 43  # RFC 7636's floor


async def test_oid_wins_over_the_pairwise_sub(oidc, idp) -> None:
    identity = await _login(oidc, idp, subject="alice")
    assert identity.subject == "alice"  # not "pairwise-alice"


async def test_sub_serves_when_there_is_no_oid(oidc, idp) -> None:
    identity = await _login(oidc, idp, drop=("oid",))
    assert identity.subject == "pairwise-user-1"


# -- every lie is refused, with a readable reason ---------------------------


@pytest.mark.parametrize(
    ("mint", "expected"),
    [
        ({"claims": {"iss": "https://evil.example"}}, "failed validation"),
        ({"claims": {"aud": "someone-else"}}, "failed validation"),
        ({"claims": {"exp": 1}}, "failed validation"),
        ({"claims": {"nonce": "stolen"}}, "nonce"),
        ({"drop": ("nonce",)}, "nonce"),
        ({"overage": True}, "groups assigned to the application"),
        ({"drop": ("oid", "sub")}, "neither oid nor sub"),
        ({"claims": {"groups": "not-a-list"}}, "not a list"),
    ],
)
async def test_bad_tokens_are_refused(oidc, idp, mint, expected) -> None:
    with pytest.raises(OidcError, match=expected):
        await _login(oidc, idp, **mint)


async def test_a_mismatched_discovery_issuer_is_refused(idp) -> None:
    client = OidcClient(issuer=f"{idp.issuer}/tenant-typo", client_id=CLIENT_ID)
    try:
        with pytest.raises(OidcError, match="discovery"):
            await client.provider()
    finally:
        await client.aclose()


# -- sessions ---------------------------------------------------------------


def _identity(**kw: object) -> Identity:
    values: dict = {"subject": "alice", "name": "Alice", "groups": ("g-hr",)}
    values.update(kw)
    return Identity(**values)


def test_a_session_round_trips_and_mints_principals(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "auth.db")
    token = store.create(_identity(), ttl_seconds=60)
    session = store.get(token)
    assert session is not None
    assert session.principals == frozenset({"authenticated", "user:alice", "group:g-hr"})
    store.delete(token)
    assert store.get(token) is None
    store.close()


def test_the_database_never_holds_the_raw_token(tmp_path: Path) -> None:
    """A copied auth.db must not be a bag of valid cookies."""
    store = SessionStore(tmp_path / "auth.db")
    token = store.create(_identity(), ttl_seconds=60)
    rows = sqlite3.connect(tmp_path / "auth.db").execute("SELECT token_hash FROM sessions")
    stored = [r[0] for r in rows]
    assert stored and all(token not in value for value in stored)
    store.close()


def test_an_expired_session_is_gone(tmp_path: Path, monkeypatch) -> None:
    from openknowledge.auth import sessions as module

    store = SessionStore(tmp_path / "auth.db")
    token = store.create(_identity(), ttl_seconds=60)
    monkeypatch.setattr(module, "_now", lambda: module.time.time() + 61)
    assert store.get(token) is None
    store.close()


def test_a_pending_login_is_single_use(tmp_path: Path, monkeypatch) -> None:
    from openknowledge.auth import sessions as module
    from openknowledge.auth.oidc import PendingLogin

    store = SessionStore(tmp_path / "auth.db")
    pending = PendingLogin(state="s1", nonce="n1", code_verifier="v1", created_at=1000.0)
    monkeypatch.setattr(module, "_now", lambda: 1001.0)
    store.save_pending(pending)
    assert store.take_pending("s1") == pending
    assert store.take_pending("s1") is None, "a state must not be redeemable twice"

    stale = PendingLogin(state="s2", nonce="n2", code_verifier="v2", created_at=1000.0)
    store.save_pending(stale)
    monkeypatch.setattr(module, "_now", lambda: 1000.0 + 601)
    assert store.take_pending("s2") is None, "a stale state must be refused"
    store.close()
