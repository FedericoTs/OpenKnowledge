"""Changing who may read a folder, without rebuilding the corpus for it.

An access rule decides a document's audience and nothing else about it: not
its text, not how it chunks, not the statistics BM25 scores with, and not
``corpus_version``, which hashes content. Applying one used to re-index -
reading every file off disk, re-parsing it, re-tokenising every passage and
re-running contradiction detection - to arrive at an index identical but for
one field per passage. Measured on 1,200 documents that was nine seconds,
inside the request the admin was waiting on, and it grew with the corpus.

Two things these tests hold. The audiences a re-stamp produces are the ones a
rebuild would have produced - if they ever differ, this is not an optimisation
but a quiet change to who can read what. And the change is still applied
before the response returns, because the reason it was synchronous was never
speed: it is that no window may exist in which a rule is stored and the index
is still serving the old audience.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi.testclient import TestClient

from fake_idp import FakeIdp
from openknowledge.api.app import create_app
from openknowledge.api.engine import build_engine
from openknowledge.config import Settings

TOKEN = "t0ken"
ADMIN = {"authorization": f"Bearer {TOKEN}"}


def _corpus(root: Path) -> Path:
    documents = root / "documents"
    for folder in ("hr", "hr/archive", "travel"):
        (documents / folder).mkdir(parents=True, exist_ok=True)
        (documents / folder / "policy.md").write_text(
            f"# {folder}\n\nParental leave is 20 weeks, fully paid.\n"
        )
    return documents


def _settings(root: Path, documents: Path, **over) -> Settings:
    values: dict = {
        "data_dir": str(root / "data"),
        "documents_dir": str(documents),
        "admin_token": TOKEN,
        "local_enabled": False,
        "embedding_enabled": False,
        "_env_file": None,
    }
    values.update(over)
    return Settings(**values)


def _audiences(engine) -> dict[str, frozenset[str]]:
    return {c.document_id: frozenset(c.allowed_principals) for c in engine.retriever.chunks}


def test_a_restamp_lands_where_a_rebuild_would_have(tmp_path: Path) -> None:
    """The property the whole change rests on. If these ever differ, this is
    not an optimisation - it is a quiet change to who can read what."""
    rebuilt = build_engine(_settings(tmp_path / "a", _corpus(tmp_path / "a")))
    restamped = build_engine(_settings(tmp_path / "b", _corpus(tmp_path / "b")))
    try:
        for engine in (rebuilt, restamped):
            engine.reindex()
            engine.knowledge.set_folder_access("hr", frozenset({"group:hr"}))
            engine.knowledge.set_folder_access("hr/archive", frozenset({"group:records"}))

        rebuilt.reindex()
        restamped.reapply_access()

        assert _audiences(restamped) == _audiences(rebuilt)
        assert restamped.retriever.corpus_version == rebuilt.retriever.corpus_version
        # And the deepest rule still wins, which is the thing a flat re-stamp
        # would be most likely to get wrong.
        assert _audiences(restamped)["hr-policy"] == frozenset({"group:hr"})
        assert _audiences(restamped)["hr-archive-policy"] == frozenset({"group:records"})
        assert _audiences(restamped)["travel-policy"] == frozenset()
    finally:
        for engine in (rebuilt, restamped):
            engine.store.close()
            engine.knowledge.close()


def test_a_document_the_rules_say_nothing_about_keeps_what_it_had(tmp_path: Path) -> None:
    """Silence is not permission. The re-stamp is told what the rules now say,
    and reading an absent entry as "open to everyone" would be a way to widen
    access by omission."""
    documents = _corpus(tmp_path)
    engine = build_engine(_settings(tmp_path, documents))
    try:
        engine.reindex()
        engine.knowledge.set_folder_access("hr", frozenset({"group:hr"}))
        engine.reapply_access()
        assert _audiences(engine)["hr-policy"] == frozenset({"group:hr"})

        # A mapping that mentions nobody must change nobody.
        assert engine.retriever.restamp({}) == 0
        assert _audiences(engine)["hr-policy"] == frozenset({"group:hr"})
    finally:
        engine.store.close()
        engine.knowledge.close()


def test_the_rule_is_in_force_before_the_response_returns(tmp_path: Path) -> None:
    """The reason this was synchronous was never speed. A rule stored while
    the index still serves the old audience is a window in which the wrong
    person is answered, and the response saying "done" is what an admin
    believes."""
    idp = FakeIdp()
    try:
        documents = _corpus(tmp_path)
        app = create_app(
            _settings(
                tmp_path,
                documents,
                auth_mode="oidc",
                oidc_issuer=idp.issuer,
                oidc_client_id="ok-restamp",
                oidc_client_secret="s3cret",
                # The listing is explicitly "what is in the documents folder,
                # as this viewer may see it" - the ACL-filtered surface a
                # person actually looks at, so it is the one to check.
                upload_enabled=True,
            )
        )
        with TestClient(app):
            outsider = TestClient(app)
            started = outsider.get("/auth/login", follow_redirects=False)
            sent = {
                k: v[0] for k, v in parse_qs(urlparse(started.headers["location"]).query).items()
            }
            code = idp.mint_code(audience="ok-restamp", nonce=sent["nonce"], subject="bob")
            outsider.get(
                f"/auth/callback?code={code}&state={sent['state']}", follow_redirects=False
            )

            listed = outsider.get("/documents").json()
            assert any("hr" in f["name"] for f in listed["files"]), "hr is readable to start with"

            applied = outsider.put(
                "/admin/access/hr", json={"principals": ["group:hr"]}, headers=ADMIN
            )
            assert applied.status_code == 200, applied.text
            assert applied.json()["corpus"]["passages_restamped"] >= 1

            # The very next request, with no re-index in between.
            after = outsider.get("/documents").json()
            assert not any("hr" in f["name"] for f in after["files"]), (
                "the rule was in force the moment the response returned"
            )
    finally:
        idp.close()


def test_opening_a_folder_again_takes_effect_the_same_way(tmp_path: Path) -> None:
    documents = _corpus(tmp_path)
    with TestClient(create_app(_settings(tmp_path, documents))) as client:
        client.put("/admin/access/hr", json={"principals": ["group:hr"]}, headers=ADMIN)
        cleared = client.delete("/admin/access/hr", headers=ADMIN)
        assert cleared.status_code == 200, cleared.text
        assert cleared.json()["corpus"]["passages_restamped"] >= 1
        assert cleared.json()["open"] is True


def test_an_access_change_evicts_no_cached_answers(tmp_path: Path) -> None:
    """corpus_version hashes content, so an access change does not invalidate
    a single cached answer - and it must not, because a cached answer's
    sources are re-checked against whoever is asking at read time. Evicting
    them would throw away the hit rate the cost model depends on for nothing.
    """
    documents = _corpus(tmp_path)
    with TestClient(create_app(_settings(tmp_path, documents))) as client:
        before = client.get("/healthz").json()["corpus_version"]
        applied = client.put(
            "/admin/access/hr", json={"principals": ["group:hr"]}, headers=ADMIN
        ).json()
        assert applied["corpus"]["answers_evicted"] == 0
        assert applied["corpus"]["corpus_version"] == before


@pytest.mark.parametrize("folder", ["hr", "hr/archive"])
def test_the_admin_log_still_records_who_changed_it(tmp_path: Path, folder: str) -> None:
    """Re-stamping instead of rebuilding must not lose the attribution."""
    documents = _corpus(tmp_path)
    with TestClient(create_app(_settings(tmp_path, documents))) as client:
        client.put(f"/admin/access/{folder}", json={"principals": ["group:x"]}, headers=ADMIN)
        log = client.get("/admin/log", headers=ADMIN).json()
    (entry,) = [e for e in log["entries"] if e["action"] == "access.set"]
    assert entry["target"] == folder
