"""Changing settings while the server runs, without lying about it.

The contract these tests hold: everything applied is persisted to the dotenv
the next start reads (a change that evaporates on restart teaches people not
to trust the page), live fields take effect on the very next request, rebuild
fields swap in a freshly built engine, and a change that cannot build leaves
the server answering exactly as before.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from openknowledge.api.app import create_app
from openknowledge.api.runtime_settings import (
    EDITABLE,
    SettingsChangeError,
    validate_changes,
)
from openknowledge.config import Settings

TOKEN = "test-admin-token"


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OK_STATE_DIR", str(tmp_path / "state"))
    settings = Settings(
        data_dir=str(tmp_path / "data"),
        documents_dir=str(tmp_path / "documents"),
        admin_token=TOKEN,
        local_enabled=False,
        embedding_enabled=False,
        _env_file=None,  # type: ignore[call-arg]
    )
    with TestClient(create_app(settings)) as c:
        yield c, settings, tmp_path / "state" / ".env"


AUTH = {"Authorization": f"Bearer {TOKEN}"}


def test_the_whitelist_is_typed_and_closed() -> None:
    assert validate_changes({"retrieval_k": "8"}) == {"retrieval_k": 8}
    with pytest.raises(SettingsChangeError, match="not editable"):
        validate_changes({"admin_token": "mine-now"})
    with pytest.raises(SettingsChangeError, match="not editable"):
        validate_changes({"data_dir": "/tmp/elsewhere"})
    with pytest.raises(SettingsChangeError, match="retrieval_k"):
        validate_changes({"retrieval_k": "a lot"})


def test_a_live_change_applies_to_the_next_request_and_survives_restart(client) -> None:
    c, settings, env_file = client
    response = c.put("/admin/settings", headers=AUTH, json={"upload_enabled": True})
    assert response.status_code == 200
    body = response.json()
    assert body["applied"] == {"upload_enabled": True}
    assert body["engine_rebuilt"] is False

    # Next request: the uploads surface exists now, with no restart.
    assert c.get("/documents").status_code == 200
    # And the change is written where the next start will read it.
    assert "OK_UPLOAD_ENABLED=true" in env_file.read_text()

    c.put("/admin/settings", headers=AUTH, json={"upload_enabled": False})
    assert c.get("/documents").status_code == 404


def test_a_rebuild_change_swaps_the_engine(client) -> None:
    c, settings, env_file = client
    before = c.app.state.engine
    response = c.put(
        "/admin/settings", headers=AUTH, json={"local_enabled": True, "retrieval_k": 4}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["engine_rebuilt"] is True
    assert c.app.state.engine is not before
    assert settings.retrieval_k == 4
    assert "OK_LOCAL_ENABLED=true" in env_file.read_text()
    assert "OK_RETRIEVAL_K=4" in env_file.read_text()

    # The swapped engine serves.
    assert c.get("/healthz").json()["status"] == "ok"


def test_an_invalid_change_is_rejected_whole_with_nothing_applied(client) -> None:
    c, settings, env_file = client
    before_k = settings.retrieval_k
    response = c.put(
        "/admin/settings",
        headers=AUTH,
        json={"retrieval_k": 9, "escalation_effort": "extreme"},
    )
    assert response.status_code == 422
    assert settings.retrieval_k == before_k, "a rejected batch half-applied"
    assert not env_file.exists() or "OK_RETRIEVAL_K" not in env_file.read_text()


def test_settings_require_the_admin_token(client) -> None:
    c, _, _ = client
    assert c.get("/admin/settings").status_code == 401
    assert c.put("/admin/settings", json={"retrieval_k": 3}).status_code == 401


def test_get_reports_every_editable_field_with_how_it_applies(client) -> None:
    c, _, _ = client
    body = c.get("/admin/settings", headers=AUTH).json()
    assert set(body["settings"]) == set(EDITABLE)
    assert body["settings"]["retrieval_k"]["applies"] == "live"
    assert body["settings"]["local_model"]["applies"] == "rebuild"
    assert "persists_to" in body


def test_every_editable_field_actually_exists_on_settings() -> None:
    """The whitelist is data; a typo in it would 500 at request time."""
    for key in EDITABLE:
        assert key in Settings.model_fields, key
