"""The health line on /manage: the endpoints, through the running app.

The probes themselves are tested against a stub in ``test_health``. This
file is the route and the page: what a curator gets back, that the cache
holds between two looks and a re-check bypasses it, that every way into the
page loads the line, and that the caution about what it costs is on the page.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from tests.test_knowledge_gaps import _boot_paths

from openknowledge.api.app import create_app
from openknowledge.config import Settings
from openknowledge.health import Reading, Target

TOKEN = "t0ken"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


def _client(tmp_path: Path) -> TestClient:
    docs = tmp_path / "documents"
    docs.mkdir()
    (docs / "leave.md").write_text(
        "# Parental Leave\nEmployees get 20 weeks fully paid.", encoding="utf-8"
    )
    settings = Settings(
        data_dir=str(tmp_path / "data"),
        documents_dir=str(docs),
        admin_token=TOKEN,
        # A model endpoint that is configured and dead: nothing listens on
        # port 9, and the refusal is immediate.
        local_enabled=True,
        local_base_url="http://127.0.0.1:9/v1",
        local_model="qwen3:8b",
        embedding_enabled=False,
        escalation_enabled=False,
        _env_file=None,  # type: ignore[call-arg]
    )
    return TestClient(create_app(settings))


def test_the_line_says_which_endpoints_answer(tmp_path: Path) -> None:
    with _client(tmp_path) as c:
        body = c.get("/admin/health", headers=AUTH).json()
    assert body["ttl_seconds"] == 30
    chat, embedding, escalation = body["endpoints"]
    assert (chat["role"], chat["configured"], chat["state"], chat["ok"]) == (
        "chat",
        True,
        "unreachable",
        False,
    )
    assert chat["host"] == "127.0.0.1:9"
    assert chat["model"] == "qwen3:8b"
    assert (embedding["role"], embedding["state"], embedding["ok"]) == ("embedding", "off", None)
    assert (escalation["role"], escalation["state"], escalation["ok"]) == (
        "escalation",
        "off",
        None,
    )
    assert "off" in escalation["detail"]
    for reading in body["endpoints"]:
        assert "api_key" not in reading, "a reading must never carry the key it was asked with"


def test_two_looks_inside_the_ttl_ask_once_and_fresh_asks_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The page is opened by every curator and admin; the endpoints must not
    be asked once per person. The button is the exception, and says so."""
    asked: list[str] = []

    def counting(target: Target, *, timeout: float) -> Reading:
        asked.append(target.role)
        return Reading(role=target.role, configured=True, state="reachable", ok=True)

    with _client(tmp_path) as c:
        monkeypatch.setattr(c.app.state.health, "_prober", counting)
        c.get("/admin/health", headers=AUTH)
        c.get("/admin/health", headers=AUTH)
        assert sorted(asked) == ["chat", "embedding", "escalation"]
        c.get("/admin/health?fresh=1", headers=AUTH)
        assert len(asked) == 6


def test_the_line_is_not_public(tmp_path: Path) -> None:
    """``fresh=1`` asks the endpoints on demand. Public, that is a way to make
    a server poke its model as often as anyone likes."""
    with _client(tmp_path) as c:
        assert c.get("/admin/health").status_code in (401, 403)
        assert c.get("/healthz").status_code == 200
        assert "endpoints" not in c.get("/healthz").json(), "/healthz stays a liveness check"


def test_every_way_into_the_page_loads_the_line(tmp_path: Path) -> None:
    with _client(tmp_path) as c:
        page = c.get("/manage").text
    assert "async function refreshHealth(" in page
    for path, loaded in _boot_paths(page).items():
        assert "refreshHealth" in loaded, f"the {path} path does not load the health line"


def test_the_page_says_a_question_never_waits_on_it(tmp_path: Path) -> None:
    with _client(tmp_path) as c:
        page = c.get("/manage").text
    assert "<h2>Is it up</h2>" in page
    assert 'id="health"' in page
    assert "a question never waits on this" in page
    assert "every 30 seconds" in page
