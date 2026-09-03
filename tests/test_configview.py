"""The effective configuration: every setting, its default, no secret.

The unit half asks ``describe`` directly with sentinels planted in every
credential. The route half goes through the app. The page half checks that
only the two admin routes into /manage load it - it names hostnames and
paths, so a curator does not get it - and that the caution is on the page.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient
from tests.test_knowledge_gaps import _boot_paths

from openknowledge.api.app import create_app
from openknowledge.api.runtime_settings import EDITABLE
from openknowledge.config import Settings
from openknowledge.configview import SECRET_NAME, describe, group_of

SECRETS = {
    "admin_token": "sekrit-admin",
    "local_api_key": "sekrit-local",
    "anthropic_api_key": "sekrit-anthropic",
    "openai_api_key": "sekrit-openai",
    "azure_openai_api_key": "sekrit-azure",
    "ladder_api_key": "sekrit-ladder",
    "oidc_client_secret": "sekrit-oidc",
    "tls_key": "/etc/sekrit/server.key",
}


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    return Settings(
        data_dir=str(tmp_path / "data"),
        documents_dir=str(tmp_path / "documents"),
        _env_file=None,  # type: ignore[call-arg]
        **overrides,  # type: ignore[arg-type]
    )


def _rows(view: dict) -> dict[str, dict]:
    return {row["name"]: row for group in view["groups"] for row in group["settings"]}


def test_every_setting_is_listed_once_and_filed(tmp_path: Path) -> None:
    """A setting that is not on the page is one somebody will set blind. The
    "Other" group exists so that a new setting with no home fails here, not
    on an admin's screen."""
    view = describe(_settings(tmp_path))
    names = [row["name"] for group in view["groups"] for row in group["settings"]]
    assert sorted(names) == sorted(Settings.model_fields)
    assert len(names) == len(set(names))
    assert "Other" not in {group["name"] for group in view["groups"]}
    assert [group["name"] for group in view["groups"]][:2] == [
        "Where things live",
        "Retrieval and grounding",
    ]


def test_no_secret_reaches_the_output(tmp_path: Path) -> None:
    """Planted in every credential the settings have, then searched for."""
    view = describe(_settings(tmp_path, **SECRETS))
    assert "sekrit" not in json.dumps(view)
    rows = _rows(view)
    for name in SECRETS:
        assert rows[name]["redacted"] is True, name
        assert rows[name]["value"] == "set", name
    # Every field whose name says credential is redacted, whatever its value.
    for name in Settings.model_fields:
        if SECRET_NAME.search(name):
            assert rows[name]["redacted"] is True, name


def test_an_unset_secret_says_so_and_a_count_of_tokens_is_not_a_secret(tmp_path: Path) -> None:
    rows = _rows(describe(_settings(tmp_path)))
    assert rows["anthropic_api_key"]["value"] == "not set"
    assert rows["oidc_client_secret"]["value"] == "not set", "the empty string is not set"
    # max_answer_tokens ends in "tokens": a size, not a credential. Hiding it
    # would hide the one number people tune when answers get cut off.
    assert rows["max_answer_tokens"]["redacted"] is False
    assert rows["max_answer_tokens"]["value"] == 1500
    assert rows["local_context_tokens"]["redacted"] is False


def test_default_is_a_fact_about_the_setting_not_the_value(tmp_path: Path) -> None:
    rows = _rows(describe(_settings(tmp_path, retrieval_k=9, embedding_enabled=False)))
    assert rows["retrieval_k"] == {
        "name": "retrieval_k",
        "env": "OK_RETRIEVAL_K",
        "value": 9,
        "redacted": False,
        "is_default": False,
        "live": "live",
    }
    assert rows["embedding_enabled"]["is_default"] is False
    assert rows["embedding_enabled"]["live"] == "rebuild"
    assert rows["local_model"]["is_default"] is True
    assert rows["local_model"]["value"] == "qwen3:8b"
    # A default that is built by a factory still counts as the default.
    assert rows["trusted_hosts"]["is_default"] is True
    assert rows["trusted_hosts"]["value"] == ["127.0.0.1", "localhost", "::1", "testserver"]
    assert rows["budget_daily_usd"] == {
        "name": "budget_daily_usd",
        "env": "OK_BUDGET_DAILY_USD",
        "value": None,
        "redacted": False,
        "is_default": True,
        "live": "rebuild",
    }
    # Paths are set here by the test, so they are not defaults - and say so.
    assert rows["data_dir"]["is_default"] is False
    assert rows["data_dir"]["live"] is None, "paths are the operator's shell, not the page's"


def test_the_groups_count_what_was_set(tmp_path: Path) -> None:
    view = describe(_settings(tmp_path, local_model="qwen3:4b", local_parallel=2))
    by_name = {g["name"]: g for g in view["groups"]}
    assert by_name["Local model"]["set"] == 2
    assert by_name["Knowledge"]["set"] == 0


def test_live_follows_the_settings_endpoint(tmp_path: Path) -> None:
    """One source for what the page can change: a setting the Settings panel
    edits is marked live or rebuild here, and nothing else is."""
    rows = _rows(describe(_settings(tmp_path)))
    assert {n for n, r in rows.items() if r["live"]} == set(EDITABLE)
    for name, how in EDITABLE.items():
        assert rows[name]["live"] == how


def test_every_current_setting_has_a_group() -> None:
    assert all(group_of(name) != "Other" for name in Settings.model_fields)
    assert group_of("something_nobody_filed") == "Other"


def test_the_state_block_names_the_file_the_next_start_reads(tmp_path: Path) -> None:
    view = describe(_settings(tmp_path))
    assert view["state"]["mode"] in {"project", "app", "override"}
    assert view["state"]["env_file"].endswith(".env")
    assert isinstance(view["state"]["env_file_exists"], bool)


# -- through the app -----------------------------------------------------------

TOKEN = "t0ken-sekrit"
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
        local_enabled=False,
        escalation_enabled=False,
        anthropic_api_key="sekrit-anthropic",
        _env_file=None,  # type: ignore[call-arg]
    )
    return TestClient(create_app(settings))


def test_the_route_carries_the_version_and_hides_the_token_that_unlocked_it(tmp_path: Path) -> None:
    from openknowledge import __version__

    with _client(tmp_path) as c:
        response = c.get("/admin/config", headers=AUTH)
    assert response.status_code == 200
    body = response.json()
    assert body["version"] == __version__
    assert "sekrit" not in response.text
    rows = _rows(body)
    assert rows["admin_token"]["value"] == "set"
    assert rows["local_enabled"] == {
        "name": "local_enabled",
        "env": "OK_LOCAL_ENABLED",
        "value": False,
        "redacted": False,
        "is_default": False,
        "live": "rebuild",
    }


def test_only_the_admin_routes_into_the_page_load_it(tmp_path: Path) -> None:
    with _client(tmp_path) as c:
        page = c.get("/manage").text
    assert "async function refreshConfig()" in page
    paths = _boot_paths(page)
    assert "refreshConfig" in paths["token"]
    assert "refreshConfig" in paths["admin session"]
    assert "refreshConfig" not in paths["curator session"], (
        "the configuration names hosts and paths; a curator does not hold governance"
    )


def test_the_page_says_what_default_and_live_mean(tmp_path: Path) -> None:
    with _client(tmp_path) as c:
        page = c.get("/manage").text
    assert "<h2>Configuration</h2>" in page
    assert 'id="config"' in page
    assert "never as values" in page
    assert "read from the process" in page
