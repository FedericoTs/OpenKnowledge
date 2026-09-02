"""HTTP surface: chat, admin auth, and the pin workflow."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from openknowledge.api.app import create_app
from openknowledge.config import Settings

TOKEN = "test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    docs = tmp_path / "documents"
    docs.mkdir()
    (docs / "leave.md").write_text(
        "# Parental Leave\nEmployees with 12 months of service get 20 weeks fully paid."
    )
    settings = Settings(
        data_dir=str(tmp_path / "data"),
        documents_dir=str(docs),
        admin_token=TOKEN,
        local_enabled=False,  # no model in tests; exercise routing only
        escalation_enabled=False,
        _env_file=None,  # type: ignore[call-arg]
    )
    with TestClient(create_app(settings)) as c:
        yield c


def test_healthz_reports_the_indexed_corpus(client: TestClient) -> None:
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert body["documents_indexed"] == 1
    assert body["escalation_enabled"] is False


def test_widget_is_served_at_the_root(client: TestClient) -> None:
    res = client.get("/")
    assert res.status_code == 200
    assert "OpenKnowledge" in res.text


def test_with_no_model_available_it_refuses_instead_of_guessing(client: TestClient) -> None:
    body = client.post("/chat", json={"question": "How much parental leave?"}).json()
    assert body["tier"] == "refused"
    assert body["grounded"] is False
    assert body["cost_usd"] == 0.0


def test_empty_question_is_rejected(client: TestClient) -> None:
    assert client.post("/chat", json={"question": ""}).status_code == 422


@pytest.mark.parametrize(
    "headers", [{}, {"Authorization": "Bearer wrong"}, {"Authorization": "not-a-bearer"}]
)
def test_admin_requires_a_valid_token(client: TestClient, headers: dict[str, str]) -> None:
    assert client.get("/admin/costs", headers=headers).status_code == 401


def test_admin_is_disabled_when_no_token_is_configured(tmp_path: Path) -> None:
    """Fail closed: an unconfigured admin API is unavailable, not open."""
    settings = Settings(
        data_dir=str(tmp_path / "d"),
        documents_dir=str(tmp_path),
        admin_token=None,
        local_enabled=False,
        _env_file=None,  # type: ignore[call-arg]
    )
    with TestClient(create_app(settings)) as c:
        res = c.get("/admin/costs", headers=AUTH)
        assert res.status_code == 503
        assert "OK_ADMIN_TOKEN" in res.json()["detail"]


def test_pinning_makes_a_question_free_and_exact(client: TestClient) -> None:
    created = client.post(
        "/admin/pins",
        headers=AUTH,
        json={
            "question": "How much parental leave do I get?",
            "answer": "20 weeks.",
            "author": "hr@example.com",
        },
    )
    assert created.status_code == 201
    assert created.json()["question"] == "how much parental leave do i get"

    # A messier phrasing of the same question must land on the same pin.
    body = client.post("/chat", json={"question": "Hi, how much PARENTAL LEAVE do I get??"}).json()
    assert body["tier"] == "pinned"
    assert body["answer"] == "20 weeks."
    assert body["cost_usd"] == 0.0
    assert body["cached"] is True


def test_citations_serialise(client: TestClient) -> None:
    from openknowledge.types import Citation

    client.post(
        "/admin/pins",
        headers=AUTH,
        json={"question": "leave?", "answer": "20 weeks."},
    )
    engine = client.app.state.engine
    engine.store.pin(
        "leave",
        "20 weeks.",
        citations=(Citation("leave", "Parental Leave", "20 weeks fully paid", "p.1"),),
    )
    body = client.post("/chat", json={"question": "Leave?"}).json()
    assert body["citations"][0]["document_title"] == "Parental Leave"
    assert body["citations"][0]["locator"] == "p.1"


def test_pins_can_be_listed_and_removed(client: TestClient) -> None:
    client.post("/admin/pins", headers=AUTH, json={"question": "q?", "answer": "a"})
    assert len(client.get("/admin/pins", headers=AUTH).json()) == 1

    removed = client.request("DELETE", "/admin/pins", headers=AUTH, params={"question": "q?"})
    assert removed.json()["removed"] is True
    assert client.get("/admin/pins", headers=AUTH).json() == []


def test_reindex_picks_up_a_new_document(client: TestClient, tmp_path: Path) -> None:
    (tmp_path / "documents" / "vpn.md").write_text("# VPN\nConnect to vpn.internal with MFA.")
    body = client.post("/admin/reindex", headers=AUTH).json()
    assert body["documents"] == 2
    assert body["corpus_version"] != "empty"


def test_cost_report_counts_free_answers(client: TestClient) -> None:
    client.post("/admin/pins", headers=AUTH, json={"question": "leave?", "answer": "20 weeks."})
    for _ in range(3):
        client.post("/chat", json={"question": "Leave?", "channel": "teams"})

    report = client.get("/admin/costs", headers=AUTH).json()
    assert report["questions"] == 3
    assert report["cost_per_question_usd"] == 0.0
    assert report["by_tier"]["pinned"]["questions"] == 3


def test_cost_report_windows_to_the_days_asked_for(client: TestClient) -> None:
    """The report the manage page reads twice: all time, and the last 30 days.

    The default stays every question this install ever answered - the number
    an owner means by "what has this cost me" - and a window narrows it
    without changing that.
    """
    client.post("/admin/pins", headers=AUTH, json={"question": "leave?", "answer": "20 weeks."})
    for _ in range(2):
        client.post("/chat", json={"question": "Leave?"})

    everything = client.get("/admin/costs", headers=AUTH).json()
    assert everything["days"] == 0
    assert everything["questions"] == 2

    recent = client.get("/admin/costs?days=30", headers=AUTH).json()
    assert recent["days"] == 30
    assert recent["questions"] == 2, "questions asked just now are inside 30 days"

    # A window that starts after everything was asked reports nothing, rather
    # than quietly falling back to the whole ledger. Asked of the store
    # directly, because the cutoff is the thing under test, not the clock.
    store = client.app.state.engine.store
    assert store.cost_report(since=time.time() + 86400)["questions"] == 0
    assert store.cost_report(since=time.time() + 86400)["cost_per_question_usd"] == 0.0


def test_questions_endpoint_surfaces_pin_candidates(client: TestClient) -> None:
    for _ in range(2):
        client.post("/chat", json={"question": "How do I get VPN access?", "channel": "web"})
    body = client.get("/admin/questions", headers=AUTH).json()
    top = body["top"][0]
    assert (top["question"], top["count"]) == ("how do i get vpn access", 2)
    assert top["pinned"] is False
    assert body["recent"][0]["channel"] == "web"


def test_admin_config_does_not_leak_secrets(client: TestClient) -> None:
    """The token that authorised this very request must not be in the reply."""
    response = client.get("/admin/config", headers=AUTH)
    assert response.status_code == 200
    assert TOKEN not in response.text
    rows = {r["name"]: r for g in response.json()["groups"] for r in g["settings"]}
    assert rows["admin_token"] == {
        "name": "admin_token",
        "env": "OK_ADMIN_TOKEN",
        "value": "set",
        "redacted": True,
        "is_default": False,
        "live": None,
    }


def test_refusal_is_not_reported_as_cached(client: TestClient) -> None:
    body = client.post("/chat", json={"question": "Something not in the docs at all"}).json()
    assert body["tier"] == "refused"
    assert body["cached"] is False


def test_a_pin_can_carry_provenance(client: TestClient) -> None:
    """A pinned answer without sources asks the reader to take it on trust."""
    client.post(
        "/admin/pins",
        headers=AUTH,
        json={
            "question": "How much parental leave?",
            "answer": "20 weeks after 12 months of service.",
            "cite": ["leave"],
            "author": "hr@example.com",
        },
    )
    body = client.post("/chat", json={"question": "How much parental leave?"}).json()
    assert body["tier"] == "pinned"
    assert body["citations"][0]["document_id"] == "leave"
    assert body["citations"][0]["document_title"] == "Parental Leave"


def test_aliases_let_one_pin_answer_several_phrasings(client: TestClient) -> None:
    created = client.post(
        "/admin/pins",
        headers=AUTH,
        json={
            "question": "How much parental leave do I get?",
            "answer": "20 weeks.",
            "aliases": [
                "what is the parental leave entitlement",
                "how much time off do I get for a new baby",
            ],
        },
    ).json()
    assert len(created["aliases"]) == 2

    for phrasing in [
        "How much parental leave do I get?",
        "What is the parental leave entitlement?",
        "how much time off do I get for a new baby",
    ]:
        body = client.post("/chat", json={"question": phrasing}).json()
        assert body["tier"] == "pinned", phrasing
        assert body["answer"] == "20 weeks."


def test_listed_pins_show_their_sources(client: TestClient) -> None:
    client.post(
        "/admin/pins",
        headers=AUTH,
        json={"question": "q?", "answer": "a", "cite": ["leave"]},
    )
    assert client.get("/admin/pins", headers=AUTH).json()[0]["cited"] == ["leave"]


def test_the_widget_has_a_tab_icon() -> None:
    """It was logging a 404 for this on every page load."""
    from fastapi.testclient import TestClient

    from openknowledge.api.app import create_app
    from openknowledge.config import Settings

    with TestClient(create_app(Settings(data_dir="./data-favicon-test"))) as client:
        response = client.get("/favicon.ico")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("image/svg+xml")
        assert response.text.startswith("<svg")


def test_health_counts_documents_and_chunks_separately(tmp_path: Path) -> None:
    """`documents_indexed` used to carry the chunk count.

    Reported from a real install: `openknowledge index` said "4 documents -> 6
    chunks" and this endpoint said six documents. They differ by roughly an
    order of magnitude on a real corpus, and the widget now shows this number
    to the person asking.
    """
    from fastapi.testclient import TestClient

    from openknowledge.api.app import create_app
    from openknowledge.config import Settings

    documents = tmp_path / "docs"
    documents.mkdir()
    # Two files, deliberately long enough to chunk into more than two pieces.
    for name in ("a.md", "b.md"):
        (documents / name).write_text(
            f"# {name}\n\n" + "\n\n".join(f"Paragraph {i} about policy. " * 60 for i in range(6))
        )

    settings = Settings(data_dir=str(tmp_path / "data"), documents_dir=str(documents))
    with TestClient(create_app(settings)) as client:
        client.app.state.engine.reindex()
        body = client.get("/healthz").json()

    assert body["documents_indexed"] == 2, "counted chunks as documents"
    assert body["chunks_indexed"] > body["documents_indexed"]
