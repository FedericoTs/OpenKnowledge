"""The most-asked list on /manage: the ledger read as a shortlist.

``openknowledge top`` has always said which questions to pin. The page had
the cost of answering them but not the list, and the list without how each
question was answered is a count with no case attached: a question the cache
answers forty times costs nothing and needs nobody, one a model answers forty
times is paid for forty times and can change when the documents do.

The browser half - see the question, pin it from there, watch both lists
agree - lives with the other Chromium tests in ``test_widget_rendering``.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from tests.test_knowledge_gaps import _boot_paths

from openknowledge.api.app import create_app
from openknowledge.cache.store import AnswerStore
from openknowledge.config import Settings

TOKEN = "t0ken"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
DAY = 86400.0


def _client(tmp_path: Path) -> TestClient:
    docs = tmp_path / "documents"
    docs.mkdir()
    (docs / "leave.md").write_text("# Parental Leave\nEmployees get 20 weeks fully paid.")
    settings = Settings(
        data_dir=str(tmp_path / "data"),
        documents_dir=str(docs),
        admin_token=TOKEN,
        local_enabled=False,
        escalation_enabled=False,
        _env_file=None,  # type: ignore[call-arg]
    )
    return TestClient(create_app(settings))


def _asked(store: AnswerStore, rows: list[tuple[float, str, str, float]]) -> None:
    """Place ledger rows by hand: the query is what is under test, and the
    window and the tie rule both need timestamps ``record()`` would not give."""
    store._conn.executemany(
        "INSERT INTO ledger (ts, canonical_query, tier, model_id, cost_usd) VALUES (?, ?, ?, ?, ?)",
        [(ts, question, tier, "m", cost) for ts, question, tier, cost in rows],
    )
    store._conn.commit()


def test_demand_is_counted_per_question_with_how_it_was_answered(tmp_path: Path) -> None:
    """Hand-computed: three questions inside the window, one outside it."""
    now = 1_700_000_000.0
    with _client(tmp_path) as c:
        store = c.app.state.engine.store
        _asked(
            store,
            [
                (now - 10, "vpn", "local", 0.01),
                (now - 20, "vpn", "local", 0.01),
                (now - 5, "vpn", "pinned", 0.0),
                (now - 100, "leave", "refused", 0.0),
                (now - 200, "leave", "refused", 0.0),
                (now - 300, "leave", "refused", 0.0),
                (now - 1, "printer", "frontier", 0.05),
                *[(now - 40 * DAY - i, "holidays", "exact", 0.0) for i in range(5)],
            ],
        )
        month = store.question_demand(limit=10, since=now - 30 * DAY)
        all_time = store.question_demand(limit=10)
        two = store.question_demand(limit=2, since=now - 30 * DAY)

    assert [(d.canonical_query, d.count) for d in month] == [
        ("vpn", 3),
        ("leave", 3),  # asked as often as vpn, but longer ago: the live one first
        ("printer", 1),
    ]
    vpn, leave, printer = month
    assert vpn.by_tier == {"local": 2, "pinned": 1}
    assert vpn.spend_usd == 0.02
    assert vpn.last_asked == now - 5
    assert leave.by_tier == {"refused": 3}
    assert leave.spend_usd == 0.0
    assert leave.last_asked == now - 100
    assert printer.by_tier == {"frontier": 1}
    assert printer.spend_usd == 0.05

    assert [(d.canonical_query, d.count) for d in all_time][0] == ("holidays", 5)
    assert [d.canonical_query for d in two] == ["vpn", "leave"]


def test_the_endpoint_says_how_each_question_went_and_whether_it_is_pinned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Through the real routes: asked twice and refused, then pinned, then
    asked once more and answered from the pin. The window is checked by
    moving the clock the endpoint reads, not by waiting."""
    with _client(tmp_path) as c:
        for _ in range(2):
            assert c.post("/chat", json={"question": "How do I get VPN access?"}).status_code == 200
        c.post(
            "/admin/pins",
            headers=AUTH,
            json={"question": "How do I get VPN access?", "answer": "Ask IT.", "cite": ["leave"]},
        )
        again = c.post("/chat", json={"question": "how do I get vpn access"}).json()
        assert again["tier"] == "pinned"

        body = c.get("/admin/questions?days=30", headers=AUTH).json()
        assert body["days"] == 30
        (top,) = body["top"]
        assert top["question"] == "how do i get vpn access"
        assert top["count"] == 3
        assert top["by_tier"] == {"refused": 2, "pinned": 1}
        assert top["pinned"] is True
        assert top["spend_usd"] == 0.0
        assert top["last_asked"] > 0
        assert set(top) == {"question", "count", "by_tier", "spend_usd", "last_asked", "pinned"}

        # Everything above was asked "now"; a month from now it is outside a
        # 30-day window and still inside all time.
        monkeypatch.setattr(AnswerStore, "now", staticmethod(lambda: time.time() + 31 * DAY))
        assert c.get("/admin/questions?days=30", headers=AUTH).json()["top"] == []
        assert len(c.get("/admin/questions", headers=AUTH).json()["top"]) == 1


def test_every_way_into_the_page_loads_the_list(tmp_path: Path) -> None:
    """Three routes into /manage; the list loads on all of them, curators
    included - pinning what is asked most is the curator's job."""
    with _client(tmp_path) as c:
        page = c.get("/manage").text
    assert "async function refreshDemand()" in page
    for path, loaded in _boot_paths(page).items():
        assert "refreshDemand" in loaded, f"the {path} path does not load the most-asked list"


def test_the_page_says_what_pinning_buys(tmp_path: Path) -> None:
    """The case for the list is on the page, where the person reading the
    count decides: paid for every time and changeable, or free and identical."""
    with _client(tmp_path) as c:
        page = c.get("/manage").text
    assert "<h2>Most asked</h2>" in page
    assert 'id="demand"' in page
    # Both halves of the case, in the hint a reader sees before the counts:
    # what a model answer costs, and what a pin buys.
    assert "paid for every" in page
    assert "free and identical" in page
    assert "Nobody's name is recorded" in page
