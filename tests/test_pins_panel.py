"""The pins panel on /manage: the write-only half of pinning, made readable.

Pinning had three ways in - a gap, a wrong-answer report, the command line -
and no way to see what had been pinned short of calling the API. A pinned
answer is served word for word before any document or model is consulted,
which makes the list of them the most consequential thing on the page and,
until now, the one thing the page did not show.

The browser half - pin, see it, edit it, take it back, watch it go - lives
with the other Chromium tests in ``test_widget_rendering``. This file is what
CI, with no browser installed, still proves.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
from tests.test_knowledge_gaps import _boot_paths

from openknowledge.api.app import create_app
from openknowledge.config import Settings

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
        local_enabled=False,
        escalation_enabled=False,
        _env_file=None,  # type: ignore[call-arg]
    )
    return TestClient(create_app(settings))


def test_the_list_says_when_and_by_whom_each_answer_was_pinned(tmp_path: Path) -> None:
    """The store always knew; the API dropped it on the way out.

    A curator reading the list needs to tell a pin written last week from one
    written before the policy changed, and whose it is. Re-pinning is how the
    page edits, so the timestamp has to follow the text, and the citation has
    to survive a change of words - a pin that loses its source on edit is a
    pin that asks to be trusted.
    """
    with _client(tmp_path) as c:
        body = {
            "question": "How much parental leave?",
            "answer": "20 weeks.",
            "author": "hr@example.com",
            "cite": ["leave"],
        }
        assert c.post("/admin/pins", headers=AUTH, json=body).status_code == 201
        (first,) = c.get("/admin/pins", headers=AUTH).json()
        assert set(first) == {"question", "answer", "author", "cited", "updated_at", "enabled"}
        assert first["question"] == "how much parental leave"
        assert first["author"] == "hr@example.com"
        assert first["cited"] == ["leave"]
        assert first["enabled"] is True
        assert first["updated_at"] > 0

        # The edit the page makes: same question, same sources, new words.
        body |= {"answer": "24 weeks.", "author": "manage-page"}
        assert c.post("/admin/pins", headers=AUTH, json=body).status_code == 201
        (second,) = c.get("/admin/pins", headers=AUTH).json()
        assert second["answer"] == "24 weeks."
        assert second["cited"] == ["leave"]
        assert second["updated_at"] > first["updated_at"], "the date did not follow the edit"


def test_every_way_into_the_page_loads_the_pins(tmp_path: Path) -> None:
    """A pin is a curator's thing, not governance: a curator can make one from
    the gaps list, so a curator must be able to see and take back the ones that
    exist. Three routes into the page, and the panel loads on all of them."""
    with _client(tmp_path) as c:
        page = c.get("/manage").text
    assert "async function refreshPins()" in page
    for path, loaded in _boot_paths(page).items():
        assert "refreshPins" in loaded, f"the {path} path does not load the pins panel"


def test_the_page_says_a_pin_is_served_word_for_word(tmp_path: Path) -> None:
    """The caution belongs on the page, where the person about to edit reads
    it, not in a docstring: a pinned answer bypasses retrieval, the model and
    the grounding check, and what is typed here is exactly what a reader gets."""
    with _client(tmp_path) as c:
        page = c.get("/manage").text
    assert "<h2>Pinned answers</h2>" in page
    assert 'id="pins"' in page
    assert "word for word" in page
    assert "grounding check never sees" in page
