"""The refusal, kept rather than forgotten.

Every other tier leaves something behind - an answer, a cache entry, a line
in the cost report. "I don't know - that isn't covered by the documents I
have" was said once and lost, so the person who owns the corpus never learned
that eleven colleagues had asked the same unanswerable question that month.

That refusal is the most useful thing this product produces. A system that
guesses has no refusals to count; this one can hand its owner the list of
documents worth writing, in the order worth writing them.
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from openknowledge.api.app import create_app
from openknowledge.cache.store import AnswerStore
from openknowledge.config import Settings
from openknowledge.types import Answer, Tier

_TOKEN = "test-admin-token"
_AUTH = {"Authorization": f"Bearer {_TOKEN}"}


def _settings(tmp_path) -> Settings:
    return Settings(
        data_dir=str(tmp_path / "data"),
        documents_dir=str(tmp_path / "documents"),
        admin_token=_TOKEN,
        local_enabled=False,
        embedding_enabled=False,
        _env_file=None,  # type: ignore[call-arg]
    )


def _answer(tier: Tier) -> Answer:
    return Answer(text="...", tier=tier, model_id="none", cache_key="k")


@pytest.fixture
def store():
    s = AnswerStore()
    yield s
    s.close()


def test_refusals_are_counted_and_ranked(store) -> None:
    for _ in range(3):
        store.record("what is the notice period for contractors", _answer(Tier.REFUSED))
    store.record("who signs off travel above eur 500", _answer(Tier.REFUSED))
    store.record("how long is parental leave", _answer(Tier.LOCAL))

    gaps = store.knowledge_gaps()

    assert [g["question"] for g in gaps] == [
        "what is the notice period for contractors",
        "who signs off travel above eur 500",
    ], "most asked first; an answered question is not a gap"
    assert gaps[0]["asked"] == 3


def test_an_answered_question_is_never_a_gap(store) -> None:
    """Including the tiers that answer for free - a gap is the absence of an
    answer, not the absence of a model call."""
    for tier in (Tier.LOCAL, Tier.EXACT_CACHE, Tier.PINNED, Tier.CORPUS):
        store.record(f"question for {tier.value}", _answer(tier))
    assert store.knowledge_gaps() == []


def test_the_window_holds(store) -> None:
    old = time.time() - 90 * 86400
    store.record("an ancient question", _answer(Tier.REFUSED))
    store._conn.execute("UPDATE ledger SET ts = ?", (old,))  # noqa: SLF001 - ageing a row
    store.record("a question from today", _answer(Tier.REFUSED))

    recent = [g["question"] for g in store.knowledge_gaps(since=time.time() - 30 * 86400)]
    assert recent == ["a question from today"]
    assert len(store.knowledge_gaps()) == 2, "no window means the whole history"


def test_the_report_cannot_name_anyone(store) -> None:
    """The property that makes this safe to ship in a product whose promise is
    privacy: the ledger it reads has no identity column, so the report can say
    a question was asked forty times and never who asked it. If a principal is
    ever added to that table, this test should be the thing that argues about
    it."""
    columns = {
        row["name"]
        for row in store._conn.execute("PRAGMA table_info(ledger)").fetchall()  # noqa: SLF001
    }
    assert not columns & {"principal", "principals", "user", "asker", "session"}

    store.record("something nobody could answer", _answer(Tier.REFUSED))
    gap = store.knowledge_gaps()[0]
    assert set(gap) == {"question", "asked", "last_asked"}


def test_the_endpoint_is_admin_only(tmp_path) -> None:
    """What colleagues are looking for is not everybody's business."""
    with TestClient(create_app(_settings(tmp_path))) as c:
        assert c.get("/admin/gaps").status_code == 401
        assert c.get("/admin/gaps", headers={"Authorization": "Bearer wrong"}).status_code == 401


def test_the_endpoint_reports_the_gaps(tmp_path) -> None:
    with TestClient(create_app(_settings(tmp_path))) as c:
        engine = c.app.state.engine
        for _ in range(2):
            engine.store.record("what is the contractor notice period", _answer(Tier.REFUSED))
        body = c.get("/admin/gaps", headers=_AUTH).json()
        assert body["gaps"][0]["question"] == "what is the contractor notice period"
        assert body["gaps"][0]["asked"] == 2


def test_the_manage_page_shows_the_gaps(tmp_path) -> None:
    """A report nobody sees is a report nobody acts on. The corpus owner
    works in /manage, so the panel lives there - and it has to be refreshed
    on both routes into the page, the pasted admin token and the signed-in
    admin session, or it silently stays empty for half the people who have
    it."""
    with TestClient(create_app(_settings(tmp_path))) as c:
        page = c.get("/manage").text
        assert "Asked and not answered" in page
        assert "async function refreshGaps()" in page
        # Both unlock paths specifically, rather than a count of every call:
        # pinning refreshes the list too, and a total would break the moment
        # somebody adds a legitimate third caller.
        assert page.count("refreshGaps(); refreshAccess();") == 2, (
            "the token path and the signed-in admin path must both load it"
        )
        # The promise the panel makes about privacy has to be on the page,
        # not only in a docstring the reader will never open.
        assert "no column for who asked" in page


def test_pinning_an_answer_closes_the_gap(tmp_path) -> None:
    """The loop, end to end: asked, refused, answered, gone.

    A list that never shrinks is worse than no list - the person working
    through it does the work and watches nothing happen. Pinning is the click
    this report exists to prompt, so it has to take effect immediately rather
    than at the next time somebody happens to ask.
    """
    with TestClient(create_app(_settings(tmp_path))) as c:
        store = c.app.state.engine.store
        for _ in range(11):
            store.record("what is the contractor notice period", _answer(Tier.REFUSED))

        before = c.get("/admin/gaps", headers=_AUTH).json()["gaps"]
        assert [g["question"] for g in before] == ["what is the contractor notice period"]

        created = c.post(
            "/admin/pins",
            headers=_AUTH,
            json={
                "question": "what is the contractor notice period",
                "answer": "Contractors give 30 days, per their engagement letter.",
                "cite": [],
                "author": "manage-page",
            },
        )
        assert created.status_code == 201, created.text

        after = c.get("/admin/gaps", headers=_AUTH).json()["gaps"]
        assert after == [], "a pinned question is not still a gap"


def test_a_gap_the_documents_now_answer_drops_off(store) -> None:
    """The other way a gap closes: somebody wrote the document. Nothing is
    re-asked of a model to find that out - the most recent time the question
    was asked, it was answered, and that is the whole signal."""
    for _ in range(4):
        store.record("how do i claim mileage", _answer(Tier.REFUSED))
    assert [g["question"] for g in store.knowledge_gaps()] == ["how do i claim mileage"]

    store.record("how do i claim mileage", _answer(Tier.LOCAL))
    assert store.knowledge_gaps() == [], "the corpus caught up; the gap is closed"


def test_a_question_that_regressed_is_a_gap_again(store) -> None:
    """And if it stops being answered - the document was deleted, or walled
    off from the people asking - it comes back. The list tracks the present,
    not a decision made once."""
    store.record("how do i claim mileage", _answer(Tier.REFUSED))
    store.record("how do i claim mileage", _answer(Tier.LOCAL))
    assert store.knowledge_gaps() == []

    store.record("how do i claim mileage", _answer(Tier.REFUSED))
    assert [g["question"] for g in store.knowledge_gaps()] == ["how do i claim mileage"]


def test_the_manage_page_can_answer_a_gap_in_place(tmp_path) -> None:
    """The box is on the gap, not somewhere else in the product."""
    with TestClient(create_app(_settings(tmp_path))) as c:
        page = c.get("/manage").text
        assert "Pin this answer" in page
        assert "/admin/pins" in page
        assert "cite no document" in page, "a citation is offered, never required"
