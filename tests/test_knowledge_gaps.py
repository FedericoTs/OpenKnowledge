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

import re
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
    # Named exactly, so adding a field to this report is a decision somebody
    # made rather than something that happened.
    assert set(gap) == {"question", "asked", "answered_in_part", "kind", "last_asked"}
    # And named badly is still named: nothing here may describe a person.
    assert not {k.lower() for k in gap} & {
        "principal",
        "principals",
        "user",
        "users",
        "asker",
        "askers",
        "session",
        "email",
        "name",
        "account",
        "who",
    }


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
        #
        # Read as "which panels does this path load", not as an exact line of
        # source. The first version of this check pinned two calls being
        # adjacent, and broke the day a third panel was slotted between them -
        # a green-to-red on a change that was correct.
        for path, loaded in _boot_paths(page).items():
            assert "refreshGaps" in loaded, f"the {path} path does not load the gaps panel"
        # The promise the panel makes about privacy has to be on the page,
        # not only in a docstring the reader will never open.
        assert "no column for who asked" in page


#: Where each route into /manage begins, in the page's own words, and the
#: role it should arrive as.
_ROUTES = {
    "token": ("$('token-input').value", "admin"),
    "admin session": ("say('Admin, via your sign-in group.'", "admin"),
    "curator session": ("say('Knowledge curator", "curator"),
}


def _boot_paths(page: str) -> dict[str, set[str]]:
    """Which refresh functions each way into /manage calls on arrival.

    Three ways in - a pasted admin token, an admin's sign-in session, a
    curator's - and a panel that loads on one and not the others is empty
    for a third of the people who have the page.

    Each route ends by handing its panel loads to ``ready(role, [...])``,
    the same signal the browser test waits on, so these bounds move with the
    code rather than being pinned to whatever line happens to sit nearby.
    """
    found: dict[str, set[str]] = {}
    for name, (start, role) in _ROUTES.items():
        at = page.find(start)
        assert at >= 0, f"the {name} route is not in the page any more"
        stop = page.find("]);", at)
        assert stop > at, f"the {name} route does not end in a ready() call"
        region = page[at:stop]
        assert f"ready('{role}'" in region, f"the {name} route no longer arrives as {role}"
        found[name] = set(re.findall(r"(refresh[A-Za-z]+)\(\)", region))
    return found


def test_every_way_into_the_page_loads_the_same_panels(tmp_path) -> None:
    """A panel that loads on one route and not the others is empty for
    however many people arrive the other way, and looks like a bug in the
    feature rather than in the wiring."""
    with TestClient(create_app(_settings(tmp_path))) as c:
        paths = _boot_paths(c.get("/manage").text)

    assert paths["token"] == paths["admin session"], (
        "the pasted token and an admin's session must load the same page"
    )
    # A curator holds no governance, so those panels are deliberately absent.
    assert paths["curator session"] < paths["admin session"]
    assert {"refreshGaps", "refreshReports"} <= paths["curator session"]


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


def _half(tier: Tier = Tier.LOCAL) -> Answer:
    """An answer that covered part of the question and named the rest."""
    return Answer(
        text="Meals are covered [expenses]. There is no information about taxis.",
        tier=tier,
        model_id="qwen3-4b",
        cache_key="k",
        declined_in_part=True,
    )


def test_a_question_answered_only_in_part_is_still_a_gap(store) -> None:
    """The debt taken in v0.2.17, repaid.

    Before that release a partial "I don't know" was fatal to the whole
    answer, so the question came back as a refusal and the report saw it.
    Making the answer survive - which was right - took the gap with it, and
    the half the documents could not answer stopped being reported at all.
    """
    for _ in range(3):
        store.record("are taxis and meals covered", _half())

    (gap,) = store.knowledge_gaps()
    assert gap["question"] == "are taxis and meals covered"
    assert gap["asked"] == 3
    assert gap["answered_in_part"] == 3
    assert gap["kind"] == "partial", "half an answer is not the same job as none"


def test_a_full_answer_is_not_a_gap(store) -> None:
    """Nothing widens into 'every question we ever answered'."""
    store.record("how much parental leave", _answer(Tier.LOCAL))
    assert store.knowledge_gaps() == []


def test_a_partial_gap_closes_when_the_document_is_written(store) -> None:
    """The same rule refusals get: the list tracks the present."""
    store.record("are taxis and meals covered", _half())
    assert store.knowledge_gaps()

    store.record("are taxis and meals covered", _answer(Tier.LOCAL))
    assert store.knowledge_gaps() == [], "answered in full, so no longer a gap"


def test_the_kind_follows_the_most_recent_ask(store) -> None:
    """A question that used to be refused and is now half-answered has moved,
    and the report should say which way."""
    store.record("are taxis and meals covered", _answer(Tier.REFUSED))
    store.record("are taxis and meals covered", _half())

    (gap,) = store.knowledge_gaps()
    assert gap["asked"] == 2
    assert gap["answered_in_part"] == 1
    assert gap["kind"] == "partial"


def test_a_ledger_written_before_partial_existed_still_reads(tmp_path) -> None:
    """An install that has been running does not get a new database.

    CREATE TABLE IF NOT EXISTS does nothing to a table that already exists, so
    a column added later never arrives without an explicit migration - and a
    row written before it existed is simply not known to be partial, which is
    what it was.
    """
    import sqlite3

    db = tmp_path / "old.db"
    old = sqlite3.connect(db)
    old.executescript(
        "CREATE TABLE ledger (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL,"
        " canonical_query TEXT NOT NULL, tier TEXT NOT NULL, model_id TEXT NOT NULL,"
        " cost_usd REAL NOT NULL DEFAULT 0.0, usage TEXT NOT NULL DEFAULT '{}', channel TEXT);"
    )
    old.execute(
        "INSERT INTO ledger (ts, canonical_query, tier, model_id) VALUES (1.0, 'old one', ?, 'm')",
        ("refused",),
    )
    old.commit()
    old.close()

    with AnswerStore(db) as store:
        (gap,) = store.knowledge_gaps()
        assert gap["question"] == "old one"
        assert gap["answered_in_part"] == 0
        assert gap["kind"] == "refused"
        # And the column is usable from here on.
        store.record("are taxis and meals covered", _half())
        assert len(store.knowledge_gaps()) == 2
