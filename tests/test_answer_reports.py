"""When the answer is wrong.

The one signal this product could not collect. A refusal leaves a trace - the
gaps report counts it and ranks it - but an answer that was confidently wrong
left nothing at all, because it looked exactly like an answer that was right.
Somebody noticed, told a colleague, and the documents never heard.

The test that matters most here is the privacy one. This table travels in
every backup and is read by whoever runs the server, so a reporter's name in
it would be a record of who complained about what. The gaps report has a test
that it cannot name anyone who asked; this is the same promise for who
objected, and it is checked against the bytes rather than the API.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi.testclient import TestClient

from fake_idp import FakeIdp
from openknowledge.api.app import create_app
from openknowledge.config import Settings
from openknowledge.knowledge.store import MAX_REPORT_NOTES, KnowledgeStore

WIDGET = Path("web/widget/index.html")


# -- the store ---------------------------------------------------------------


def test_a_report_keeps_the_answer_and_the_reason(tmp_path: Path) -> None:
    with KnowledgeStore(tmp_path / "k.db") as store:
        store.report_answer(
            "how much parental leave",
            "How much parental leave?",
            "16 weeks, fully paid.",
            tier="pinned",
            corpus_version="abc",
            note="It went to 20 weeks in April.",
        )
        (report,) = store.answer_reports()

    assert report.question == "How much parental leave?"
    assert report.answer == "16 weeks, fully paid."
    assert report.tier == "pinned"
    assert report.notes == ("It went to 20 weeks in April.",)
    assert report.reports == 1
    assert report.status == "open"


def test_the_same_wrong_answer_is_one_line_with_a_count(tmp_path: Path) -> None:
    """A hundred colleagues agreeing is one thing to act on, not a hundred."""
    with KnowledgeStore(tmp_path / "k.db") as store:
        for note in ("wrong", "still wrong", "wrong"):
            store.report_answer("q", "Q?", "16 weeks.", note=note)
        (report,) = store.answer_reports()

    assert report.reports == 3
    assert report.notes == ("wrong", "still wrong"), "the same note is not kept twice"


def test_a_different_answer_to_the_same_question_is_its_own_report(tmp_path: Path) -> None:
    with KnowledgeStore(tmp_path / "k.db") as store:
        store.report_answer("q", "Q?", "16 weeks.")
        store.report_answer("q", "Q?", "18 weeks.")
        assert len(store.answer_reports()) == 2


def test_notes_are_capped_so_one_report_cannot_grow_without_end(tmp_path: Path) -> None:
    with KnowledgeStore(tmp_path / "k.db") as store:
        for i in range(MAX_REPORT_NOTES + 10):
            store.report_answer("q", "Q?", "16 weeks.", note=f"note {i}")
        (report,) = store.answer_reports()

    assert len(report.notes) == MAX_REPORT_NOTES
    assert report.notes[-1] == f"note {MAX_REPORT_NOTES + 9}", "the newest are kept"
    assert report.reports == MAX_REPORT_NOTES + 10, "every report is still counted"


def test_an_empty_note_adds_a_count_and_nothing_else(tmp_path: Path) -> None:
    with KnowledgeStore(tmp_path / "k.db") as store:
        store.report_answer("q", "Q?", "16 weeks.", note="   ")
        (report,) = store.answer_reports()
    assert report.notes == ()
    assert report.reports == 1


def test_the_most_reported_answer_is_read_first(tmp_path: Path) -> None:
    with KnowledgeStore(tmp_path / "k.db") as store:
        store.report_answer("rare", "Rare?", "a")
        for _ in range(4):
            store.report_answer("common", "Common?", "b")
        assert [r.canonical_query for r in store.answer_reports()] == ["common", "rare"]


def test_closing_a_report_takes_it_off_the_list(tmp_path: Path) -> None:
    with KnowledgeStore(tmp_path / "k.db") as store:
        report = store.report_answer("q", "Q?", "16 weeks.")
        assert store.resolve_report(report.id, status="fixed", resolution="pinned 20")
        assert store.answer_reports() == ()
        assert not store.resolve_report(report.id, status="fixed"), "already closed"
        (closed,) = store.answer_reports(status="fixed")
        assert closed.resolution == "pinned 20"


def test_reporting_a_fixed_answer_again_reopens_it(tmp_path: Path) -> None:
    """If it is still wrong, the fix did not work, and a closed row hides that."""
    with KnowledgeStore(tmp_path / "k.db") as store:
        report = store.report_answer("q", "Q?", "16 weeks.")
        store.resolve_report(report.id, status="fixed")
        store.report_answer("q", "Q?", "16 weeks.", note="no, still 16")
        (again,) = store.answer_reports()

    assert again.status == "open"
    assert again.reports == 2
    assert again.resolution is None
    assert "no, still 16" in again.notes


def test_a_report_is_fixed_or_dismissed_and_nothing_else(tmp_path: Path) -> None:
    with KnowledgeStore(tmp_path / "k.db") as store:
        report = store.report_answer("q", "Q?", "a")
        with pytest.raises(ValueError, match="fixed or dismissed"):
            store.resolve_report(report.id, status="maybe")


def test_an_unreadable_notes_column_does_not_hide_the_report(tmp_path: Path) -> None:
    path = tmp_path / "k.db"
    with KnowledgeStore(path) as store:
        store.report_answer("q", "Q?", "a", note="real")
        raw = sqlite3.connect(path)
        raw.execute("UPDATE answer_reports SET notes = 'not json'")
        raw.commit()
        raw.close()
        (report,) = store.answer_reports()
    assert report.notes == ()
    assert report.reports == 1


# -- through HTTP ------------------------------------------------------------

TOKEN = "t0ken"
ADMIN = {"authorization": f"Bearer {TOKEN}"}


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    documents = tmp_path / "documents"
    documents.mkdir()
    (documents / "handbook.md").write_text(
        "# Handbook\n\nParental leave is 20 weeks, fully paid.\n", encoding="utf-8"
    )
    return documents


@pytest.fixture
def app(tmp_path: Path, corpus: Path):
    return create_app(
        Settings(
            data_dir=str(tmp_path / "data"),
            documents_dir=str(corpus),
            admin_token=TOKEN,
            upload_enabled=True,
            local_enabled=False,
            embedding_enabled=False,
            _env_file=None,  # type: ignore[call-arg]
        )
    )


def _answer(client: TestClient, question: str) -> dict:
    body = client.post("/chat", json={"question": question})
    assert body.status_code == 200, body.text
    return body.json()


def test_a_reader_reports_a_wrong_answer_without_being_an_admin(app) -> None:
    """A report that needs an admin is a report nobody files."""
    with TestClient(app) as client:
        client.post(
            "/admin/pins",
            json={"question": "how much parental leave?", "answer": "16 weeks.", "cite": []},
            headers=ADMIN,
        )
        served = _answer(client, "how much parental leave?")
        sent = client.post(
            "/report",
            json={
                "question": "how much parental leave?",
                "answer": served["answer"],
                "tier": served["tier"],
                "note": "It went to 20 weeks in April.",
            },
        )
        assert sent.status_code == 201, sent.text
        seen = client.get("/admin/reports", headers=ADMIN).json()["reports"]

    assert [r["answer"] for r in seen] == ["16 weeks."]
    assert seen[0]["notes"] == ["It went to 20 weeks in April."]
    assert seen[0]["stale"] is False


def test_only_answers_this_install_gave_can_be_reported(app) -> None:
    """Otherwise the table holds whatever anybody posts, and an admin's
    morning is spent reading it."""
    with TestClient(app) as client:
        refused = client.post(
            "/report",
            json={"question": "what is the airspeed of a swallow?", "answer": "African?"},
        )
        assert refused.status_code == 404
        assert "as it was asked" in refused.json()["detail"]
        assert client.get("/admin/reports", headers=ADMIN).json()["reports"] == []


def test_a_report_becomes_stale_when_the_documents_change(app, corpus: Path) -> None:
    """The complaint may already be answered. Worth knowing before spending
    a morning on it, and worth not deleting."""
    with TestClient(app) as client:
        served = _answer(client, "how much parental leave?")
        client.post(
            "/report", json={"question": "how much parental leave?", "answer": served["answer"]}
        )
        assert client.get("/admin/reports", headers=ADMIN).json()["reports"][0]["stale"] is False

        (corpus / "handbook.md").write_text(
            "# Handbook\n\nParental leave is 22 weeks.\n", encoding="utf-8"
        )
        assert client.post("/admin/reindex", headers=ADMIN).status_code == 200
        assert client.get("/admin/reports", headers=ADMIN).json()["reports"][0]["stale"] is True


def test_closing_a_report_is_recorded_with_who_closed_it(app) -> None:
    with TestClient(app) as client:
        served = _answer(client, "how much parental leave?")
        client.post(
            "/report", json={"question": "how much parental leave?", "answer": served["answer"]}
        )
        report_id = client.get("/admin/reports", headers=ADMIN).json()["reports"][0]["id"]
        done = client.post(
            f"/admin/reports/{report_id}/resolve",
            json={"status": "dismissed", "note": "the handbook does say 20"},
            headers=ADMIN,
        )
        assert done.status_code == 200 and done.json()["status"] == "dismissed"
        assert client.get("/admin/reports", headers=ADMIN).json()["reports"] == []
        log = client.get("/admin/log", headers=ADMIN).json()

    (entry,) = [e for e in log["entries"] if e["action"] == "report.dismissed"]
    assert entry["target"] == str(report_id)


def test_a_report_that_is_already_closed_is_a_404(app) -> None:
    with TestClient(app) as client:
        served = _answer(client, "how much parental leave?")
        client.post(
            "/report", json={"question": "how much parental leave?", "answer": served["answer"]}
        )
        rid = client.get("/admin/reports", headers=ADMIN).json()["reports"][0]["id"]
        assert (
            client.post(f"/admin/reports/{rid}/resolve", json={}, headers=ADMIN).status_code == 200
        )
        assert (
            client.post(f"/admin/reports/{rid}/resolve", json={}, headers=ADMIN).status_code == 404
        )


def test_the_whole_loop_from_a_wrong_answer_to_a_right_one(app) -> None:
    """What the feature is for, end to end and through the same endpoints
    the page uses."""
    with TestClient(app) as client:
        client.post(
            "/admin/pins",
            json={"question": "how much parental leave?", "answer": "16 weeks.", "cite": []},
            headers=ADMIN,
        )
        assert _answer(client, "how much parental leave?")["answer"] == "16 weeks."

        client.post(
            "/report",
            json={
                "question": "how much parental leave?",
                "answer": "16 weeks.",
                "note": "20 since April",
            },
        )
        report = client.get("/admin/reports", headers=ADMIN).json()["reports"][0]

        client.post(
            "/admin/pins",
            json={"question": report["question"], "answer": "20 weeks, fully paid.", "cite": []},
            headers=ADMIN,
        )
        client.post(
            f"/admin/reports/{report['id']}/resolve",
            json={"status": "fixed", "note": "pinned the right figure"},
            headers=ADMIN,
        )

        assert _answer(client, "how much parental leave?")["answer"] == "20 weeks, fully paid."
        assert client.get("/admin/reports", headers=ADMIN).json()["reports"] == []


# -- the promise the table has to keep --------------------------------------


def test_the_report_cannot_name_who_objected(tmp_path: Path, corpus: Path) -> None:
    """Checked against the bytes, not the API.

    This table travels in every backup and is read by whoever runs the
    server. A reporter's name in it would turn "what did we get wrong" into
    "who complained", which is a different product and a worse one. The
    schema has no column for it; this proves nothing sneaks in through the
    fields that do exist.
    """
    idp = FakeIdp()
    try:
        settings = Settings(
            data_dir=str(tmp_path / "data"),
            documents_dir=str(corpus),
            auth_mode="oidc",
            oidc_issuer=idp.issuer,
            oidc_client_id="ok-report-test",
            oidc_client_secret="s3cret",
            admin_token=TOKEN,
            local_enabled=False,
            embedding_enabled=False,
            _env_file=None,  # type: ignore[call-arg]
        )
        app = create_app(settings)
        with TestClient(app) as client:
            started = client.get("/auth/login", follow_redirects=False)
            sent = {
                k: v[0] for k, v in parse_qs(urlparse(started.headers["location"]).query).items()
            }
            code = idp.mint_code(
                audience="ok-report-test",
                nonce=sent["nonce"],
                subject="alice-oid-8821",
                name="Alice Moreau",
                groups=(),
            )
            client.get(f"/auth/callback?code={code}&state={sent['state']}", follow_redirects=False)
            served = _answer(client, "how much parental leave?")
            assert (
                client.post(
                    "/report",
                    json={
                        "question": "how much parental leave?",
                        "answer": served["answer"],
                        "note": "this is out of date",
                    },
                ).status_code
                == 201
            )
            seen = client.get("/admin/reports", headers=ADMIN).json()
    finally:
        idp.close()

    assert seen["reports"], "the report was recorded at all"
    assert "Alice" not in json.dumps(seen)
    assert "alice-oid-8821" not in json.dumps(seen)

    raw = sqlite3.connect(tmp_path / "data" / "knowledge.db")
    try:
        dumped = "\n".join(raw.iterdump())
    finally:
        raw.close()
    assert "answer_reports" in dumped, "the table is in the file being searched"
    assert "Alice" not in dumped
    assert "alice-oid-8821" not in dumped


# -- roles -------------------------------------------------------------------


def test_reading_reports_is_a_curators_job_not_an_askers(app) -> None:
    with TestClient(app) as client:
        assert client.get("/admin/reports").status_code == 401
        assert client.get("/admin/reports", headers=ADMIN).status_code == 200


# -- the guard that runs where no browser is installed -----------------------


def test_the_chat_card_still_offers_the_button() -> None:
    """Proves only that the calls are wired; the browser test measures the
    behaviour. It exists so deleting the control cannot pass CI silently."""
    source = WIDGET.read_text(encoding="utf-8")
    assert "reportWrong(wrong, box, data, asked)" in source
    assert "'/report'" in source
