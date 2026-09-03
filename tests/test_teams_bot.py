"""The bot endpoint through the running app.

A bot endpoint is a public URL that anyone who finds it can post to, so most
of this is about what it refuses. The rest is the one thing that matters when
it does answer: the asker sees only what their own groups let them see, and
the answer arrives with its sources through the connector rather than in the
HTTP response, because a self-hosted model takes longer than Teams waits.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from tests.fake_botframework import (
    APP_ID,
    APP_PASSWORD,
    OTHER_TENANT,
    TENANT_ID,
    FakeBotFramework,
)
from tests.fakes import FakeProvider

from openknowledge.api import engine as engine_module
from openknowledge.api.app import create_app
from openknowledge.channels.teams import LIMITED_NOTE
from openknowledge.config import Settings

ALICE = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
HR = "11111111-1111-1111-1111-111111111111"
TOKEN = "t0ken"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
#: What the model says. The trailing marker is how an answer cites its
#: source; the reply that reaches Teams must not still carry it.
LEAVE = "Employees with 12 months of service get 20 weeks of fully paid parental leave. [hr-leave]"


@pytest.fixture
def bot() -> Iterator[FakeBotFramework]:
    b = FakeBotFramework()
    b.memberships[ALICE] = [HR]
    try:
        yield b
    finally:
        b.close()


def _settings(tmp_path: Path, bot: FakeBotFramework, **overrides: object) -> Settings:
    docs = tmp_path / "documents"
    (docs / "hr").mkdir(parents=True)
    (docs / "hr" / "leave.md").write_text(
        "# Parental Leave\nEmployees with 12 months of service get 20 weeks of fully "
        "paid parental leave.\n",
        encoding="utf-8",
    )
    (docs / "handbook.md").write_text("# Handbook\nThe office closes at 18:00.\n", encoding="utf-8")
    values: dict[str, object] = {
        "data_dir": str(tmp_path / "data"),
        "documents_dir": str(docs),
        "admin_token": TOKEN,
        "local_enabled": True,
        "embedding_enabled": False,
        "escalation_enabled": False,
        "teams_enabled": True,
        "teams_app_id": APP_ID,
        "teams_app_password": APP_PASSWORD,
        "teams_tenant_id": TENANT_ID,
        "teams_metadata_url": f"{bot.base}/v1/.well-known/openidconfiguration",
        "teams_issuer": bot.issuer,
        "teams_graph_url": bot.base,
        "teams_login_url": bot.base,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)  # type: ignore[call-arg, arg-type]


@pytest.fixture
def client(
    tmp_path: Path, bot: FakeBotFramework, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    fake = FakeProvider(replies=[LEAVE] * 6)
    monkeypatch.setattr(engine_module, "_build_local", lambda settings: fake)
    with TestClient(create_app(_settings(tmp_path, bot))) as c:
        # HR's folder is HR's. The bot must honour this exactly as the widget does.
        c.put("/admin/access/hr", headers=AUTH, json={"principals": [f"group:{HR}"]})
        yield c


def _post(client: TestClient, bot: FakeBotFramework, activity: dict, **mint: object):
    token = bot.mint(service_url=activity.get("serviceUrl"), **mint)  # type: ignore[arg-type]
    return client.post(
        "/teams/messages", json=activity, headers={"Authorization": f"Bearer {token}"}
    )


def test_a_question_is_answered_through_the_connector_with_its_sources(
    client: TestClient, bot: FakeBotFramework
) -> None:
    activity = bot.activity("how much parental leave do I get?", user_id=ALICE)
    response = _post(client, bot, activity)
    assert response.status_code == 200
    assert response.content in (b"", b"null"), "the answer does not ride in the response"

    kinds = [a["type"] for _, a in bot.replies]
    assert kinds == ["typing", "message"], "typing first, because the model takes a while"
    text = bot.replies[1][1]["text"]
    assert "20 weeks" in text
    assert "[hr-leave]" not in text, "the citation marker is machinery, not something to read"
    assert "**Sources**" in text and "Parental Leave" in text
    assert LIMITED_NOTE not in text
    assert bot.replies[1][0] == activity["conversation"]["id"]


def test_the_asker_sees_only_what_their_groups_allow(
    client: TestClient, bot: FakeBotFramework
) -> None:
    """The whole reason identity is looked up. Alice is in HR and gets the
    answer; the same question from somebody who is not gets a refusal, from
    the same corpus and the same model."""
    bob = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
    bot.memberships[bob] = []
    _post(client, bot, bot.activity("how much parental leave do I get?", user_id=ALICE))
    answered = bot.replies[-1][1]["text"]
    bot.replies.clear()

    _post(client, bot, bot.activity("how much parental leave do I get?", user_id=bob))
    refused = bot.replies[-1][1]["text"]
    assert "20 weeks" in answered
    assert "20 weeks" not in refused, refused
    assert "don't know" in refused.lower() or "isn't covered" in refused.lower()


def test_a_failed_group_lookup_answers_narrowly_and_says_so(
    client: TestClient, bot: FakeBotFramework
) -> None:
    bot.graph_status = 503
    _post(client, bot, bot.activity("how much parental leave do I get?", user_id=ALICE))
    text = bot.replies[-1][1]["text"]
    assert LIMITED_NOTE in text
    assert "20 weeks" not in text, "failing closed means the HR document stays unseen"


def test_an_activity_without_a_valid_token_is_refused_and_answered_to_nobody(
    client: TestClient, bot: FakeBotFramework
) -> None:
    activity = bot.activity("how much parental leave do I get?", user_id=ALICE)
    unsigned = client.post("/teams/messages", json=activity)
    assert unsigned.status_code == 401
    forged = _post(client, bot, activity, audience="another-bot")
    assert forged.status_code == 401
    expired = _post(client, bot, activity, expires_in=-3600.0)
    assert expired.status_code == 401
    assert bot.replies == [], "nothing was said to anybody"
    assert unsigned.json()["detail"] == "unauthorised", "and nothing useful was disclosed"


def test_another_tenants_activity_is_refused(client: TestClient, bot: FakeBotFramework) -> None:
    """A bot registration can be added by any tenant that finds it. The
    documents behind this one belong to exactly one company."""
    activity = bot.activity("how much leave?", user_id=ALICE, tenant_id=OTHER_TENANT)
    response = _post(client, bot, activity)
    assert response.status_code == 403
    assert bot.replies == []


def test_a_join_or_a_reaction_is_ignored_quietly(client: TestClient, bot: FakeBotFramework) -> None:
    activity = bot.activity("", user_id=ALICE, activity_type="conversationUpdate")
    assert _post(client, bot, activity).status_code == 200
    assert bot.replies == [], "a join is not a question"


def test_one_asker_cannot_spend_everybodys_budget(
    tmp_path: Path, bot: FakeBotFramework, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeProvider(replies=[LEAVE] * 6)
    monkeypatch.setattr(engine_module, "_build_local", lambda settings: fake)
    settings = _settings(tmp_path, bot, asker_questions_per_minute=1)
    with TestClient(create_app(settings)) as c:
        activity = bot.activity("how much parental leave?", user_id=ALICE)
        assert _post(c, bot, activity).status_code == 200
        bot.replies.clear()
        assert _post(c, bot, activity).status_code == 200
    ((_, told),) = bot.replies
    assert "limit per person" in told["text"] and "Try again in" in told["text"]


def test_the_ledger_knows_the_question_came_from_teams(
    client: TestClient, bot: FakeBotFramework
) -> None:
    _post(client, bot, bot.activity("how much parental leave do I get?", user_id=ALICE))
    engine = client.app.state.engine
    assert [e.channel for e in engine.store.recent_questions(5)] == ["teams"]


def test_with_the_bot_off_there_is_no_bot_endpoint(tmp_path: Path, bot: FakeBotFramework) -> None:
    """Not registered rather than registered and refusing: an endpoint that
    exists is an endpoint somebody probes."""
    with TestClient(create_app(_settings(tmp_path, bot, teams_enabled=False))) as c:
        assert c.post("/teams/messages", json={}).status_code == 404


def test_half_configured_refuses_to_start_rather_than_accepting_activities(
    tmp_path: Path, bot: FakeBotFramework
) -> None:
    settings = _settings(tmp_path, bot, teams_app_password=None)
    with pytest.raises(RuntimeError, match="OK_TEAMS_APP_PASSWORD"):
        create_app(settings)
