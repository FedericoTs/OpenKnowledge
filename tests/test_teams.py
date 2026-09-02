"""The Teams channel's own parts: what it believes, what it asks, what it says.

The refusals are the point. A bot endpoint is a public URL, so each way a
forged or replayed activity could get an answer is minted here deliberately
and has to be turned away.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator

import pytest
from tests.fake_botframework import (
    APP_ID,
    APP_PASSWORD,
    TENANT_ID,
    FakeBotFramework,
)

from openknowledge.channels.teams import (
    LIMITED_NOTE,
    Connector,
    Conversation,
    GroupLookup,
    TeamsChannel,
    TeamsConfig,
    TeamsError,
    TokenValidator,
    _text_of,
)
from openknowledge.types import Answer, Citation, Tier

ALICE = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
HR = "11111111-1111-1111-1111-111111111111"
FINANCE = "22222222-2222-2222-2222-222222222222"


@pytest.fixture
def bot() -> Iterator[FakeBotFramework]:
    b = FakeBotFramework()
    b.memberships[ALICE] = [HR, FINANCE]
    try:
        yield b
    finally:
        b.close()


def _config(bot: FakeBotFramework) -> TeamsConfig:
    return TeamsConfig(
        app_id=APP_ID,
        app_password=APP_PASSWORD,
        tenant_id=TENANT_ID,
        metadata_url=f"{bot.base}/v1/.well-known/openidconfiguration",
        issuer=bot.issuer,
        graph_url=bot.base,
        login_url=bot.base,
    )


# -- what is read out of an activity -------------------------------------------------


def test_the_bots_own_mention_is_not_part_of_the_question(bot: FakeBotFramework) -> None:
    """In a channel every message names the bot. Left in, the same question
    asked in a channel and in a chat would canonicalise differently and miss
    the cache that makes it free."""
    activity = bot.activity(
        "<at>OpenKnowledge</at> how much parental leave do I get?",
        user_id=ALICE,
        entities=[
            {
                "type": "mention",
                "text": "<at>OpenKnowledge</at>",
                "mentioned": {"id": f"28:{APP_ID}", "name": "OpenKnowledge"},
            }
        ],
    )
    assert _text_of(activity) == "how much parental leave do I get?"
    assert TeamsChannel().parse(activity).text == "how much parental leave do I get?"


def test_an_activity_that_is_not_a_question_is_refused_not_answered(
    bot: FakeBotFramework,
) -> None:
    channel = TeamsChannel()
    for activity in (
        bot.activity("hello", user_id=ALICE, activity_type="conversationUpdate"),
        bot.activity("", user_id=ALICE),
        {**bot.activity("hi", user_id=ALICE), "from": {}},
    ):
        with pytest.raises(TeamsError):
            channel.parse(activity)


def test_the_sender_and_the_tenant_are_read_off_the_activity(bot: FakeBotFramework) -> None:
    activity = bot.activity("how much leave?", user_id=ALICE)
    channel = TeamsChannel()
    message = channel.parse(activity)
    assert (message.user_id, message.channel) == (ALICE, "teams")
    assert message.thread_id == "a:conversation-1"
    assert message.principals is None, "a permission is never decided by parsing"
    assert channel.tenant_of(activity) == TENANT_ID
    assert channel.conversation_of(activity) == Conversation(bot.base, "a:conversation-1")


# -- proving the request ---------------------------------------------------------------


def _claims(validator: TokenValidator, token: str, activity: dict) -> dict:
    return asyncio.run(validator.claims(f"Bearer {token}", activity))


def test_a_correctly_signed_token_is_accepted(bot: FakeBotFramework) -> None:
    validator = TokenValidator(_config(bot))
    activity = bot.activity("q", user_id=ALICE)
    claims = _claims(validator, bot.mint(service_url=bot.base), activity)
    assert claims["aud"] == APP_ID
    asyncio.run(validator.aclose())


@pytest.mark.parametrize(
    ("what", "kwargs"),
    [
        ("another bot's audience", {"audience": "someone-else"}),
        ("an issuer that is not the Bot Service", {"issuer": "https://evil.example"}),
        ("an expired token", {"expires_in": -3600.0}),
        ("a key the Bot Service does not publish", {"kid": "not-a-key"}),
    ],
)
def test_a_token_that_is_wrong_in_any_way_is_refused(
    bot: FakeBotFramework, what: str, kwargs: dict
) -> None:
    validator = TokenValidator(_config(bot))
    activity = bot.activity("q", user_id=ALICE)
    with pytest.raises(TeamsError):
        _claims(validator, bot.mint(service_url=bot.base, **kwargs), activity)
    asyncio.run(validator.aclose())


def test_a_token_cannot_be_replayed_with_a_rewritten_reply_address(
    bot: FakeBotFramework,
) -> None:
    """The serviceUrl claim binds a token to where its replies go. Without the
    check, a token minted for this bot could arrive with a serviceUrl pointing
    at an attacker's server and the answer would follow it there."""
    validator = TokenValidator(_config(bot))
    activity = {**bot.activity("q", user_id=ALICE), "serviceUrl": "https://evil.example"}
    with pytest.raises(TeamsError, match="serviceUrl"):
        _claims(validator, bot.mint(service_url=bot.base), activity)
    asyncio.run(validator.aclose())


def test_no_token_at_all_is_refused(bot: FakeBotFramework) -> None:
    validator = TokenValidator(_config(bot))
    activity = bot.activity("q", user_id=ALICE)
    for header in (None, "", "Basic abc", "Bearer not-a-jwt"):
        with pytest.raises(TeamsError):
            asyncio.run(validator.claims(header, activity))
    asyncio.run(validator.aclose())


def test_the_keys_are_fetched_once_and_kept(bot: FakeBotFramework) -> None:
    validator = TokenValidator(_config(bot))
    activity = bot.activity("q", user_id=ALICE)
    for _ in range(3):
        _claims(validator, bot.mint(service_url=bot.base), activity)
    assert sum("/v1/keys" in r for r in bot.requests) == 1
    assert sum("openidconfiguration" in r for r in bot.requests) == 1
    asyncio.run(validator.aclose())


# -- who is asking -----------------------------------------------------------------------


def test_the_askers_groups_come_from_graph_and_are_cached(bot: FakeBotFramework) -> None:
    lookup = GroupLookup(_config(bot))
    channel = TeamsChannel()
    message = channel.parse(bot.activity("q", user_id=ALICE))

    principals, complete = asyncio.run(channel.principals(message, lookup))
    assert complete is True
    assert principals == frozenset(
        {"authenticated", f"user:{ALICE}", f"group:{HR}", f"group:{FINANCE}"}
    )

    asyncio.run(channel.principals(message, lookup))
    assert sum("transitiveMemberOf" in r for r in bot.requests) == 1, "the second ask is cached"
    assert bot.token_calls == 1
    asyncio.run(lookup.aclose())


def test_a_failed_group_lookup_answers_as_nobody_in_particular(bot: FakeBotFramework) -> None:
    """Fail closed: a lookup that cannot answer must not become "everyone".
    The asker keeps only what every employee holds, and the reply says so."""
    bot.graph_status = 403
    lookup = GroupLookup(_config(bot))
    channel = TeamsChannel()
    message = channel.parse(bot.activity("q", user_id=ALICE))
    principals, complete = asyncio.run(channel.principals(message, lookup))
    assert complete is False
    assert principals == frozenset({"authenticated", f"user:{ALICE}"})
    assert f"group:{HR}" not in principals
    asyncio.run(lookup.aclose())


def test_the_group_cache_expires(bot: FakeBotFramework) -> None:
    now = [1000.0]
    lookup = GroupLookup(_config(bot), clock=lambda: now[0], ttl=900.0)
    asyncio.run(lookup.groups(ALICE))
    now[0] += 899
    asyncio.run(lookup.groups(ALICE))
    assert lookup.lookups == 1
    now[0] += 2
    bot.memberships[ALICE] = [HR]
    assert asyncio.run(lookup.groups(ALICE)) == frozenset({HR}), "a removal takes effect"
    assert lookup.lookups == 2
    asyncio.run(lookup.aclose())


# -- what the asker sees ------------------------------------------------------------------


def _answer(text: str, *, tier: Tier = Tier.LOCAL, citations: tuple[Citation, ...] = ()) -> Answer:
    return Answer(text=text, tier=tier, model_id="qwen3", cache_key="k", citations=citations)


def test_the_reply_carries_the_answer_and_where_it_came_from() -> None:
    answer = _answer(
        "Twenty weeks, fully paid.",
        citations=(
            Citation(
                document_id="hr-leave",
                document_title="Parental Leave",
                snippet="…",
                locator="p.2",
                url="https://contoso.sharepoint.com/leave.docx",
            ),
        ),
    )
    reply = TeamsChannel().reply(answer)
    assert reply["type"] == "message" and reply["textFormat"] == "markdown"
    assert "Twenty weeks, fully paid." in reply["text"]
    assert "**Sources**" in reply["text"]
    assert "[Parental Leave](https://contoso.sharepoint.com/leave.docx) (p.2)" in reply["text"]
    assert LIMITED_NOTE not in reply["text"]


def test_a_local_file_citation_is_named_but_not_linked() -> None:
    """A file:// link is the server's path, not the reader's - it would open
    nothing on their machine and leak the server's layout to everyone."""
    answer = _answer(
        "EUR 500.",
        citations=(
            Citation(
                document_id="expenses",
                document_title="Expenses Policy",
                snippet="…",
                url="file:///srv/documents/expenses.md",
            ),
        ),
    )
    text = TeamsChannel().reply(answer)["text"]
    assert "- Expenses Policy" in text and "file:///" not in text


def test_a_citation_marker_never_reaches_the_reader() -> None:
    """A model cites by writing [hr-leave] after a claim, which is how the
    grounding gate checks it. The widget strips those in the browser; a Teams
    reader would otherwise have been the only one who saw the machinery. A
    bracket the document itself wrote is not a marker and stays."""
    answer = _answer(
        "Install the client [vpn-access]. Approved by IT Operations [vpn-access], "
        "and clause [7] applies.",
        citations=(
            Citation(document_id="vpn-access", document_title="Remote Access", snippet="…"),
        ),
    )
    text = TeamsChannel().reply(answer)["text"]
    assert "[vpn-access]" not in text
    assert "Install the client." in text
    assert "IT Operations," in text
    assert "clause [7] applies." in text, "a bracket that is not a citation is the document's"


def test_a_limited_answer_says_it_is_limited() -> None:
    text = TeamsChannel().reply(_answer("A partial answer."), limited=True)["text"]
    assert LIMITED_NOTE in text


def test_an_answer_with_no_source_says_that_rather_than_nothing() -> None:
    text = TeamsChannel().reply(_answer("Something."))["text"]
    assert "No source was cited" in text


def test_a_refusal_is_not_decorated_with_a_missing_source_note() -> None:
    text = TeamsChannel().reply(_answer("I don't know.", tier=Tier.REFUSED))["text"]
    assert "No source was cited" not in text and "Sources" not in text


# -- talking back ----------------------------------------------------------------------------


def test_the_connector_authenticates_and_delivers(bot: FakeBotFramework) -> None:
    connector = Connector(_config(bot))
    where = Conversation(bot.base, "a:conversation-1")
    asyncio.run(connector.typing(where))
    asyncio.run(connector.send(where, {"type": "message", "text": "hello"}))
    assert [a["type"] for _, a in bot.replies] == ["typing", "message"]
    assert bot.replies[1][0] == "a:conversation-1"
    assert connector.token_fetches == 1, "one token for both calls"
    asyncio.run(connector.aclose())


def test_a_connector_that_cannot_deliver_does_not_raise(bot: FakeBotFramework) -> None:
    """The answer was produced and paid for; a lost reply is a warning in the
    log, not an exception that takes a request down with it."""
    connector = Connector(_config(bot))
    asyncio.run(connector.send(Conversation("http://127.0.0.1:9", "c"), {"type": "message"}))
    assert bot.replies == []
    asyncio.run(connector.aclose())
