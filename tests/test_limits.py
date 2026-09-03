"""One asker cannot spend everybody else's day.

The budget governor already stops a flood becoming an invoice - its ceiling is
remaining budget over questions still expected, so a thousand questions lower
what any one of them may cost rather than running the bill up. What it cannot
do is decide *whose* questions those were, so a looping caller drags the shared
ceiling down for the whole company and the first anybody notices is that
answers got worse at eleven on a Tuesday.

The test that matters most here is the pair: the flood is cut off, and the
colleague who has asked nothing is served normally in the same breath. A limit
that stopped everybody would be an outage wearing a cost control's clothes.

The second is the privacy one. Enforcing a limit needs to know that this caller
has asked twelve times in the last minute; it never needs to know who they are,
and it must not quietly become the log that says so.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi.testclient import TestClient

from fake_idp import FakeIdp
from openknowledge.api.app import create_app
from openknowledge.config import Settings
from openknowledge.limits import AskerLimiter
from openknowledge.metrics import Sample, render

# -- the limiter -------------------------------------------------------------


def test_it_allows_up_to_the_limit_and_then_stops() -> None:
    limiter = AskerLimiter(3)
    assert [limiter.check("someone").allowed for _ in range(5)] == [
        True,
        True,
        True,
        False,
        False,
    ]
    assert limiter.refused == 2


def test_the_window_slides_rather_than_resetting_on_the_hour() -> None:
    """A fixed bucket lets a caller spend the whole limit at 10:59:59 and the
    whole limit again at 11:00:00, which is the burst this exists to stop."""
    limiter = AskerLimiter(2, window_seconds=60.0)
    assert limiter.check("someone", now=0.0).allowed
    assert limiter.check("someone", now=30.0).allowed
    assert not limiter.check("someone", now=59.0).allowed

    # The first question leaves the window; room for exactly one more.
    assert limiter.check("someone", now=61.0).allowed
    assert not limiter.check("someone", now=61.5).allowed


def test_it_says_when_to_come_back() -> None:
    limiter = AskerLimiter(1, window_seconds=60.0)
    limiter.check("someone", now=0.0)
    refused = limiter.check("someone", now=10.0)
    assert not refused.allowed
    assert refused.retry_after == pytest.approx(50.0), "when the oldest falls out"


def test_one_asker_is_not_another(tmp_path: Path) -> None:
    limiter = AskerLimiter(1)
    assert limiter.check("alice").allowed
    assert limiter.check("bob").allowed, "bob has asked nothing"
    assert not limiter.check("alice").allowed


def test_zero_turns_it_off_entirely() -> None:
    """The right default for a desktop install, where the only asker is the
    person whose laptop it is."""
    limiter = AskerLimiter(0)
    assert not limiter.enabled
    assert all(limiter.check("someone").allowed for _ in range(100))
    assert limiter.refused == 0


def test_the_limiter_never_holds_the_asker(tmp_path: Path) -> None:
    """It counts askers without naming them. The keys are a salted digest, so
    the plaintext is nowhere in this object to be dumped, logged or exported."""
    limiter = AskerLimiter(5)
    limiter.check("user:alice-oid-8821")

    state = repr(limiter.__dict__)
    assert "alice-oid-8821" not in state
    assert "user:alice-oid-8821" not in limiter._seen  # noqa: SLF001 - the point
    assert len(limiter._seen) == 1  # noqa: SLF001 - it did count somebody


def test_two_processes_do_not_share_a_key_for_the_same_person() -> None:
    """The salt is per-process and random, so a key cannot be correlated
    across a restart or between two installs."""
    assert AskerLimiter(5).key("user:alice") != AskerLimiter(5).key("user:alice")


def test_the_same_asker_is_the_same_key_within_a_process() -> None:
    limiter = AskerLimiter(5)
    assert limiter.key("user:alice") == limiter.key("user:alice")
    assert limiter.key("user:alice") != limiter.key("user:bob")


def test_askers_who_stopped_asking_are_forgotten(monkeypatch) -> None:
    """Otherwise the dict is a slow leak keyed by everybody who ever asked."""
    import openknowledge.limits as limits

    monkeypatch.setattr(limits, "_SWEEP_ABOVE", 4)
    limiter = AskerLimiter(5, window_seconds=10.0)
    for i in range(6):
        limiter.check(f"asker-{i}", now=0.0)
    limiter.check("someone new", now=100.0)

    assert len(limiter._seen) == 1, "the sweep dropped everyone out of the window"  # noqa: SLF001


# -- through HTTP ------------------------------------------------------------

TOKEN = "t0ken"
ADMIN = {"authorization": f"Bearer {TOKEN}"}


@pytest.fixture
def idp():
    provider = FakeIdp()
    yield provider
    provider.close()


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    documents = tmp_path / "documents"
    documents.mkdir()
    (documents / "handbook.md").write_text(
        "# Handbook\n\nParental leave is 20 weeks.\n", encoding="utf-8"
    )
    return documents


def _app(tmp_path: Path, corpus: Path, idp: FakeIdp | None = None, **over):
    values: dict = {
        "data_dir": str(tmp_path / "data"),
        "documents_dir": str(corpus),
        "admin_token": TOKEN,
        "local_enabled": False,
        "embedding_enabled": False,
        "_env_file": None,
    }
    if idp is not None:
        values.update(
            auth_mode="oidc",
            oidc_issuer=idp.issuer,
            oidc_client_id="ok-limit-test",
            oidc_client_secret="s3cret",
        )
    values.update(over)
    return create_app(Settings(**values))


def _person(app, idp: FakeIdp, subject: str) -> TestClient:
    client = TestClient(app)
    started = client.get("/auth/login", follow_redirects=False)
    sent = {k: v[0] for k, v in parse_qs(urlparse(started.headers["location"]).query).items()}
    code = idp.mint_code(audience="ok-limit-test", nonce=sent["nonce"], subject=subject)
    client.get(f"/auth/callback?code={code}&state={sent['state']}", follow_redirects=False)
    return client


def test_a_flood_is_stopped_and_a_colleague_is_not(tmp_path: Path, corpus: Path, idp) -> None:
    """The whole point. A limit that stopped everybody would be an outage
    wearing a cost control's clothes."""
    app = _app(tmp_path, corpus, idp, asker_questions_per_minute=3)
    with TestClient(app):
        bot = _person(app, idp, "bot-oid")
        colleague = _person(app, idp, "dana-oid")

        codes = [bot.post("/chat", json={"question": f"q {i}"}).status_code for i in range(6)]
        assert codes == [200, 200, 200, 429, 429, 429]

        served = colleague.post("/chat", json={"question": "how much parental leave?"})
        assert served.status_code == 200, "the colleague has asked nothing"


def test_the_refusal_says_what_happened_and_when_to_retry(
    tmp_path: Path, corpus: Path, idp
) -> None:
    app = _app(tmp_path, corpus, idp, asker_questions_per_minute=1)
    with TestClient(app):
        asker = _person(app, idp, "someone")
        asker.post("/chat", json={"question": "first"})
        refused = asker.post("/chat", json={"question": "second"})

    assert refused.status_code == 429
    assert "limit per person" in refused.json()["detail"]
    assert int(refused.headers["retry-after"]) >= 1


def test_the_streaming_endpoint_is_limited_too(tmp_path: Path, corpus: Path, idp) -> None:
    """The widget streams; a limit only on /chat would be a limit on nothing."""
    app = _app(tmp_path, corpus, idp, asker_questions_per_minute=1)
    with TestClient(app):
        asker = _person(app, idp, "someone")
        assert asker.post("/chat/stream", json={"question": "first"}).status_code == 200
        assert asker.post("/chat/stream", json={"question": "second"}).status_code == 429


def test_the_limit_can_be_changed_without_a_restart(tmp_path: Path, corpus: Path) -> None:
    """The lever an operator reaches for while a caller is looping. A rebuild
    would make them wait for it."""
    with TestClient(_app(tmp_path, corpus, asker_questions_per_minute=0)) as client:
        assert all(
            client.post("/chat", json={"question": f"q {i}"}).status_code == 200 for i in range(5)
        )
        applied = client.put(
            "/admin/settings", json={"asker_questions_per_minute": 2}, headers=ADMIN
        )
        assert applied.status_code == 200, applied.text
        codes = [client.post("/chat", json={"question": f"r {i}"}).status_code for i in range(4)]

    assert codes == [200, 200, 429, 429]


def test_off_by_default(tmp_path: Path, corpus: Path) -> None:
    """A desktop install must not meet a limit it never asked for."""
    with TestClient(_app(tmp_path, corpus)) as client:
        codes = {client.post("/chat", json={"question": f"q {i}"}).status_code for i in range(40)}
    assert codes == {200}


# -- metrics -----------------------------------------------------------------


def test_a_metric_family_is_written_in_one_block() -> None:
    """The format requires it, and writing samples in arrival order does not.
    Interleaving two families - which happens the moment a second window or
    label is added - makes a strict scraper reject the page as a duplicate."""
    text = render(
        [
            Sample("a_total", "A.", "counter", 1.0, (("w", "all"),)),
            Sample("b_total", "B.", "counter", 2.0, (("w", "all"),)),
            Sample("a_total", "A.", "counter", 3.0, (("w", "day"),)),
        ]
    )
    lines = [line for line in text.splitlines() if not line.startswith("#")]
    assert lines == [
        'a_total{w="all"} 1',
        'a_total{w="day"} 3',
        'b_total{w="all"} 2',
    ]
    assert text.count("# TYPE a_total") == 1, "one header per family"


def test_a_label_cannot_break_out_of_its_quotes() -> None:
    text = render([Sample("m", "M.", "gauge", 1.0, (("v", 'a"b\\c\nd'),))])
    assert 'm{v="a\\"b\\\\c\\nd"} 1' in text


def test_metrics_are_admin_only(tmp_path: Path, corpus: Path) -> None:
    """Spend and volume are not everybody's business; /healthz stays open."""
    with TestClient(_app(tmp_path, corpus)) as client:
        assert client.get("/metrics").status_code == 401
        assert client.get("/healthz").status_code == 200
        assert client.get("/metrics", headers=ADMIN).status_code == 200


def test_metrics_report_what_actually_happened(tmp_path: Path, corpus: Path) -> None:
    with TestClient(_app(tmp_path, corpus, asker_questions_per_minute=2)) as client:
        client.post(
            "/admin/pins",
            json={"question": "how much parental leave?", "answer": "20 weeks.", "cite": []},
            headers=ADMIN,
        )
        for _ in range(4):
            client.post("/chat", json={"question": "how much parental leave?"})
        body = client.get("/metrics", headers=ADMIN)

    assert body.headers["content-type"].startswith("text/plain")
    text = body.text
    assert 'openknowledge_questions_total{tier="pinned",window="all"} 2' in text
    assert "openknowledge_rate_limited_total 2" in text
    assert "openknowledge_asker_limit_per_minute 2" in text
    assert "openknowledge_documents_indexed 1" in text


def test_metrics_carry_no_question_and_no_person(tmp_path: Path, corpus: Path, idp) -> None:
    """A metric with the question in it is a log of what people asked,
    published to whatever scrapes it - and this page is scraped on a timer by
    something nobody reads until it is too late."""
    app = _app(tmp_path, corpus, idp, asker_questions_per_minute=1)
    with TestClient(app):
        asker = _person(app, idp, "alice-oid-8821")
        asker.post("/chat", json={"question": "what is the secret handshake"})
        asker.post("/chat", json={"question": "what is the secret handshake"})
        text = TestClient(app).get("/metrics", headers=ADMIN).text

    assert "openknowledge_rate_limited_total 1" in text, "it did record the refusal"
    assert "secret handshake" not in text
    assert "alice-oid-8821" not in text
