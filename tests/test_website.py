"""The website, and the one place this project collects anything.

OpenKnowledge's promise is that nothing leaves a deployment. A marketing site
with a contact form is the obvious place to quietly break that - a third-party
form service, an analytics tag, a font from a CDN - so these tests hold the page
and the endpoint to it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from openknowledge.api.app import create_app
from openknowledge.contacts import ContactError, ContactStore, clean

SITE = Path(__file__).resolve().parent.parent / "web" / "site" / "index.html"


@pytest.fixture
def page() -> str:
    return SITE.read_text(encoding="utf-8")


# -- the page makes no third-party requests ---------------------------------


def test_the_page_loads_nothing_from_anywhere_else(page: str) -> None:
    """A page advertising that your documents stay put must not call four other
    servers to render itself.

    Checks what a browser would *fetch*, not what words appear. A first version
    searched for "analytics" anywhere in the file and failed on the sentence
    promising there is none of it - the wrong thing to test, and the wrong way
    to fail.
    """
    fetched = re.findall(r'\ssrc\s*=\s*"([^"]+)"', page, flags=re.IGNORECASE)
    fetched += re.findall(
        r'<link[^>]+rel="(?:stylesheet|preload|preconnect)"[^>]+href="([^"]+)"',
        page,
        flags=re.IGNORECASE,
    )
    remote = [url for url in fetched if url.startswith(("http://", "https://", "//"))]
    assert remote == [], f"page fetches from third parties: {remote}"

    lowered = page.lower()
    for tag in ("<script src", "<iframe", "googletagmanager", "google-analytics", "fonts."):
        assert tag not in lowered, f"page embeds {tag}"


def test_the_form_posts_to_the_same_origin(page: str) -> None:
    """Submissions must reach the operator's own container, not a form service."""
    action = re.search(r'<form[^>]+action="([^"]+)"', page)
    assert action is not None
    assert action.group(1).startswith("/"), "relative action keeps it same-origin"


def test_the_page_states_its_own_limits(page: str) -> None:
    """The honest section is the reason to trust the rest of the page, so it is
    not something a later edit should be able to quietly drop."""
    lowered = page.lower()
    assert "does not do yet" in lowered
    for gap in ("sharepoint", "ocr", "unmeasured"):
        assert gap in lowered, f"the limits section no longer mentions {gap}"


# -- validation -------------------------------------------------------------


def test_a_usable_submission_is_accepted() -> None:
    fields = clean({"name": " Ada ", "email": "ada@example.com", "message": " hello "})
    assert fields["name"] == "Ada" and fields["message"] == "hello"


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        ({"name": "", "email": "a@b.co"}, "a name is required"),
        ({"name": "Ada", "email": "nope"}, "email address"),
        ({"name": "Ada", "email": ""}, "email address"),
        ({"name": "A" * 201, "email": "a@b.co"}, "longer than"),
    ],
)
def test_unusable_submissions_are_rejected_with_a_reason(
    payload: dict[str, str], reason: str
) -> None:
    with pytest.raises(ContactError, match=reason):
        clean(payload)


def test_an_unusual_but_valid_address_is_accepted() -> None:
    """Rejecting a real address is a worse failure than accepting a junk one,
    which a human deletes in a second."""
    for address in ("a+tag@sub.example.co.uk", "o'brien@example.ie", "ünïcode@example.com"):
        assert clean({"name": "X", "email": address})["email"] == address


# -- storage ----------------------------------------------------------------


def test_contacts_are_stored_and_read_back_newest_first() -> None:
    store = ContactStore(":memory:")
    for name in ("First", "Second", "Third"):
        store.add(clean({"name": name, "email": f"{name}@example.com"}))
    assert [c.name for c in store.recent()] == ["Third", "Second", "First"]
    assert store.count() == 3


# -- the endpoint -----------------------------------------------------------


@pytest.fixture
def client(tmp_path, monkeypatch) -> TestClient:
    monkeypatch.setenv("OK_WEBSITE_ENABLED", "true")
    monkeypatch.setenv("OK_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("OK_DOCUMENTS_DIR", str(tmp_path / "docs"))
    monkeypatch.setenv("OK_LOCAL_ENABLED", "false")
    (tmp_path / "docs").mkdir()
    return TestClient(create_app())


def test_the_form_is_off_unless_an_operator_turns_it_on(tmp_path, monkeypatch) -> None:
    """A running answer engine has no business accepting public writes by
    default."""
    monkeypatch.setenv("OK_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("OK_DOCUMENTS_DIR", str(tmp_path))
    monkeypatch.setenv("OK_LOCAL_ENABLED", "false")
    monkeypatch.delenv("OK_WEBSITE_ENABLED", raising=False)

    with TestClient(create_app()) as off:
        assert off.post("/api/contact", json={"name": "A", "email": "a@b.co"}).status_code == 404
        assert off.get("/site").status_code == 404


def test_a_submission_is_accepted_and_stored(client: TestClient) -> None:
    response = client.post(
        "/api/contact",
        json={"name": "Ada", "email": "ada@example.com", "interest": "audit"},
    )
    assert response.status_code == 201
    assert response.json() == {"received": True}

    stored = client.app.state.contacts.recent()
    assert [(c.name, c.email, c.interest) for c in stored] == [("Ada", "ada@example.com", "audit")]


def test_a_bot_that_fills_the_honeypot_is_thanked_and_dropped(client: TestClient) -> None:
    """Telling it what failed is how it learns to pass."""
    response = client.post(
        "/api/contact",
        json={"name": "Bot", "email": "bot@example.com", "website": "http://spam"},
    )
    assert response.status_code == 201
    assert client.app.state.contacts.count() == 0


def test_a_bad_address_is_refused_with_a_reason(client: TestClient) -> None:
    response = client.post("/api/contact", json={"name": "Ada", "email": "nope"})
    assert response.status_code == 422
    assert "email" in response.json()["detail"].lower()


def test_submissions_are_rate_limited(client: TestClient, monkeypatch) -> None:
    """A public write endpoint without one is an invitation."""
    client.app.state.settings.contact_max_per_hour = 2
    for n in range(2):
        assert (
            client.post(
                "/api/contact", json={"name": f"P{n}", "email": f"p{n}@example.com"}
            ).status_code
            == 201
        )
    blocked = client.post("/api/contact", json={"name": "P3", "email": "p3@example.com"})
    assert blocked.status_code == 429


def test_the_site_is_served_when_enabled(client: TestClient) -> None:
    response = client.get("/site")
    assert response.status_code == 200
    assert "OpenKnowledge" in response.text


def test_the_audit_output_on_the_page_is_what_the_command_prints(page: str) -> None:
    """The terminal block must be real output, not a tidied-up version of it.

    A first draft simplified the document ids and showed one contradiction where
    the command reports two. On a page whose whole argument is that its numbers
    come from something you can run, an invented terminal block is the worst
    thing it could contain - so this pins it to the command.
    """
    from openknowledge.audit import audit_folder, render

    corpus = Path(__file__).resolve().parent.parent / "evals" / "corpus" / "aveline"
    actual = render(audit_folder(corpus))

    blocks = [
        body
        for body in re.findall(r"<pre[^>]*>(.*?)</pre>", page, re.S)
        if "OpenKnowledge audit" in body
    ]
    assert blocks, "the page no longer shows audit output"

    # The terminal is coloured with spans; strip them back to what it reads as.
    shown = re.sub(r"</?span[^>]*>", "", blocks[0])
    shown = shown.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")

    # Two lines are legitimately edited: the shell prompt, and the path, which
    # is shown as a plausible install path rather than this checkout.
    for line in (row.strip() for row in shown.splitlines()):
        if not line or line.startswith(("$ ", "OpenKnowledge audit -")):
            continue
        assert line in actual, f"page shows a line the command never prints: {line!r}"


def test_the_page_quotes_the_live_run_numbers_it_recorded(page: str) -> None:
    """Same rule for the results table: the figures must match the run."""
    import json

    recorded = json.loads(
        (
            Path(__file__).resolve().parent.parent / "evals" / "measured" / "first-live-run.json"
        ).read_text()
    )
    assert f"{recorded['accuracy']:.1%}" in page
    assert f"{recorded['determinism']:.1%}" in page
    assert f"{recorded['paraphrase_consistency']:.1%}" in page


def test_the_page_only_advertises_commands_that_exist(page: str) -> None:
    """A landing page that shows a command the tool does not have is a lie.

    An earlier draft of this page advertised `openknowledge model use` before it
    was written, on the reasoning that it would exist shortly. This test is what
    stops that reasoning being used again.
    """
    import re as _re

    from openknowledge.cli import main

    shown = set(_re.findall(r"\bopenknowledge ([a-z-]+(?: [a-z-]+)?)", page))
    assert shown, "the page no longer shows any commands"

    for invocation in sorted(shown):
        with pytest.raises(SystemExit) as exited:
            main([*invocation.split(), "--help"])
        assert exited.value.code == 0, f"the page advertises `openknowledge {invocation}`"


def test_the_page_offers_an_installer_that_is_there(page: str) -> None:
    root = Path(__file__).resolve().parent.parent
    script = root / "install.sh"
    assert script.exists(), "the page's one-line install points at a file that is missing"
    assert script.stat().st_mode & 0o111, "install.sh is not executable"

    # The page tells the reader how long it is, so they know what they are
    # piping into a shell. Keep that number honest.
    claimed = re.search(r"the script is (\d+) lines", page)
    assert claimed is not None
    assert int(claimed.group(1)) == len(script.read_text().splitlines())
