"""What the answer card actually shows, driven in a real browser.

Two field defects, both photographed in a running 0.2.13 and both invisible to
every test that stopped at the JSON:

* a refusal was printed twice - once struck through under "Withdrawn - this did
  not pass the grounding check", once plain - because the model declined, the
  gate reported the decline, and the honest refusal that replaced the draft was
  word for word the draft. The product at its most characteristic looked like a
  crash.
* the ``[doc-id]`` markers the model writes to cite, and the grounding gate
  reads, were shown to the reader as if they were prose.

The stream here is the real one: the real router, the real grounding gate, the
real SSE endpoint, the real widget in Chromium. Only the model is a fake, and
only so that "the model declines" is a fact rather than a hope.

Skips when Playwright or a browser is missing, as the other browser test does -
which means CI, where neither is installed, does not run these. The source
assertion at the bottom is the guard that survives there, and it is honest
about proving only that the calls are still wired.
"""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from tests.test_auth_browser import _chromium
from tests.test_streaming import StreamingFakeProvider

from openknowledge.config import Settings
from openknowledge.prompts import REFUSAL_TEXT

WIDGET = Path("web/widget/index.html")


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _serve(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake) -> Iterator[str]:
    """A real server on loopback whose only fake part is the model."""
    uvicorn = pytest.importorskip("uvicorn")
    from openknowledge.api import app as app_module
    from openknowledge.api import engine as engine_module

    documents = tmp_path / "documents"
    documents.mkdir()
    (documents / "handbook.md").write_text(
        "# Handbook\n\nThe office closes at 18:00.\n", encoding="utf-8"
    )
    monkeypatch.setattr(engine_module, "_build_local", lambda settings: fake)

    settings = Settings(
        data_dir=str(tmp_path / "data"),
        documents_dir=str(documents),
        local_enabled=True,
        embedding_enabled=False,
        _env_file=None,  # type: ignore[call-arg]
    )
    port = _free_port()
    server = uvicorn.Server(
        uvicorn.Config(
            app_module.create_app(settings), host="127.0.0.1", port=port, log_level="warning"
        )
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 15
    while not server.started:
        if time.time() > deadline:
            pytest.fail("uvicorn did not start")
        time.sleep(0.05)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)


@pytest.fixture
def declining_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """A server whose local model always declines - the field case, on demand.

    The model answers "I don't know", which the gate reads as an abstention and
    the router turns into REFUSAL_TEXT: the same sentence, which is exactly
    what made the draft and the answer duplicates.
    """
    yield from _serve(tmp_path, monkeypatch, StreamingFakeProvider(replies=[REFUSAL_TEXT] * 3))


@pytest.fixture
def dying_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """A server whose local model dies between announcing itself and its first
    token - the second way a retraction ends up owed for nothing."""
    yield from _serve(tmp_path, monkeypatch, StreamingFakeProvider(fail=True))


def _page(pw, base: str):
    executable = _chromium()
    if executable is None:
        pytest.skip("no chromium available")
    browser = pw.chromium.launch(executable_path=executable, args=["--no-sandbox"])
    page = browser.new_page(viewport={"width": 1280, "height": 900})
    page.goto(base + "/")
    return browser, page


def test_a_refusal_is_printed_once(declining_app) -> None:
    """The photographed defect, reproduced and then absent.

    Before this fix the card carried the refusal twice with a "Withdrawn"
    banner between them. Nothing vanished from the reader's view - the draft
    and the answer are the same sentence - so there is nothing to account for.
    """
    sync_playwright = pytest.importorskip(
        "playwright.sync_api", reason="playwright is not installed"
    ).sync_playwright

    with sync_playwright() as pw:
        browser, page = _page(pw, declining_app)
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        try:
            # A question the handbook does retrieve for: retrieval has to
            # find something or the model is never called, and then there is
            # no draft to withdraw and nothing to duplicate. Measured while
            # writing this: the first question here matched no document, so
            # the stream was a bare `final` and the test passed against the
            # unfixed widget.
            page.fill("#q", "when does the office close?")
            page.click("#send")
            page.wait_for_selector(".msg.a .badge", timeout=60000)
            card = page.inner_text(".msg.a")

            assert card.count("that isn't covered by the documents") == 1, (
                f"the refusal was printed more than once:\n{card}"
            )
            assert "Withdrawn" not in card, (
                "a draft that says what the answer says needs no retraction notice"
            )
            assert page.query_selector(".msg.a .retracted") is None
            assert errors == [], f"page errors: {errors}"
        finally:
            browser.close()


def test_a_withdrawn_draft_that_said_something_else_is_still_shown(declining_app) -> None:
    """The bargain the fix must not break.

    Only duplicates are dropped. A draft the reader watched appear, saying
    something the answer does not say, is still withdrawn in front of them.
    """
    sync_playwright = pytest.importorskip(
        "playwright.sync_api", reason="playwright is not installed"
    ).sync_playwright

    with sync_playwright() as pw:
        browser, page = _page(pw, declining_app)
        try:
            kept = page.evaluate(
                """() => {
                  const pending = document.createElement('div');
                  const draft = document.createElement('div');
                  draft.className = 'retracted';
                  const p = document.createElement('p');
                  p.textContent = 'Contractors may be paid in Bitcoin up to 5,000 EUR.';
                  draft.appendChild(p);
                  pending.appendChild(draft);
                  return retractionsWorthKeeping(pending, "I don't know - that isn't "
                    + "covered by the documents I have.").length;
                }"""
            )
            assert kept == 1, "an invented draft must still be shown withdrawn"

            # And the same draft, punctuated differently, is still a duplicate.
            same = page.evaluate(
                """() => sayTheSameThing(
                     "I don't know \\u2014 that isn't covered by the documents I have",
                     "I don't know - that isn't covered by the documents I have.")"""
            )
            assert same is True
        finally:
            browser.close()


def test_citation_markers_do_not_reach_the_reader(declining_app) -> None:
    """renderAnswer itself, called with an answer shaped like the field's."""
    sync_playwright = pytest.importorskip(
        "playwright.sync_api", reason="playwright is not installed"
    ).sync_playwright

    with sync_playwright() as pw:
        browser, page = _page(pw, declining_app)
        try:
            rendered = page.evaluate(
                """() => {
                  const box = renderAnswer({
                    answer: 'Install the GlobalProtect client [vpn-access]. Requests are '
                          + 'approved by IT Operations [vpn-access], and clause [7] applies.',
                    tier: 'local', model: 'qwen3-4b', cost_usd: 0, grounded: true,
                    cached: false, support: 0.97, notes: [],
                    citations: [{ document_id: 'vpn-access', document_title: 'Remote Access',
                                  snippet: 'To connect from outside...', locator: 'chunk 1' }],
                  });
                  return box.querySelector('p').textContent;
                }"""
            )
            assert "[vpn-access]" not in rendered
            assert "Install the GlobalProtect client." in rendered
            assert "approved by IT Operations," in rendered
            # A bracket that is not a citation is the document's own text.
            assert "clause [7] applies." in rendered
        finally:
            browser.close()


def test_the_widget_still_routes_both_decisions_through_the_helpers() -> None:
    """The guard that runs where no browser is installed.

    This proves only that the calls are wired, not that they behave - the
    browser tests above are what measure the behaviour. It exists so that
    deleting the fix cannot pass CI silently.
    """
    source = WIDGET.read_text(encoding="utf-8")
    assert "withoutCitationMarkers(data.answer, data.citations)" in source
    assert "retractionsWorthKeeping(pending, event.response.answer)" in source


def test_a_retraction_of_nothing_is_not_shown(dying_app) -> None:
    """The second receipt owed for nothing, photographed live.

    The model announced itself and died before its first token, so the reader
    watched no text appear and none disappear. The card still carried
    "Withdrawn - this did not pass the grounding check", which describes
    neither what happened nor anything the reader saw. Why the answer is a
    refusal stays in the notes underneath, where it belongs.
    """
    sync_playwright = pytest.importorskip(
        "playwright.sync_api", reason="playwright is not installed"
    ).sync_playwright

    with sync_playwright() as pw:
        browser, page = _page(pw, dying_app)
        try:
            page.fill("#q", "when does the office close?")
            page.click("#send")
            page.wait_for_selector(".msg.a .badge", timeout=60000)
            card = page.inner_text(".msg.a")

            assert "Withdrawn" not in card, f"withdrew nothing, said so anyway:\n{card}"
            assert page.query_selector(".msg.a .retracted") is None
            # The reason survives, in the notes rather than as a banner.
            assert "never read" in card
        finally:
            browser.close()
