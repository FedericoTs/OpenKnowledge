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

import json
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


def _needs_a_browser() -> None:
    """Skip before doing any work, not after.

    These fixtures used to start a real uvicorn server and only then let the
    test body decide it had no browser to drive - so every one of them booted
    an app in CI, where no browser is installed, purely to skip. On a loaded
    Windows runner one of those servers missed its start deadline and failed
    the build. Deciding here costs nothing and starts nothing.
    """
    pytest.importorskip("playwright.sync_api", reason="playwright is not installed")
    if _chromium() is None:
        pytest.skip("no chromium available")


def _serve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake, admin_token: str = ""
) -> Iterator[str]:
    """A real server on loopback whose only fake part is the model."""
    _needs_a_browser()
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
        admin_token=admin_token,
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
    deadline = time.time() + 60
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


@pytest.fixture
def managed_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """The same server with an admin token, so /manage can be unlocked."""
    yield from _serve(
        tmp_path, monkeypatch, StreamingFakeProvider(replies=[REFUSAL_TEXT] * 6), "t0ken"
    )


def _page(pw, base: str):
    # The fixture already established there is one - see _needs_a_browser.
    browser = pw.chromium.launch(executable_path=_chromium(), args=["--no-sandbox"])
    page = browser.new_page(viewport={"width": 1280, "height": 900})
    page.goto(base + "/")
    return browser, page


def test_a_refusal_is_printed_once(declining_app) -> None:
    """The photographed defect, reproduced and then absent.

    Before this fix the card carried the refusal twice with a "Withdrawn"
    banner between them. Nothing vanished from the reader's view - the draft
    and the answer are the same sentence - so there is nothing to account for.
    """
    sync_playwright = pytest.importorskip("playwright.sync_api").sync_playwright

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
    sync_playwright = pytest.importorskip("playwright.sync_api").sync_playwright

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
    sync_playwright = pytest.importorskip("playwright.sync_api").sync_playwright

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


def test_the_page_says_which_build_it_is_before_anyone_touches_it(declining_app) -> None:
    """The third field defect of this kind, and the one that hid a fix.

    ``refreshUpdateChip()`` and the fetch that fills the version label had
    drifted inside ``removeDocument``'s try block, so they ran only after
    somebody deleted a document. On a fresh page the footer was blank and the
    update chip was empty - which is the second reason "I still don't see the
    update button" was true, the first being installs that never received a
    new version at all. Both looked identical from the outside.
    """
    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        browser, page = _page(pw, declining_app)
        try:
            page.wait_for_function(
                "document.getElementById('build-version').textContent.trim() !== ''",
                timeout=15000,
            )
            version = page.inner_text("#build-version")
            chip = page.inner_text("#update-chip")
        finally:
            browser.close()

    assert version.startswith("OpenKnowledge v"), version
    assert chip.strip(), "the update chip never ran"


def _blanked(source: str) -> str:
    """The source with quoted text and comments replaced by spaces.

    Length-preserving, so an offset found in the original still points at
    the same place here. That is the whole trick: find the statement in the
    real text, count braces in the blanked text.
    """
    out = list(source)
    i, n, quote = 0, len(source), ""
    while i < n:
        c = source[i]
        if quote:
            out[i] = " "
            if c == "\\" and i + 1 < n:
                out[i + 1] = " "
                i += 2
                continue
            if c == quote:
                quote = ""
        elif c in "'\"`":
            quote = c
            out[i] = " "
        elif source[i : i + 2] == "//":
            while i < n and source[i] != "\n":
                out[i] = " "
                i += 1
            continue
        elif source[i : i + 2] == "/*":
            stop = source.find("*/", i)
            stop = n if stop < 0 else stop + 2
            for j in range(i, stop):
                out[j] = " "
            i = stop
            continue
        i += 1
    return "".join(out)


def _nesting_of(source: str, needle: str) -> list[int]:
    """How deeply nested each occurrence of ``needle`` is: unclosed braces
    between the start of the file and that occurrence.

    Written for one question and no more: the widget's script is flat, and
    the defect this guards against is a statement that fell inside a
    function while still starting at column zero. A call can legitimately
    appear at several depths - what matters is whether one of them runs at
    load, which is depth zero.
    """
    blanked = _blanked(source)
    depths, at = [], source.find(needle)
    assert at >= 0, f"{needle!r} is not in the widget at all"
    while at >= 0:
        depths.append(blanked.count("{", 0, at) - blanked.count("}", 0, at))
        at = source.find(needle, at + 1)
    return depths


def test_a_reader_can_say_the_answer_is_wrong(managed_app) -> None:
    """The control, the form and the confirmation, clicked in Chromium.

    The server contract has its own tests. This is the half only a browser
    proves: that the button is on the card, that it knows which question it
    is reporting, and that a reader is told their name was not sent - the
    reason to use it rather than telling a colleague.
    """
    import httpx
    from playwright.sync_api import sync_playwright

    httpx.post(
        managed_app + "/admin/pins",
        json={"question": "how much parental leave?", "answer": "16 weeks.", "cite": []},
        headers={"authorization": "Bearer t0ken"},
        timeout=20,
    )
    with sync_playwright() as pw:
        browser, page = _page(pw, managed_app)
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        try:
            page.fill("#q", "how much parental leave?")
            page.press("#q", "Enter")
            page.wait_for_selector(".msg.a .meta", timeout=20000)
            page.click("text=This is wrong")
            page.fill(".report input", "It went to 20 weeks in April.")
            page.click(".report button:not(.ghost)")
            page.wait_for_selector(".report.done", timeout=15000)
            confirmation = page.inner_text(".report.done")
            labels = page.eval_on_selector_all(".msg.a .again", "b => b.map(x => x.textContent)")
        finally:
            browser.close()

    assert "Nobody is named" in confirmation
    assert "Reported" in labels, "the button does not say it was sent"
    assert errors == [], f"page errors: {errors}"

    seen = httpx.get(
        managed_app + "/admin/reports",
        headers={"authorization": "Bearer t0ken"},
        timeout=20,
    ).json()["reports"]
    assert [r["answer"] for r in seen] == ["16 weeks."]
    assert seen[0]["notes"] == ["It went to 20 weeks in April."]


def test_the_boot_calls_are_at_the_top_level() -> None:
    """The guard that runs where no browser is installed.

    The defect above was scope, not logic: two statements that fell inside
    ``removeDocument``'s try block while still starting at column zero. A
    check for "appears in the source", or for the indentation, passes against
    the bug - both were true of it. Nesting depth is the thing that was
    wrong, so nesting depth is what this asserts.
    """
    source = WIDGET.read_text(encoding="utf-8")
    assert 0 in _nesting_of(source, "refreshUpdateChip();"), "nothing boots the update chip at load"
    assert 0 in _nesting_of(source, "fetch('healthz')"), "nothing fetches the version at load"


def test_a_retraction_of_nothing_is_not_shown(dying_app) -> None:
    """The second receipt owed for nothing, photographed live.

    The model announced itself and died before its first token, so the reader
    watched no text appear and none disappear. The card still carried
    "Withdrawn - this did not pass the grounding check", which describes
    neither what happened nor anything the reader saw. Why the answer is a
    refusal stays in the notes underneath, where it belongs.
    """
    sync_playwright = pytest.importorskip("playwright.sync_api").sync_playwright

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


def test_asking_again_reports_identical(declining_app) -> None:
    """The claim on the front of the product, demonstrated rather than asserted.

    "Asked twice, it answers identically" was a sentence on the landing page
    and nothing else. The card now repeats the identical request and compares
    the two answers character for character, in front of the person who asked.
    """
    sync_playwright = pytest.importorskip("playwright.sync_api").sync_playwright

    with sync_playwright() as pw:
        browser, page = _page(pw, declining_app)
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        try:
            page.fill("#q", "when does the office close?")
            page.click("#send")
            page.wait_for_selector(".msg.a .again", timeout=60000)
            page.click(".msg.a .again")
            page.wait_for_selector(".msg.a .verdict", timeout=60000)

            verdict = page.inner_text(".msg.a .verdict")
            assert "identical, character for character" in verdict, verdict
            assert page.query_selector(".msg.a .verdict.same") is not None
            # The second answer cost nothing, and the card says which it was.
            assert "$0.00" in verdict, verdict
            assert errors == [], f"page errors: {errors}"
        finally:
            browser.close()


def test_asking_again_shows_a_changed_answer_in_full(declining_app) -> None:
    """The half that makes the other half worth anything.

    A product whose argument is that it does not paper over what it cannot do
    has no business hiding a difference here. The response is intercepted and
    altered so the comparison genuinely fails, and the card must say so and
    show the answer it got the second time.
    """
    sync_playwright = pytest.importorskip("playwright.sync_api").sync_playwright

    with sync_playwright() as pw:
        browser, page = _page(pw, declining_app)
        try:
            page.fill("#q", "when does the office close?")
            page.click("#send")
            page.wait_for_selector(".msg.a .again", timeout=60000)

            # Only the recheck is tampered with: the first answer stands.
            def tamper(route):
                response = route.fetch()
                body = response.json()
                body["answer"] = "The office closes at 17:00 on Fridays."
                route.fulfill(json=body)

            page.route("**/chat", tamper)
            page.click(".msg.a .again")
            page.wait_for_selector(".msg.a .verdict", timeout=60000)

            card = page.inner_text(".msg.a")
            assert "the answer changed" in card
            assert page.query_selector(".msg.a .verdict.diff") is not None
            assert "The office closes at 17:00 on Fridays." in card, (
                "a changed answer must be shown, not summarised away"
            )
        finally:
            browser.close()


def test_a_recheck_is_recorded_as_a_recheck(declining_app) -> None:
    """The repeat is a real question and the ledger says what it was for."""
    sync_playwright = pytest.importorskip("playwright.sync_api").sync_playwright

    with sync_playwright() as pw:
        browser, page = _page(pw, declining_app)
        seen: list[str] = []
        try:
            page.on(
                "request",
                lambda r: (
                    seen.append(r.post_data or "")
                    if r.url.endswith("/chat") and r.method == "POST"
                    else None
                ),
            )
            page.fill("#q", "when does the office close?")
            page.click("#send")
            page.wait_for_selector(".msg.a .again", timeout=60000)
            page.click(".msg.a .again")
            page.wait_for_selector(".msg.a .verdict", timeout=60000)

            assert len(seen) == 1, "the recheck should be exactly one extra ask"
            body = json.loads(seen[0])
            assert body["channel"] == "recheck"
            assert body["question"] == "when does the office close?"
        finally:
            browser.close()


def test_the_receipts_line_says_only_what_is_true(declining_app) -> None:
    """Zero is zero, and "no model call" is a claim that has to be earned.

    A refusal used to read "none · $0.00000": five decimal places of nothing,
    and a model name that is not one. It is also not "no model call" - the
    local rung generated a draft that the grounding gate threw out, and
    claiming otherwise would misdescribe the one mechanism this product sells.
    """
    sync_playwright = pytest.importorskip("playwright.sync_api").sync_playwright

    with sync_playwright() as pw:
        browser, page = _page(pw, declining_app)
        try:
            lines = page.evaluate(
                """() => ({
                  cached: costLine({ model: 'qwen3-4b', cost_usd: 0 }, true),
                  refused: costLine({ model: 'none', cost_usd: 0 }, false),
                  cheap: costLine({ model: 'gpt-4o-mini', cost_usd: 0.00146 }, false),
                  dear: costLine({ model: 'gpt-4o', cost_usd: 0.0325 }, false),
                })"""
            )
            assert lines["cached"] == "no model call · $0.00"
            assert lines["refused"] == "no model answered · $0.00"
            assert lines["cheap"] == "gpt-4o-mini · $0.00146"
            assert lines["dear"] == "gpt-4o · $0.03"
            assert "0.00000" not in "".join(lines.values())
        finally:
            browser.close()


def test_the_cost_panel_shows_the_numbers_not_the_json(managed_app) -> None:
    """The money, promoted out of a JSON dump.

    /manage already fetched the cost report and printed
    ``JSON.stringify(report, null, 2)`` into a box at the bottom of the page -
    the machinery rather than the number. The figures are now the first thing
    an owner sees after unlocking.
    """
    sync_playwright = pytest.importorskip("playwright.sync_api").sync_playwright

    with sync_playwright() as pw:
        browser, page = _page(pw, managed_app)
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        try:
            # Two questions, so the ledger has something to report.
            for _ in range(2):
                page.fill("#q", "when does the office close?")
                page.click("#send")
                page.wait_for_timeout(300)
            page.wait_for_selector(".msg.a .badge", timeout=60000)

            page.goto(managed_app + "/manage")
            page.fill("#token-input", "t0ken")
            page.click("#unlock")
            page.wait_for_selector("#costs .figure b", timeout=30000)

            panel = page.inner_text("#costs")
            assert "questions asked" in panel
            assert "spent, in total" in panel
            assert "Last 30 days" in panel
            assert "{" not in panel, f"the report is still being dumped as JSON:\n{panel}"
            assert '"by_tier"' not in panel

            figures = page.eval_on_selector_all(
                "#costs .figure b", "els => els.map(e => e.textContent)"
            )
            assert figures[0] == "2", f"two questions were asked, panel says {figures}"
            assert figures[1] == "$0.00"
            assert errors == [], f"page errors: {errors}"
        finally:
            browser.close()


def test_a_citation_names_where_a_reader_can_look(declining_app) -> None:
    """ "chunk 1" is a position in our index, not in anybody's document.

    It stays on the wire, because the grounding gate resolves a model's
    "(chunk 4)" against it. It stops being shown, because the reader cannot
    open chunk 1 of anything.
    """
    sync_playwright = pytest.importorskip("playwright.sync_api").sync_playwright

    with sync_playwright() as pw:
        browser, page = _page(pw, declining_app)
        try:
            rendered = page.evaluate(
                """() => {
                  const box = renderAnswer({
                    answer: 'Meals are reimbursed up to EUR 45 per day.',
                    tier: 'local', model: 'qwen3-4b', cost_usd: 0, grounded: true,
                    cached: false, support: 0.9, notes: [],
                    citations: [
                      { document_id: 'expenses', document_title: 'Expenses Policy',
                        snippet: 'Meals are reimbursed up to EUR 45 per day.',
                        section: 'Expenses Policy > Meals and subsistence',
                        locator: 'chunk 2' },
                      { document_id: 'vpn', document_title: 'Remote Access and VPN',
                        snippet: 'Install the client.',
                        section: 'Remote Access and VPN', locator: 'chunk 1' },
                      { document_id: 'handbook', document_title: 'Handbook',
                        snippet: 'Anything.', section: 'Leave', locator: 'page 3' },
                      { document_id: 'plain', document_title: 'Notes',
                        snippet: 'Anything.', section: null, locator: 'chunk 7' },
                    ],
                  });
                  return [...box.querySelectorAll('.cite b')].map(e => e.textContent);
                }"""
            )
            assert rendered[0] == "Expenses Policy · Meals and subsistence"
            # A section that only repeats the title adds nothing.
            assert rendered[1] == "Remote Access and VPN"
            # A locator a reader can act on is kept, alongside the section.
            assert rendered[2] == "Handbook · Leave · page 3"
            # No structure at all: the quoted passage is the only locator there is.
            assert rendered[3] == "Notes"
            assert not any("chunk" in r for r in rendered), rendered
        finally:
            browser.close()


def test_a_pin_can_be_read_edited_and_taken_back_from_the_page(managed_app) -> None:
    """The loop the API had and the page did not: see it, change it, take it back.

    The server contract has its own tests. What only a browser proves is that
    the panel shows the question a curator would look for, with its source and
    its author; that Save re-pins under the same question with the same
    citation rather than minting a second pin or dropping the source; and that
    Unpin asks first, deletes only when told to, and leaves the panel saying so
    - after which the store is empty and the admin log names the deletion.
    """
    import httpx
    from playwright.sync_api import sync_playwright

    auth = {"authorization": "Bearer t0ken"}
    httpx.post(
        managed_app + "/admin/pins",
        json={
            "question": "when does the office close?",
            "answer": "At 18:00.",
            "cite": ["handbook"],
            "author": "cli",
        },
        headers=auth,
        timeout=20,
    )

    def pins() -> list[dict]:
        return httpx.get(managed_app + "/admin/pins", headers=auth, timeout=20).json()

    with sync_playwright() as pw:
        browser, page = _page(pw, managed_app)
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        try:
            page.goto(managed_app + "/manage")
            page.fill("#token-input", "t0ken")
            page.click("#unlock")
            page.wait_for_selector('body[data-ready="admin"]', timeout=30000)

            summary = page.inner_text("#pins details summary")
            assert "when does the office close" in summary, summary
            assert "cites handbook" in summary, summary
            assert "by cli" in summary, summary

            # The controls sit inside the folded item, as in the other panels.
            page.click("#pins details summary")

            # Declining the confirmation must delete nothing.
            page.once("dialog", lambda d: d.dismiss())
            page.click("#pins button:has-text('Unpin')")

            # Editing: the round trip is proved by the author changing to the
            # page's own name once the panel re-renders from the server.
            page.fill("#pins textarea", "At 18:00, and at 17:00 on Fridays.")
            page.click("#pins button:has-text('Save')")
            page.wait_for_selector("#pins summary:has-text('by manage-page')", timeout=15000)
            (pin,) = pins()
            assert pin["answer"] == "At 18:00, and at 17:00 on Fridays."
            assert pin["cited"] == ["handbook"], "the edit lost the citation"

            page.click("#pins details summary")
            page.once("dialog", lambda d: d.accept())
            page.click("#pins button:has-text('Unpin')")
            page.wait_for_selector("#pins .empty", timeout=15000)
            assert "Nothing is pinned" in page.inner_text("#pins .empty")
        finally:
            browser.close()

    assert errors == [], f"page errors: {errors}"
    assert pins() == []
    log = httpx.get(managed_app + "/admin/log", headers=auth, timeout=20).json()
    deleted = [e for e in log["entries"] if e["action"] == "pin.delete"]
    assert [e["target"] for e in deleted] == ["when does the office close"]


def test_the_most_asked_list_can_pin_from_where_it_counts(managed_app) -> None:
    """The ledger read as a shortlist, with the fix one click away.

    Two people ask the same thing and the model declines both times. The
    panel has to show that question with its count and how it went, and
    pinning it from there has to mark it pinned in this list and put it in
    the pinned list - one store, so one page cannot say two things.
    """
    import httpx
    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        browser, page = _page(pw, managed_app)
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        try:
            # Asked twice, each answer waited for, so both reach the ledger
            # before the page that counts them is opened.
            for asked in (1, 2):
                page.fill("#q", "when does the office close?")
                page.click("#send")
                page.wait_for_function(
                    f"document.querySelectorAll('.msg.a .badge').length >= {asked}", timeout=60000
                )

            page.goto(managed_app + "/manage")
            page.fill("#token-input", "t0ken")
            page.click("#unlock")
            page.wait_for_selector('body[data-ready="admin"]', timeout=30000)

            summary = page.inner_text("#demand details summary")
            assert "2 times" in summary, summary
            assert "when does the office close" in summary, summary
            assert "pinned" not in summary, summary
            page.click("#demand details summary")
            assert "2 refused" in page.inner_text("#demand details .body")

            page.fill("#demand textarea", "At 18:00.")
            page.click("#demand button:has-text('Pin this answer')")
            page.wait_for_selector("#demand summary:has-text('pinned')", timeout=15000)
            page.wait_for_selector(
                "#pins summary:has-text('when does the office close')", timeout=15000
            )
        finally:
            browser.close()

    assert errors == [], f"page errors: {errors}"
    (pin,) = httpx.get(
        managed_app + "/admin/pins", headers={"authorization": "Bearer t0ken"}, timeout=20
    ).json()
    assert (pin["question"], pin["answer"]) == ("when does the office close", "At 18:00.")
