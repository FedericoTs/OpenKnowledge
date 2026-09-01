"""The sign-in flow in a real browser, riding real redirects and cookies.

TestClient proves the HTTP contract; only a browser proves the part people
actually touch - the redirect chain out to the provider and back, the cookie
surviving it, the sidebar naming who signed in, sign-out ending the session.
The provider is the same loopback fake the rest of the auth suite uses, in
auto-approve mode, which is exactly how corporate SSO feels: you never see a
login page, you are simply back and known.

Skips when Playwright or a browser is missing, so `make check` on a bare
checkout still passes.
"""

from __future__ import annotations

import shutil
import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from fake_idp import FakeIdp
from openknowledge.api.app import create_app
from openknowledge.config import Settings

CLIENT_ID = "ok-browser-test"


def _chromium() -> str | None:
    for candidate in Path("/opt/pw-browsers").glob("chromium-*/chrome-linux/chrome"):
        return str(candidate)
    return shutil.which("chromium") or shutil.which("google-chrome")


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture
def stack(tmp_path: Path) -> Iterator[tuple[str, FakeIdp]]:
    """A signed-in-capable server on loopback, backed by the fake IdP."""
    uvicorn = pytest.importorskip("uvicorn")
    idp = FakeIdp()
    idp.auto_identity = {"subject": "priya", "name": "Priya Tester", "groups": ("g-hr",)}
    documents = tmp_path / "documents"
    documents.mkdir()
    (documents / "handbook.md").write_text(
        "# Handbook\n\nThe office closes at 18:00.\n", encoding="utf-8"
    )
    settings = Settings(
        data_dir=str(tmp_path / "data"),
        documents_dir=str(documents),
        auth_mode="oidc",
        oidc_issuer=idp.issuer,
        oidc_client_id=CLIENT_ID,
        oidc_client_secret="s3cret",
        oidc_admin_group="g-admins",
        oidc_curator_group="g-curators",
        upload_enabled=True,
        local_enabled=False,
        embedding_enabled=False,
        _env_file=None,  # type: ignore[call-arg]
    )
    port = _free_port()
    server = uvicorn.Server(
        uvicorn.Config(create_app(settings), host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 15
    while not server.started:
        if time.time() > deadline:
            pytest.fail("uvicorn did not start")
        time.sleep(0.05)
    try:
        yield f"http://127.0.0.1:{port}", idp
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        idp.close()


def test_a_browser_signs_in_uses_the_app_and_signs_out(stack) -> None:
    sync_playwright = pytest.importorskip(
        "playwright.sync_api", reason="playwright is not installed"
    ).sync_playwright
    executable = _chromium()
    if executable is None:
        pytest.skip("no chromium available")
    base, idp = stack

    with sync_playwright() as pw:
        browser = pw.chromium.launch(executable_path=executable, args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1280, "height": 860})
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        try:
            # Out through the provider and back in one navigation - the
            # whole chain: / -> /auth/login -> authorize -> callback -> /.
            page.goto(base + "/")
            page.wait_for_selector("#who b", timeout=15000)
            assert page.url.rstrip("/") == base, "did not land back on the app"
            assert page.inner_text("#who b") == "Priya Tester"

            # The session actually works: the sidebar loads the corpus.
            page.wait_for_selector("#doc-list li", timeout=15000)
            assert "handbook.md" in page.inner_text("#doc-list")

            # Sign out. With auto-approve off, the login redirect dead-ends
            # at the provider - proof the session is gone and a new one
            # cannot be minted silently.
            idp.auto_identity = None
            page.click("#who button")
            page.wait_for_url("**/authorize**", timeout=15000)
            assert errors == [], f"page errors: {errors}"
        finally:
            browser.close()


def test_the_manage_page_shows_a_curator_only_what_they_may_change(stack) -> None:
    """The role split lives half in JavaScript: nothing in Python notices if
    the governance panels stop being hidden, and a curator would meet three
    separate 403s instead of one sentence.

    It checks both roles against the same page, because the failure that
    matters is not "the curator saw an error" - it is "the curator saw the
    access rules", which only shows up next to what an admin sees.
    """
    sync_playwright = pytest.importorskip(
        "playwright.sync_api", reason="playwright is not installed"
    ).sync_playwright
    executable = _chromium()
    if executable is None:
        pytest.skip("no chromium available")
    base, idp = stack

    def manage_page_for(browser, groups: tuple[str, ...]) -> dict[str, str]:
        idp.auto_identity = {"subject": "p", "name": "Pat Curator", "groups": groups}
        context = browser.new_context()
        page = context.new_page()
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(base + "/manage")
        page.wait_for_selector("#who-line b", timeout=15000)
        page.wait_for_function(
            "document.getElementById('status').textContent.includes('via your sign-in group')",
            timeout=15000,
        )
        seen = {
            "status": page.inner_text("#status"),
            "access": page.inner_text("#access"),
            "settings": page.inner_text("#settings"),
            "log": page.inner_text("#admin-log"),
        }
        assert errors == [], f"page errors: {errors}"
        context.close()
        return seen

    with sync_playwright() as pw:
        browser = pw.chromium.launch(executable_path=executable, args=["--no-sandbox"])
        try:
            admin = manage_page_for(browser, ("g-admins",))
            curator = manage_page_for(browser, ("g-curators",))
        finally:
            browser.close()

    assert "Admin" in admin["status"]
    assert "retrieval_k" in admin["settings"], "an admin sees the settings themselves"
    assert "Unlock to load" not in admin["log"], "and the log loaded"

    assert "curator" in curator["status"].lower()
    for panel in ("access", "settings", "log"):
        assert "an administrator does" in curator[panel], f"{panel} was not withheld"
    assert "retrieval_k" not in curator["settings"]
