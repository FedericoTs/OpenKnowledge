"""One-click verified updates: offered, never imposed, and never unverified.

The properties under test are the safety ones. A newer release without a
digest-verified installer is not an update this code will offer; a download
that hashes wrong is deleted and refused loudly; a server that is not the
desktop app refuses to update itself; and the check fails soft, because an
offline install is a supported way to run this product.
"""

from __future__ import annotations

import hashlib
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from openknowledge.api.app import create_app
from openknowledge.config import Settings
from openknowledge.desktop import update
from openknowledge.desktop.update import (
    HANDOFF,
    CheckResult,
    UpdateError,
    check_latest,
    current_version,
    download_and_verify,
    spawn_command,
)

INSTALLER_BYTES = b"MZ this is definitely a windows installer\n" * 64
INSTALLER_SHA = hashlib.sha256(INSTALLER_BYTES).hexdigest()


def release_json(tag: str, *, digest: str | None = INSTALLER_SHA) -> dict:
    asset = {
        "name": f"OpenKnowledge-Setup-{tag.lstrip('v')}.exe",
        "browser_download_url": "http://unused.invalid/installer",
        "digest": f"sha256:{digest}" if digest else "",
    }
    return {"tag_name": tag, "html_url": f"https://example.invalid/{tag}", "assets": [asset]}


# -- the check ---------------------------------------------------------------


def test_a_newer_release_is_offered(tmp_path: Path) -> None:
    result = check_latest(state_dir=tmp_path, fetch=lambda: release_json("v99.0.0"))
    assert result.update_available
    assert result.latest == "99.0.0"
    assert result.sha256 == INSTALLER_SHA


def test_the_current_release_is_not_an_update(tmp_path: Path) -> None:
    result = check_latest(state_dir=tmp_path, fetch=lambda: release_json(f"v{current_version()}"))
    assert not result.update_available


def test_a_release_without_a_digest_is_not_offered(tmp_path: Path) -> None:
    """Newer but unverifiable means "not an update", said in words - never a
    download this code cannot prove."""
    result = check_latest(state_dir=tmp_path, fetch=lambda: release_json("v99.0.0", digest=None))
    assert not result.update_available
    assert "no digest-verified installer" in result.error


def test_an_unparseable_tag_fails_soft(tmp_path: Path) -> None:
    result = check_latest(state_dir=tmp_path, fetch=lambda: release_json("vNext-final-FINAL"))
    assert not result.update_available
    assert "cannot compare versions" in result.error


def test_a_dead_network_is_a_note_not_a_crash(tmp_path: Path) -> None:
    def boom() -> dict:
        raise OSError("no route to host")

    result = check_latest(state_dir=tmp_path, fetch=boom)
    assert not result.update_available
    assert "update check failed" in result.error


def test_the_check_runs_at_most_once_a_day(tmp_path: Path) -> None:
    calls = {"n": 0}

    def counted() -> dict:
        calls["n"] += 1
        return release_json("v99.0.0")

    first = check_latest(state_dir=tmp_path, fetch=counted, now=1000.0)
    again = check_latest(state_dir=tmp_path, fetch=counted, now=2000.0)
    later = check_latest(state_dir=tmp_path, fetch=counted, now=1000.0 + 25 * 3600)
    forced = check_latest(state_dir=tmp_path, fetch=counted, now=2000.0, force=True)
    assert calls["n"] == 3  # first, past-the-day, forced - never the cached one
    assert first.update_available and again.update_available
    assert later.update_available and forced.update_available


# -- the download ------------------------------------------------------------


class _Serving:
    def __init__(self, body: bytes) -> None:
        serving = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: object) -> None:  # noqa: D102
                pass

            def do_GET(self) -> None:  # noqa: N802 - http.server API
                self.send_response(200)
                self.send_header("Content-Length", str(len(serving.body)))
                self.end_headers()
                self.wfile.write(serving.body)

        self.body = body
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}/installer"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def _result(url: str, sha256: str = INSTALLER_SHA) -> CheckResult:
    return CheckResult(
        current="0.0.1",
        latest="99.0.0",
        update_available=True,
        installer_name="OpenKnowledge-Setup-99.0.0.exe",
        installer_url=url,
        sha256=sha256,
        release_url="https://example.invalid/v99.0.0",
    )


def test_a_verified_download_lands_and_matches(tmp_path: Path) -> None:
    serving = _Serving(INSTALLER_BYTES)
    try:
        dest = download_and_verify(_result(serving.url), dest_dir=tmp_path)
        assert dest.read_bytes() == INSTALLER_BYTES
    finally:
        serving.close()


def test_a_tampered_download_is_deleted_and_refused(tmp_path: Path) -> None:
    serving = _Serving(INSTALLER_BYTES + b"one extra byte")
    try:
        with pytest.raises(UpdateError, match="does not match the release's SHA-256"):
            download_and_verify(_result(serving.url), dest_dir=tmp_path)
        assert not list(tmp_path.iterdir()), "nothing unverified may remain on disk"
    finally:
        serving.close()


# -- the endpoints -----------------------------------------------------------


@pytest.fixture
def client(tmp_path: Path):
    settings = Settings(
        data_dir=str(tmp_path / "data"),
        documents_dir=str(tmp_path / "documents"),
        local_enabled=False,
        embedding_enabled=False,
        _env_file=None,  # type: ignore[call-arg]
    )
    with TestClient(create_app(settings)) as c:
        yield c


def test_a_server_install_reports_it_cannot_apply(client, monkeypatch) -> None:
    monkeypatch.setattr(
        update,
        "check_latest",
        lambda **kw: _result("http://unused.invalid/installer"),
    )
    status = client.get("/update/status").json()
    assert status["update_available"] is True
    assert status["can_apply"] is False

    refused = client.post("/update/apply")
    assert refused.status_code == 409
    assert "operator" in refused.json()["detail"]


def test_disabled_checks_phone_nowhere(client, monkeypatch) -> None:
    client.app.state.settings.update_check = False

    def explode(**kw: object) -> CheckResult:
        raise AssertionError("the check ran despite OK_UPDATE_CHECK=false")

    monkeypatch.setattr(update, "check_latest", explode)
    status = client.get("/update/status").json()
    assert status["disabled"] is True and status["update_available"] is False
    assert client.post("/update/apply").status_code == 409


def test_the_desktop_apply_downloads_verifies_and_hands_off(
    client, tmp_path: Path, monkeypatch
) -> None:
    serving = _Serving(INSTALLER_BYTES)
    quit_event = threading.Event()
    HANDOFF.bind(quit_event)
    try:
        monkeypatch.setattr(update, "check_latest", lambda **kw: _result(serving.url))
        response = client.post("/update/apply")
        assert response.status_code == 200, response.text
        assert response.json()["applying"] == "99.0.0"

        installer = HANDOFF.installer()
        assert installer is not None and installer.read_bytes() == INSTALLER_BYTES
        assert quit_event.wait(timeout=5), "the clean-shutdown signal must follow the response"
    finally:
        serving.close()
        HANDOFF._quit = None  # noqa: SLF001 - reset the module singleton for other tests
        HANDOFF._installer = None  # noqa: SLF001


def test_apply_requires_an_admin_when_sign_in_is_on(client) -> None:
    client.app.state.settings.auth_mode = "oidc"
    try:
        assert client.post("/update/apply").status_code == 403
    finally:
        client.app.state.settings.auth_mode = "off"


# -- the helper --------------------------------------------------------------


def test_the_helper_installs_silently_then_relaunches() -> None:
    cmd = spawn_command(Path("C:/x/Setup.exe"), Path("C:/apps/OpenKnowledge.exe"))
    script = cmd[-1]
    assert "/VERYSILENT" in script and "-Wait" in script
    assert script.index("Setup.exe") < script.index("OpenKnowledge.exe"), (
        "install must finish before the relaunch"
    )


def test_refresh_bypasses_the_daily_throttle(client, tmp_path: Path, monkeypatch) -> None:
    """The explicit "check now" a person is entitled to: without it, a
    release published hours after the last check hides behind the throttle
    for up to a day."""
    calls = {"n": 0}

    def counted(*, state_dir: Path, force: bool = False) -> CheckResult:
        calls["n"] += 1
        calls["force"] = force
        return CheckResult(current="0.0.1")

    monkeypatch.setattr(update, "check_latest", counted)
    client.get("/update/status")
    assert calls["force"] is False
    client.get("/update/status?refresh=1")
    assert calls["force"] is True
