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

    update.forget_launch_check()
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


def test_starting_the_app_always_checks_for_real(tmp_path: Path) -> None:
    """The field complaint, pinned: a stamp written before a release hid it
    for the rest of the day, and restarting the app - the obvious remedy -
    changed nothing, because the stamp outlives the process. The first check
    of a launch ignores the stamp; the ones after it do not."""
    calls = {"n": 0}

    def counted() -> dict:
        calls["n"] += 1
        return release_json("v99.0.0")

    update.forget_launch_check()
    check_latest(state_dir=tmp_path, fetch=counted, now=1000.0)
    check_latest(state_dir=tmp_path, fetch=counted, now=1100.0)
    assert calls["n"] == 1, "the second check of a launch obeys the throttle"

    update.forget_launch_check()  # a restart
    result = check_latest(state_dir=tmp_path, fetch=counted, now=1200.0)
    assert calls["n"] == 2, "restarting the app must re-check, stamp or no stamp"
    assert result.update_available


# -- the install can always say what it is (field regression) -----------------
#
# Four rounds of "I still don't see the update button" turned on nobody being
# able to say which build was answering. Neither /healthz nor the UI reported
# a version, so every diagnosis started with a guess.


def test_healthz_reports_the_running_version(client) -> None:
    """Public, no sign-in, one URL. The question "what am I running?" must
    never again need a code reading to answer."""
    from openknowledge import __version__

    body = client.get("/healthz").json()
    assert body["version"] == __version__


def test_the_widget_page_is_never_cached(client) -> None:
    """The page carries the update UI, so a browser holding yesterday's copy
    would hide the control that replaces it."""
    assert client.get("/").headers["cache-control"] == "no-store"


def test_the_widget_shows_the_version_independently_of_the_update_check(client) -> None:
    """Read from /healthz, not /update/status: the version has to appear even
    when update checks are off, blocked, or failing - which is exactly when
    someone needs to know it."""
    page = client.get("/").text
    assert 'id="build-version"' in page
    marker = page.index('id="build-version"')
    assert "healthz" in page[marker:], "the version line must not depend on the update endpoint"


def test_the_helper_survives_an_apostrophe_in_the_account_name() -> None:
    """O'Brien could never update, and nothing in the app said why.

    Both paths in the handoff are built from the Windows account name -
    the installer sits under %LOCALAPPDATA% and the relaunch target is
    sys.executable under %LOCALAPPDATA%\\Programs - so an apostrophe in the
    account name landed inside a PowerShell single-quoted string and ended
    it early. The rest of the path parsed as bare tokens and the update
    silently did nothing.
    """
    installer = Path(r"C:\Users\O'Brien\AppData\Local\OpenKnowledge\Setup.exe")
    relaunch = Path(r"C:\Users\O'Brien\AppData\Local\Programs\OpenKnowledge\OpenKnowledge.exe")
    script = spawn_command(installer, relaunch)[-1]

    # PowerShell escapes a literal quote by doubling it, so every quote pairs.
    assert script.count("'") % 2 == 0, script
    assert "O''Brien" in script, "the apostrophe must be doubled, not dropped"
    assert r"C:\Users\O'Brien\AppData\Local\OpenKnowledge\Setup.exe" not in script, (
        "the raw path would end the string at the O"
    )


def test_only_a_plain_installer_filename_is_ever_offered() -> None:
    """The one place this product turns remote data into a running program.

    The asset name becomes a file we write into the state directory and then
    execute. It used to be accepted on its ends alone - starts with the
    prefix, ends with .exe - which says nothing about the middle, so a
    release naming its asset with a separator could have written outside the
    directory the state dir owns.
    """
    from openknowledge.desktop.update import _read_release

    def release(asset_name: str) -> CheckResult:
        return _read_release(
            "0.1.0",
            {
                "tag_name": "v9.9.9",
                "html_url": "https://example.invalid/r",
                "assets": [
                    {
                        "name": asset_name,
                        "browser_download_url": "https://example.invalid/a.exe",
                        "digest": "sha256:" + "a" * 64,
                    }
                ],
            },
        )

    assert release("OpenKnowledge-Setup-0.2.18.exe").update_available

    for hostile in (
        "OpenKnowledge-Setup-../../evil.exe",
        r"OpenKnowledge-Setup-..\..\evil.exe",
        "OpenKnowledge-Setup-x'; Start-Process calc; '.exe",
        "OpenKnowledge-Setup-/etc/passwd.exe",
    ):
        result = release(hostile)
        assert not result.update_available, hostile
        assert not result.installer_name, hostile
        assert "no digest-verified installer" in result.error, hostile


def test_the_installer_clears_the_version_it_is_replacing() -> None:
    """The guard for the defect that made every update invisible.

    The frozen app reads its version from bundled .dist-info, whose directory
    name carries that version, and Inno never removes files the new build no
    longer has. An upgrade left openknowledge-0.2.18.dist-info beside
    openknowledge-0.2.19.dist-info, importlib.metadata answered 0.2.18, and
    the updater offered the same release for ever.

    This asserts only that the removal is still declared; the CI
    windows-upgrade job is what measures that an install actually receives a
    build.
    """
    iss = Path(__file__).resolve().parent.parent / "packaging" / "windows" / "installer.iss"
    text = iss.read_text(encoding="utf-8")
    assert "[InstallDelete]" in text
    assert r"{app}\_internal\openknowledge-*.dist-info" in text


def test_the_helper_waits_for_the_app_to_let_go_of_its_own_files() -> None:
    """The race the update path had been winning by luck.

    spawn_installer is called from the launcher's `finally`, so the process
    calling it is still running - still holding its own executable and every
    DLL under _internal, none of which Windows lets an installer replace. It
    worked because PowerShell takes a moment to start and Inno takes longer
    to unpack itself than the process takes to exit. Losing that race leaves
    half a version on disk.
    """
    script = spawn_command(Path("C:/x/Setup.exe"), Path("C:/a/App.exe"), wait_for_pid=4242)[-1]

    assert "Wait-Process -Id 4242" in script
    assert script.index("Wait-Process") < script.index("Setup.exe"), (
        "the wait must come before the installer touches anything"
    )
    # Neither a process that has already gone nor a timeout is a reason to
    # abandon the update.
    assert "-ErrorAction SilentlyContinue" in script
    assert "-Timeout" in script


def test_the_pid_is_a_number_and_nothing_else() -> None:
    """It reaches a shell, so it is an int before it is a string."""
    script = spawn_command(Path("C:/x/S.exe"), Path("C:/a/A.exe"), wait_for_pid=True)[-1]
    assert "Wait-Process -Id 1 " in script


# -- which executable a relaunch produces ------------------------------------


def test_a_relaunch_after_updating_starts_the_app_not_the_cli(tmp_path: Path) -> None:
    """The bundle ships two executables and only one of them is the app.

    Started from the Start-menu shortcut, sys.executable is already the
    windowed entry and nothing needs deciding. Started as `openknowledge
    desktop` - which the installer's PATH option and every terminal user
    make reachable - sys.executable is the CLI, and the CLI with no
    subcommand prints a usage error and exits. Relaunching that closes the
    app for an update and never brings it back.
    """
    cli = tmp_path / "openknowledge.exe"
    app = tmp_path / update.WINDOWED_EXE
    cli.write_bytes(b"cli")
    app.write_bytes(b"app")

    assert update.relaunch_target(cli) == app, "the CLI must hand off to the app"
    assert update.relaunch_target(app) == app, "the app relaunches itself"


def test_a_layout_without_the_windowed_build_is_left_alone(tmp_path: Path) -> None:
    """A dev checkout, or any layout this does not recognise, gets what it
    gave rather than a guess at a file that is not there."""
    lonely = tmp_path / "openknowledge.exe"
    lonely.write_bytes(b"cli")
    assert update.relaunch_target(lonely) == lonely


def test_the_windowed_name_is_the_one_the_bundle_builds() -> None:
    """Pinned to the spec: a rename there without one here would ship an
    update that closes the app and reopens nothing."""
    spec = Path("packaging/pyinstaller/openknowledge.spec").read_text(encoding="utf-8")
    assert f'APP_NAME = "{update.WINDOWED_EXE.removesuffix(".exe")}"' in spec
