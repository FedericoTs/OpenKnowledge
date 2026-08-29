"""One-click verified updates for the desktop app.

Every field iteration this week cost a manual loop: find the release, download
the installer, check its hash, run the wizard. This closes the loop from inside
the app - but deliberately as *notify and one click*, not a silent background
updater. The binaries are not yet code-signed, and enterprise IT rightly bans
software that changes itself unannounced; until signing lands, a pinned repo,
an explicit click, and a hash verified against the release's own digest is the
defensible posture.

What one click does: download the new installer to the state directory, verify
its SHA-256 against the digest GitHub records for that exact asset, hand the
path to the launcher, shut everything down cleanly, run the installer silently,
and relaunch. The 2.6 GB of models never re-download - they live outside the
install directory and every release keeps using them.

The check itself is an outbound call to api.github.com and is documented as
one: it runs at most once a day, sends nothing but the request itself, and
``OK_UPDATE_CHECK=false`` turns it off entirely for privacy-strict or
IT-managed fleets. Honesty about the trust model: verifying the digest defeats
a corrupted download or a tampering mirror, not a compromised publisher
account - that is what code signing will be for, and it stays on the roadmap.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

#: The one place updates may come from. Never configurable at runtime: an env
#: var that redirects "where do you fetch executables from" is a gift to
#: anyone who can edit an env file.
REPO = "FedericoTs/OpenKnowledge"

_API_LATEST = f"https://api.github.com/repos/{REPO}/releases/latest"

#: At most one check per this many seconds, remembered across restarts.
CHECK_INTERVAL_SECONDS = 24 * 3600


class UpdateError(RuntimeError):
    """A failure worth telling the person about, in a full sentence."""


def current_version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("openknowledge")
    except PackageNotFoundError:  # pragma: no cover - source checkout
        from openknowledge import __version__

        return __version__


def _parse(tag: str) -> tuple[int, ...] | None:
    cleaned = tag.strip().lstrip("vV")
    parts = cleaned.split(".")
    try:
        return tuple(int(p) for p in parts)
    except ValueError:
        return None


@dataclass(frozen=True, slots=True)
class CheckResult:
    current: str
    latest: str = ""
    update_available: bool = False
    installer_name: str = ""
    installer_url: str = ""
    sha256: str = ""
    release_url: str = ""
    #: Why there is no verdict, when there is none. Empty on a clean check.
    error: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "current": self.current,
            "latest": self.latest,
            "update_available": self.update_available,
            "release_url": self.release_url,
            "error": self.error,
        }


def check_latest(
    *,
    state_dir: Path,
    force: bool = False,
    now: float | None = None,
    fetch: object | None = None,
) -> CheckResult:
    """What the newest release is, at most once a day, failing soft.

    An offline install is a supported way to run this product, so a failed
    check is a note, never an error dialog and never a crash. The throttle
    stamp lives beside the rest of the state; ``force`` belongs to the
    explicit "check now" click and the apply path.
    """
    current = current_version()
    moment = time.time() if now is None else now

    stamp = state_dir / "update-check.json"
    if not force:
        try:
            saved = json.loads(stamp.read_text(encoding="utf-8"))
            if moment - float(saved.get("checked_at", 0)) < CHECK_INTERVAL_SECONDS:
                return CheckResult(**{**saved.get("result", {}), "current": current})
        except (OSError, ValueError, TypeError):
            pass

    try:
        if fetch is not None:
            data = fetch()  # type: ignore[operator]
        else:  # pragma: no cover - exercised in the field, not in tests
            response = httpx.get(
                _API_LATEST,
                headers={"Accept": "application/vnd.github+json"},
                timeout=10.0,
                follow_redirects=True,
            )
            response.raise_for_status()
            data = response.json()
    except Exception as exc:  # noqa: BLE001 - failing soft is the contract
        return CheckResult(current=current, error=f"update check failed: {exc}")

    result = _read_release(current, data)
    try:
        state_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "checked_at": moment,
            "result": {
                "latest": result.latest,
                "update_available": result.update_available,
                "installer_name": result.installer_name,
                "installer_url": result.installer_url,
                "sha256": result.sha256,
                "release_url": result.release_url,
                "error": result.error,
            },
        }
        stamp.write_text(json.dumps(payload), encoding="utf-8")
    except OSError:  # pragma: no cover - a stamp that cannot write is a shrug
        pass
    return result


def _read_release(current: str, data: object) -> CheckResult:
    if not isinstance(data, dict):
        return CheckResult(current=current, error="update check failed: unexpected response")
    tag = str(data.get("tag_name", ""))
    latest = _parse(tag)
    mine = _parse(current)
    if latest is None or mine is None:
        return CheckResult(
            current=current,
            latest=tag.lstrip("vV"),
            error=f"update check failed: cannot compare versions {current!r} and {tag!r}",
        )

    installer_name = ""
    installer_url = ""
    sha256 = ""
    for asset in data.get("assets") or []:
        name = str(asset.get("name", ""))
        if name.startswith("OpenKnowledge-Setup-") and name.endswith(".exe"):
            installer_name = name
            installer_url = str(asset.get("browser_download_url", ""))
            digest = str(asset.get("digest", ""))
            if digest.startswith("sha256:"):
                sha256 = digest.removeprefix("sha256:")
            break

    available = latest > mine and bool(installer_url) and bool(sha256)
    error = ""
    if latest > mine and not available:
        # A newer release without a verifiable installer is not an update
        # this code will offer - say why instead of failing quietly.
        error = "a newer release exists but carries no digest-verified installer"
    return CheckResult(
        current=current,
        latest=".".join(str(p) for p in latest),
        update_available=available,
        installer_name=installer_name,
        installer_url=installer_url,
        sha256=sha256,
        release_url=str(data.get("html_url", "")),
        error=error,
    )


def download_and_verify(
    result: CheckResult, *, dest_dir: Path, client: httpx.Client | None = None
) -> Path:
    """Fetch the installer and prove it is the one the release recorded.

    A mismatch deletes the file and refuses loudly: serving a person an
    executable that does not hash to what the release says is the one
    failure this feature must never have.
    """
    if not result.update_available:
        raise UpdateError("no verified update is available to download")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / result.installer_name
    digest = hashlib.sha256()
    owns_client = client is None
    http = client or httpx.Client(follow_redirects=True, timeout=60.0)
    try:
        with http.stream("GET", result.installer_url) as response:
            if response.status_code != 200:
                raise UpdateError(f"the installer download answered HTTP {response.status_code}")
            with dest.open("wb") as out:
                for piece in response.iter_bytes():
                    digest.update(piece)
                    out.write(piece)
    finally:
        if owns_client:
            http.close()

    got = digest.hexdigest()
    if got != result.sha256.lower():
        dest.unlink(missing_ok=True)
        raise UpdateError(
            "the downloaded installer does not match the release's SHA-256 "
            f"(expected {result.sha256[:12]}…, got {got[:12]}…), so it was deleted "
            "and nothing will run it. Try again; if it repeats, download manually "
            f"from {result.release_url}"
        )
    return dest


class _Handoff:
    """The meeting point between the web endpoint and the launcher.

    Same pattern as first-run setup: the endpoint records what should happen,
    signals the launcher's quit event, and the launcher - after it has shut
    the web server and the model servers down cleanly - runs the installer.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._quit: threading.Event | None = None
        self._installer: Path | None = None

    def bind(self, quit_event: threading.Event) -> None:
        with self._lock:
            self._quit = quit_event

    @property
    def bound(self) -> bool:
        with self._lock:
            return self._quit is not None

    def request(self, installer: Path, *, delay_seconds: float = 0.7) -> None:
        """Record the installer and schedule the shutdown signal.

        The delay exists so the HTTP response announcing "updating" reaches
        the browser before the server it came from starts dying.
        """
        with self._lock:
            if self._quit is None:
                raise UpdateError("updates are applied by the desktop app, not this server")
            self._installer = installer
            quit_event = self._quit
        timer = threading.Timer(delay_seconds, quit_event.set)
        timer.daemon = True
        timer.start()

    def installer(self) -> Path | None:
        with self._lock:
            return self._installer


HANDOFF = _Handoff()


def spawn_command(installer: Path, relaunch: Path) -> list[str]:
    """The detached helper: silent install, then start the new build."""
    script = (
        f"Start-Process -FilePath '{installer}' "
        "-ArgumentList '/VERYSILENT','/SUPPRESSMSGBOXES','/NORESTART' -Wait; "
        f"Start-Process -FilePath '{relaunch}'"
    )
    return ["powershell", "-NoProfile", "-WindowStyle", "Hidden", "-Command", script]


def spawn_installer(installer: Path, *, relaunch: Path) -> None:  # pragma: no cover - windows
    if sys.platform != "win32":
        raise UpdateError("silent self-update is a Windows desktop feature")
    flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
    subprocess.Popen(  # noqa: S603 - fixed command, paths from our own state
        spawn_command(installer, relaunch),
        creationflags=flags,
        close_fds=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
