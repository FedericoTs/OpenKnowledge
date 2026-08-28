"""First-run state, shared between the launcher and the page in the browser.

The first field test of the installer settled how first run must work: the
person saw a native dialog per network stall and had to relaunch the app to
resume a 2.6 GB download. Now the app serves immediately, the browser asks
for consent before anything is downloaded ("asking is the feature - 2.6 GB
is not a surprise you spring on someone's data plan"), progress lives on a
page, stalls retry themselves, and a download that exhausts its retries
waits for a click on Resume rather than for a relaunch.

This module is the meeting point: the launcher's setup thread writes state
here, the web app's ``/setup/status`` route reads it, and the ``/setup``
page's buttons signal back through the events. On a plain server deployment
nothing ever touches this and the state stays ``ready``.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from .manifest import ModelFile


@dataclass
class FileProgress:
    filename: str
    done: int
    total: int


@dataclass
class _State:
    #: ready | waiting | downloading | stalled | starting | failed
    state: str = "ready"
    message: str = ""
    files: list[FileProgress] = field(default_factory=list)


class SetupStatus:
    """Thread-safe first-run state with the two signals the page can send."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state = _State()
        self._proceed = threading.Event()

    # -- what the page reads -------------------------------------------------

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "state": self._state.state,
                "message": self._state.message,
                "files": [
                    {"filename": f.filename, "done": f.done, "total": f.total}
                    for f in self._state.files
                ],
            }

    # -- what the page signals -----------------------------------------------

    def request_proceed(self) -> bool:
        """The Download or Resume button. True if there was anything to signal."""
        with self._lock:
            if self._state.state not in ("waiting", "stalled"):
                return False
        self._proceed.set()
        return True

    # -- what the setup thread drives ----------------------------------------

    def set_waiting(self, models: tuple[ModelFile, ...]) -> None:
        with self._lock:
            self._state = _State(
                state="waiting",
                message="These models answer your questions on this machine. "
                "Downloaded once, verified against pinned hashes.",
                files=[FileProgress(m.filename, 0, m.size_bytes) for m in models],
            )
        self._proceed.clear()

    def wait_for_proceed(self, stop: threading.Event, poll_seconds: float = 0.25) -> bool:
        """Block until the page signals, or ``stop`` ends the whole app."""
        while not stop.is_set():
            if self._proceed.wait(timeout=poll_seconds):
                self._proceed.clear()
                return True
        return False

    def set_downloading(self) -> None:
        with self._lock:
            self._state.state = "downloading"
            self._state.message = ""

    def progress(self, filename: str, done: int, total: int) -> None:
        with self._lock:
            for f in self._state.files:
                if f.filename == filename:
                    f.done, f.total = done, total
                    return
            self._state.files.append(FileProgress(filename, done, total))

    def set_stalled(self, message: str) -> None:
        with self._lock:
            self._state.state = "stalled"
            self._state.message = message
        self._proceed.clear()

    def set_starting(self, message: str) -> None:
        with self._lock:
            self._state.state = "starting"
            self._state.message = message

    def set_failed(self, message: str) -> None:
        with self._lock:
            self._state.state = "failed"
            self._state.message = message

    def set_ready(self) -> None:
        with self._lock:
            self._state = _State()


#: The one instance both sides talk through.
STATUS = SetupStatus()
