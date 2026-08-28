"""From a double-click to a chatbot - with first run living in the browser.

The order of operations is the contract:

1. **Plan** - read the state ``.env`` and decide, as a pure function, what to
   write and which llama-servers this launcher owns. First run writes the
   measured defaults; a person who has re-pointed a setting at their own
   Ollama keeps their choice, and the launcher simply does not spawn what it
   no longer manages.
2. **Provision** - the plan's keys land in the state ``.env`` through the
   same atomic writer everything else uses.
3. **Serve immediately** - the same FastAPI app the CLI serves, on
   127.0.0.1, before any model exists. The browser opens at once.
4. **First run in the page** - when models are missing, the widget hands
   over to ``/setup``: the person consents to the 2.6 GB download, watches
   progress, and a connection that keeps dropping ends in a Resume button,
   never a native dialog and never a relaunch. The downloader retries and
   resumes by itself; the field test that shaped this was a laptop whose
   connection died every ~190 MB.
5. **Swap in the engine** - once models are on disk the llama-servers start
   and a freshly built engine replaces the one that booted without them,
   exactly the way the settings page swaps engines.
6. **Tray** - where available; a plain wait-for-Ctrl+C otherwise.

If OpenKnowledge is already serving on the app port, the launcher opens the
browser at it and exits: two instances would fight over the same SQLite
files, and the person's intent - "I want the chatbot" - is already met.
"""

from __future__ import annotations

import os
import sys
import threading
import time
import webbrowser
from dataclasses import dataclass
from pathlib import Path

import httpx

from ..models import write_env
from ..paths import StatePaths, state_paths
from . import llama
from .download import (
    DownloadError,
    TransientDownloadError,
    already_verified,
    ensure_model,
)
from .llama import LlamaError, LlamaServer
from .manifest import CHAT_MODEL, EMBEDDING_MODEL, ModelFile
from .setup import STATUS

APP_PORT = 8080
CHAT_PORT = 8091
EMBED_PORT = 8092

_CHAT_MANAGED = f"http://127.0.0.1:{CHAT_PORT}/v1"
_EMBED_MANAGED = f"http://127.0.0.1:{EMBED_PORT}/v1"


@dataclass(frozen=True)
class LaunchPlan:
    """What this launch will write and which servers it owns."""

    provision: dict[str, str]
    spawn_chat: bool
    spawn_embed: bool
    notes: tuple[str, ...]


def desktop_defaults(models_dir: Path) -> dict[str, str]:
    """The measured configuration, expressed as state-env keys."""
    return {
        "OK_LOCAL_ENABLED": "true",
        "OK_LOCAL_MODEL": str(models_dir / CHAT_MODEL.filename),
        "OK_LOCAL_BASE_URL": _CHAT_MANAGED,
        "OK_LOCAL_CONTEXT_TOKENS": str(CHAT_MODEL.context_tokens),
        "OK_EMBEDDING_ENABLED": "true",
        "OK_EMBEDDING_MODEL": "nomic-embed-text-v1.5",
        "OK_EMBEDDING_BASE_URL": _EMBED_MANAGED,
    }


def read_env_file(path: Path) -> dict[str, str]:
    """The state .env as written: KEY=value lines, comments ignored.

    This reads the file, not the environment - the plan must see what is
    *persisted*, because that is what the person chose or what a previous
    launch provisioned.
    """
    values: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return values
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip("'\"")
    return values


def plan_launch(existing: dict[str, str], models_dir: Path) -> LaunchPlan:
    """Decide writes and ownership from the persisted state alone."""
    defaults = desktop_defaults(models_dir)
    provision = {key: value for key, value in defaults.items() if key not in existing}
    effective = {**defaults, **existing}

    notes: list[str] = []

    def _owns(url_key: str, enabled_key: str, managed: str, what: str) -> bool:
        if effective.get(enabled_key, "true").strip().lower() in ("false", "0", "no"):
            notes.append(f"{what} is disabled in settings; not started.")
            return False
        url = effective.get(url_key, "").strip().rstrip("/")
        if url == managed.rstrip("/"):
            return True
        notes.append(
            f"{what} points at {url or '(unset)'} - your setting, not the launcher's, "
            "so it is expected to already be running."
        )
        return False

    return LaunchPlan(
        provision=provision,
        spawn_chat=_owns("OK_LOCAL_BASE_URL", "OK_LOCAL_ENABLED", _CHAT_MANAGED, "the chat model"),
        spawn_embed=_owns(
            "OK_EMBEDDING_BASE_URL", "OK_EMBEDDING_ENABLED", _EMBED_MANAGED, "embeddings"
        ),
        notes=tuple(notes),
    )


def models_needed(plan: LaunchPlan) -> tuple[ModelFile, ...]:
    needed: list[ModelFile] = []
    if plan.spawn_chat:
        needed.append(CHAT_MODEL)
    if plan.spawn_embed:
        needed.append(EMBEDDING_MODEL)
    return tuple(needed)


class ConsoleProgress:
    """One redrawn console line per file - visible when run from a terminal."""

    def __init__(self) -> None:
        self._last = ""

    def update(self, filename: str, done: int, total: int) -> None:
        line = f"{filename}: {done / 1_000_000:,.0f} / {total / 1_000_000:,.0f} MB"
        if line != self._last:
            print("\r" + line.ljust(len(self._last)), end="", flush=True)
            self._last = line

    def finish(self) -> None:
        if self._last:
            print(flush=True)
            self._last = ""


def _first_run(
    exe: Path,
    needed: tuple[ModelFile, ...],
    models_dir: Path,
    state: StatePaths,
    servers: list[LlamaServer],
    stop: threading.Event,
) -> None:
    """Download (with consent), start llama-servers, swap the engine in.

    Runs on a background thread while the app already serves the setup page.
    Every state change lands in setup.STATUS, which is what the page shows.
    """
    console = ConsoleProgress()
    missing = [m for m in needed if not already_verified(m, models_dir)]

    if missing:
        STATUS.set_waiting(needed)
        if not STATUS.wait_for_proceed(stop):
            return
        STATUS.set_downloading()
        for model in needed:
            if already_verified(model, models_dir):
                STATUS.progress(model.filename, model.size_bytes, model.size_bytes)

        def report(model: ModelFile, done: int, total: int) -> None:
            STATUS.progress(model.filename, done, total)
            console.update(model.filename, done, total)

        for model in missing:
            while not stop.is_set():
                try:
                    ensure_model(model, models_dir, progress=report)
                    console.finish()
                    break
                except TransientDownloadError as error:
                    # The downloader already retried with resume; reaching
                    # here means the network needs a human moment. The page
                    # shows Resume; nothing is lost while it waits.
                    console.finish()
                    STATUS.set_stalled(str(error))
                    print(str(error), file=sys.stderr)
                    if not STATUS.wait_for_proceed(stop):
                        return
                    STATUS.set_downloading()
                except DownloadError as error:
                    console.finish()
                    STATUS.set_failed(str(error))
                    print(str(error), file=sys.stderr)
                    return
            if stop.is_set():
                return

    STATUS.set_starting("loading the chat and embedding models")
    log_dir = state.data_dir / "logs"
    ports = {CHAT_MODEL.purpose: CHAT_PORT, EMBEDDING_MODEL.purpose: EMBED_PORT}
    try:
        for model in needed:
            servers.append(
                llama.spawn(exe, models_dir / model.filename, model, ports[model.purpose], log_dir)
            )
        for server in servers:
            llama.wait_ready(server)
    except LlamaError as error:
        STATUS.set_failed(str(error))
        print(str(error), file=sys.stderr)
        llama.terminate(servers)
        servers.clear()
        return

    try:
        _swap_running_engine()
    except Exception as error:  # a bad engine build must not kill the page
        STATUS.set_failed(f"models are running but the engine did not rebuild: {error}")
        print(f"engine rebuild failed: {error}", file=sys.stderr)
        return

    STATUS.set_ready()
    print("first run complete - models ready", flush=True)


def _swap_running_engine() -> None:
    """Rebuild the served app's engine now that the models answer.

    The same build-first-then-swap the settings page uses: the old engine
    keeps serving until the new one exists.
    """
    from ..api import app as app_module
    from ..api.engine import build_engine

    application = app_module.app
    fresh = build_engine(application.state.settings)
    old = application.state.engine
    application.state.engine = fresh
    old.store.close()
    old.knowledge.close()
    app_module._warm_the_model_in_the_background(application.state.settings)


def _already_serving(port: int) -> bool:
    try:
        with httpx.Client(timeout=2.0) as client:
            body = client.get(f"http://127.0.0.1:{port}/healthz").json()
        return "corpus_version" in body or "documents_indexed" in body
    except Exception:
        return False


def _fail(message: str) -> int:
    print(message, file=sys.stderr)
    if sys.platform == "win32":  # pragma: no cover - native message box
        import ctypes

        ctypes.windll.user32.MessageBoxW(None, message, "OpenKnowledge", 0x10)
    return 1


def _serve_app(state: StatePaths) -> tuple[object, threading.Thread]:
    """Start uvicorn on a thread; returns (server, thread) for shutdown."""
    import uvicorn

    os.environ["OK_BIND_HOST"] = "127.0.0.1"
    config = uvicorn.Config(
        "openknowledge.api.app:app", host="127.0.0.1", port=APP_PORT, log_config=None
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, name="openknowledge-web", daemon=True)
    thread.start()
    return server, thread


def _wait_app_ready(timeout_seconds: float = 120.0) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if _already_serving(APP_PORT):
            return True
        time.sleep(0.5)
    return False


def _tray_or_wait(state: StatePaths, on_quit: threading.Event) -> None:
    """Sit in the system tray; without one, wait for Ctrl+C."""
    try:
        _run_tray(state, on_quit)
    except Exception:
        print("OpenKnowledge is running at the address above. Ctrl+C stops it.", flush=True)
        try:
            while not on_quit.is_set():
                time.sleep(0.5)
        except KeyboardInterrupt:
            on_quit.set()


def _run_tray(state: StatePaths, on_quit: threading.Event) -> None:
    import pystray
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (64, 64), (16, 82, 60))
    draw = ImageDraw.Draw(image)
    draw.rectangle((14, 22, 50, 27), fill=(240, 240, 235))
    draw.rectangle((14, 34, 50, 39), fill=(240, 240, 235))
    draw.rectangle((14, 46, 38, 51), fill=(240, 240, 235))

    def open_chat(*_: object) -> None:
        webbrowser.open(f"http://127.0.0.1:{APP_PORT}/")

    def open_manage(*_: object) -> None:
        webbrowser.open(f"http://127.0.0.1:{APP_PORT}/manage")

    def open_documents(*_: object) -> None:
        path = state.documents_dir
        path.mkdir(parents=True, exist_ok=True)
        if sys.platform == "win32":  # pragma: no cover - windows shell
            os.startfile(path)  # type: ignore[attr-defined]  # noqa: S606
        else:
            import subprocess

            subprocess.Popen(["xdg-open", str(path)])

    def quit_(icon: object, *_: object) -> None:
        on_quit.set()
        icon.stop()  # type: ignore[attr-defined]

    icon = pystray.Icon(
        "openknowledge",
        image,
        "OpenKnowledge",
        menu=pystray.Menu(
            pystray.MenuItem("Open chat", open_chat, default=True),
            pystray.MenuItem("Manage knowledge", open_manage),
            pystray.MenuItem("Documents folder", open_documents),
            pystray.MenuItem("Quit", quit_),
        ),
    )
    icon.run()


def main() -> int:
    if _already_serving(APP_PORT):
        webbrowser.open(f"http://127.0.0.1:{APP_PORT}/")
        print(
            f"OpenKnowledge is already running on port {APP_PORT}; opened the browser.",
            flush=True,
        )
        return 0

    state = state_paths()
    models_dir = state.data_dir / "models"
    plan = plan_launch(read_env_file(state.env_file), models_dir)
    for note in plan.notes:
        print(note, flush=True)

    if plan.provision:
        state.root.mkdir(parents=True, exist_ok=True)
        write_env(state.env_file, plan.provision)

    needed = models_needed(plan)
    servers: list[LlamaServer] = []
    on_quit = threading.Event()

    web_server, web_thread = _serve_app(state)
    try:
        if not _wait_app_ready():
            return _fail(
                f"The web server did not come up. The state and logs are under {state.data_dir}."
            )

        if needed:
            exe = llama.find_llama_server()
            if exe is None:
                message = (
                    "The bundled llama-server was not found. Reinstalling OpenKnowledge "
                    "restores it, or set OK_LLAMA_SERVER to a llama-server executable."
                )
                STATUS.set_failed(message)
                print(message, file=sys.stderr)
            else:
                threading.Thread(
                    target=_first_run,
                    args=(exe, needed, models_dir, state, servers, on_quit),
                    name="openknowledge-first-run",
                    daemon=True,
                ).start()
        else:
            STATUS.set_ready()

        url = f"http://127.0.0.1:{APP_PORT}/"
        print(f"OpenKnowledge is serving at {url}", flush=True)
        webbrowser.open(url)

        _tray_or_wait(state, on_quit)
        return 0
    finally:
        on_quit.set()
        web_server.should_exit = True  # type: ignore[attr-defined]
        web_thread.join(timeout=10)
        llama.terminate(servers)
