"""The desktop launcher: downloads that resume, servers that report, plans that respect settings.

The scenarios here are the ones that actually happen to installed software:
the download dies at 94% and must resume, the upstream file changes and must
be refused, the person re-points a setting at their own Ollama and the
launcher must keep its hands off, llama-server crashes on load and the error
must carry the log instead of "timeout".
"""

from __future__ import annotations

import hashlib
import stat
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from openknowledge.desktop import download as download_module
from openknowledge.desktop import llama
from openknowledge.desktop.download import DownloadError, ensure_model
from openknowledge.desktop.launcher import (
    APP_PORT,
    CHAT_PORT,
    EMBED_PORT,
    desktop_defaults,
    models_needed,
    plan_launch,
    read_env_file,
)
from openknowledge.desktop.manifest import MODELS, ModelFile

# ---- manifest --------------------------------------------------------------


def test_manifest_is_well_formed() -> None:
    purposes = set()
    for model in MODELS:
        assert model.filename.endswith(".gguf")
        assert model.url.startswith("https://huggingface.co/")
        assert model.url.endswith(model.filename), "URL must resolve the pinned filename"
        assert len(model.sha256) == 64 and int(model.sha256, 16) >= 0
        assert model.size_bytes > 1_000_000
        assert model.license == "Apache-2.0"
        assert model.context_tokens >= 2048
        purposes.add(model.purpose)
    assert purposes == {"chat", "embedding"}


def test_the_three_ports_are_distinct() -> None:
    assert len({APP_PORT, CHAT_PORT, EMBED_PORT}) == 3


# ---- download: a local server that behaves like a CDN ----------------------

_CONTENT = bytes(range(256)) * 400  # 102,400 deterministic bytes


def _fake_model(sha256: str | None = None, size: int | None = None) -> ModelFile:
    return ModelFile(
        filename="fake-model.gguf",
        url="https://huggingface.co/unused/resolve/main/fake-model.gguf",
        sha256=sha256 or hashlib.sha256(_CONTENT).hexdigest(),
        size_bytes=size or len(_CONTENT),
        purpose="chat",
        license="Apache-2.0",
        context_tokens=4096,
    )


class _CdnHandler(BaseHTTPRequestHandler):
    content: bytes = _CONTENT
    ignore_range = False
    #: While positive, each request promises the full body, delivers half,
    #: and hangs up - the shape of a stalled connection as the client sees it.
    fail_first = 0
    #: When set, every request gets this status and an empty body.
    status_override: int | None = None
    ranges_seen: list[str] = []

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        cls = type(self)
        header = self.headers.get("Range") or ""
        cls.ranges_seen.append(header)
        if cls.status_override is not None:
            self.send_response(cls.status_override)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        start = 0
        if header and not cls.ignore_range:
            start = int(header.removeprefix("bytes=").split("-")[0])
            self.send_response(206)
        else:
            self.send_response(200)
        body = cls.content[start:]
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if cls.fail_first > 0:
            cls.fail_first -= 1
            self.wfile.write(body[: max(1, len(body) // 2)])
            self.wfile.flush()
            self.connection.close()
            return
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:  # keep test output clean
        pass


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    """Retries are the product; waiting between them is not the test's job."""
    monkeypatch.setattr(download_module, "_BACKOFF_SECONDS", (0.0,))


@pytest.fixture()
def cdn():
    class Handler(_CdnHandler):
        content = _CONTENT
        ignore_range = False
        fail_first = 0
        status_override: int | None = None
        ranges_seen: list[str] = []

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}", Handler
    server.shutdown()


def test_download_verifies_and_marks(tmp_path: Path, cdn) -> None:
    base_url, handler = cdn
    model = _fake_model()
    path = ensure_model(model, tmp_path, base_url=base_url)
    assert path.read_bytes() == _CONTENT
    assert (tmp_path / "fake-model.gguf.sha256-ok").read_text(encoding="utf-8") == model.sha256
    assert not (tmp_path / "fake-model.gguf.part").exists()


def test_verified_file_costs_no_network(tmp_path: Path, cdn) -> None:
    base_url, handler = cdn
    model = _fake_model()
    ensure_model(model, tmp_path, base_url=base_url)
    handler.ranges_seen.clear()
    ensure_model(model, tmp_path, base_url=base_url)
    assert handler.ranges_seen == [], "a verified file must not touch the network again"


def test_download_resumes_from_partial_bytes(tmp_path: Path, cdn) -> None:
    base_url, handler = cdn
    model = _fake_model()
    (tmp_path / "fake-model.gguf.part").write_bytes(_CONTENT[:40_000])
    path = ensure_model(model, tmp_path, base_url=base_url)
    assert path.read_bytes() == _CONTENT
    assert handler.ranges_seen == ["bytes=40000-"], "the request must ask only for the rest"


def test_server_ignoring_range_restarts_clean(tmp_path: Path, cdn) -> None:
    base_url, handler = cdn
    handler.ignore_range = True
    model = _fake_model()
    (tmp_path / "fake-model.gguf.part").write_bytes(_CONTENT[:40_000])
    path = ensure_model(model, tmp_path, base_url=base_url)
    assert path.read_bytes() == _CONTENT, "a 200 reply replaces the partial bytes, never appends"


def test_hash_mismatch_refuses_and_discards(tmp_path: Path, cdn) -> None:
    base_url, _ = cdn
    model = _fake_model(sha256="0" * 64)
    with pytest.raises(DownloadError, match="pins"):
        ensure_model(model, tmp_path, base_url=base_url)
    assert not (tmp_path / "fake-model.gguf").exists()
    assert not (tmp_path / "fake-model.gguf.part").exists(), "wrong bytes must not survive"


def test_short_body_retries_then_fails_keeping_bytes(tmp_path: Path, cdn) -> None:
    """A server that always closes early exhausts the retries; the final
    error names the attempts and the partial bytes survive for next launch."""
    base_url, handler = cdn
    handler.content = _CONTENT[:60_000]
    model = _fake_model()
    with pytest.raises(DownloadError, match="still failing after"):
        ensure_model(model, tmp_path, base_url=base_url)
    assert len(handler.ranges_seen) == download_module._ATTEMPTS
    part = tmp_path / "fake-model.gguf.part"
    assert part.stat().st_size == 60_000, "partial bytes are the resume point, keep them"


def test_a_stalled_download_retries_and_resumes_by_itself(tmp_path: Path, cdn, monkeypatch) -> None:
    """The first field report: a read timeout at 58% shown to a person as a
    dialog. A transient failure must retry with resume, not ask for help.

    The chunk size shrinks so the test has real-file proportions: chunks
    must land before the stall, or there is no progress to resume from -
    at 1 MB chunks over a 100 KB fixture the failure arrives before the
    first yield, which is not the shape of a 2.5 GB download."""
    base_url, handler = cdn
    handler.fail_first = 2
    monkeypatch.setattr(download_module, "_CHUNK", 16_384)
    model = _fake_model()
    path = ensure_model(model, tmp_path, base_url=base_url)
    assert path.read_bytes() == _CONTENT, "the retries must converge on the right bytes"
    assert len(handler.ranges_seen) == 3, "two stalls, then the request that finished"
    assert handler.ranges_seen[0] == ""
    assert all(r.startswith("bytes=") for r in handler.ranges_seen[1:]), (
        "every retry must resume, never restart"
    )


def test_a_permanent_error_is_not_retried(tmp_path: Path, cdn) -> None:
    """A 404 is the manifest's problem, not the network's - retrying would
    only turn a clear error into a slow one."""
    base_url, handler = cdn
    handler.status_override = 404
    with pytest.raises(DownloadError, match="HTTP 404"):
        ensure_model(_fake_model(), tmp_path, base_url=base_url)
    assert len(handler.ranges_seen) == 1, "a permanent error must fail once, loudly"


def test_a_server_error_is_retried(tmp_path: Path, cdn) -> None:
    """A 503 is the server's bad afternoon; the retries must outlast it."""
    base_url, handler = cdn
    handler.status_override = 503
    with pytest.raises(DownloadError, match="still failing after"):
        ensure_model(_fake_model(), tmp_path, base_url=base_url)
    assert len(handler.ranges_seen) == download_module._ATTEMPTS


def test_right_size_wrong_bytes_is_redownloaded(tmp_path: Path, cdn) -> None:
    base_url, handler = cdn
    model = _fake_model()
    (tmp_path / "fake-model.gguf").write_bytes(b"\0" * len(_CONTENT))
    path = ensure_model(model, tmp_path, base_url=base_url)
    assert path.read_bytes() == _CONTENT
    assert handler.ranges_seen, "the corrupt file must have been replaced over the network"


# ---- llama-server supervision ----------------------------------------------


def _write_stub(tmp_path: Path, body: str) -> Path:
    """A fake llama-server the supervisor can actually spawn, on any OS.

    The Python body carries the behaviour; a .bat (Windows) or sh (POSIX)
    shim makes it an executable the same way the real llama-server is one.
    """
    script = tmp_path / "stub.py"
    script.write_text(body)
    if sys.platform == "win32":
        stub = tmp_path / "llama-server.bat"
        stub.write_text(f'@echo off\r\n"{sys.executable}" "{script}" %*\r\n')
    else:
        stub = tmp_path / "llama-server"
        stub.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n')
        stub.chmod(stub.stat().st_mode | stat.S_IXUSR)
    return stub


_READY_STUB = """\
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

port = int(sys.argv[sys.argv.index("--port") + 1])


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200 if self.path == "/health" else 404)
        self.end_headers()

    def log_message(self, *args):
        pass


HTTPServer(("127.0.0.1", port), Handler).serve_forever()
"""

_DYING_STUB = """\
import sys

print("error: model file is not a valid GGUF", file=sys.stderr)
sys.exit(7)
"""


def test_find_llama_server_env_override_and_bundle(tmp_path: Path, monkeypatch) -> None:
    named = tmp_path / ("llama-server.exe" if sys.platform == "win32" else "llama-server")
    named.write_bytes(b"not really a binary, but findable")
    monkeypatch.setenv("OK_LLAMA_SERVER", str(named))
    assert llama.find_llama_server() == named
    monkeypatch.setenv("OK_LLAMA_SERVER", str(tmp_path / "absent"))
    assert llama.find_llama_server() is None
    monkeypatch.delenv("OK_LLAMA_SERVER")
    assert llama.find_llama_server(bundle_dir=tmp_path) == named
    assert llama.find_llama_server(bundle_dir=tmp_path / "nowhere") is None


def test_spawn_wait_ready_and_terminate(tmp_path: Path) -> None:
    exe = _write_stub(tmp_path, _READY_STUB)
    model = _fake_model()
    (tmp_path / model.filename).write_bytes(b"gguf")
    server = llama.spawn(exe, tmp_path / model.filename, model, 18391, tmp_path / "logs")
    try:
        llama.wait_ready(server, timeout_seconds=15.0)
        assert server.base_url == "http://127.0.0.1:18391/v1"
    finally:
        llama.terminate([server])
    assert server.process.poll() is not None, "terminate must actually stop the process"


def test_a_dying_server_reports_its_own_log(tmp_path: Path) -> None:
    exe = _write_stub(tmp_path, _DYING_STUB)
    model = _fake_model()
    server = llama.spawn(exe, tmp_path / model.filename, model, 18392, tmp_path / "logs")
    with pytest.raises(llama.LlamaError) as error:
        llama.wait_ready(server, timeout_seconds=15.0)
    message = str(error.value)
    assert "exited with code 7" in message
    assert "not a valid GGUF" in message, "the error must carry the log, not just 'it died'"


# ---- the launch plan -------------------------------------------------------


def test_first_run_provisions_everything_and_owns_both(tmp_path: Path) -> None:
    plan = plan_launch({}, tmp_path / "models")
    assert plan.provision == desktop_defaults(tmp_path / "models")
    assert plan.spawn_chat and plan.spawn_embed
    assert plan.notes == ()
    assert [m.purpose for m in models_needed(plan)] == ["chat", "embedding"]


def test_a_repointed_endpoint_is_not_touched(tmp_path: Path) -> None:
    existing = {"OK_LOCAL_BASE_URL": "http://localhost:11434/v1"}
    plan = plan_launch(existing, tmp_path / "models")
    assert not plan.spawn_chat, "the person chose Ollama; the launcher must not fight it"
    assert plan.spawn_embed
    assert "OK_LOCAL_BASE_URL" not in plan.provision, "their value must never be overwritten"
    assert any("11434" in note for note in plan.notes), "silence would read as a hang"
    assert [m.purpose for m in models_needed(plan)] == ["embedding"]


def test_disabled_features_are_not_spawned(tmp_path: Path) -> None:
    plan = plan_launch({"OK_EMBEDDING_ENABLED": "false"}, tmp_path / "models")
    assert plan.spawn_chat and not plan.spawn_embed


def test_managed_url_with_trailing_slash_still_owned(tmp_path: Path) -> None:
    plan = plan_launch(
        {"OK_LOCAL_BASE_URL": f"http://127.0.0.1:{CHAT_PORT}/v1/"}, tmp_path / "models"
    )
    assert plan.spawn_chat


def test_read_env_file_reads_what_write_env_writes(tmp_path: Path) -> None:
    from openknowledge.models import write_env

    env = tmp_path / ".env"
    env.write_text("# a comment\nOK_KEEP='quoted'\n\nbroken line\n")
    write_env(env, {"OK_LOCAL_BASE_URL": "http://127.0.0.1:8091/v1"})
    values = read_env_file(env)
    assert values["OK_KEEP"] == "quoted"
    assert values["OK_LOCAL_BASE_URL"] == "http://127.0.0.1:8091/v1"
    assert read_env_file(tmp_path / "absent.env") == {}


def test_setup_status_walks_the_first_run() -> None:
    """The browser-first first run: waiting needs consent, stalls need a
    click on Resume, and only the setup thread moves the state forward."""
    from openknowledge.desktop.setup import SetupStatus

    status = SetupStatus()
    assert status.snapshot()["state"] == "ready"
    assert not status.request_proceed(), "nothing to consent to yet"

    model = _fake_model()
    status.set_waiting((model,))
    body = status.snapshot()
    assert body["state"] == "waiting"
    assert body["files"] == [{"filename": model.filename, "done": 0, "total": model.size_bytes}]

    assert status.request_proceed(), "the Download button must land"
    stop = threading.Event()
    assert status.wait_for_proceed(stop), "the signal must be received"

    status.set_downloading()
    status.progress(model.filename, 1234, model.size_bytes)
    assert status.snapshot()["files"][0]["done"] == 1234

    status.set_stalled("the connection kept dropping")
    assert status.snapshot()["state"] == "stalled"
    assert status.request_proceed(), "Resume must land too"

    status.set_starting("loading models")
    assert not status.request_proceed(), "no button applies while starting"
    status.set_ready()
    assert status.snapshot() == {"state": "ready", "message": "", "files": []}


def test_wait_for_proceed_yields_to_shutdown() -> None:
    """Quitting the app while the page waits for consent must not hang."""
    from openknowledge.desktop.setup import SetupStatus

    status = SetupStatus()
    status.set_waiting((_fake_model(),))
    stop = threading.Event()
    stop.set()
    assert status.wait_for_proceed(stop) is False


def test_console_progress_prints(capsys) -> None:
    from openknowledge.desktop.launcher import ConsoleProgress

    console = ConsoleProgress()
    console.update("fake-model.gguf", 0, 100)
    console.update("fake-model.gguf", 100, 100)
    console.finish()
    out = capsys.readouterr().out
    assert "fake-model.gguf" in out


def test_already_verified_requires_size_and_marker(tmp_path: Path, cdn) -> None:
    from openknowledge.desktop.download import already_verified

    base_url, _ = cdn
    model = _fake_model()
    assert not already_verified(model, tmp_path)
    ensure_model(model, tmp_path, base_url=base_url)
    assert already_verified(model, tmp_path)
    (tmp_path / (model.filename + ".sha256-ok")).unlink()
    assert not already_verified(model, tmp_path), "no marker, no trust"


def test_spawn_asks_for_one_slot_and_carries_extra_args(tmp_path: Path, monkeypatch) -> None:
    """Field lesson: llama-server's default four slots quadruple the KV
    buffer, and a laptop iGPU refused exactly that 1 GiB allocation. One
    slot is all this app uses; extra_args is the CPU-fallback channel."""
    captured: list[list[str]] = []

    class FakeProcess:
        def poll(self):
            return None

    def fake_popen(args, **kwargs):  # type: ignore[no-untyped-def]
        captured.append(list(args))
        return FakeProcess()

    monkeypatch.setattr(llama.subprocess, "Popen", fake_popen)
    model = _fake_model()
    llama.spawn(tmp_path / "llama-server", tmp_path / model.filename, model, 18400, tmp_path)
    llama.spawn(
        tmp_path / "llama-server",
        tmp_path / model.filename,
        model,
        18401,
        tmp_path,
        extra_args=("-ngl", "0"),
    )
    first, second = captured
    assert [
        first[first.index("--parallel")],
        first[first.index("--parallel") + 1],
    ] == ["--parallel", "1"]
    assert second[-2:] == ["-ngl", "0"] or ("-ngl" in second and "0" in second)


def test_gpu_out_of_memory_falls_back_to_cpu(tmp_path: Path, monkeypatch) -> None:
    """A GPU that cannot hold the model must not end first run: the same
    server is retried with every layer on the CPU, and the page says so."""
    from openknowledge.desktop import launcher

    calls: list[tuple[str, tuple[str, ...]]] = []
    terminated: list[object] = []

    class FakeServer:
        def __init__(self, tag: str) -> None:
            self.tag = tag

    def fake_spawn(exe, model_path, model, port, log_dir, extra_args=()):  # type: ignore[no-untyped-def]
        calls.append(("spawn", tuple(extra_args)))
        return FakeServer("cpu" if extra_args else "gpu")

    def fake_wait_ready(server, timeout_seconds=420.0):  # type: ignore[no-untyped-def]
        if server.tag == "gpu":
            raise llama.LlamaError(
                "llama-server (chat) exited with code 1 while loading. Log tail:\n"
                "ggml_vulkan: vk::Device::allocateMemory: ErrorOutOfDeviceMemory"
            )

    monkeypatch.setattr(launcher.llama, "spawn", fake_spawn)
    monkeypatch.setattr(launcher.llama, "wait_ready", fake_wait_ready)
    monkeypatch.setattr(launcher.llama, "terminate", lambda servers: terminated.extend(servers))

    model = _fake_model()
    server = launcher._start_with_cpu_fallback(
        tmp_path / "llama-server", tmp_path / model.filename, model, 18402, tmp_path
    )
    assert server.tag == "cpu", "the CPU retry must be what actually serves"
    assert calls == [("spawn", ()), ("spawn", ("-ngl", "0"))]
    assert len(terminated) == 1 and terminated[0].tag == "gpu"
