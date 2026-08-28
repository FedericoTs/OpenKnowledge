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
    ranges_seen: list[str] = []

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        cls = type(self)
        header = self.headers.get("Range") or ""
        cls.ranges_seen.append(header)
        start = 0
        if header and not cls.ignore_range:
            start = int(header.removeprefix("bytes=").split("-")[0])
            self.send_response(206)
        else:
            self.send_response(200)
        body = cls.content[start:]
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:  # keep test output clean
        pass


@pytest.fixture()
def cdn():
    class Handler(_CdnHandler):
        content = _CONTENT
        ignore_range = False
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
    assert (tmp_path / "fake-model.gguf.sha256-ok").read_text() == model.sha256
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


def test_short_body_fails_but_keeps_bytes_for_resume(tmp_path: Path, cdn) -> None:
    base_url, handler = cdn
    handler.content = _CONTENT[:60_000]
    model = _fake_model()
    with pytest.raises(DownloadError, match="server sent"):
        ensure_model(model, tmp_path, base_url=base_url)
    part = tmp_path / "fake-model.gguf.part"
    assert part.stat().st_size == 60_000, "partial bytes are the resume point, keep them"


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


def test_progress_callback_reaches_the_reporter(tmp_path: Path, cdn) -> None:
    """fetch_models wires download progress through to a reporter."""
    base_url, _ = cdn
    from openknowledge.desktop import firstrun

    seen: list[tuple[str, int, int]] = []

    class Recorder:
        def update(self, model: ModelFile, done: int, total: int, position: str) -> None:
            seen.append((position, done, total))

        def close(self) -> None:
            seen.append(("closed", 0, 0))

    real_ensure = firstrun.ensure_model

    def patched(model: ModelFile, into: Path, progress=None):  # type: ignore[no-untyped-def]
        return real_ensure(model, into, progress, base_url=base_url)

    reporter = Recorder()
    original_make, firstrun.make_reporter = firstrun.make_reporter, lambda: reporter
    firstrun.ensure_model = patched
    try:
        firstrun.fetch_models((_fake_model(),), tmp_path)
    finally:
        firstrun.make_reporter = original_make
        firstrun.ensure_model = real_ensure
    assert seen[-1][0] == "closed"
    mid = [s for s in seen if s[0] == "model 1 of 1"]
    assert mid and mid[-1][1] == len(_CONTENT), "the bar must reach the end"


def test_console_reporter_survives_no_tty(capsys) -> None:
    from openknowledge.desktop.firstrun import ConsoleReporter

    reporter = ConsoleReporter()
    reporter.update(_fake_model(), 0, 100, "model 1 of 1")
    reporter.update(_fake_model(), 100, 100, "model 1 of 1")
    reporter.close()
    out = capsys.readouterr().out
    assert "fake-model.gguf" in out
