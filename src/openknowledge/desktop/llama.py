"""Supervising the bundled llama-server processes.

The installer ships llama.cpp's ``llama-server`` (the win-vulkan build: GPU
through Vulkan where a driver exists, runtime-dispatched CPU paths where
not). The desktop app runs two of them, exactly the shape every measured
number was produced on: one serving chat completions, one serving
embeddings, both loopback-only. One process per model is deliberate - it is
the configuration the golden runs used, and a wedged chat model cannot take
retrieval's embeddings down with it.

Nothing here is Windows-specific except cosmetics (hiding the console
windows); the tests drive it with a stub server on Linux.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from .manifest import ModelFile


class LlamaError(Exception):
    """A llama-server could not be found, started, or become ready."""


@dataclass
class LlamaServer:
    """One running llama-server process and where it listens."""

    process: subprocess.Popen[bytes]
    port: int
    purpose: str
    log_path: Path

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"


def find_llama_server(bundle_dir: Path | None = None) -> Path | None:
    """Locate the llama-server executable, bundled or supplied.

    ``OK_LLAMA_SERVER`` wins so a person can point at their own build. After
    that, the installer's layout: a ``llama`` folder next to the executable
    (PyInstaller onedir), then next to this file's bundle root.
    """
    override = os.environ.get("OK_LLAMA_SERVER", "")
    if override:
        path = Path(override)
        return path if path.is_file() else None

    exe = "llama-server.exe" if sys.platform == "win32" else "llama-server"
    roots = []
    if bundle_dir is not None:
        roots.append(bundle_dir)
    if getattr(sys, "frozen", False):  # pragma: no cover - only true in a bundle
        roots.append(Path(sys.executable).resolve().parent / "llama")
    for root in roots:
        candidate = root / exe
        if candidate.is_file():
            return candidate
    return None


def spawn(
    exe: Path,
    model_path: Path,
    model: ModelFile,
    port: int,
    log_dir: Path,
    extra_args: tuple[str, ...] = (),
    parallel: int = 1,
) -> LlamaServer:
    """Start one llama-server for ``model`` and return the handle.

    Flags are the measured configuration: an explicit context size (``-c``)
    because a per-launch flag is the whole reason llama-server replaced the
    Ollama derived-model workaround, loopback host, no web UI. Embedding
    servers get ``--embedding``; pooling comes from the model's own
    metadata.

    One slot by default is a field lesson: llama-server defaults to four,
    each with the full context, and the resulting KV buffer - a single 1 GiB
    Vulkan allocation for the chat model - is exactly what a laptop's
    integrated GPU refused.

    The earlier version of this note said the app "serializes its requests
    through the cascade anyway". That was an assumption about one person
    using a laptop, not a property of the code, and on a shared server it
    was false: measured with four simultaneous questions, three had their
    streams severed and came back as a model that could not be reached.
    Requests now queue in the provider, sized by ``OK_LOCAL_PARALLEL`` -
    the same number passed here, so the queue and the slots can never
    disagree.

    ``extra_args`` is how the launcher retries on CPU (``-ngl 0``) when a
    GPU cannot hold the model.
    """
    args = [
        str(exe),
        "-m",
        str(model_path),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "-c",
        str(model.context_tokens),
        "--parallel",
        str(parallel),
        "--no-webui",
        *extra_args,
    ]
    if model.purpose == "embedding":
        args.append("--embedding")

    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"llama-{model.purpose}.log"
    creationflags = 0
    if sys.platform == "win32":  # pragma: no cover - windows cosmetics
        creationflags = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
    try:
        with log_path.open("ab") as log:
            process = subprocess.Popen(
                args, stdout=log, stderr=subprocess.STDOUT, creationflags=creationflags
            )
    except OSError as exc:
        raise LlamaError(f"could not start {exe.name} for {model.purpose}: {exc}") from exc
    return LlamaServer(process=process, port=port, purpose=model.purpose, log_path=log_path)


def wait_ready(server: LlamaServer, timeout_seconds: float = 420.0) -> None:
    """Block until the server answers, or explain why it never will.

    llama-server exposes ``/health`` and returns 503 while the model loads;
    a dead process is reported with the tail of its own log rather than a
    bare timeout, because "it did not start" without the reason is exactly
    the kind of error this project refuses to emit.
    """
    deadline = time.monotonic() + timeout_seconds
    url = f"http://127.0.0.1:{server.port}/health"
    with httpx.Client(timeout=2.0) as client:
        while time.monotonic() < deadline:
            code = server.process.poll()
            if code is not None:
                raise LlamaError(
                    f"llama-server ({server.purpose}) exited with code {code} while "
                    f"loading. Log tail:\n{_tail(server.log_path)}"
                )
            try:
                if client.get(url).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.5)
    raise LlamaError(
        f"llama-server ({server.purpose}) did not become ready within "
        f"{timeout_seconds:.0f}s. Log tail:\n{_tail(server.log_path)}"
    )


def terminate(servers: list[LlamaServer], grace_seconds: float = 10.0) -> None:
    """Stop every server: ask nicely once, then insist."""
    for server in servers:
        if server.process.poll() is None:
            server.process.terminate()
    deadline = time.monotonic() + grace_seconds
    for server in servers:
        remaining = max(0.1, deadline - time.monotonic())
        try:
            server.process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            server.process.kill()
            server.process.wait(timeout=5.0)


def _tail(log_path: Path, lines: int = 8) -> str:
    try:
        text = log_path.read_text(errors="replace", encoding="utf-8")
    except OSError:
        return "(no log was written)"
    tail = text.strip().splitlines()[-lines:]
    return "\n".join(tail) if tail else "(log is empty)"
