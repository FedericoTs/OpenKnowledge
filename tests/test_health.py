"""The model endpoints, asked whether they answer.

No mocking of HTTP: the thing under test is which URL is called with which
header and how each reply is read, and a mock would only assert this file
agrees with itself. A stub on a real socket answers like Ollama, like an
OpenAI-compatible server, like a paid API that checks its key, and like
Azure's legacy surface - one handler, a few switches.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse

import pytest

from openknowledge.config import Settings
from openknowledge.health import (
    ANTHROPIC_API,
    HealthMonitor,
    Reading,
    Target,
    probe,
    targets,
)


class _Stub:
    """What the stub endpoint believes about itself."""

    def __init__(self) -> None:
        self.ollama = True
        self.models = ["qwen3:8b", "nomic-embed-text"]
        #: When set, /v1/models wants this key as a bearer or as x-api-key.
        self.key: str | None = None
        #: Azure's data plane wants this in an ``api-key`` header.
        self.azure_key = "az-key"
        self.status = 200
        self.requests: list[tuple[str, dict[str, str]]] = []


@pytest.fixture
def stub() -> _Stub:
    return _Stub()


@pytest.fixture
def base(stub: _Stub) -> Iterator[str]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_: object) -> None:
            pass

        def _send(self, code: int, body: object) -> None:
            payload = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:  # noqa: N802 - http.server's spelling
            headers = {k.lower(): v for k, v in self.headers.items()}
            stub.requests.append((self.path, headers))
            path = urlparse(self.path).path
            if path == "/api/version" and stub.ollama:
                return self._send(200, {"version": "0.12.0"})
            if path == "/api/tags" and stub.ollama:
                # Ollama reports an untagged model as name:latest.
                rows = [{"model": m if ":" in m else f"{m}:latest", "size": 1} for m in stub.models]
                return self._send(200, {"models": rows})
            if path == "/v1/models":
                bearer, direct = headers.get("authorization"), headers.get("x-api-key")
                if stub.key and bearer != f"Bearer {stub.key}" and direct != stub.key:
                    return self._send(401, {"error": "bad key"})
                return self._send(stub.status, {"data": [{"id": m} for m in stub.models]})
            if path == "/openai/models":
                if headers.get("api-key") != stub.azure_key:
                    return self._send(401, {"error": "bad key"})
                return self._send(200, {"data": [{"id": "gpt-4o"}]})
            self._send(404, {"error": "not found"})

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()


# -- self-hosted runtimes -------------------------------------------------


def test_a_running_runtime_is_reported_with_its_model(base: str) -> None:
    reading = probe(Target("chat", "runtime", f"{base}/v1", "qwen3:8b"))
    assert (reading.state, reading.ok) == ("reachable", True)
    assert (reading.runtime, reading.version) == ("ollama", "0.12.0")
    assert reading.model_served is True
    assert reading.host == urlparse(base).netloc, "the host is what the page names"
    assert reading.latency_ms is not None and reading.latency_ms >= 0
    assert reading.checked_at > 0


def test_an_untagged_model_is_found_under_the_tag_ollama_gives_it(base: str) -> None:
    """The store says ``nomic-embed-text``; Ollama lists ``nomic-embed-text:latest``.
    The same model, and `model status` once called it missing over this."""
    reading = probe(Target("embedding", "runtime", f"{base}/v1", "nomic-embed-text"))
    assert reading.model_served is True
    assert reading.ok is True


def test_a_missing_model_is_not_a_healthy_endpoint(base: str) -> None:
    """Up is not the same as usable: a server that answers but does not serve
    the configured model refuses every question just as surely."""
    reading = probe(Target("chat", "runtime", f"{base}/v1", "qwen3:14b"))
    assert reading.state == "reachable"
    assert reading.model_served is False
    assert reading.ok is False
    assert "qwen3:14b" in reading.detail


def test_a_plain_openai_compatible_server_is_read_by_its_model_list(base: str, stub: _Stub) -> None:
    stub.ollama = False
    reading = probe(Target("chat", "runtime", f"{base}/v1", "qwen3:8b"))
    assert (reading.runtime, reading.model_served, reading.ok) == ("openai-compatible", True, True)


def test_nothing_listening_is_unreachable_within_the_timeout() -> None:
    started = time.perf_counter()
    reading = probe(Target("chat", "runtime", "http://127.0.0.1:9/v1", "qwen3:8b"), timeout=1.0)
    assert (reading.state, reading.ok) == ("unreachable", False)
    assert reading.configured is True
    assert "127.0.0.1:9" in reading.detail
    assert time.perf_counter() - started < 5, "a dead endpoint must not hang the reading"


# -- paid APIs ---------------------------------------------------------------


def test_a_paid_api_that_refuses_the_key_is_said_so(base: str, stub: _Stub) -> None:
    stub.key = "good"
    refused = probe(Target("escalation", "openai", f"{base}/v1", "gpt-x", api_key="bad"))
    assert (refused.state, refused.ok) == ("key refused", False)
    assert "401" in refused.detail

    accepted = probe(Target("escalation", "openai", f"{base}/v1", "qwen3:8b", api_key="good"))
    assert (accepted.state, accepted.ok, accepted.model_served) == ("reachable", True, True)
    assert stub.requests[-1][1]["authorization"] == "Bearer good"


def test_a_paid_api_that_does_not_list_the_model_is_not_ok(base: str, stub: _Stub) -> None:
    reading = probe(Target("escalation", "openai", f"{base}/v1", "gpt-nope", api_key="k"))
    assert (reading.state, reading.model_served, reading.ok) == ("reachable", False, False)
    assert "gpt-nope" in reading.detail


def test_a_paid_api_error_is_an_error_not_a_refusal(base: str, stub: _Stub) -> None:
    stub.status = 503
    reading = probe(Target("escalation", "openai", f"{base}/v1", "gpt-x", api_key="k"))
    assert (reading.state, reading.ok) == ("error", False)
    assert "503" in reading.detail


def test_an_azure_deployment_is_probed_through_its_models_list(base: str, stub: _Stub) -> None:
    """Azure's legacy URL names the deployment; the model list sits one level
    up and wants the dated api-version and an ``api-key`` header, not a bearer.
    Its list names models rather than deployments, so "served" is unknowable
    and must be reported as such rather than as missing."""
    target = Target(
        "escalation",
        "azure",
        f"{base}/openai/deployments/my-gpt",
        "my-gpt",
        api_key="az-key",
        api_version="2024-06-01",
    )
    reading = probe(target)
    assert (reading.state, reading.ok, reading.model_served) == ("reachable", True, None)
    path, headers = stub.requests[-1]
    assert path == "/openai/models?api-version=2024-06-01"
    assert headers["api-key"] == "az-key"
    assert "authorization" not in headers


def test_azures_v1_surface_is_probed_under_its_own_root(base: str, stub: _Stub) -> None:
    stub.key = None
    target = Target(
        "escalation", "azure", f"{base}/openai/v1", "my-gpt", api_key="az-key", api_version="v1"
    )
    # /openai/v1/models is not a route the stub has; what matters is the URL asked.
    probe(target)
    assert stub.requests[-1][0] == "/openai/v1/models"
    assert stub.requests[-1][1]["api-key"] == "az-key"


def test_anthropic_is_asked_with_its_own_headers(base: str, stub: _Stub) -> None:
    stub.key = "sk-ant"
    reading = probe(Target("escalation", "anthropic", base, "qwen3:8b", api_key="sk-ant"))
    assert (reading.state, reading.ok) == ("reachable", True)
    path, headers = stub.requests[-1]
    assert path == "/v1/models"
    assert headers["x-api-key"] == "sk-ant"
    assert headers["anthropic-version"], "the Messages API refuses a call without a version"


def test_a_reading_never_carries_the_key(base: str, stub: _Stub) -> None:
    stub.key = "sekrit-key"
    for target in (
        Target("escalation", "openai", f"{base}/v1", "qwen3:8b", api_key="sekrit-key"),
        Target("escalation", "openai", f"{base}/v1", "qwen3:8b", api_key="wrong-sekrit"),
    ):
        assert "sekrit" not in json.dumps(probe(target).as_dict())


# -- what is not configured ---------------------------------------------------


def test_off_is_a_reading_with_a_reason() -> None:
    reading = probe(Target("escalation", "off", detail="escalation is off"))
    assert (reading.configured, reading.state, reading.ok) == (False, "off", None)
    assert reading.detail == "escalation is off"


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    return Settings(
        data_dir=str(tmp_path / "data"),
        documents_dir=str(tmp_path / "documents"),
        _env_file=None,  # type: ignore[call-arg]
        **overrides,  # type: ignore[arg-type]
    )


def test_targets_follow_the_settings(tmp_path: Path) -> None:
    settings = _settings(
        tmp_path,
        local_enabled=True,
        local_base_url="http://models.internal:8081/v1",
        local_model="qwen3-4b-instruct",
        embedding_enabled=True,
        embedding_base_url="",
        embedding_model="nomic-embed-text",
        escalation_enabled=False,
    )
    chat, embedding, escalation = targets(settings, None)
    assert (chat.role, chat.kind, chat.base_url, chat.model) == (
        "chat",
        "runtime",
        "http://models.internal:8081/v1",
        "qwen3-4b-instruct",
    )
    # Embeddings run on the chat endpoint unless told otherwise, and the
    # health line has to ask the endpoint they actually use.
    assert (embedding.kind, embedding.base_url, embedding.model) == (
        "runtime",
        "http://models.internal:8081/v1",
        "nomic-embed-text",
    )
    assert (escalation.kind, escalation.role) == ("off", "escalation")
    assert "off" in escalation.detail

    apart = _settings(tmp_path, embedding_base_url="http://embed.internal:8082/v1")
    assert targets(apart, None)[1].base_url == "http://embed.internal:8082/v1"


def test_escalation_on_without_a_provider_is_reported_as_missing(tmp_path: Path) -> None:
    """Enabled in settings, nothing built: the key or endpoint is absent. That
    is the state an operator most needs to see, and it is not "off"."""
    settings = _settings(tmp_path, escalation_enabled=True)
    escalation = targets(settings, None)[2]
    assert escalation.kind == "off"
    assert "missing" in escalation.detail


def test_the_escalation_target_is_read_off_the_provider_that_was_built(tmp_path: Path) -> None:
    from openknowledge.providers.anthropic_provider import AnthropicProvider
    from openknowledge.providers.azure_openai import AzureOpenAIProvider
    from openknowledge.providers.openai_compat import OpenAICompatProvider

    settings = _settings(tmp_path, escalation_enabled=True)
    azure = AzureOpenAIProvider(
        endpoint="https://r.openai.azure.com", deployment="my-gpt", api_key="az"
    )
    target = targets(settings, azure)[2]
    assert (target.kind, target.model, target.api_key, target.api_version) == (
        "azure",
        "my-gpt",
        "az",
        "2024-06-01",
    )
    assert target.base_url == "https://r.openai.azure.com/openai/deployments/my-gpt"

    anthropic = AnthropicProvider(model_id="claude-opus-5", api_key="sk")
    target = targets(settings, anthropic)[2]
    assert (target.kind, target.base_url, target.model, target.api_key) == (
        "anthropic",
        ANTHROPIC_API,
        "claude-opus-5",
        "sk",
    )

    compat = OpenAICompatProvider(
        model_id="gpt-x", base_url="https://api.openai.com/v1", api_key="oa", tier="frontier"
    )
    target = targets(settings, compat)[2]
    assert (target.kind, target.base_url, target.api_key) == (
        "openai",
        "https://api.openai.com/v1",
        "oa",
    )


# -- the cache ---------------------------------------------------------------


def test_readings_stand_for_the_ttl_and_fresh_asks_again() -> None:
    clock = [1000.0]
    calls: list[str] = []

    def prober(target: Target, *, timeout: float) -> Reading:
        calls.append(target.role)
        return Reading(role=target.role, configured=True, state="reachable", ok=True)

    monitor = HealthMonitor(ttl=30.0, clock=lambda: clock[0], prober=prober)
    chat = Target("chat", "runtime", "http://x/v1", "m")

    asyncio.run(monitor.readings([chat]))
    asyncio.run(monitor.readings([chat]))
    assert calls == ["chat"], "a second look inside the TTL must not ask the endpoint again"

    clock[0] += 29.0
    asyncio.run(monitor.readings([chat]))
    assert len(calls) == 1
    clock[0] += 1.5
    asyncio.run(monitor.readings([chat]))
    assert len(calls) == 2, "past the TTL the endpoint is asked again"

    asyncio.run(monitor.readings([chat], fresh=True))
    assert len(calls) == 3, "an explicit re-check bypasses the cache"

    # A changed setting is a different target, asked at once.
    asyncio.run(monitor.readings([Target("chat", "runtime", "http://y/v1", "m")]))
    assert len(calls) == 4


def test_every_target_is_asked_and_answered_in_order() -> None:
    def prober(target: Target, *, timeout: float) -> Reading:
        return Reading(role=target.role, configured=True, state="reachable", ok=True)

    monitor = HealthMonitor(prober=prober)
    wanted = [Target("chat", "runtime"), Target("embedding", "off"), Target("escalation", "off")]
    readings = asyncio.run(monitor.readings(wanted))
    assert [r.role for r in readings] == ["chat", "embedding", "escalation"]
