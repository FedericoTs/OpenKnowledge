"""Switching the local model, against a stub that answers like Ollama does.

There is no mocking here on purpose. The thing being tested is a conversation
with an HTTP API - which endpoint, which JSON shape, which field the window
comes back in - and a mock would only assert that this file agrees with itself.
A real server on a real socket catches a wrong URL or a wrong payload key, which
is exactly the class of bug that would make `model use --context` look like it
worked and change nothing.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from openknowledge.models import (
    ModelError,
    derived_name,
    installed,
    probe,
    switch,
    write_env,
)


class _State:
    """What the stub runtime believes about itself."""

    def __init__(self) -> None:
        self.models: dict[str, int] = {"qwen3:8b": 40_960}
        self.parameters: dict[str, str] = {}
        self.created: list[dict[str, object]] = []
        self.pulled: list[str] = []
        self.is_ollama = True


@pytest.fixture
def state() -> _State:
    return _State()


@pytest.fixture
def base_url(state: _State) -> Iterator[str]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_: object) -> None:  # keep the test output readable
            pass

        def _send(self, code: int, body: object) -> None:
            payload = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:  # noqa: N802 - http.server's spelling
            if self.path == "/api/version" and state.is_ollama:
                return self._send(200, {"version": "0.12.0"})
            if self.path == "/v1/models":
                return self._send(200, {"data": [{"id": name} for name in sorted(state.models)]})
            if self.path == "/api/tags" and state.is_ollama:
                return self._send(
                    200,
                    {
                        "models": [
                            {"model": name, "size": 5_200_000_000} for name in sorted(state.models)
                        ]
                    },
                )
            self._send(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")

            if self.path == "/api/show":
                name = body.get("model", "")
                if name not in state.models:
                    return self._send(404, {"error": "model not found"})
                return self._send(
                    200,
                    {
                        "parameters": state.parameters.get(name, ""),
                        "model_info": {
                            "general.architecture": "qwen3",
                            "qwen3.context_length": state.models[name],
                        },
                    },
                )
            if self.path == "/api/pull":
                state.pulled.append(body.get("model", ""))
                state.models[body["model"]] = 40_960
                return self._send(200, {"status": "success"})
            if self.path == "/api/create":
                state.created.append(body)
                name = body["model"]
                state.models[name] = state.models.get(body.get("from", ""), 40_960)
                state.parameters[name] = f"num_ctx {body['parameters']['num_ctx']}"
                return self._send(200, {"status": "success"})
            self._send(404, {"error": "not found"})

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/v1"
    server.shutdown()
    server.server_close()


def test_it_recognises_ollama_rather_than_assuming_it(base_url: str) -> None:
    runtime = probe(base_url)
    assert runtime.is_ollama
    assert runtime.version == "0.12.0"
    assert runtime.root.endswith("/v1") is False


def test_a_runtime_without_ollamas_api_is_still_usable(base_url: str, state: _State) -> None:
    """vLLM and llama.cpp serve /v1/models and nothing else. That is not a failure."""
    state.is_ollama = False
    runtime = probe(base_url)
    assert runtime.kind == "openai-compatible"
    assert {m.name for m in installed(runtime)} == {"qwen3:8b"}


def test_nothing_listening_is_reported_not_guessed() -> None:
    runtime = probe("http://127.0.0.1:9/v1", timeout=1.0)
    assert not runtime.reachable
    with pytest.raises(ModelError, match="nothing is answering"):
        switch(runtime, "qwen3:8b")


def test_switching_to_an_installed_model_reads_its_window(base_url: str) -> None:
    result = switch(probe(base_url), "qwen3:8b")
    assert result.model == "qwen3:8b"
    assert result.context == 40_960  # asked for, not assumed
    assert not result.pulled


def test_an_absent_model_is_downloaded_first(base_url: str, state: _State) -> None:
    result = switch(probe(base_url), "qwen3:14b")
    assert state.pulled == ["qwen3:14b"]
    assert result.pulled


def test_downloading_can_be_refused(base_url: str, state: _State) -> None:
    with pytest.raises(ModelError, match="not installed"):
        switch(probe(base_url), "qwen3:14b", allow_download=False)
    assert state.pulled == []


def test_a_bigger_window_builds_a_model_that_carries_it(base_url: str, state: _State) -> None:
    """The point of the whole module: Ollama has no per-call context setting.

    If this ever silently stopped issuing /api/create, `model use --context`
    would keep printing a large number and the model would keep running with
    its default window - which is the failure mode the command exists to close.
    """
    result = switch(probe(base_url), "qwen3:30b", context=131_072)

    assert state.created == [
        {
            "model": "qwen3-30b-ok131072",
            "from": "qwen3:30b",
            "parameters": {"num_ctx": 131_072},
            "stream": False,
        }
    ]
    assert result.model == "qwen3-30b-ok131072"
    assert result.context == 131_072
    assert result.derived_from == "qwen3:30b"
    assert any("beyond the 40,960" in note for note in result.notes)


def test_a_window_the_model_already_has_builds_nothing(base_url: str, state: _State) -> None:
    result = switch(probe(base_url), "qwen3:8b", context=32_768)
    assert state.created == []
    assert result.context == 32_768
    assert result.model == "qwen3:8b"


def test_the_derived_model_reports_the_window_it_was_built_with(
    base_url: str, state: _State
) -> None:
    """A derived model's num_ctx must win over what the weights declare.

    Otherwise `model status` reads back the base model's 40,960 and reports a
    mismatch against the 131,072 that is genuinely in force.
    """
    from openknowledge.models import context_window

    runtime = probe(base_url)
    switch(runtime, "qwen3:30b", context=131_072)
    assert context_window(runtime, "qwen3-30b-ok131072") == 131_072


def test_a_non_ollama_runtime_says_what_to_relaunch(base_url: str, state: _State) -> None:
    state.is_ollama = False
    result = switch(probe(base_url), "qwen3:8b", context=65_536)
    assert result.context == 65_536
    # All three spellings, because a flag that is nearly right fails at launch.
    assert any("--max-model-len 65536" in note for note in result.notes)
    assert any("--ctx-size 65536" in note for note in result.notes)
    assert any("--n_ctx 65536" in note for note in result.notes)


def test_derived_names_are_stable_and_legal() -> None:
    assert derived_name("qwen3:30b", 131_072) == "qwen3-30b-ok131072"
    assert derived_name("Qwen3:30B", 131_072) == derived_name("qwen3:30b", 131_072)
    assert (
        derived_name("hf.co/user/Model-GGUF:Q4_K_M", 8192) == "hf.co-user-model-gguf-q4_k_m-ok8192"
    )


def test_recording_the_choice_keeps_the_operators_file_intact(tmp_path: Path) -> None:
    """An operator's .env has their comments and ordering in it. Keep both."""
    env = tmp_path / ".env"
    env.write_text(
        "# my notes\nOK_DOCUMENTS_DIR=./policies\nOK_LOCAL_MODEL=qwen3:4b\n# trailing note\n"
    )
    changed = write_env(env, {"OK_LOCAL_MODEL": "qwen3:8b", "OK_LOCAL_CONTEXT_TOKENS": "40960"})

    assert changed == ["OK_LOCAL_MODEL", "OK_LOCAL_CONTEXT_TOKENS"]
    assert env.read_text() == (
        "# my notes\n"
        "OK_DOCUMENTS_DIR=./policies\n"
        "OK_LOCAL_MODEL=qwen3:8b\n"
        "# trailing note\n"
        "OK_LOCAL_CONTEXT_TOKENS=40960\n"
    )


def test_recording_an_unchanged_value_reports_no_change(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("OK_LOCAL_MODEL=qwen3:8b\n")
    assert write_env(env, {"OK_LOCAL_MODEL": "qwen3:8b"}) == []


# --- the reason the window is recorded at all --------------------------------


def test_a_prompt_too_big_for_the_window_is_refused_not_truncated() -> None:
    """The whole point of recording the window.

    A local runtime given an over-long prompt does not error - it drops tokens
    off the front, which is where the grounding rules are, and answers from the
    remainder. That answer looks ordinary and is ungrounded. Refusing here makes
    the cascade treat the rung as failed and move up, instead of caching a
    confident wrong answer.
    """
    import asyncio

    from openknowledge.providers.base import ProviderError
    from openknowledge.providers.openai_compat import OpenAICompatProvider

    provider = OpenAICompatProvider(
        model_id="qwen3:8b",
        base_url="http://127.0.0.1:9/v1",  # never reached: the check fires first
        context_tokens=4096,
    )
    with pytest.raises(ProviderError) as raised:
        asyncio.run(
            provider.complete(system="s" * 400, context="c" * 12_000, question="q", max_tokens=1500)
        )
    message = str(raised.value)
    assert "window is 4,096" in message
    assert "--context 8192" in message  # the next size up that would fit


def test_an_unrecorded_window_checks_nothing() -> None:
    """Zero means unknown. Refusing on a guess would be worse than not checking."""
    import asyncio

    from openknowledge.providers.base import ProviderError
    from openknowledge.providers.openai_compat import OpenAICompatProvider

    provider = OpenAICompatProvider(
        model_id="qwen3:8b", base_url="http://127.0.0.1:9/v1", context_tokens=0
    )
    with pytest.raises(ProviderError) as raised:
        asyncio.run(provider.complete(system="s" * 400, context="c" * 99_000, question="q"))
    assert "window" not in str(raised.value)  # it failed on the connection, not the fit
