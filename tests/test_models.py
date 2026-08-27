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
        #: When set, /api/pull streams this error instead of succeeding.
        self.fail_pull_with: str | None = None


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
                if state.fail_pull_with:
                    return self._send(200, {"error": state.fail_pull_with})
                state.models[body["model"]] = 40_960
                if not body.get("stream", True):
                    return self._send(200, {"status": "success"})
                # Ollama streams newline-delimited progress, not one object at
                # the end. The stub does too, or the progress path is untested.
                lines = [
                    {"status": "pulling manifest"},
                    {"status": "pulling 1a2b", "completed": 1_000_000, "total": 5_000_000},
                    {"status": "pulling 1a2b", "completed": 5_000_000, "total": 5_000_000},
                    {"status": "success"},
                ]
                payload = ("\n".join(json.dumps(row) for row in lines) + "\n").encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return None
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


def test_the_plain_command_pins_a_window_rather_than_trusting_the_default(
    base_url: str, state: _State
) -> None:
    """`model use qwen3:8b` has to produce a configuration that is actually true.

    Two wrong answers were possible here and this used to give the first:

      - record the declared 40,960, which Ollama will not use. The fit check
        then waves through prompts ten times too large for what is allocated,
        and they get truncated from the front - where the grounding rules are.
      - record nothing, and leave the runtime on its 4,096 default. That is
        honest but marginal: k=6 retrieval plus 1,500 answer tokens comes to
        about 3,800, which fits until one longer document does not.

    So the plain command pins a real window and says it did.
    """
    result = switch(probe(base_url), "qwen3:8b")

    assert state.created, "left the window to chance"
    assert state.created[0]["parameters"] == {"num_ctx": 8192}
    assert result.model == "qwen3-8b-ok8192"
    assert result.context == 8192
    assert result.native_context == 40_960  # what the weights allow, for the note
    assert any("pinned at 8,192" in note for note in result.notes)
    assert any("--no-pin" in note for note in result.notes)


def test_pinning_can_be_declined(base_url: str, state: _State) -> None:
    """--no-pin leaves the runtime alone, and then records nothing at all.

    Recording a number without pinning it would describe nothing, so the fit
    check stays off rather than checking against a guess.
    """
    result = switch(probe(base_url), "qwen3:8b", pin=False)

    assert state.created == []
    assert result.model == "qwen3:8b"
    assert result.context is None
    assert any("no window recorded" in note for note in result.notes)


def test_a_model_already_pinned_is_left_as_it_is(base_url: str, state: _State) -> None:
    """Re-running the plain command must not re-pin a deliberate choice back to
    the default."""
    runtime = probe(base_url)
    first = switch(runtime, "qwen3:8b", context=32_768)
    state.created.clear()

    again = switch(runtime, first.model)
    assert state.created == []
    assert again.context == 32_768


def test_an_absent_model_is_downloaded_first(base_url: str, state: _State) -> None:
    result = switch(probe(base_url), "qwen3:14b")
    assert state.pulled == ["qwen3:14b"]
    assert result.pulled


def test_the_download_reports_progress_while_it_runs(base_url: str) -> None:
    """Five gigabytes behind a silent request looks exactly like a hung command.

    Asking Ollama for a pull without `stream` returns one response at the end,
    which is what this used to do: the terminal sat blank for ten minutes with
    no way to tell a slow download from a dead one.
    """
    seen: list[tuple[str, int, int]] = []
    switch(probe(base_url), "qwen3:14b", on_progress=lambda *row: seen.append(row))

    assert len(seen) >= 3, "progress arrived only at the end, or not at all"
    assert seen[0][0] == "pulling manifest"
    assert seen[-1][0] == "success"
    # Enough to render a percentage, not just a spinner.
    assert any(total and 0 < done < total for _, done, total in seen)


def test_a_failed_download_is_reported_from_the_stream(base_url: str, state: _State) -> None:
    """Ollama reports a bad model name inside the stream, with a 200 status."""
    from openknowledge.models import pull

    state.fail_pull_with = "model 'nope:1b' not found"
    with pytest.raises(ModelError, match="not found"):
        pull(probe(base_url), "nope:1b")


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


def test_a_window_under_the_declared_length_is_still_pinned(base_url: str, state: _State) -> None:
    """Asking for less than the weights allow still has to build the model.

    This used to skip the build, reasoning that a smaller window than the
    weights allow is "a downgrade dressed as a setting". That was wrong:
    pinning num_ctx is the only thing that decides what Ollama allocates, so
    recording 32,768 without pinning it described nothing at all.
    """
    result = switch(probe(base_url), "qwen3:8b", context=32_768)

    assert state.created, "recorded a window without making it true"
    assert state.created[0]["parameters"] == {"num_ctx": 32_768}
    assert result.model == "qwen3-8b-ok32768"
    assert result.context == 32_768


def test_asking_for_the_window_already_pinned_builds_nothing(base_url: str, state: _State) -> None:
    """Idempotent: re-running the same command is a no-op, not a rebuild."""
    runtime = probe(base_url)
    first = switch(runtime, "qwen3:8b", context=16_384)
    state.created.clear()

    again = switch(runtime, first.model, context=16_384)
    assert state.created == []
    assert again.model == first.model
    assert again.context == 16_384


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

    window = context_window(runtime, "qwen3-30b-ok131072")
    assert window.pinned == 131_072, "the pinned window is what it will run with"
    assert window.declared == 40_960, "and the weights still say what they say"


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


def test_download_progress_is_not_gated_on_having_a_terminal(
    base_url: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reported from Git Bash: `model use` printed nothing at all.

    The progress line was only emitted when stderr.isatty(), and MinTTY is a
    named pipe, so Python reports False there - making a five-gigabyte download
    completely silent on the platform least able to tell it apart from a hung
    command. Rewriting a line in place needs a terminal; saying something does
    not, and only the first is conditional now.
    """
    from openknowledge.cli import main

    monkeypatch.setenv("OK_LOCAL_BASE_URL", base_url)
    assert main(["model", "use", "qwen3:14b", "--env-file", str(tmp_path / ".env")]) == 0

    # capsys makes stderr a pipe, which is exactly the condition that silenced it.
    progress = capsys.readouterr().err
    assert "%" in progress, f"no progress reached a non-terminal stderr: {progress!r}"
    assert "pulling" in progress


def test_interrupting_a_download_says_what_survives_it(
    base_url: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ctrl-C during a five-gigabyte pull is normal, not a crash.

    A traceback here reads as "something went wrong and you have probably lost
    it all", when in fact Ollama keeps every block already fetched.
    """
    from openknowledge import models as local_models
    from openknowledge.cli import main

    monkeypatch.setenv("OK_LOCAL_BASE_URL", base_url)

    def interrupted(*_: object, **__: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(local_models, "pull", interrupted)
    code = main(["model", "use", "qwen3:14b", "--env-file", str(tmp_path / ".env")])

    assert code == 130  # the conventional exit status for an interrupt
    message = capsys.readouterr().err
    assert "stopped" in message
    assert "resumes rather than starting over" in message
    assert not (tmp_path / ".env").exists(), "an interrupted switch must record nothing"
