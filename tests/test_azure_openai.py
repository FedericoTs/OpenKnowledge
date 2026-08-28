"""Azure OpenAI as the escalation rung, against a dialect we control.

The provider is a thin subclass - deployments URL, api-version query,
api-key header - so the tests hold exactly those three to the wire, then
prove the part that matters commercially: the operator's own price is the
one the ledger uses, an unpriced deployment is flagged rather than
guessed, and a real escalated answer through /chat is grounded, billed,
and byte-identical from cache on the second ask.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi.testclient import TestClient

from openknowledge.api.app import create_app
from openknowledge.api.engine import _build_frontier
from openknowledge.cascade.router import _price
from openknowledge.config import Settings
from openknowledge.costs import Usage
from openknowledge.providers.azure_openai import AzureOpenAIProvider

ANSWER = "Meals are reimbursed up to EUR 45 per day. [expenses-policy]"


class FakeAzure:
    """A loopback server speaking Azure OpenAI's chat-completions dialect.

    ``reasoning=True`` makes it behave like a gpt-5-family deployment: it
    refuses `max_tokens` (naming `max_completion_tokens`) and refuses a
    pinned `temperature`, one 400 at a time, exactly the way the live
    service teaches its dialect. ``empty_length=True`` answers with no text
    and finish_reason "length" - the reasoning-budget failure shape.
    """

    def __init__(self, *, reasoning: bool = False, empty_length: bool = False) -> None:
        self.requests: list[dict] = []
        self.reasoning = reasoning
        self.empty_length = empty_length
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: object) -> None:  # noqa: D102 - quiet
                pass

            def do_POST(self) -> None:  # noqa: N802 - http.server API
                length = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(length))
                parsed = urlparse(self.path)
                fake.requests.append(
                    {
                        "path": parsed.path,
                        "query": {k: v[0] for k, v in parse_qs(parsed.query).items()},
                        "api_key": self.headers.get("api-key"),
                        "authorization": self.headers.get("Authorization"),
                        "payload": payload,
                    }
                )
                if fake.reasoning and "max_tokens" in payload:
                    self._reject("unsupported_parameter", "max_tokens")
                elif fake.reasoning and "temperature" in payload:
                    self._reject("unsupported_value", "temperature")
                elif fake.empty_length:
                    self._empty_length()
                elif payload.get("stream"):
                    self._stream()
                else:
                    self._complete()

            def _reject(self, code: str, param: str) -> None:
                body = json.dumps(
                    {
                        "error": {
                            "message": f"Unsupported: {param!r} is not supported with this model.",
                            "type": "invalid_request_error",
                            "param": param,
                            "code": code,
                        }
                    }
                ).encode()
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _empty_length(self) -> None:
                body = json.dumps(
                    {
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": ""},
                                "finish_reason": "length",
                            }
                        ],
                        "usage": {"prompt_tokens": 812, "completion_tokens": 1500},
                    }
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _complete(self) -> None:
                body = json.dumps(
                    {
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": ANSWER},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {"prompt_tokens": 812, "completion_tokens": 46},
                    }
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _stream(self) -> None:
                chunks = [
                    {"choices": [{"delta": {"content": "Meals are reimbursed"}}]},
                    {
                        "choices": [
                            {
                                "delta": {"content": " up to EUR 45 per day."},
                                "finish_reason": "stop",
                            }
                        ]
                    },
                    {"choices": [], "usage": {"prompt_tokens": 812, "completion_tokens": 46}},
                ]
                body = (
                    "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.endpoint = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def azure():
    fake = FakeAzure()
    yield fake
    fake.close()


def _provider(azure: FakeAzure, **kw: object) -> AzureOpenAIProvider:
    values: dict = {
        "endpoint": azure.endpoint,
        "deployment": "kb-answers",
        "api_key": "azure-key-1",
    }
    values.update(kw)
    return AzureOpenAIProvider(**values)


# -- the dialect, held to the wire ------------------------------------------


async def test_the_dialect_is_azure_shaped(azure) -> None:
    completion = await _provider(azure).complete(
        system="Answer from the passages.", context="passages", question="meal limit?"
    )
    sent = azure.requests[0]
    assert sent["path"] == "/openai/deployments/kb-answers/chat/completions"
    assert sent["query"] == {"api-version": "2024-06-01"}
    assert sent["api_key"] == "azure-key-1"
    assert sent["authorization"] is None, "Azure wants api-key, not a bearer"
    assert sent["payload"]["temperature"] == 0
    assert sent["payload"]["messages"][0]["role"] == "system"
    assert completion.text == ANSWER
    assert completion.model_id == "kb-answers"
    assert completion.usage == Usage(input_tokens=812, output_tokens=46)


async def test_streaming_speaks_the_same_dialect(azure) -> None:
    deltas: list[str] = []
    final = None
    async for event in _provider(azure).stream(system="s", context="c", question="q"):
        if isinstance(event, str):
            deltas.append(event)
        else:
            final = event
    assert "".join(deltas) == "Meals are reimbursed up to EUR 45 per day."
    assert final is not None and final.usage == Usage(input_tokens=812, output_tokens=46)


def test_azure_is_never_treated_as_self_hosted(azure) -> None:
    # The fake lives on loopback, which the URL heuristic calls self-hosted;
    # Azure bills per token whatever the hostname looks like.
    assert _provider(azure).self_hosted is False


# -- the price is the operator's, or it is flagged --------------------------


def test_the_operators_price_is_the_one_the_ledger_uses(azure) -> None:
    priced = _provider(azure, input_per_mtok=2.50, output_per_mtok=10.00)
    usage = Usage(input_tokens=812, output_tokens=46)
    dollars, notes = _price(usage, priced)
    assert dollars == pytest.approx((812 * 2.50 + 46 * 10.00) / 1_000_000)
    assert notes == ()


def test_an_unpriced_deployment_is_flagged_not_guessed(azure) -> None:
    dollars, notes = _price(Usage(input_tokens=812, output_tokens=46), _provider(azure))
    assert dollars == 0.0
    assert notes and "cost not counted" in notes[0]


def test_half_a_price_is_no_price(azure) -> None:
    assert _provider(azure, input_per_mtok=2.50).price_override is None


# -- the frontier builder ---------------------------------------------------


def _azure_settings(azure: FakeAzure, **overrides: object) -> Settings:
    values: dict = {
        "escalation_enabled": True,
        "escalation_provider": "azure",
        "azure_openai_endpoint": azure.endpoint,
        "azure_openai_deployment": "kb-answers",
        "azure_openai_api_key": "azure-key-1",
        "local_enabled": False,
        "embedding_enabled": False,
        "_env_file": None,
    }
    values.update(overrides)
    return Settings(**values)


def test_the_frontier_builds_only_when_azure_is_fully_named(azure, tmp_path) -> None:
    built = _build_frontier(_azure_settings(azure, data_dir=str(tmp_path)))
    assert isinstance(built, AzureOpenAIProvider)
    assert built.model_id == "kb-answers"
    assert (
        _build_frontier(_azure_settings(azure, data_dir=str(tmp_path), azure_openai_api_key=None))
        is None
    ), "a half-configured escalation must stay local, loudly"


# -- the whole cascade, escalated -------------------------------------------


def test_an_escalated_answer_is_grounded_billed_and_then_cached(azure, tmp_path) -> None:
    docs = tmp_path / "documents"
    docs.mkdir()
    (docs / "expenses-policy.md").write_text(
        "# Expenses Policy\n\nMeals are reimbursed up to EUR 45 per day.\n",
        encoding="utf-8",
    )
    settings = _azure_settings(
        azure,
        data_dir=str(tmp_path / "data"),
        documents_dir=str(docs),
        azure_openai_input_per_mtok=2.50,
        azure_openai_output_per_mtok=10.00,
    )
    with TestClient(create_app(settings)) as client:
        first = client.post(
            "/chat", json={"question": "How much are meals reimbursed per day?"}
        ).json()
        assert first["tier"] == "frontier"
        assert first["model"] == "kb-answers"
        assert first["grounded"] is True
        assert "45" in first["answer"]
        assert first["cost_usd"] == pytest.approx((812 * 2.50 + 46 * 10.00) / 1_000_000)

        again = client.post(
            "/chat", json={"question": "How much are meals reimbursed per day?"}
        ).json()
        assert again["tier"] == "exact"
        assert again["answer"] == first["answer"], "cache must be byte-identical"
        assert again["cost_usd"] == 0.0
        assert len([r for r in azure.requests if not r["payload"].get("stream")]) == 1, (
            "the second ask must not reach Azure at all"
        )


# -- reasoning-family deployments (gpt-5*, o*) -------------------------------


@pytest.fixture
def reasoning_azure():
    fake = FakeAzure(reasoning=True)
    yield fake
    fake.close()


async def test_a_reasoning_deployment_teaches_its_dialect(reasoning_azure) -> None:
    """gpt-5-family refuses `max_tokens` and a pinned temperature. The 400s
    name the offender, the provider adopts the dialect and retries - and the
    next call speaks it directly, no wasted round trips."""
    provider = _provider(reasoning_azure)
    completion = await provider.complete(system="s", context="c", question="q", max_tokens=350)
    assert completion.text == ANSWER

    payloads = [r["payload"] for r in reasoning_azure.requests]
    assert len(payloads) == 3  # refused max_tokens, refused temperature, accepted
    final = payloads[-1]
    assert "max_tokens" not in final and "temperature" not in final
    # The thinking spends from the same budget as the answer; headroom is
    # what stops a 350-token cap from producing an empty reply.
    assert final["max_completion_tokens"] == 350 + 1500

    await provider.complete(system="s", context="c", question="q", max_tokens=350)
    assert len(reasoning_azure.requests) == 4  # exactly one more - remembered


async def test_the_stream_path_learns_the_same_dialect(reasoning_azure) -> None:
    deltas: list[str] = []
    final = None
    async for event in _provider(reasoning_azure).stream(system="s", context="c", question="q"):
        if isinstance(event, str):
            deltas.append(event)
        else:
            final = event
    assert final is not None and "EUR 45" in final.text
    accepted = reasoning_azure.requests[-1]["payload"]
    assert accepted["stream"] is True
    assert "max_tokens" not in accepted and "max_completion_tokens" in accepted


async def test_an_empty_reply_that_ran_out_of_budget_names_the_fix() -> None:
    from openknowledge.providers.base import ProviderError

    fake = FakeAzure(empty_length=True)
    try:
        with pytest.raises(ProviderError, match="OK_MAX_ANSWER_TOKENS"):
            await _provider(fake).complete(system="s", context="c", question="q")
    finally:
        fake.close()


async def test_v1_selects_the_unversioned_path(azure) -> None:
    completion = await _provider(azure, api_version="v1").complete(
        system="s", context="c", question="q"
    )
    sent = azure.requests[0]
    assert sent["path"] == "/openai/v1/chat/completions"
    assert sent["query"] == {}
    assert sent["payload"]["model"] == "kb-answers"
    assert sent["api_key"] == "azure-key-1"
    assert completion.text == ANSWER
