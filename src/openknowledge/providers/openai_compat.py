"""OpenAI-compatible chat-completions provider.

One adapter covers a surprising amount of ground: OpenAI itself, plus Ollama,
vLLM, LM Studio, llama.cpp's server, and most on-prem inference stacks, all of
which expose ``/v1/chat/completions``. That is why this is the local path - an
admin who wants a private model points ``base_url`` at their own box and nothing
else in OpenKnowledge changes.

The same adapter also reaches the open-weight serverless providers - Together,
Groq, DeepInfra, Novita, Fireworks - which are the cheapest way to run a small
model when the documents are allowed to leave the building. That is a feature
and a hazard: those endpoints *bill per token*, and treating them as "local"
would report $0 for calls that really cost money.

So this provider decides, from its own base URL, whether it is talking to a box
you own or a service that invoices you, and says so through :attr:`self_hosted`.
Anything that is not unmistakably a loopback or private address is treated as
billed. Getting it wrong in that direction is cheap - a cost note - while the
other direction silently corrupts the ledger.
"""

from __future__ import annotations

import ipaddress
import json
from collections.abc import AsyncIterator
from urllib.parse import urlparse

import httpx

from ..costs import Usage
from .base import Completion, Message, ProviderError


def is_self_hosted(base_url: str) -> bool:
    """Whether ``base_url`` is a machine the operator runs themselves.

    Loopback, link-local, and RFC1918 addresses are self-hosted; so is a bare
    hostname with no dots, which is how a container reaches a sibling service.
    Everything else - every public hostname - is assumed to bill per token.
    """
    host = urlparse(base_url).hostname
    if not host:
        return False
    if host in ("localhost", "host.docker.internal") or host.endswith(".local"):
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        # A bare name with no dot is a container or LAN hostname, not a vendor.
        return "." not in host
    return address.is_loopback or address.is_private or address.is_link_local


class OpenAICompatProvider:
    """Chat provider for any ``/v1/chat/completions`` endpoint."""

    def __init__(
        self,
        *,
        model_id: str,
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        tier: str = "local",
        self_hosted: bool | None = None,
        timeout: float = 120.0,
        context_tokens: int = 0,
    ) -> None:
        self.model_id = model_id
        self.tier = tier
        self.base_url = base_url.rstrip("/")
        #: The window this endpoint will run with, when it is known. Zero means
        #: unknown and nothing is checked - see :meth:`_check_fit`.
        self.context_tokens = context_tokens
        #: True when there is no per-token invoice behind this endpoint. Decides
        #: whether the cascade prices a call at zero or at this model's rate.
        self.self_hosted = is_self_hosted(base_url) if self_hosted is None else self_hosted
        self._api_key = api_key
        self._timeout = timeout

    def _check_fit(self, prompt_chars: int, max_tokens: int) -> None:
        """Decide locally whether the prompt fits, rather than finding out remotely.

        Runtimes disagree about what an over-long prompt means, and nothing in
        the OpenAI-compatible API says which kind you have. Measured against
        llama-cpp-python 0.3.35 at an 8,192-token window: 7,916 prompt tokens
        answered normally, and past the window it returned HTTP 400
        ``context_length_exceeded`` and answered nothing. That is the good case.
        The bad case is a runtime that trims the prompt to fit instead, because
        trimming takes tokens off the *front* - where the system prompt's
        grounding rules are - and the answer that comes back looks completely
        ordinary and is ungrounded.

        This check does not assume either behaviour. It makes the outcome the
        same on both: the rung fails, the cascade moves up or refuses, and no
        ungrounded answer is written to the cache. Where the runtime would have
        rejected it anyway, the cost is one saved round trip and an error naming
        the window and the next size up instead of a vendor string.

        Only ever runs when the window is known, which is what `openknowledge
        model use` records. Four characters per token is the same rough figure
        the budget forecast uses; it is an estimate, so the check keeps a tenth
        of the window in hand rather than cutting it fine.
        """
        if self.context_tokens <= 0:
            return
        estimate = prompt_chars // 4 + max_tokens
        usable = int(self.context_tokens * 0.9)
        if estimate <= usable:
            return
        # Round the suggestion up to the next power of two: the sizes runtimes
        # are actually configured with, and it leaves headroom for the estimate.
        suggested = 1 << max(estimate - 1, 1).bit_length()
        raise ProviderError(
            f"{self.model_id}: the prompt needs about {estimate:,} tokens and the window "
            f"is {self.context_tokens:,}, so it was not sent - a runtime that trims rather "
            f"than rejects would have answered from a prompt with its start missing. "
            f"Either give the model a larger window "
            f"(`openknowledge model use {self.model_id} --context {suggested}`) "
            f"or lower OK_RETRIEVAL_K."
        )

    def _messages(
        self, system: str, context: str, question: str, history: tuple[Message, ...]
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [{"role": "system", "content": system}]
        messages.extend({"role": m.role, "content": m.content} for m in history)
        messages.append({"role": "user", "content": f"{context}\n\n---\n\nQuestion: {question}"})
        return messages

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _url(self) -> str:
        """The completions endpoint. One seam, so a dialect that shapes its
        URL differently (Azure's deployments path) overrides this alone."""
        return f"{self.base_url}/chat/completions"

    async def stream(
        self,
        *,
        system: str,
        context: str,
        question: str,
        history: tuple[Message, ...] = (),
        max_tokens: int = 1500,
    ) -> AsyncIterator[str | Completion]:
        """Yield the answer as it generates: text deltas, then one Completion.

        This exists for one reason: on a laptop CPU the local model produces
        six tokens a second, and fifteen silent seconds behind a spinner is
        indistinguishable from a hang. Streaming does not make the answer
        arrive sooner - it makes the wait legible.

        The final Completion carries whatever usage the server reported in its
        terminal chunk (asked for via ``stream_options``). Runtimes that do not
        send one yield zero usage - which is why the cascade streams only the
        self-hosted rung, where nothing is billed per token: a billed rung with
        unreported usage would put a zero into the ledger, and a ledger that
        understates is worse than a spinner.
        """
        self._check_fit(
            len(system) + len(context) + len(question) + sum(len(m.content) for m in history),
            max_tokens,
        )
        payload = {
            "model": self.model_id,
            "messages": self._messages(system, context, question, history),
            "max_tokens": max_tokens,
            "temperature": 0,
            "seed": 0,
            "stream": True,
            # Ask for usage in the final chunk; servers that predate the option
            # ignore it rather than erroring.
            "stream_options": {"include_usage": True},
        }

        collected: list[str] = []
        usage = Usage()
        stop_reason: str | None = None
        try:
            async with (
                httpx.AsyncClient(timeout=self._timeout) as client,
                client.stream(
                    "POST",
                    self._url(),
                    json=payload,
                    headers=self._headers(),
                ) as response,
            ):
                if response.status_code != 200:
                    body = (await response.aread()).decode("utf-8", "replace")[:300]
                    raise ProviderError(f"{self.model_id}: HTTP {response.status_code}: {body}")
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:") :].strip()
                    if not data or data == "[DONE]":
                        continue
                    try:
                        event = json.loads(data)
                    except ValueError as exc:
                        raise ProviderError(
                            f"{self.model_id}: unparseable stream chunk: {data[:120]!r}"
                        ) from exc
                    raw_usage = event.get("usage")
                    if raw_usage:
                        usage = Usage(
                            input_tokens=int(raw_usage.get("prompt_tokens", 0)),
                            output_tokens=int(raw_usage.get("completion_tokens", 0)),
                        )
                    for choice in event.get("choices") or []:
                        if choice.get("finish_reason"):
                            stop_reason = choice["finish_reason"]
                        delta = (choice.get("delta") or {}).get("content")
                        if delta:
                            collected.append(delta)
                            yield delta
        except httpx.TimeoutException as exc:
            raise ProviderError(
                f"{self.model_id}: no response within {self._timeout:.0f}s "
                f"({type(exc).__name__}) while streaming."
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"{self.model_id}: {exc or type(exc).__name__}") from exc

        yield Completion(
            text="".join(collected).strip(),
            usage=usage,
            model_id=self.model_id,
            stop_reason=stop_reason,
        )

    async def complete(
        self,
        *,
        system: str,
        context: str,
        question: str,
        history: tuple[Message, ...] = (),
        max_tokens: int = 1500,
    ) -> Completion:
        self._check_fit(
            len(system) + len(context) + len(question) + sum(len(m.content) for m in history),
            max_tokens,
        )
        payload = {
            "model": self.model_id,
            "messages": self._messages(system, context, question, history),
            "max_tokens": max_tokens,
            "temperature": 0,
            "seed": 0,  # honoured by some backends; harmless where it is not
        }

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(self._url(), json=payload, headers=self._headers())
                resp.raise_for_status()
                data = resp.json()
        except httpx.TimeoutException as exc:
            # httpx timeouts stringify to "", which reached an operator as
            # `local tier unavailable: qwen3-8b-ok8192:` after four minutes of
            # waiting - a blank where the reason should be. Say what happened,
            # and the thing most likely to explain it: the first call after a
            # model changes includes loading it into memory, which on a laptop
            # is minutes before a single token is generated.
            raise ProviderError(
                f"{self.model_id}: no response within {self._timeout:.0f}s "
                f"({type(exc).__name__}). The first call after a model changes also "
                "loads it into memory, which can take minutes on a CPU - raise "
                "OK_LOCAL_TIMEOUT_SECONDS if that is what happened."
            ) from exc
        except httpx.HTTPError as exc:
            # Some httpx errors also carry no message. A class name is a worse
            # reason than a sentence and a better one than nothing.
            raise ProviderError(f"{self.model_id}: {exc or type(exc).__name__}") from exc

        try:
            choice = data["choices"][0]
            text = choice["message"]["content"] or ""
        except (KeyError, IndexError) as exc:
            raise ProviderError(f"{self.model_id}: unexpected response shape: {data!r}") from exc

        raw_usage = data.get("usage") or {}
        usage = Usage(
            input_tokens=int(raw_usage.get("prompt_tokens", 0)),
            output_tokens=int(raw_usage.get("completion_tokens", 0)),
        )
        return Completion(
            text=text.strip(),
            usage=usage,
            model_id=self.model_id,
            stop_reason=choice.get("finish_reason"),
        )
