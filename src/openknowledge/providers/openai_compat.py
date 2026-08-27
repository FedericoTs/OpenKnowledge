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
    ) -> None:
        self.model_id = model_id
        self.tier = tier
        self.base_url = base_url.rstrip("/")
        #: True when there is no per-token invoice behind this endpoint. Decides
        #: whether the cascade prices a call at zero or at this model's rate.
        self.self_hosted = is_self_hosted(base_url) if self_hosted is None else self_hosted
        self._api_key = api_key
        self._timeout = timeout

    async def complete(
        self,
        *,
        system: str,
        context: str,
        question: str,
        history: tuple[Message, ...] = (),
        max_tokens: int = 1500,
    ) -> Completion:
        messages: list[dict[str, str]] = [{"role": "system", "content": system}]
        messages.extend({"role": m.role, "content": m.content} for m in history)
        messages.append({"role": "user", "content": f"{context}\n\n---\n\nQuestion: {question}"})

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        payload = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0,
            "seed": 0,  # honoured by some backends; harmless where it is not
        }

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    f"{self.base_url}/chat/completions", json=payload, headers=headers
                )
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as exc:
            raise ProviderError(f"{self.model_id}: {exc}") from exc

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
