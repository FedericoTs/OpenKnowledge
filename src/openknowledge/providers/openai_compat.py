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
        """Refuse a prompt too large for the window, instead of having it truncated.

        Local runtimes do not error on an over-long prompt: they drop tokens off
        the front, which is where the system prompt's grounding rules live, and
        answer from the remainder. The answer that comes back looks ordinary and
        is ungrounded. Raising here is loud, and the cascade treats it as this
        rung failing - so the question moves up a rung rather than being answered
        badly and cached.

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
            f"is {self.context_tokens:,}. The runtime would drop the start of it without "
            f"saying so, so this was not sent. Either give the model a larger window "
            f"(`openknowledge model use {self.model_id} --context {suggested}`) "
            f"or lower OK_RETRIEVAL_K."
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
