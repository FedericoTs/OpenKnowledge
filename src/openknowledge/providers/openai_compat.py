"""OpenAI-compatible chat-completions provider.

One adapter covers a surprising amount of ground: OpenAI itself, plus Ollama,
vLLM, LM Studio, llama.cpp's server, and most on-prem inference stacks, all of
which expose ``/v1/chat/completions``. That is why this is the local path - an
admin who wants a private model points ``base_url`` at their own box and nothing
else in OpenKnowledge changes.
"""

from __future__ import annotations

import httpx

from ..costs import Usage
from .base import Completion, Message, ProviderError


class OpenAICompatProvider:
    """Chat provider for any ``/v1/chat/completions`` endpoint."""

    def __init__(
        self,
        *,
        model_id: str,
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        tier: str = "local",
        timeout: float = 120.0,
    ) -> None:
        self.model_id = model_id
        self.tier = tier
        self.base_url = base_url.rstrip("/")
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
