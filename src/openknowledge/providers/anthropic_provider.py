"""Anthropic provider, with prompt caching wired up correctly.

Caching is the single biggest lever on the paid tier, and it is easy to get
backwards here. Our prompt has the shape "big fixed instructions, then retrieved
context, then this user's question" - a shared prefix with a varying suffix. So:

* The **system prompt** is byte-identical on every call and carries an explicit
  cache breakpoint. Repeat calls read it at ~0.1x input price.
* The **retrieved context and the question** get no breakpoint. They differ every
  call, so marking them would pay the ~1.25x write premium on bytes nobody ever
  reads back - a pure surcharge that looks like caching in the code and shows up
  as a bigger bill.

That is also why we do not use top-level automatic caching: it places its
breakpoint at the end of the prompt, which here is the unique question.

Two other things this file is deliberate about:

* No ``temperature``. Current Anthropic models reject sampling parameters
  outright, and reproducibility comes from the cache anyway.
* ``effort="low"``. Grounded extractive Q&A over retrieved text is not a
  reasoning-heavy route; high effort buys nothing here and costs output tokens.
"""

from __future__ import annotations

from typing import Any

from ..costs import Usage
from .base import Completion, Message, ProviderError

#: Below this the API silently declines to cache (varies by model; 512 is the
#: Opus 5 floor). A system prompt shorter than this gets no cache benefit however
#: the cache_control marker is placed, which is worth surfacing rather than
#: leaving as a mystery in the usage numbers - and worth checking against, which
#: `tools/measure_prompts.py` does. This project's own system prompt currently
#: measures under it, so prompt caching is priced at zero rather than assumed.
CACHE_MIN_TOKENS = 512


class AnthropicProvider:
    """Frontier tier backed by the Anthropic Messages API."""

    def __init__(
        self,
        *,
        model_id: str = "claude-opus-5",
        api_key: str | None = None,
        tier: str = "frontier",
        effort: str = "low",
        timeout: float = 120.0,
    ) -> None:
        self.model_id = model_id
        self.tier = tier
        self._effort = effort
        self._timeout = timeout
        self._api_key = api_key
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from anthropic import AsyncAnthropic
            except ImportError as exc:  # pragma: no cover - depends on extras
                raise ProviderError(
                    "The Anthropic provider needs the SDK: pip install 'openknowledge[anthropic]'"
                ) from exc
            kwargs: dict[str, Any] = {"timeout": self._timeout}
            if self._api_key:
                kwargs["api_key"] = self._api_key
            self._client = AsyncAnthropic(**kwargs)
        return self._client

    async def complete(
        self,
        *,
        system: str,
        context: str,
        question: str,
        history: tuple[Message, ...] = (),
        max_tokens: int = 1500,
    ) -> Completion:
        client = self._get_client()

        messages: list[dict[str, Any]] = [
            {"role": m.role, "content": [{"type": "text", "text": m.content}]} for m in history
        ]
        messages.append(
            {
                "role": "user",
                "content": [
                    # No cache_control on either block: both vary per question.
                    {"type": "text", "text": context},
                    {"type": "text", "text": f"Question: {question}"},
                ],
            }
        )

        try:
            resp = await client.messages.create(
                model=self.model_id,
                max_tokens=max_tokens,  # deliberate cost cap; answers are short
                system=[
                    {
                        "type": "text",
                        "text": system,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=messages,
                thinking={"type": "adaptive"},
                output_config={"effort": self._effort},
            )
        except Exception as exc:  # noqa: BLE001 - surfaced to the cascade as one type
            raise ProviderError(f"{self.model_id}: {exc}") from exc

        # Always check stop_reason before reading content: a safety refusal is an
        # HTTP 200 with no usable answer, and treating it as text would put the
        # refusal string into the cache as if it were a policy answer.
        stop_reason = getattr(resp, "stop_reason", None)
        if stop_reason == "refusal":
            raise ProviderError(f"{self.model_id}: request refused by the model")

        text = "".join(
            block.text for block in resp.content if getattr(block, "type", None) == "text"
        ).strip()

        raw = resp.usage
        usage = Usage(
            input_tokens=getattr(raw, "input_tokens", 0) or 0,
            output_tokens=getattr(raw, "output_tokens", 0) or 0,
            cache_read_tokens=getattr(raw, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(raw, "cache_creation_input_tokens", 0) or 0,
        )
        return Completion(text=text, usage=usage, model_id=self.model_id, stop_reason=stop_reason)

    @staticmethod
    def cache_health(usage: Usage) -> str:
        """Human-readable check that caching is actually working.

        The costliest caching failure is silent - requests keep succeeding, the
        bill is just higher. If reads are zero across repeated calls, something
        upstream is rewriting the system prompt (an interpolated timestamp is the
        classic culprit).
        """
        if usage.cache_read_tokens > 0:
            return f"ok: {usage.cache_read_tokens} tokens served from cache"
        if usage.cache_write_tokens > 0:
            return "cold: cache written this call, reads should appear on the next one"
        return (
            "MISS: no cache activity. Either the system prompt changes between calls "
            f"(check for interpolated dates/IDs) or it is under ~{CACHE_MIN_TOKENS} "
            "tokens, below the minimum cacheable prefix."
        )
