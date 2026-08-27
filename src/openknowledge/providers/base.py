"""Provider interface.

Everything the cascade needs from a model is "given a fixed system prompt and
some retrieved context, produce an answer, and tell me what it cost". Keeping
that surface small is what lets a self-hosted 8B model and a frontier API sit in
the same pipeline behind the same grounding checks.

All providers run at temperature 0. That is necessary for reproducibility but
not sufficient - batching and kernel non-determinism mean even a greedy decode
can vary between runs. Reproducibility comes from the cache, not from the
sampler; temperature 0 just stops us making it worse.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from ..costs import Usage


class ProviderError(RuntimeError):
    """A provider call failed in a way the cascade should handle (usually: escalate)."""


@dataclass(frozen=True, slots=True)
class Message:
    role: Literal["user", "assistant"]
    content: str


@dataclass(frozen=True, slots=True)
class Completion:
    text: str
    usage: Usage
    model_id: str
    #: Provider-reported reason the generation stopped, for debugging truncation.
    stop_reason: str | None = None


@runtime_checkable
class ChatProvider(Protocol):
    """Minimal contract every backend implements."""

    model_id: str
    tier: str

    async def complete(
        self,
        *,
        system: str,
        context: str,
        question: str,
        history: tuple[Message, ...] = (),
        max_tokens: int = 1500,
    ) -> Completion:
        """Answer ``question`` using ``context``, following ``system``.

        ``system`` and ``context`` are passed separately rather than
        pre-concatenated so providers that support prompt caching can place a
        cache breakpoint between them - ``system`` is identical on every call and
        is worth caching; ``context`` is not.
        """
        ...
