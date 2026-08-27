"""Test doubles."""

from __future__ import annotations

from dataclasses import dataclass, field

from openknowledge.costs import Usage
from openknowledge.providers.base import Completion, Message, ProviderError


@dataclass
class FakeProvider:
    """A scripted provider, so cascade tests exercise routing rather than a model."""

    model_id: str = "fake"
    tier: str = "local"
    replies: list[str] = field(default_factory=list)
    fail: bool = False
    calls: list[str] = field(default_factory=list)
    usage: Usage = field(default_factory=lambda: Usage(input_tokens=3000, output_tokens=200))

    async def complete(
        self,
        *,
        system: str,
        context: str,
        question: str,
        history: tuple[Message, ...] = (),
        max_tokens: int = 1500,
    ) -> Completion:
        self.calls.append(question)
        if self.fail:
            raise ProviderError(f"{self.model_id} is unavailable")
        reply = self.replies.pop(0) if self.replies else "I don't know."
        return Completion(text=reply, usage=self.usage, model_id=self.model_id)
