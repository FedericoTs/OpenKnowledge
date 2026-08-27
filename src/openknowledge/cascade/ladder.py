"""The escalation ladder: which models this deployment may climb, in order.

The cascade used to have exactly two model tiers, a cheap one and a frontier
one, which made every grounding failure cost frontier prices. Measured on a real
corpus that is $0.037 a question, against $0.0009 for a mid-size open-weight
model that would have answered most of them - so the gap between the rungs was
costing more than the rungs themselves.

A ladder fixes that by letting an operator put rungs wherever they like:

    self-hosted -> gpt-oss-20b -> gpt-oss-120b -> claude-opus-5

Each rung answers from the same retrieved passages under the same system prompt
and is graded by the same grounding gate. That invariant is the whole reason
climbing is safe: a rung's answer either passes the gate or nobody sees it, so
adding a cheap rung can lower the bill but cannot lower the standard.

A rung may narrow the evidence it is shown - `k` - which exists for models whose
context window cannot take the full retrieved set, not as a cost lever. Showing
a *more* expensive rung *less* evidence than the rung that already failed is
almost always a mistake, so `Ladder` says so at construction.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass

from ..providers.base import ChatProvider
from ..types import Tier

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Rung:
    """One model the cascade may try, and what it is allowed to see."""

    name: str
    provider: ChatProvider
    tier: Tier = Tier.FRONTIER
    #: Chunks shown to this rung. None means "everything the cascade retrieved".
    #: Set it only when a model's context window forces the issue.
    k: int | None = None
    #: Output cap. None means the deployment-wide setting.
    max_tokens: int | None = None

    @property
    def model_id(self) -> str:
        return str(getattr(self.provider, "model_id", self.name))

    @property
    def is_free(self) -> bool:
        """True when there is no per-token invoice behind this rung."""
        return bool(getattr(self.provider, "self_hosted", False)) or (
            getattr(self.provider, "tier", "") == "local"
            and getattr(self.provider, "self_hosted", True)
        )


@dataclass(frozen=True, slots=True)
class Ladder:
    """An ordered, cheapest-first sequence of rungs."""

    rungs: tuple[Rung, ...] = ()

    def __post_init__(self) -> None:
        widths = [(r.name, r.k) for r in self.rungs if r.k is not None]
        for (earlier, wide), (later, narrow) in zip(widths, widths[1:], strict=False):
            if narrow < wide:
                log.warning(
                    "ladder rung %r sees %d chunks but %r below it saw %d. A rung is only "
                    "reached because the one below it could not ground an answer, so showing "
                    "it less evidence usually turns an escalation into a refusal.",
                    later,
                    narrow,
                    earlier,
                    wide,
                )

    def __bool__(self) -> bool:
        return bool(self.rungs)

    def __len__(self) -> int:
        return len(self.rungs)

    def __iter__(self) -> Iterator[Rung]:
        return iter(self.rungs)

    def widest_k(self, default: int) -> int:
        """How many chunks to retrieve so every rung can be served from one search.

        Retrieval happens once and every rung reads a prefix of it. Searching per
        rung would let two rungs answer from different evidence, which is exactly
        the property that makes climbing safe to lose.
        """
        return max([default, *[r.k for r in self.rungs if r.k is not None]])

    def describe(self) -> str:
        return " -> ".join(r.name for r in self.rungs) if self.rungs else "(no models configured)"
