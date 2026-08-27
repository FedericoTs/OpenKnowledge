"""Shared result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from .costs import Usage


class Tier(StrEnum):
    """Which stage of the cascade produced an answer.

    Ordered cheapest to most expensive. The whole system is an argument for
    resolving as far up this list as possible.
    """

    PINNED = "pinned"  # human-authored canonical answer; $0, exact
    EXACT_CACHE = "exact"  # byte-identical question seen before; $0
    SEMANTIC_CACHE = "semantic"  # near-identical question; $0
    DRAFT = "draft"  # auto-drafted at ingest, gate-passed, unreviewed; $0
    LOCAL = "local"  # self-hosted model; no per-token invoice
    FRONTIER = "frontier"  # paid API call
    REFUSED = "refused"  # nothing grounded enough to answer with
    CONTESTED = "contested"  # the sources disagree and nobody has resolved it

    @property
    def is_cache_hit(self) -> bool:
        """Answered without calling a model at all - always free.

        REFUSED is deliberately excluded. A refusal can be expensive: it is
        what happens when the escalation tier was called and its answer failed
        the grounding gate, so the tokens were spent and the ledger has to say
        so.
        """
        return self in (
            Tier.PINNED,
            Tier.EXACT_CACHE,
            Tier.SEMANTIC_CACHE,
            Tier.DRAFT,
        )


@dataclass(frozen=True, slots=True)
class Citation:
    """A pointer back into the source corpus.

    Every factual claim in an answer must be traceable to one of these. An
    answer without citations is a guess, and this tool's whole promise is that
    it does not guess about company policy.
    """

    document_id: str
    document_title: str
    snippet: str
    locator: str | None = None  # page number, heading anchor, sheet name...
    url: str | None = None


@dataclass(frozen=True, slots=True)
class Answer:
    """A complete response, with its provenance and its price attached."""

    text: str
    tier: Tier
    model_id: str
    cache_key: str
    citations: tuple[Citation, ...] = ()
    usage: Usage = field(default_factory=Usage)
    cost_usd: float = 0.0
    grounded: bool = True
    escalated_from: Tier | None = None
    notes: tuple[str, ...] = ()

    @property
    def is_answerable(self) -> bool:
        return self.tier not in (Tier.REFUSED, Tier.CONTESTED)

    @property
    def needs_review(self) -> bool:
        """True for an answer a machine drafted that no human has approved."""
        return self.tier is Tier.DRAFT
