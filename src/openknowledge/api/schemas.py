"""Request and response models for the HTTP API."""

from __future__ import annotations

from dataclasses import asdict

from pydantic import BaseModel, Field

from ..types import Answer


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    #: Groups the asker belongs to. The chat surface supplies these from the
    #: identity it already has (Teams tenant groups, SSO claims). ``None`` means
    #: unrestricted and is only appropriate for a single-tenant internal deploy.
    principals: list[str] | None = None
    channel: str | None = None


class CitationOut(BaseModel):
    document_id: str
    document_title: str
    snippet: str
    locator: str | None = None
    url: str | None = None


class ChatResponse(BaseModel):
    answer: str
    tier: str
    model: str
    cost_usd: float
    grounded: bool
    cached: bool
    citations: list[CitationOut]
    notes: list[str] = []
    #: How closely to read this answer, in [0, 1]. Free to compute: every input
    #: is already produced by the gate and the retriever, so it costs no extra
    #: model call. Not a probability - an ordering, where lower deserves a look.
    confidence: float = 1.0
    #: Why it is not 1.0, in words a reader can act on.
    confidence_reasons: list[str] = []

    @classmethod
    def from_answer(cls, answer: Answer) -> ChatResponse:
        return cls(
            answer=answer.text,
            tier=answer.tier.value,
            model=answer.model_id,
            cost_usd=round(answer.cost_usd, 6),
            grounded=answer.grounded,
            cached=answer.tier.is_cache_hit,
            # Citation is a slots dataclass, so it has no __dict__ to splat.
            citations=[CitationOut(**asdict(c)) for c in answer.citations],
            confidence=round(answer.confidence, 3),
            confidence_reasons=list(answer.confidence_reasons),
            notes=list(answer.notes),
        )


class PinRequest(BaseModel):
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    author: str | None = None
    #: Document ids this answer comes from. A pinned answer without provenance
    #: asks the reader to trust it, which is the thing this project avoids.
    cite: list[str] = []
    #: Other phrasings that should resolve to the same answer.
    aliases: list[str] = []


class ReindexResponse(BaseModel):
    documents: int
    chunks: int
    corpus_version: str
    evicted_cache_entries: int


class ReviewRequest(BaseModel):
    reviewer: str | None = None
    note: str | None = None


class ResolveRequest(BaseModel):
    #: The document that is authoritative. Recorded, not enforced - the system
    #: cannot edit your SharePoint for you, and pretending otherwise would be
    #: worse than being explicit about it.
    keep: str | None = None
    note: str | None = None
    reviewer: str | None = None


class LearnRequest(BaseModel):
    max_documents: int | None = None
