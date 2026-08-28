"""Request and response models for the HTTP API."""

from __future__ import annotations

from dataclasses import asdict
from typing import Literal

from pydantic import BaseModel, Field

from ..types import Answer


class Turn(BaseModel):
    """One earlier exchange in this conversation, supplied by the client.

    The server stays stateless about conversations on purpose: the transcript
    lives where the person is (their browser tab, their Teams thread), and the
    answer cache stays keyed on standalone questions. History is only material
    for interpreting a follow-up, never part of the key.
    """

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    #: Recent turns, oldest first. Optional; only consulted when the question
    #: reads as a follow-up ("what about contractors?").
    history: list[Turn] | None = Field(default=None, max_length=20)
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
    #: Share of the answer's content words that appear in the text it cited,
    #: from the grounding gate. A fact about how closely the answer tracks its
    #: sources, not a prediction that it is right. Null where there is no answer.
    support: float | None = None

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
            support=answer.support,
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


class ContactRequest(BaseModel):
    """A submission from the website's contact form.

    Lengths are capped here as well as in `contacts.clean`, so an oversized body
    is rejected before it is parsed rather than after.
    """

    name: str = Field(max_length=200)
    email: str = Field(max_length=320)
    organisation: str = Field(default="", max_length=200)
    interest: str = Field(default="", max_length=60)
    message: str = Field(default="", max_length=4000)
    #: Honeypot. A human never sees this field, so anything in it is a bot -
    #: which is answered with a success it will not benefit from.
    website: str = ""


class ContactResponse(BaseModel):
    received: bool = True
