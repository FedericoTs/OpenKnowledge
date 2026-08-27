"""Retrieval interfaces and document chunking."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

_WORD_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens. Shared by the index and the grounding check so
    both agree on what a word is."""
    return _WORD_RE.findall(text.lower())


@dataclass(frozen=True, slots=True)
class Document:
    """A source file pulled from a connector."""

    document_id: str
    title: str
    text: str
    url: str | None = None
    #: Connector-supplied ACL principals. Retrieval filters on these so the bot
    #: can never surface a document the asker could not open themselves.
    allowed_principals: frozenset[str] = field(default_factory=frozenset)

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class Chunk:
    """A retrievable window of a document."""

    chunk_id: str
    document_id: str
    document_title: str
    text: str
    locator: str | None = None
    url: str | None = None
    allowed_principals: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    chunk: Chunk
    score: float


@runtime_checkable
class Retriever(Protocol):
    def index(self, documents: list[Document]) -> None: ...

    def search(
        self, query: str, *, k: int = 6, principals: frozenset[str] | None = None
    ) -> list[ScoredChunk]: ...

    @property
    def corpus_version(self) -> str: ...


def chunk_document(
    doc: Document, *, target_words: int = 350, overlap_words: int = 60
) -> list[Chunk]:
    """Split a document into overlapping windows.

    Overlap exists so a policy sentence that straddles a boundary is still
    retrievable in one piece - a rule split across two chunks is how you get an
    answer that quotes half a condition.
    """
    if overlap_words >= target_words:
        raise ValueError("overlap_words must be smaller than target_words")

    words = doc.text.split()
    if not words:
        return []

    chunks: list[Chunk] = []
    step = target_words - overlap_words
    for i, start in enumerate(range(0, len(words), step)):
        window = words[start : start + target_words]
        if not window:
            break
        chunks.append(
            Chunk(
                chunk_id=f"{doc.document_id}#{i}",
                document_id=doc.document_id,
                document_title=doc.title,
                text=" ".join(window),
                locator=f"chunk {i + 1}",
                url=doc.url,
                allowed_principals=doc.allowed_principals,
            )
        )
        if start + target_words >= len(words):
            break
    return chunks
