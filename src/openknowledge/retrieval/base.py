"""Retrieval interfaces and document chunking."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from ..documents.blocks import Block, BlockKind

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
    #: Structure recovered by the parser: headings, paragraphs, list items and
    #: table rows, each with its heading trail and a real locator. Empty for a
    #: document supplied as bare text, which then falls back to word windows.
    blocks: tuple[Block, ...] = field(default_factory=tuple)

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
    """Split a document into retrievable chunks.

    Uses the parser's structure when there is any, and falls back to overlapping
    word windows when there is not. The structured path matters for accuracy,
    not just tidiness: a window boundary that lands mid-rule produces a chunk
    stating a threshold without the condition attached to it, and the grounding
    gate cannot tell that something is missing - the answer it checks is
    faithful to the chunk it was given.
    """
    if doc.blocks:
        return chunk_blocks(doc, target_words=target_words, overlap_words=overlap_words)
    return _chunk_words(doc, target_words=target_words, overlap_words=overlap_words)


def chunk_blocks(doc: Document, *, target_words: int = 350, overlap_words: int = 60) -> list[Chunk]:
    """Group parsed blocks into chunks along their own structure.

    Three rules, each protecting something the grounding gate depends on:

    * **A heading starts a chunk.** Content under different headings is about
      different things, and merging them lets retrieval return a chunk whose
      heading contradicts half its body.
    * **Atomic blocks are never split.** A table row cut in half is a number
      with no label - precisely the input that makes a numeric claim look
      unsupported, or worse, supported by the wrong figure.
    * **Every chunk carries its heading trail**, so a passage retrieved on its
      own still says what it is about.
    """
    chunks: list[Chunk] = []
    current: list[Block] = []
    current_words = 0
    pending_heading: Block | None = None

    def flush() -> None:
        nonlocal current_words, pending_heading
        if not current:
            return
        text = "\n".join(b.contextual_text for b in current).strip()
        if text:
            locator = next((b.locator for b in current if b.locator), None)
            chunks.append(_make_chunk(doc, len(chunks), text, locator))
        current.clear()
        current_words = 0
        pending_heading = None

    for block in doc.blocks:
        if block.kind is BlockKind.HEADING:
            flush()
            pending_heading = block
            continue

        words = len(block.text.split())

        # An over-long paragraph is the one case that still needs windowing;
        # everything else stays whole.
        if words > target_words and not block.kind.is_atomic:
            flush()
            for piece in _windows(block.text, target_words, overlap_words):
                prefix = " > ".join(block.heading_path)
                text = f"{prefix}: {piece}" if prefix else piece
                chunks.append(_make_chunk(doc, len(chunks), text, block.locator))
            continue

        if current and current_words + words > target_words:
            flush()

        if pending_heading is not None and not current:
            # Keep the heading with the content beneath it: alone it is a fine
            # retrieval signal and a useless answer.
            current.append(pending_heading)
            current_words += len(pending_heading.text.split())
            pending_heading = None

        current.append(block)
        current_words += words

    flush()
    return chunks


def _make_chunk(doc: Document, index: int, text: str, locator: str | None) -> Chunk:
    return Chunk(
        chunk_id=f"{doc.document_id}#{index}",
        document_id=doc.document_id,
        document_title=doc.title,
        text=text,
        locator=locator or f"chunk {index + 1}",
        url=doc.url,
        allowed_principals=doc.allowed_principals,
    )


def _windows(text: str, target_words: int, overlap_words: int) -> list[str]:
    words = text.split()
    step = max(1, target_words - overlap_words)
    pieces: list[str] = []
    for start in range(0, len(words), step):
        window = words[start : start + target_words]
        if not window:
            break
        pieces.append(" ".join(window))
        if start + target_words >= len(words):
            break
    return pieces


def _chunk_words(doc: Document, *, target_words: int = 350, overlap_words: int = 60) -> list[Chunk]:
    """Overlapping word windows, for documents with no recovered structure.

    Overlap exists so a policy sentence that straddles a boundary is still
    retrievable in one piece.
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
