"""Retrieval interfaces and document chunking."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from ..documents.blocks import Block, BlockKind

_WORD_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens. Shared by the index and the grounding check so
    both agree on what a word is."""
    return _WORD_RE.findall(text.lower())


#: Longest prefix compared when matching words across phrasings. Folding to a
#: prefix beats suffix stripping: a stripper turns "expenses" into "expens"
#: while leaving "expense" alone, so the two stop matching - which is worse
#: than not normalising at all.
_FOLD_PREFIX = 5

#: Never fold these. Negations and modals decide what a rule means, and a
#: normaliser that mangles them could match a claim to its own opposite.
_NEVER_FOLD = frozenset(
    {"is", "was", "has", "does", "goes", "less", "unless", "yes", "this", "its", "as"}
)


def fold_word(word: str) -> str:
    """Reduce a word to a comparable form.

    Strips a plural ending, then truncates, so "expense"/"expenses",
    "submit"/"submitted" and "day"/"days" all land on the same token. Short
    words are left alone - truncating them would collide unrelated terms.

    One folder, used everywhere phrasings are matched - conflict relevance
    and tag routing both - because two subtly different normalisers is how
    "travelling" matches a contested claim but misses the document tagged
    "travel".
    """
    if word in _NEVER_FOLD or len(word) <= 3:
        return word
    if word.endswith("es") and len(word) > 5:
        word = word[:-2]
    elif word.endswith("s") and not word.endswith("ss"):
        word = word[:-1]
    return word[:_FOLD_PREFIX]


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
    #: The document declares itself retired - "SUPERSEDED by v4.1", "Retained
    #: for audit only". Detected from its own head at parse time, never
    #: guessed from dates or folder names. Retrieval demotes these so a stale
    #: archived figure cannot outshout the current policy; the document stays
    #: indexed, listed, and part of conflict detection.
    superseded: bool = False

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
    #: The heading trail this passage sits under, "Leave > Parental leave".
    #: What a person needs to find it again; the locator is what the grounding
    #: gate resolves "chunk 4" against, and the two are not the same job.
    section: str | None = None
    url: str | None = None
    allowed_principals: frozenset[str] = field(default_factory=frozenset)
    #: Inherited from the document, so ranking can see it without a lookup.
    superseded: bool = False


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    chunk: Chunk
    score: float


def demote_superseded(ranked: list[ScoredChunk], k: int) -> list[ScoredChunk]:
    """Prefer documents that still claim to be in force.

    A document that opens by declaring itself superseded has already answered
    the question of whether it should compete with the current copy: it
    should not. Whenever any current document matches the query, chunks from
    self-declared superseded documents are *excluded*, not merely downranked -
    a downranked stale figure still lands in the model's context, and the
    model is then asked to adjudicate a versioning question retrieval already
    knows the answer to. Measured on the Aveline corpus: with both copies
    retrieved, the local model led with the archived CFO limit and disclosed
    the archived meal rate, both of which the golden set forbids.

    Fails open: a query that only the superseded documents match - a corpus
    whose sole document on some topic was retired without replacement - still
    gets them, cited as what they are, rather than a blind refusal.
    """
    current = [s for s in ranked if not s.chunk.superseded]
    if not current:
        return ranked[:k]
    return current[:k]


@runtime_checkable
class Retriever(Protocol):
    """What the cascade needs from a retriever, and nothing more.

    ADR 0004 put lexical search behind this so a dense half could be added
    later without the cascade knowing. Declaring it was not enough - the
    cascade and the engine were annotated with the concrete BM25 class, so the
    first thing to implement the protocol did not type-check. The annotations
    now name this.
    """

    def index(self, documents: list[Document]) -> None: ...

    def search(
        self, query: str, *, k: int = 6, principals: frozenset[str] | None = None
    ) -> list[ScoredChunk]: ...

    def restamp(self, principals: Mapping[str, frozenset[str]]) -> int: ...

    def documents_visible_to(self, principals: frozenset[str] | None) -> tuple[list[str], int]: ...

    def visible_to(self, document_ids: set[str], principals: frozenset[str] | None) -> bool: ...

    def describe_document(self, document_id: str) -> tuple[str, str] | None: ...

    def __len__(self) -> int: ...

    @property
    def corpus_version(self) -> str: ...

    @property
    def document_count(self) -> int: ...


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
        text, section = _passage(current)
        if text:
            locator = next((b.locator for b in current if b.locator), None)
            chunks.append(_make_chunk(doc, len(chunks), text, locator, section))
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
            section = " > ".join(block.heading_path)
            for piece in _windows(block.text, target_words, overlap_words):
                text = f"{section}: {piece}" if section else piece
                chunks.append(_make_chunk(doc, len(chunks), text, block.locator, section or None))
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


def _passage(blocks: list[Block]) -> tuple[str, str | None]:
    """One chunk's text, with its heading trail stated once, and that trail.

    Every block used to be rendered through ``Block.contextual_text``, which
    prefixes each with its own copy of the trail. Under one H1 that put the
    heading in a chunk once as itself and again in front of every paragraph -
    on the four-line sample VPN policy, three times in a three-line chunk.
    Measured on the aveline corpus: 2851 words indexed before, 2436 after.
    **14.6% of what we indexed was one heading, said again**, inflating the
    BM25 term frequency of heading words and spending context on nothing.

    Saying it once keeps what the trail is for - a passage retrieved alone
    still states what it is about - and drops the repetition, which never
    told anything anything it did not already know.
    """
    trail: tuple[str, ...] = ()
    for block in blocks:
        if block.kind is not BlockKind.HEADING and block.heading_path:
            trail = tuple(block.heading_path)
            break

    lines: list[str] = []
    if trail:
        lines.append(" > ".join(trail) + ":")
    for block in blocks:
        # The heading's own line is already the last step of the trail above.
        if block.kind is BlockKind.HEADING and trail and block.text == trail[-1]:
            continue
        # A heading always starts a chunk, so every content block here shares
        # one trail and the line above already states it. Anything that does
        # not share it keeps its own label rather than being filed under a
        # heading it was never under - the invariant is the parser's, and this
        # does not need to depend on it holding.
        own = tuple(block.heading_path)
        lines.append(block.text if not own or own == trail else block.contextual_text)
    return "\n".join(lines).strip(), (" > ".join(trail) if trail else None)


def _make_chunk(
    doc: Document, index: int, text: str, locator: str | None, section: str | None = None
) -> Chunk:
    return Chunk(
        chunk_id=f"{doc.document_id}#{index}",
        document_id=doc.document_id,
        document_title=doc.title,
        text=text,
        locator=locator or f"chunk {index + 1}",
        section=section,
        url=doc.url,
        allowed_principals=doc.allowed_principals,
        superseded=doc.superseded,
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
                superseded=doc.superseded,
            )
        )
        if start + target_words >= len(words):
            break
    return chunks
