"""What one document contributes to the index, kept so a rebuild reuses it.

Every upload and every delete re-indexes the whole corpus, synchronously,
inside the request. Measured, once parses were cached and PDFs batched and
that was the cost left:

    400 documents  (4,812 chunks)    1.04 s
    1,200 documents (14,412 chunks)  3.42 s
    2,400 documents (28,812 chunks)  7.23 s

Linear, about 3 ms a document, so five thousand policies is fifteen seconds
of waiting after dragging in one file - and the file that changed is one of
them.

Almost none of that work is new. Chunking a document, tokenising its
passages and counting its words depend on **that document alone**; only the
tf-idf ranking behind its tags needs the rest of the corpus, because a word
is distinctive against the documents that never use it. So the per-document
half is remembered here and the corpus-wide half is recomputed every time,
which keeps tags exactly as correct as a full rebuild made them.

**The key is everything a chunk is made of**, not just the text.
``content_hash`` covers ``document.text`` and would have been the obvious
key; it is the wrong one twice over. A chunk carries ``allowed_principals``,
so an entry keyed on text alone could hand back a passage stamped with a
folder's *previous* audience - the worst mistake this product can make. And
chunking reads ``blocks``, not ``text``: a heading rewritten as a paragraph
with the same words leaves the text identical and the chunks different.
Hashing all of it costs 22 ms per 1,200 documents against the seconds it
saves, so there is nothing to trade off.

**Deliberately in memory, not on disk.** This holds the index's own
contents; writing it out would roughly double the stored state to save a
rebuild that now takes a fraction of a second. A restart pays one full
derivation, from parses that are themselves cached.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass, field

from .base import Chunk, Document, chunk_document, tokenize
from .tags import folded_vocabulary, tag_body, tag_sources


@dataclass(frozen=True, slots=True)
class Derived:
    """One document's contribution to the index, in the shapes it is needed."""

    chunks: tuple[Chunk, ...]
    term_freqs: tuple[Counter[str], ...]
    lengths: tuple[int, ...]
    #: How many of this document's chunks each term appears in - what the
    #: corpus-wide ``doc_freq`` is the sum of.
    chunk_frequency: Counter[str]
    #: Its folded vocabulary, which is what a corpus document-frequency counts
    #: one of per document.
    folded_words: frozenset[str]
    #: The tag sources that need no corpus: id, title and headings, folded
    #: form to display form, in the order a reader would trust them.
    tag_sources: tuple[tuple[str, str], ...]
    #: Body word counts, ranked against the corpus at index time.
    tag_body: Counter[str]


def fingerprint(document: Document, *, target_words: int, overlap_words: int) -> str:
    """Everything the derivation reads, in one hash.

    Includes the chunker's own settings: the same document chunked at a
    different window is a different set of passages, and an entry that
    ignored that would survive a settings change it should not have.
    """
    parts = [
        document.document_id,
        document.title,
        document.url or "",
        "1" if document.superseded else "0",
        str(target_words),
        str(overlap_words),
        *sorted(document.allowed_principals),
        document.text,
    ]
    for block in document.blocks:
        parts += [
            block.kind.value,
            block.text,
            block.locator or "",
            str(block.level),
            *block.heading_path,
        ]
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:32]


def derive(document: Document, *, target_words: int, overlap_words: int) -> Derived:
    """Do the per-document work once."""
    chunks = tuple(chunk_document(document, target_words=target_words, overlap_words=overlap_words))
    term_freqs = tuple(Counter(tokenize(chunk.text)) for chunk in chunks)
    chunk_frequency: Counter[str] = Counter()
    for frequencies in term_freqs:
        chunk_frequency.update(frequencies.keys())
    return Derived(
        chunks=chunks,
        term_freqs=term_freqs,
        lengths=tuple(sum(frequencies.values()) for frequencies in term_freqs),
        chunk_frequency=chunk_frequency,
        folded_words=folded_vocabulary(document),
        tag_sources=tag_sources(document),
        tag_body=tag_body(document),
    )


@dataclass
class DerivedCache:
    """Per-document index contributions, by fingerprint. Never wrong, only cold."""

    entries: dict[str, Derived] = field(default_factory=dict)
    hits: int = 0
    misses: int = 0

    def get(self, document: Document, *, target_words: int, overlap_words: int) -> Derived:
        key = fingerprint(document, target_words=target_words, overlap_words=overlap_words)
        found = self.entries.get(key)
        if found is not None:
            self.hits += 1
            return found
        self.misses += 1
        derived = derive(document, target_words=target_words, overlap_words=overlap_words)
        self.entries[key] = derived
        return derived

    def keep_only(self, keys: set[str]) -> int:
        """Forget documents the corpus no longer has in that form.

        Called with everything one whole index build saw. Without it this
        grows by an entry per edit for ever, and each entry holds a copy of
        the document's passages - which is the one thing here big enough to
        matter.
        """
        stale = [key for key in self.entries if key not in keys]
        for key in stale:
            del self.entries[key]
        return len(stale)

    def __len__(self) -> int:
        return len(self.entries)
