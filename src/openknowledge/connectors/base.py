"""Document sources.

A connector's job is narrow: enumerate documents, return their text, and say who
is allowed to read each one. That last part is not optional. An enterprise search
tool that ignores permissions is a data-leak generator, and permissions have to
arrive with the document - reconstructing them later is guesswork.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..retrieval.base import Document


@runtime_checkable
class Connector(Protocol):
    """A source of documents."""

    name: str

    def fetch(self) -> list[Document]:
        """Return every document this connector can currently see.

        Full enumeration rather than a delta feed, because ``corpus_version`` is
        a hash of the whole corpus - which is what makes a deleted document
        actually disappear from answers instead of lingering in the index.
        Incremental sync is a later optimisation (ROADMAP), not a change to this
        contract.
        """
        ...
