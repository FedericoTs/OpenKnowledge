"""Cache key derivation.

The key is the whole determinism story, so it is worth being explicit about
what goes into it. An answer is only reusable if *everything that produced it*
is unchanged:

``corpus_version``
    A fingerprint of the indexed documents. When someone updates the expenses
    policy in SharePoint, this changes, and every cached answer derived from
    the old corpus stops being reachable. This is the property that keeps a
    cache from quietly serving last year's rules - stale answers are a bigger
    risk in this product than cache misses are.

``prompt_version``
    The system prompt and answer template. Admins can edit these, and an edited
    prompt produces different answers, so it must break the key.

``policy_version``
    Retrieval and grounding settings (how many chunks, the citation threshold).
    Same reasoning.

``model_id``
    Different models give different answers to the same question.

Because all four are in the hash, "clear the cache" is almost never the right
operation - bumping the relevant version does it precisely and reversibly.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from ..canonical import canonicalize_query

_SEP = b"\x1f"  # ASCII unit separator: cannot occur in the canonical fields


@dataclass(frozen=True, slots=True)
class KeyContext:
    """Everything other than the question that affects the answer."""

    corpus_version: str
    prompt_version: str = "v1"
    policy_version: str = "v1"
    model_id: str = "local"

    def parts(self) -> tuple[str, ...]:
        return (self.corpus_version, self.prompt_version, self.policy_version, self.model_id)


def answer_key(query: str, ctx: KeyContext) -> str:
    """Return the stable cache key for ``query`` under ``ctx``.

    Fields are joined with a separator that cannot appear inside them, so
    ``("ab", "c")`` and ``("a", "bc")`` cannot collide into one key.
    """
    canonical = canonicalize_query(query)
    payload = _SEP.join(part.encode("utf-8") for part in (canonical, *ctx.parts()))
    return hashlib.sha256(payload).hexdigest()


def corpus_fingerprint(document_versions: dict[str, str]) -> str:
    """Fold ``{document_id: content_hash}`` into one corpus version string.

    Sorted before hashing so the fingerprint depends on the corpus contents and
    not on the order the connector happened to enumerate files in - otherwise
    every re-sync would invalidate every cached answer for no reason.
    """
    digest = hashlib.sha256()
    for doc_id in sorted(document_versions):
        digest.update(doc_id.encode("utf-8"))
        digest.update(_SEP)
        digest.update(document_versions[doc_id].encode("utf-8"))
        digest.update(_SEP)
    return digest.hexdigest()[:16]
