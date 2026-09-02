"""A document retired by the one that replaced it.

``declares_superseded`` reads a document's own head for "SUPERSEDED by v4.1"
and trusts it, which is right - the document has already answered the
versioning question about itself. What it needs is for somebody to have gone
back and **edited the old file**, and in practice nobody does. The statement
that gets written is in the *new* document, in its header:

    **Supersedes:** Expenses Policy v3.0 (January 2023)

That line is already in this project's own sample corpus and has been ignored
until now, because the self-declaration matcher deliberately skips
"Supersedes:" - there it is the current copy talking about a *different*
document. Here that is exactly what makes it useful.

**Only an authored declaration, never an inference.** A withdrawal announced
in prose - "the travel policy was withdrawn in March" - is left alone, and
that is a decision rather than an omission. Retrieval does not downrank a
superseded document, it **excludes** it whenever any current document matches
the query (`demote_superseded`), so the cost of getting this wrong is a live
policy going invisible on almost every question. A header field written by
the person who replaced the document is a statement; a sentence in the body
is a guess, and a guess is not worth that.

Three things keep the resolution from hitting the wrong document:

* **the announcer is never its own target**, or a document whose title is a
  prefix of what it names would retire itself;
* **every significant word of the target's title must appear in the named
  phrase**, so a shared word cannot carry a match on its own;
* **the best match must be unique**, and a tie resolves to nothing.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import replace

from ..retrieval.base import Document, tokenize

#: How far into a document a replacement notice is looked for. Headers live at
#: the top; a mention of some other policy on page nine is prose.
HEAD_CHARS = 800

#: Words that carry nothing on their own: grammar, and the version markers a
#: replacement notice is full of.
#:
#: Deliberately NOT the domain words. Stripping "policy" and "guidelines"
#: looked right - almost every title in a policy folder has one - and made
#: the whole thing inert: "Travel Policy" reduced to {travel}, a single word,
#: which the two-word floor below then rejected. Most real titles are exactly
#: "<something> Policy", so that ruled out the common case, and the tests did
#: not notice because their fixtures hit the same wall.
#:
#: They stay, and the floor plus the uniqueness rule do the discriminating:
#: two documents sharing only "policy" both fail the subset test, and two that
#: match equally well are a tie, which retires neither.
_UNDISTINGUISHING = frozenset(
    {"the", "a", "an", "and", "or", "for", "of", "to", "in", "v", "vs", "version", "no"}
)


#: `Supersedes: X`, `**Replaces:** X` - the header field, however markdown
#: has decorated it.
#:
#: **The colon is what makes this a field rather than a sentence**, and it is
#: required. Anchoring to the start of a line would have been the obvious way
#: to say "header" and does not work: the parser merges a metadata block into
#: one paragraph, so by the time this reads the text "**Supersedes:**" sits
#: mid-line after the owner and version. Checked against this project's own
#: sample corpus, where the anchored form matched nothing at all.
_ANNOUNCEMENT = re.compile(
    r"\b(?:supersedes(?:\s+and\s+replaces)?|replaces)\b[\s*_-]*:[\s*_-]*(?P<named>[^\n]{2,140})",
    re.IGNORECASE,
)


def _significant(text: str) -> frozenset[str]:
    return frozenset(w for w in tokenize(text) if w not in _UNDISTINGUISHING and len(w) > 1)


def announced_by(documents: Iterable[Document]) -> dict[str, str]:
    """Per retired document, the id of the one that says it replaced it.

    Empty for a corpus where nobody wrote a replacement notice, which is the
    common case and costs nothing.
    """
    corpus = list(documents)
    found: dict[str, str] = {}
    for announcer in corpus:
        for match in _ANNOUNCEMENT.finditer(announcer.text[:HEAD_CHARS]):
            named = _significant(match.group("named"))
            if not named:
                continue
            target = _resolve(named, corpus, announcer.document_id)
            if target is not None:
                found[target] = announcer.document_id
    return found


def _resolve(named: frozenset[str], corpus: list[Document], announcer: str) -> str | None:
    """Which document the phrase names, or None if that is not clear.

    "Expenses Policy v3.0 (January 2023)" has to reach "Expenses Policy
    (2023)" and not the current "Expenses Policy", both of whose titles fit
    inside it. The more specific title wins - it accounts for more of what was
    written - and a tie means nobody can tell, so nothing happens.
    """
    ranked: list[tuple[int, str]] = []
    for document in corpus:
        if document.document_id == announcer:
            continue
        title = _significant(document.title)
        if len(title) < 2 or not title <= named:
            continue
        ranked.append((len(title), document.document_id))
    if not ranked:
        return None
    best = max(count for count, _ in ranked)
    winners = [doc_id for count, doc_id in ranked if count == best]
    return winners[0] if len(winners) == 1 else None


def apply(documents: list[Document]) -> tuple[list[Document], dict[str, str]]:
    """Mark the documents another document says it replaced.

    Returns the corpus and what was marked, so a scan can report it. A
    document already declaring itself superseded is unchanged and not
    reported: it was not this that retired it.
    """
    announced = announced_by(documents)
    if not announced:
        return documents, {}
    marked = {
        doc.document_id: announced[doc.document_id]
        for doc in documents
        if doc.document_id in announced and not doc.superseded
    }
    if not marked:
        return documents, {}
    return [
        replace(doc, superseded=True) if doc.document_id in marked else doc for doc in documents
    ], marked
