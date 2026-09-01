"""Which documents have any business agreeing with each other.

The detector's largest error, and it is not a threshold that needed tuning:
it assumes every document in the folder speaks for the same authority about
the same world. Run it over fifteen real vendor contracts and it compares
one supplier's uptime commitment against another's, one licence's notice
period against another's. Those documents are not disagreeing. They are
about different agreements with different companies, and no amount of
weighting shared words can see that, because the words really are shared -
contracts are written from the same boilerplate.

The signal is the parties an agreement is **between**. Two contracts naming
disjoint counterparties govern different relationships, so a figure in one
cannot contradict a figure in the other.

Three things keep this from hiding a real contradiction, which is the
failure that matters here - a missed disagreement means somebody is told the
wrong policy with a citation attached:

**Only parties, never mentions.** A name counts when the document puts it in
a party-defining position - "between X and Y", or `X ("the Supplier")` - and
never because it appears somewhere in the prose. So an expenses policy that
happens to name a travel booking company is not thereby scoped to it, and
two versions of that policy are still compared. Policy corpora, which is
what most of this product's users have, get no scope at all and behave
exactly as before.

**Only the head of the document.** Contracts name their parties up front.
A counterparty named in a schedule on page forty is a mention.

**Only names that are not everywhere.** Your own company is a party to every
contract you hold, so a name in half the corpus or more is not what
distinguishes one agreement from another. Dropping it is what makes the
remaining sets disjoint; without it every pair would share your own name and
nothing would ever be scoped apart. It also lands the right way round on the
cases that matter: three contracts with the *same* vendor leave that vendor
in every document, so it is dropped and they are compared - which is correct,
because they really are about one relationship.

And the rule itself is conservative: two documents are compared unless
**both** name parties and they share none. Silence is not a scope.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from ..retrieval.base import Document

#: How far into a document its parties are looked for. An agreement names them
#: in its opening; a name deep in a schedule is a mention.
HEAD_CHARS = 1200

#: The legal forms a company name ends in. Requiring one keeps this from
#: reading an ordinary capitalised phrase as a party. Case-insensitive only
#: here: the patterns below are deliberately case-*sensitive*, because the
#: capital letter a name starts with is doing real work.
_FORMS = (
    "(?i:ltd|limited|llc|inc|incorporated|corp|corporation|plc"
    "|gmbh|ag|nv|bv|sa|sas|spa|a/s|ab|oy|oyj|pty|pte|sarl|srl"
    r"|l\.l\.c\.|n\.v\.|b\.v\.|s\.a\.|s\.a\.s\.|s\.p\.a\.|k\.k\.|s\.r\.l\.)"
)

#: A company name: a capitalised word, at most four more, then the legal form.
#:
#: Both halves of that are load-bearing. The **word cap** stops the role
#: pattern reading backwards across a preamble - "March 2025 by and between
#: Northwind Systems Ltd" is seven words, so it cannot match and the engine
#: moves on to the three that are the name. The **capital letter** is why
#: these patterns do not carry ``re.IGNORECASE``: with it, ``[A-Z]`` matches
#: anything and "by and between Northwind Systems Ltd" becomes a company.
_NAME = r"[A-Z][\w&.'\u2019-]*(?:[ ,]+[\w&.'\u2019-]+){0,4}?[ ,]+" + _FORMS + r"\b"

#: "This Agreement is made by and between Acme Analytics Ltd and Globex Inc."
#: Both sides are captured; either may be your own company, which the
#: most-common-party rule removes rather than this pattern.
_BETWEEN = re.compile(
    r"(?i:between)\s+(?P<left>" + _NAME + r")\.?[\s,]+(?i:and)\s+(?P<right>" + _NAME + r")"
)

#: `Acme Analytics Ltd ("Supplier")` and `Acme Analytics Ltd, a Delaware
#: corporation ("Vendor")` - the definition parenthetical, which is how a
#: contract says which role a named company plays. The optional full stop is
#: for "Inc." and "Ltd.", where the form has already consumed the letters and
#: the abbreviating period is still there.
_ROLE = re.compile(
    r"(?P<name>" + _NAME + r")\.?"
    r"\s*[,(]\s*(?:(?i:a)\s[\w\s]{1,40}?\s)?[(\"\u201c'\u2018]*\s*(?:(?i:the)\s+)?"
    r"(?i:supplier|vendor|customer|client|licensor|licensee|provider|contractor"
    r"|buyer|seller|purchaser|subscriber|processor|controller)\b"
)


def _tidy(name: str) -> str:
    """One spelling for one company, so two mentions can be compared."""
    cleaned = re.sub(r"\s+", " ", name).strip(" ,.;:\"'“”‘’()").strip()
    cleaned = re.sub(r"^(?:the|and)\s+", "", cleaned, flags=re.IGNORECASE)
    return cleaned.casefold().rstrip(".")


def named_parties(document: Document) -> frozenset[str]:
    """The companies this document declares itself to be an agreement between.

    Empty for anything that is not written like an agreement, which is the
    common case and the safe one: a document with no parties is compared with
    everything, exactly as it was before this existed.
    """
    head = document.text[:HEAD_CHARS]
    found: set[str] = set()
    for match in _BETWEEN.finditer(head):
        found.add(_tidy(match.group("left")))
        found.add(_tidy(match.group("right")))
    for match in _ROLE.finditer(head):
        found.add(_tidy(match.group("name")))
    return frozenset(name for name in found if name)


def counterparties(documents: Iterable[Document]) -> dict[str, frozenset[str]]:
    """Per document, the parties that distinguish it from the other agreements.

    Your own company is a party to every contract you hold, so it is the one
    name that tells two of them apart from nothing. It is found rather than
    configured: **the party appearing in the most documents is not a
    counterparty**, and neither is anything tied with it.

    That rule lands the right way round on the cases that decide whether this
    is safe. Fifteen contracts, your company in all fifteen and each vendor in
    one: your company is dropped and the vendors are disjoint, so none of them
    is compared with another. Three contracts with the *same* vendor: that
    vendor is as common as you are, both are dropped, and the three are
    compared - which is correct, because they really are about one
    relationship.

    Nothing is dropped when no name appears twice. There is then no common
    party to remove, and removing the unique ones would erase the only signal
    there was.
    """
    corpus = list(documents)
    parties = {doc.document_id: named_parties(doc) for doc in corpus}
    if not corpus:
        return parties

    appearances: dict[str, int] = {}
    for names in parties.values():
        for name in names:
            appearances[name] = appearances.get(name, 0) + 1
    if not appearances:
        return parties

    most = max(appearances.values())
    ours = {name for name, count in appearances.items() if count == most} if most > 1 else set()

    return {document_id: frozenset(names - ours) for document_id, names in parties.items()}


def comparable(left: frozenset[str], right: frozenset[str]) -> bool:
    """Whether two documents are about the same world at all.

    Only a *confident* difference suppresses: both sides must name parties and
    share none of them. One silent side means no scope was established, and
    an unestablished scope must never be the reason a contradiction goes
    unreported.
    """
    if not left or not right:
        return True
    return bool(left & right)


def out_of_scope_pairs(documents: Iterable[Document]) -> int:
    """How many document pairs were left uncompared, and why nobody has to guess.

    A suppression nobody can see is how a detector quietly stops detecting.
    The scan reports this figure so an admin reading "no contradictions" knows
    whether that means the corpus agrees or that half of it was never
    compared.

    Costs nothing on the corpus most people have: when no document names a
    party there is no scope to apply, and this returns before touching a
    single pair.
    """
    scopes = counterparties(documents)
    if not any(scopes.values()):
        return 0
    document_ids = sorted(scopes)
    return sum(
        1
        for i, left in enumerate(document_ids)
        for right in document_ids[i + 1 :]
        if not comparable(scopes[left], scopes[right])
    )
